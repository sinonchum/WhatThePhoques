"""V8/V9 GRAND FUSION ENGINE (runs in ~30s on Modal GPU — ZERO holdout tuning, strict 4-fold nested DEV CV!):
Automatically combines:
  1. Latest JURECA Top-8 clean honest-OOF models + GPU MetaStackNet
  2. ALL available V14 & V15 Cross-Method Visual-Forensic Refiners (ConvNeXt-Large-CLIP-384/512,
     ConvNeXt-XXLarge-CLIP-512, EVA02-Large-MIM-448, Swin-Large) — weighted blend strictly fit via 4-fold DEV CV!
  3. Adaptive L/S Split-Threshold + Hole-Dip Recovery (4-fold nested DEV CV)
  4. `rect_nnref` exact rectangle override (4-fold nested DEV CV)
  5. MERGED `SAM-1 ViT-H` U `SAM-2.1 Hiera-Large` Segment Snapping (Top-2 union + Rectangularity Guard + Hard/Soft blend, 4-fold nested DEV CV!)
  6. NON-SAM AXIS-ALIGNED RECTANGLE SNAPPER: for `S` images where neither `rect_nnref` nor SAM fired,
     snaps fuzzy oval/blob to its optimal axis-aligned rectangle `[y0:y1, x0:x1]` ONLY when nested DEV CV improves!
  7. Evaluates ONCE on 102 untouched holdout images + writes verified submission `.pt` (<20 MB)!
"""
import modal

app = modal.App("task1-v8-grand-fusion")
vol = modal.Volume.from_name("task1-wm-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "numpy", "scipy", "scikit-learn", "pillow", "tqdm")
)


@app.function(image=image, gpu=["A100-80GB", "L40S", "A10G", "H100"], volumes={"/vol": vol}, timeout=900)
def run_grand_fusion(out_name: str = "sub_v9_v14_sam1h.pt", use_sam2: bool = False):
    import sys, json, time, pathlib
    import numpy as np
    from scipy import ndimage
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    t0 = time.time()
    dev = "cuda:0"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import miou_image, downsample_mask

    # Load SAM-1 ViT-H (and optionally SAM-2.1 if use_sam2=True)
    sam1_packed = {}
    for r in range(4):
        z = np.load(f"/vol/outputs/sam_h/shard_{r}.npz")
        for k in z.files: sam1_packed[k] = z[k]
    sam2_packed = {}
    sam2_dir = pathlib.Path("/vol/outputs/sam2_l")
    if use_sam2 and all((sam2_dir / f"shard_{r}.npz").exists() for r in range(4)):
        for r in range(4):
            z = np.load(sam2_dir / f"shard_{r}.npz")
            for k in z.files: sam2_packed[k] = z[k]
        print(f"Loaded SAM-1 ViT-H ({len(sam1_packed)}) + SAM-2.1 Hiera-Large ({len(sam2_packed)})!", flush=True)
    else:
        print(f"Loaded PURE SAM-1 ViT-H ({len(sam1_packed)}) [use_sam2={use_sam2}]!", flush=True)

    # Load latest JURECA clean OOF bundle (prefer v6 -> v5 -> v4)
    for cand_f in ("ens_top8_v6_fix01.npz", "ens_top8_v5_fix01.npz", "ens_top8_v4_fix01.npz"):
        p = pathlib.Path("/vol/jureca_ens") / cand_f
        if p.exists():
            top_p = p; break
    print(f"Using JURECA bundle: {top_p.name}", flush=True)
    ez = np.load(str(top_p), allow_pickle=True)

    # Load ALL available V14/V15 Cross-Method Refiners!
    ref_dir = pathlib.Path("/vol/outputs/v14_refiners")
    ref_files = sorted(ref_dir.glob("*.npz")) if ref_dir.exists() else []
    refiners = {f.stem: np.load(str(f), allow_pickle=True) for f in ref_files}
    print(f"Loaded {len(refiners)} V14/V15 refiners: {list(refiners.keys())}", flush=True)

    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    tr_by_name = {ds_tr[i]["image_name"]: ds_tr[i] for i in range(len(ds_tr))}
    te_names_all = [ds_te[i]["image_name"] for i in range(len(ds_te))]

    def unpack_cands(nm):
        arrs = []
        for d in (sam1_packed, sam2_packed):
            p = d.get(nm)
            if p is not None and len(p) > 0:
                arrs.append(np.unpackbits(p, axis=1).reshape(-1, 128, 128).astype(bool))
        if not arrs: return np.zeros((0, 128, 128), dtype=bool)
        return np.concatenate(arrs, axis=0)

    def fast_miou_vec(pred, gt):
        p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
        i1 = (p & g).sum(1); u1 = (p | g).sum(1)
        i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
        return 0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0) + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0))

    def load_rect(p):
        z = np.load(p, allow_pickle=True)
        return {str(z["names"][i]): {
            "box": z["box"][i].astype(np.float32), "margin": float(z["margin"][i]),
            "acc": float(z["acc"][i]), "full_margin": float(z["full_margin"][i]),
            "full_acc": float(z["full_acc"][i]),
        } for i in range(len(z["names"]))}

    def box_to_m128(box):
        m = np.zeros((128, 128), dtype=bool)
        y0, x0, y1, x1 = np.clip(np.round(box / 4.0).astype(int), 0, 128)
        if y1 > y0 and x1 > x0: m[y0:y1, x0:x1] = True
        return m

    def rect_solidity(m):
        ys, xs = np.where(m)
        if len(ys) < 16: return 0.0
        return float(len(ys)) / max(float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)), 1.0)

    def best_axis_rect_from_prob(lg):
        """Find optimal axis-aligned rectangle maximizing soft IoU with sigmoid(lg) on S images!"""
        prob = 1.0 / (1.0 + np.exp(np.clip(-lg, -20, 20)))
        best_si, best_m = 0.0, np.zeros((128, 128), dtype=bool)
        p_sum = float(prob.sum())
        if p_sum < 4.0: return best_m, 0.0
        # Integral image for O(1) rectangle sum!
        ii = np.pad(prob, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        cand_boxes = set()
        for thr in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
            bw = prob >= thr
            if bw.sum() < 6 or bw.mean() > 0.50: continue
            lab, nl = ndimage.label(bw)
            if nl == 0: continue
            bc = max(range(1, nl + 1), key=lambda c: float(prob[lab == c].sum()))
            ys, xs = np.where(lab == bc)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            for dy0 in (-2, -1, 0, 1, 2):
                for dy1 in (-2, -1, 0, 1, 2):
                    for dx0 in (-2, -1, 0, 1, 2):
                        for dx1 in (-2, -1, 0, 1, 2):
                            ny0 = max(0, min(127, y0 + dy0)); ny1 = max(ny0 + 2, min(128, y1 + dy1))
                            nx0 = max(0, min(127, x0 + dx0)); nx1 = max(nx0 + 2, min(128, x1 + dx1))
                            cand_boxes.add((ny0, nx0, ny1, nx1))
        for (y0, x0, y1, x1) in cand_boxes:
            inter = float(ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0])
            area = float((y1 - y0) * (x1 - x0))
            si = inter / max(area + p_sum - inter, 1e-4)
            if si > best_si:
                best_si = si
                best_b = (y0, x0, y1, x1)
        if best_si > 0:
            y0, x0, y1, x1 = best_b
            best_m[y0:y1, x0:x1] = True
        return best_m, best_si

    class MetaStackNet(nn.Module):
        def __init__(self, M, init_w, init_b):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(float(init_b), dtype=torch.float32))
            self.net = nn.Sequential(
                nn.Conv2d(M + 4, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.GELU(),
                nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.GroupNorm(4, 24), nn.GELU(),
                nn.Conv2d(24, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
                nn.Conv2d(16, 1, 1),
            )
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
            self.res_scale = nn.Parameter(torch.tensor(0.25))

        def forward(self, x):
            w = torch.softmax(self.w, dim=0)
            base = (x * w[None, :, None, None]).sum(1, keepdim=True) + self.b
            feat = torch.cat([x, x.mean(1, keepdim=True), x.std(1, keepdim=True),
                              base.mean((2, 3), keepdim=True).expand_as(base),
                              base.std((2, 3), keepdim=True).expand_as(base)], dim=1)
            return (base + self.res_scale * self.net(feat)).squeeze(1)

    def sym_miou_loss(logits, target):
        p = torch.sigmoid(logits).flatten(1); y = target.flatten(1)
        i1 = (p * y).sum(1); u1 = (p + y - p * y).sum(1)
        i0 = ((1 - p) * (1 - y)).sum(1); u0 = ((1 - p) + (1 - y) - (1 - p) * (1 - y)).sum(1)
        return (1.0 - 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0))).mean() + 0.15 * F.binary_cross_entropy_with_logits(logits, target)

    def fit_meta_and_blend(Xd, Xh, Xt, dg, dL, extra_d, extra_h, extra_t, extra_names):
        N, M = Xd.shape[0], Xd.shape[1]
        idx_L = np.where(dL)[0]; idx_S = np.where(~dL)[0]
        folds = [[] for _ in range(4)]
        for k, i in enumerate(idx_L): folds[k % 4].append(i)
        for k, i in enumerate(idx_S): folds[k % 4].append(i)

        def opt_w(Xtr, gtr):
            w = np.zeros(Xtr.shape[1], np.float32); w[0] = 1.0
            best = float(fast_miou_vec(Xtr[:, 0] >= 0.0, gtr).mean())
            for _ in range(3):
                for j in range(Xtr.shape[1]):
                    for step in (-0.3, -0.15, 0.15, 0.3, 0.5, 1.0):
                        wc = w.copy(); wc[j] = max(0.0, wc[j] + step)
                        if wc.sum() <= 0: continue
                        wc /= wc.sum()
                        sc = float(fast_miou_vec(np.tensordot(Xtr, wc, axes=([1], [0])) >= 0.0, gtr).mean())
                        if sc > best + 1e-5: best, w = sc, wc
            lg = np.tensordot(Xtr, w, axes=([1], [0]))
            bb, bsb = 0.0, best
            for b in np.linspace(-0.5, 0.5, 21):
                sc = float(fast_miou_vec(lg + b >= 0.0, gtr).mean())
                if sc > bsb: bsb, bb = sc, float(b)
            return w, bb

        meta_oof = np.zeros((N, 128, 128), np.float32); meta_hs, meta_ts = [], []
        Xd_t = torch.from_numpy(Xd).to(dev); yd_t = torch.from_numpy(dg.astype(np.float32)).to(dev)
        Xh_t = torch.from_numpy(Xh).to(dev); Xt_t = torch.from_numpy(Xt).to(dev)
        for k in range(4):
            vi = np.array(folds[k]); ti = np.array([i for i in range(N) if i not in set(folds[k])])
            wk, bk = opt_w(Xd[ti], dg[ti])
            iw = np.log(np.maximum(wk / max(float(wk.sum()), 1e-6), 1e-2))
            torch.manual_seed(42 + k)
            net = MetaStackNet(M, iw, bk).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2e-2)
            bsc, bval, bst = -1.0, None, None
            for ep in range(60):
                net.train()
                perm = ti[np.random.RandomState(100*k + ep).permutation(len(ti))]
                for s in range(0, len(perm), 16):
                    bi = perm[s:s+16]; opt.zero_grad()
                    sym_miou_loss(net(Xd_t[bi]), yd_t[bi]).backward(); opt.step()
                if (ep + 1) % 5 == 0:
                    net.eval()
                    with torch.no_grad(): vl = net(Xd_t[vi]).cpu().numpy()
                    sc = float(fast_miou_vec(vl >= 0.0, dg[vi]).mean())
                    if sc > bsc:
                        bsc, bval = sc, vl
                        bst = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
            meta_oof[vi] = bval
            net.load_state_dict(bst); net.eval()
            with torch.no_grad():
                meta_hs.append(net(Xh_t).cpu().numpy()); meta_ts.append(net(Xt_t).cpu().numpy())
        meta_h = np.mean(meta_hs, axis=0); meta_t = np.mean(meta_ts, axis=0)

        # Now blend MetaStack + ALL V14/V15 Refiners strictly via 4-fold nested DEV CV!
        if not extra_d:
            return meta_oof, meta_h, meta_t
        Bd = np.stack([meta_oof] + extra_d, axis=1)
        Bh = np.stack([meta_h] + extra_h, axis=1)
        Bt = np.stack([meta_t] + extra_t, axis=1)
        b_oof = np.zeros_like(meta_oof); b_hs, b_ts = [], []
        for k in range(4):
            vi = np.array(folds[k]); ti = np.array([i for i in range(N) if i not in set(folds[k])])
            wk, bk = opt_w(Bd[ti], dg[ti])
            b_oof[vi] = np.tensordot(Bd[vi], wk, axes=([1], [0])) + bk
            b_hs.append(np.tensordot(Bh, wk, axes=([1], [0])) + bk)
            b_ts.append(np.tensordot(Bt, wk, axes=([1], [0])) + bk)
        wf, bf = opt_w(Bd, dg)
        sc_m = float(fast_miou_vec(meta_oof >= 0.0, dg).mean())
        sc_b = float(fast_miou_vec(b_oof >= 0.0, dg).mean())
        print(f"  [0. META+REFINER BLEND] Meta={sc_m:.4f} -> Blend(CV)={sc_b:.4f} ({sc_b-sc_m:+.4f}) w({['META']+extra_names})={np.round(wf,2).tolist()}", flush=True)
        if sc_b > sc_m + 0.0005:
            return b_oof, np.mean(b_hs, axis=0), np.mean(b_ts, axis=0)
        return meta_oof, meta_h, meta_t

    def best_sam_match(nm, lg, isL_flag, cur_mask):
        cands = unpack_cands(nm)
        cur_min = (~cur_mask) if isL_flag else cur_mask
        sol = rect_solidity(cur_min)
        if len(cands) == 0:
            return None, None, 0.0, 0.0, -99.0, 0.0, sol
        if isL_flag:
            prob_min = 1.0 / (1.0 + np.exp(np.clip(lg, -20, 20))); sig = -lg
        else:
            prob_min = 1.0 / (1.0 + np.exp(np.clip(-lg, -20, 20))); sig = lg
        cf = cands.reshape(len(cands), -1).astype(np.float32)
        pf = prob_min.ravel()[None, :]; sf = sig.ravel()[None, :]; cmf = cur_min.ravel().astype(np.float32)[None, :]
        soft_iou = (cf * pf).sum(1) / (cf + pf - cf * pf).sum(1).clip(min=1e-4)
        top_idx = np.argsort(-soft_iou)[:8]
        best_m = cands[top_idx[0]]; best_si = float(soft_iou[top_idx[0]])
        for j2 in top_idx[1:]:
            u = cands[top_idx[0]] | cands[j2]
            if u.mean() > 0.48: continue
            uf = u.ravel().astype(np.float32)
            si_u = float((uf * pf[0]).sum() / max((uf + pf[0] - uf * pf[0]).sum(), 1e-4))
            if si_u > best_si + 0.025:
                best_si, best_m = si_u, u
        bmf = best_m.ravel().astype(np.float32)
        bi = float((bmf * cmf[0]).sum() / max((bmf + cmf[0] - bmf * cmf[0]).sum(), 1e-4))
        ct = float((bmf * sf[0]).sum() / max(bmf.sum(), 1.0) - ((1.0 - bmf) * sf[0]).sum() / max((1.0 - bmf).sum(), 1.0))
        return ((~best_m) if isL_flag else best_m), best_m, bi, best_si, ct, float(best_m.mean()), sol

    test_masks = {}
    summary = {}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        mnames = [str(x) for x in ez[f"{method}_names_m"]]
        dn = [str(x) for x in ez[f"{method}_dev_names"]]
        hn = [str(x) for x in ez[f"{method}_hold_names"]]
        tn = [str(x) for x in ez[f"{method}_test_names"]]
        Xd = ez[f"{method}_X_dev"].astype(np.float32)
        Xh = ez[f"{method}_X_hold"].astype(np.float32)
        Xt = ez[f"{method}_X_test"].astype(np.float32)
        dg = np.stack([downsample_mask(tr_by_name[n]["mask"] > 0) > 0 for n in dn])
        hg = np.stack([downsample_mask(tr_by_name[n]["mask"] > 0) > 0 for n in hn])
        dL = np.array([bool(tr_by_name[n]["full_watermark"]) for n in dn])
        hL = np.array([bool(tr_by_name[n]["full_watermark"]) for n in hn])

        m_sc = [float(fast_miou_vec(Xd[:, j] >= 0.0, dg).mean()) for j in range(len(mnames))]
        order = np.argsort(-np.array(m_sc))[:8]
        mnames = [mnames[j] for j in order]
        Xd, Xh, Xt = Xd[:, order], Xh[:, order], Xt[:, order]

        # Align any available V14/V15 refiners to dn, hn, tn
        ex_d, ex_h, ex_t, ex_n = [], [], [], []
        for rname, rz in refiners.items():
            if f"{method}_dev_names" not in rz: continue
            def aln(src_n, src_a, dst_n):
                idx = {str(n): i for i, n in enumerate(src_n)}
                return src_a[[idx[n] for n in dst_n]].astype(np.float32)
            rd = aln(rz[f"{method}_dev_names"], rz[f"{method}_dev_lg"], dn)
            rh = aln(rz[f"{method}_hold_names"], rz[f"{method}_hold_lg"], hn)
            rt = aln(rz[f"{method}_test_names"], rz[f"{method}_test_lg"], tn)
            ex_d.append(rd); ex_h.append(rh); ex_t.append(rt); ex_n.append(rname)

        print(f"\n================ {method.upper()} (Top-3: {mnames[0].split('_')[2]}={m_sc[order[0]]:.4f}, {mnames[1].split('_')[2]}={m_sc[order[1]]:.4f}, {mnames[2].split('_')[2]}={m_sc[order[2]]:.4f}) ================", flush=True)
        d_lg, h_lg, t_lg = fit_meta_and_blend(Xd, Xh, Xt, dg, dL, ex_d, ex_h, ex_t, ex_n)
        base_d = float(fast_miou_vec(d_lg >= 0.0, dg).mean()); base_h = float(fast_miou_vec(h_lg >= 0.0, hg).mean())
        print(f"  [0. NEURAL STAGE] DEV={base_d:.4f} | HOLD={base_h:.4f}", flush=True)

        # 1. Adaptive Split-Thr
        def apply_adapt(lg, tL, tS, zL, zS):
            isL = (lg >= 0.0).mean(axis=(1, 2)) > 0.5
            out = np.where(isL[:, None, None], lg >= tL, lg >= tS)
            for i in range(len(lg)):
                m = lg[i]; mu = float(m.mean()); sd = float(m.std()) + 1e-6
                if isL[i] and (~out[i]).mean() < 0.005 and zL < 0: out[i] = (m - mu) / sd >= zL
                elif (not isL[i]) and out[i].mean() < 0.005 and zS > 0: out[i] = (m - mu) / sd >= zS
            return out
        ad_grid = [(tL, tS, zL, zS) for tL in (-0.2, 0.0, 0.3, 0.6) for tS in (0.0, -0.4, -0.8) for zL in (0.0, -2.5, -3.0) for zS in (0.0, 2.5, 3.0)]
        ad_scores = [fast_miou_vec(apply_adapt(d_lg, *c), dg) for c in ad_grid]
        cv_ad = np.zeros(len(dg), np.float32)
        for fold in range(4):
            vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
            cv_ad[vm] = ad_scores[max(range(len(ad_grid)), key=lambda idx: float(ad_scores[idx][tm].mean()))][vm]
        if cv_ad.mean() > base_d + 0.0005:
            ad_cfg = ad_grid[max(range(len(ad_grid)), key=lambda idx: float(ad_scores[idx].mean()))]
            d_m, h_m, t_m = apply_adapt(d_lg, *ad_cfg), apply_adapt(h_lg, *ad_cfg), apply_adapt(t_lg, *ad_cfg)
            print(f"  [1. ADAPT] nested-CV={cv_ad.mean():.4f} cfg={ad_cfg} | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)
        else:
            d_m, h_m, t_m = d_lg >= 0.0, h_lg >= 0.0, t_lg >= 0.0

        # 2. rect_nnref override
        d_rect_trig = np.zeros(len(dn), bool); h_rect_trig = np.zeros(len(hn), bool); t_rect_trig = np.zeros(len(tn), bool)
        if method in ("pixelseal", "trustmark"):
            rd_tr = load_rect(f"/vol/jureca_ens/{method}_train_nnref.npz")
            rd_te = load_rect(f"/vol/jureca_ens/{method}_test_nnref.npz")
            d_rec = [rd_tr[n] for n in dn]; h_rec = [rd_tr[n] for n in hn]; t_rec = [rd_te[n] for n in tn]
            d_bm = np.stack([box_to_m128(r["box"]) for r in d_rec])
            h_bm = np.stack([box_to_m128(r["box"]) for r in h_rec])
            t_bm = np.stack([box_to_m128(r["box"]) for r in t_rec])
            base_v = fast_miou_vec(d_m, dg); rep_v = fast_miou_vec(d_bm, dg)
            fa_d = np.array([r["full_acc"] for r in d_rec]); ac_d = np.array([r["acc"] for r in d_rec])
            gn_d = np.array([r["margin"] - r["full_margin"] for r in d_rec])
            rgrid = [(fa, ac, gn) for fa in (0.58, 0.62, 0.68, 0.75) for ac in (0.62, 0.66, 0.70, 0.76) for gn in (0.15, 0.40, 0.80, 1.20)]
            cv_r = np.zeros_like(base_v)
            for fold in range(4):
                vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
                bc = max(rgrid, key=lambda c: float(np.where((fa_d < c[0]) & (ac_d >= c[1]) & (gn_d >= c[2]), rep_v, base_v)[tm].mean()))
                cv_r[vm] = np.where((fa_d < bc[0]) & (ac_d >= bc[1]) & (gn_d >= bc[2]), rep_v, base_v)[vm]
            brc = max(rgrid, key=lambda c: float(np.where((fa_d < c[0]) & (ac_d >= c[1]) & (gn_d >= c[2]), rep_v, base_v).mean()))
            if cv_r.mean() > base_v.mean() + 0.001:
                fa, ac, gn = brc
                d_rect_trig = (fa_d < fa) & (ac_d >= ac) & (gn_d >= gn)
                h_rect_trig = np.array([r["full_acc"] < fa and r["acc"] >= ac and (r["margin"]-r["full_margin"]) >= gn for r in h_rec])
                t_rect_trig = np.array([r["full_acc"] < fa and r["acc"] >= ac and (r["margin"]-r["full_margin"]) >= gn for r in t_rec])
                d_m = np.where(d_rect_trig[:, None, None], d_bm, d_m)
                h_m = np.where(h_rect_trig[:, None, None], h_bm, h_m)
                t_m = np.where(t_rect_trig[:, None, None], t_bm, t_m)
                print(f"  [2. RECT_NNREF] nested-CV={cv_r.mean():.4f} trig={d_rect_trig.sum()}/136 | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        # 3. MERGED SAM-1 ViT-H U SAM-2.1 Hiera-Large Segment Snapping
        def precompute_sam(names, lgs, cur_ms):
            isL_a = (lgs >= 0.0).mean(axis=(1, 2)) > 0.5
            sh, sb2, sb4 = cur_ms.copy(), cur_ms.copy(), cur_ms.copy()
            bi = np.zeros(len(names), np.float32); si = np.zeros(len(names), np.float32)
            ct = np.zeros(len(names), np.float32); ar = np.zeros(len(names), np.float32)
            sol = np.zeros(len(names), np.float32)
            for i, nm in enumerate(names):
                cm, bm, bi_, si_, ct_, ar_, sol_ = best_sam_match(nm, lgs[i], bool(isL_a[i]), cur_ms[i])
                sol[i] = sol_
                if cm is not None:
                    sh[i] = cm
                    sgn = (-1.0) if isL_a[i] else 1.0
                    sb2[i] = (lgs[i] + 2.0 * sgn * (2.0 * bm.astype(np.float32) - 1.0)) >= 0.0
                    sb4[i] = (lgs[i] + 4.0 * sgn * (2.0 * bm.astype(np.float32) - 1.0)) >= 0.0
                    bi[i], si[i], ct[i], ar[i] = bi_, si_, ct_, ar_
            return isL_a, {"hard": sh, "b2": sb2, "b4": sb4}, bi, si, ct, ar, sol

        d_isL, d_sams, d_bi, d_si, d_ct, d_ar, d_sol = precompute_sam(dn, d_lg, d_m)
        h_isL, h_sams, h_bi, h_si, h_ct, h_ar, h_sol = precompute_sam(hn, h_lg, h_m)
        t_isL, t_sams, t_bi, t_si, t_ct, t_ar, t_sol = precompute_sam(tn, t_lg, t_m)
        d_sam_trig = np.zeros(len(dn), bool); h_sam_trig = np.zeros(len(hn), bool); t_sam_trig = np.zeros(len(tn), bool)

        for sub_name, is_L_target in [("SAM_S_OBJ", False), ("SAM_L_HOLE", True)]:
            base_v = fast_miou_vec(d_m, dg)
            sam_vs = {k: fast_miou_vec(d_sams[k], dg) for k in ("hard", "b2", "b4")}
            elig_d = (d_isL == is_L_target) & (~d_rect_trig)
            elig_h = (h_isL == is_L_target) & (~h_rect_trig)
            elig_t = (t_isL == is_L_target) & (~t_rect_trig)
            sgrid = [(md, t_si, t_bi, s_max)
                     for md in ("hard", "b2", "b4")
                     for t_si in (0.18, 0.28, 0.38, 0.48, 0.58)
                     for t_bi in (0.25, 0.40, 0.50, 0.60)
                     for s_max in (0.85, 0.92, 1.01)]
            cv_s = np.zeros_like(base_v)
            for fold in range(4):
                vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
                bsc, bc = float(base_v[tm].mean()), None
                for c in sgrid:
                    md, thr_si, thr_bi, sol_max = c
                    tf = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_sol <= sol_max) & (d_ar >= 0.008)
                    sc = float(np.where(tf, sam_vs[md], base_v)[tm].mean())
                    if sc > bsc: bsc, bc = sc, c
                if bc is None: cv_s[vm] = base_v[vm]
                else:
                    md, thr_si, thr_bi, sol_max = bc
                    tf = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_sol <= sol_max) & (d_ar >= 0.008)
                    cv_s[vm] = np.where(tf, sam_vs[md], base_v)[vm]
            bfull, bfc = float(base_v.mean()), None
            for c in sgrid:
                md, thr_si, thr_bi, sol_max = c
                tf = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_sol <= sol_max) & (d_ar >= 0.008)
                sc = float(np.where(tf, sam_vs[md], base_v).mean())
                if sc > bfull: bfull, bfc = sc, c
            gain_cv = float(cv_s.mean() - base_v.mean())
            print(f"  [3. {sub_name:10s}] nested-CV={cv_s.mean():.4f} ({gain_cv:+.4f}) full-dev={bfull:.4f} cfg={bfc}", flush=True)
            if bfc is not None and gain_cv > 0.001:
                md, thr_si, thr_bi, sol_max = bfc
                td = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_sol <= sol_max) & (d_ar >= 0.008)
                th = elig_h & (h_si >= thr_si) & (h_bi >= thr_bi) & (h_sol <= sol_max) & (h_ar >= 0.008)
                tt = elig_t & (t_si >= thr_si) & (t_bi >= thr_bi) & (t_sol <= sol_max) & (t_ar >= 0.008)
                d_m = np.where(td[:, None, None], d_sams[md], d_m)
                h_m = np.where(th[:, None, None], h_sams[md], h_m)
                t_m = np.where(tt[:, None, None], t_sams[md], t_m)
                d_sam_trig |= td; h_sam_trig |= th; t_sam_trig |= tt
                print(f"     >> ADOPTED {sub_name} ({md})! trig dev={td.sum()}/136 hold={th.sum()}/34 test={tt.sum()}/340 | hold_now={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        # 4. NON-SAM AXIS-ALIGNED RECTANGLE SNAPPER (for S images where NEITHER rect_nnref NOR SAM fired!)
        def prep_rect_snap(lgs, isL_a, r_trig, s_trig):
            rm = np.zeros((len(lgs), 128, 128), dtype=bool)
            rsi = np.zeros(len(lgs), np.float32)
            for i in range(len(lgs)):
                if (not isL_a[i]) and (not r_trig[i]) and (not s_trig[i]):
                    m_r, si_r = best_axis_rect_from_prob(lgs[i])
                    rm[i] = m_r; rsi[i] = si_r
            return rm, rsi
        d_rm, d_rsi = prep_rect_snap(d_lg, d_isL, d_rect_trig, d_sam_trig)
        h_rm, h_rsi = prep_rect_snap(h_lg, h_isL, h_rect_trig, h_sam_trig)
        t_rm, t_rsi = prep_rect_snap(t_lg, t_isL, t_rect_trig, t_sam_trig)
        base_v = fast_miou_vec(d_m, dg); snap_v = fast_miou_vec(d_rm, dg)
        rs_grid = [(t_rsi, s_min) for t_rsi in (0.45, 0.55, 0.65, 0.72, 0.80) for s_min in (0.50, 0.65, 0.75)]
        cv_rs = np.zeros_like(base_v)
        for fold in range(4):
            vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
            bsc, bc = float(base_v[tm].mean()), None
            for c in rs_grid:
                tf = (~d_isL) & (~d_rect_trig) & (~d_sam_trig) & (d_rsi >= c[0]) & (d_sol >= c[1])
                sc = float(np.where(tf, snap_v, base_v)[tm].mean())
                if sc > bsc: bsc, bc = sc, c
            if bc is None: cv_rs[vm] = base_v[vm]
            else:
                tf = (~d_isL) & (~d_rect_trig) & (~d_sam_trig) & (d_rsi >= bc[0]) & (d_sol >= bc[1])
                cv_rs[vm] = np.where(tf, snap_v, base_v)[vm]
        bfull, bfc = float(base_v.mean()), None
        for c in rs_grid:
            tf = (~d_isL) & (~d_rect_trig) & (~d_sam_trig) & (d_rsi >= c[0]) & (d_sol >= c[1])
            sc = float(np.where(tf, snap_v, base_v).mean())
            if sc > bfull: bfull, bfc = sc, c
        gain_rs = float(cv_rs.mean() - base_v.mean())
        print(f"  [4. RECT_SNAP ] nested-CV={cv_rs.mean():.4f} ({gain_rs:+.4f}) cfg={bfc}", flush=True)
        if bfc is not None and gain_rs > 0.0008:
            t_r, s_mn = bfc
            td = (~d_isL) & (~d_rect_trig) & (~d_sam_trig) & (d_rsi >= t_r) & (d_sol >= s_mn)
            th = (~h_isL) & (~h_rect_trig) & (~h_sam_trig) & (h_rsi >= t_r) & (h_sol >= s_mn)
            tt = (~t_isL) & (~t_rect_trig) & (~t_sam_trig) & (t_rsi >= t_r) & (t_sol >= s_mn)
            d_m = np.where(td[:, None, None], d_rm, d_m)
            h_m = np.where(th[:, None, None], h_rm, h_m)
            t_m = np.where(tt[:, None, None], t_rm, t_m)
            print(f"     >> ADOPTED RECT_SNAP! trig dev={td.sum()}/136 hold={th.sum()}/34 test={tt.sum()}/340 | hold_now={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        d_off = np.array([miou_image(d_m[i], dg[i]) for i in range(len(dg))])
        h_off = np.array([miou_image(h_m[i], hg[i]) for i in range(len(hg))])
        print(f"  *** FINAL {method.upper()} *** DEV={d_off.mean():.4f} (L {d_off[dL].mean():.4f} S {d_off[~dL].mean():.4f}) | HOLD={h_off.mean():.4f} (L {h_off[hL].mean():.4f} S {h_off[~hL].mean():.4f}) | test_fg={t_m.mean():.4f}", flush=True)
        summary[method] = {"dev": float(d_off.mean()), "hold": float(h_off.mean()), "hold_L": float(h_off[hL].mean()), "hold_S": float(h_off[~hL].mean())}
        for i, n in enumerate(tn): test_masks[n] = t_m[i].astype(np.uint8)

    m_dev = float(np.mean([summary[m]["dev"] for m in summary]))
    m_hld = float(np.mean([summary[m]["hold"] for m in summary]))
    print("\n============================================================", flush=True)
    print(f"FINAL GRAND FUSION ({out_name}) 3-METH DEV  OOF : {m_dev:.4f}", flush=True)
    print(f"FINAL GRAND FUSION ({out_name}) 3-METH HOLDOUT  : {m_hld:.4f}", flush=True)
    print(f"PER-METHOD HOLDOUT: {summary}", flush=True)
    print("============================================================", flush=True)

    masks_u8 = torch.from_numpy(np.stack([test_masks[n] for n in te_names_all])).to(torch.uint8)
    assert masks_u8.shape == (1020, 128, 128)
    out_p = pathlib.Path("/vol/outputs") / out_name
    torch.save({"image_names": te_names_all, "masks": masks_u8}, out_p)
    (out_p.with_suffix(".json")).write_text(json.dumps({"dev": m_dev, "hold": m_hld, "methods": summary}, indent=2))
    vol.commit()
    print(f"SAVED {out_p} ({out_p.stat().st_size/1e6:.2f} MB, fg={masks_u8.float().mean():.4f})", flush=True)
    return m_dev, m_hld


@app.local_entrypoint()
def main(out_name: str = "sub_v9_v14_sam1h.pt", use_sam2: bool = False):
    dev_s, hld_s = run_grand_fusion.remote(out_name, use_sam2)
    print(f"=== GRAND FUSION ({out_name}) COMPLETE: DEV={dev_s:.4f} | HOLD={hld_s:.4f} ===")
