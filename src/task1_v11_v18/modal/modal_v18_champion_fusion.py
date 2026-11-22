"""V18 CHAMPION GRAND FUSION ENGINE (Strict 4-Fold Nested DEV CV — ZERO Holdout Leakage!):
Combines EVERY breakthrough from the hackathon:
  1. Stage 0: All 41 clean JURECA models (MetaStack) + ALL 5/6 Modal Monster Refiners
     (`v14_cnxl_clip384`, `v15_cnxl_clip512`, `v16_cnxl_occ512`, `v17_dinov2l_ms_apex518`, `v17_cnxxl_846m_apex512`)
     Evaluates BOTH:
       (a) Weighted Linear Blend (strict 4-fold nested DEV CV)
       (b) 2nd-Stage Spatial MetaStacker-V3 over all Refiners + TruFor/Noiseprint++ + MMFusion + DenseOcc + SAM-1!
     Picks strictly whichever wins honest 4-fold DEV OOF!
  2. Stage 1: Adaptive L/S Split-Threshold + Hole-Dip Recovery (strict 4-fold nested DEV CV)
  3. Stage 2: `rect_nnref` exact white-box rectangle override (strict 4-fold nested DEV CV)
  4. Stage 3: JOINT `SAM-1 ViT-H` (base U V17-reprompted!) vs `AXIS-RECT` ARBITRATION!
     - Never lets SAM overwrite a PixelSeal/StegaStamp rectangle when an axis-aligned rectangle fits P_nn better!
     - Supports Hard-Snap + Soft-Blend (b2, b4) for both S-object AND L-hole!
  5. Stage 4: Connected-Component Tiny-Speck Cleaning (removes tiny isolated noise specks < K px, nested DEV CV!)
  6. Evaluates ONCE on untouched 102 holdout + saves verified submission `.pt` (<20 MB)!
"""
import modal

app = modal.App("task1-v18-champion-fusion")
vol = modal.Volume.from_name("task1-wm-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "numpy", "scipy", "scikit-learn", "pillow", "tqdm")
)


@app.function(image=image, gpu=["A100-80GB", "L40S", "A10G", "H100"], volumes={"/vol": vol}, timeout=900)
def run_champion_fusion(out_name: str = "sub_v18_champion.pt"):
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

    # 1. Load PURE SAM-1 ViT-H + V17-RePrompted SAM-1 ViT-H (if ready!)
    sam1_packed = {}
    for r in range(4):
        z = np.load(f"/vol/outputs/sam_h/shard_{r}.npz")
        for k in z.files: sam1_packed[k] = z[k]
    sam1_v17_packed = {}
    v17_sdir = pathlib.Path("/vol/outputs/sam_h_v17")
    if all((v17_sdir / f"shard_{r}.npz").exists() for r in range(4)):
        for r in range(4):
            z = np.load(v17_sdir / f"shard_{r}.npz")
            for k in z.files: sam1_v17_packed[k] = z[k]
        print(f"Loaded SAM-1 ViT-H ({len(sam1_packed)}) + V17-RePrompted SAM-1 ({len(sam1_v17_packed)})!", flush=True)
    else:
        print(f"Loaded PURE SAM-1 ViT-H ({len(sam1_packed)})!", flush=True)

    # 2. Load Forensics + DenseOcc for Spatial MetaStacker-V3! (load arrays ONCE!)
    occ_by_name = {}
    for m in ("pixelseal", "trustmark"):
        z = np.load(f"/vol/outputs/dense_occ/{m}.npz", allow_pickle=True)
        zn = z["names"]; zf = z["fine"].astype(np.float32); zc = z["coarse"].astype(np.float32)
        for i, nm in enumerate(zn):
            f = zf[i]; c = zc[i]
            occ_by_name[str(nm)] = (
                np.clip((f - f.mean()) / (f.std() + 1e-5), -4, 4) / 2.0,
                np.clip((c - c.mean()) / (c.std() + 1e-5), -4, 4) / 2.0,
            )
    tf_z = np.load("/vol/outputs/forensics/trufor_1530.npz", allow_pickle=True)
    tf_n = tf_z["names"]; tf_m = tf_z["map"].astype(np.float32); tf_c = tf_z["conf"].astype(np.float32)
    tf_by_name = {str(tf_n[i]): (tf_m[i]*2-1, tf_c[i]*2-1) for i in range(len(tf_n))}
    mm_z = np.load("/vol/outputs/forensics/mmfusion_1530.npz", allow_pickle=True)
    mm_n = mm_z["names"]; mm_m = mm_z["map"].astype(np.float32)
    mm_by_name = {str(mm_n[i]): mm_m[i]*2-1 for i in range(len(mm_n))}

    # 3. Load latest JURECA clean OOF bundle + ALL V14/V15/V16/V17 refiners!
    for cf in ("ens_top8_v7_fix01.npz", "ens_top8_v6_fix01.npz", "ens_top8_v5_fix01.npz", "ens_top8_v4_fix01.npz"):
        p = pathlib.Path("/vol/jureca_ens") / cf
        if p.exists(): top_p = p; break
    ez = np.load(str(top_p), allow_pickle=True)
    ref_dir = pathlib.Path("/vol/outputs/v14_refiners")
    ref_files = sorted([f for f in ref_dir.glob("*.npz") if "swinl" not in f.stem])
    refiners = {f.stem: np.load(str(f), allow_pickle=True) for f in ref_files}
    print(f"Bundle={top_p.name} | Loaded {len(refiners)} monster refiners: {list(refiners.keys())}", flush=True)

    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    tr_by_name = {ds_tr[i]["image_name"]: ds_tr[i] for i in range(len(ds_tr))}
    te_names_all = [ds_te[i]["image_name"] for i in range(len(ds_te))]

    def unpack_cands(nm):
        arrs = []
        for d in (sam1_packed, sam1_v17_packed):
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
        prob = 1.0 / (1.0 + np.exp(np.clip(-lg, -20, 20)))
        best_si, best_m = 0.0, np.zeros((128, 128), dtype=bool)
        p_sum = float(prob.sum())
        if p_sum < 4.0: return best_m, 0.0
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
                best_si = si; best_b = (y0, x0, y1, x1)
        if best_si > 0:
            y0, x0, y1, x1 = best_b
            best_m[y0:y1, x0:x1] = True
        return best_m, best_si

    def clean_specks(masks, isL_a, min_px):
        if min_px <= 0: return masks
        out = masks.copy()
        for i in range(len(masks)):
            m = out[i]
            minor = (~m) if isL_a[i] else m
            lab, nl = ndimage.label(minor)
            if nl <= 1: continue
            sizes = ndimage.sum(minor, lab, range(1, nl + 1))
            max_sz = float(np.max(sizes))
            keep = np.zeros_like(minor)
            for c_i, sz in enumerate(sizes, 1):
                if sz >= min_px or sz == max_sz:
                    keep |= (lab == c_i)
            out[i] = (~keep) if isL_a[i] else keep
        return out

    class MetaStackNet(nn.Module):
        def __init__(self, M, init_w, init_b, extra_ch=0):
            super().__init__()
            self.M = M
            self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(float(init_b), dtype=torch.float32))
            in_ch = M + extra_ch + 4
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.GELU(),
                nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.GroupNorm(4, 24), nn.GELU(),
                nn.Conv2d(24, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
                nn.Conv2d(16, 1, 1),
            )
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
            self.res_scale = nn.Parameter(torch.tensor(0.25))

        def forward(self, x_all):
            xm = x_all[:, :self.M]
            w = torch.softmax(self.w, dim=0)
            base = (xm * w[None, :, None, None]).sum(1, keepdim=True) + self.b
            feat = torch.cat([x_all, xm.mean(1, keepdim=True), xm.std(1, keepdim=True),
                              base.mean((2, 3), keepdim=True).expand_as(base),
                              base.std((2, 3), keepdim=True).expand_as(base)], dim=1)
            return (base + self.res_scale * self.net(feat)).squeeze(1)

    def sym_miou_loss(logits, target):
        p = torch.sigmoid(logits).flatten(1); y = target.flatten(1)
        i1 = (p * y).sum(1); u1 = (p + y - p * y).sum(1)
        i0 = ((1 - p) * (1 - y)).sum(1); u0 = ((1 - p) + (1 - y) - (1 - p) * (1 - y)).sum(1)
        return (1.0 - 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0))).mean() + 0.15 * F.binary_cross_entropy_with_logits(logits, target)

    def opt_linear_w(Xtr, gtr):
        M = Xtr.shape[1]
        scs = [float(fast_miou_vec(Xtr[:, j] >= 0.0, gtr).mean()) for j in range(M)]
        w = np.zeros(M, np.float32); w[int(np.argmax(scs))] = 1.0
        best = max(scs)
        for _ in range(2):
            for j in range(M):
                for step in (-0.25, 0.25, 0.5, 1.0):
                    wc = w.copy(); wc[j] = max(0.0, wc[j] + step)
                    if wc.sum() <= 0: continue
                    wc /= wc.sum()
                    sc = float(fast_miou_vec(np.tensordot(Xtr, wc, axes=([1], [0])) >= 0.0, gtr).mean())
                    if sc > best + 1e-5: best, w = sc, wc
        lg = np.tensordot(Xtr, w, axes=([1], [0]))
        bb, bsb = 0.0, best
        for b in np.linspace(-0.4, 0.4, 9):
            sc = float(fast_miou_vec(lg + b >= 0.0, gtr).mean())
            if sc > bsb: bsb, bb = sc, float(b)
        return w, bb

    def run_stack_cv(Xd, Xh, Xt, dg, folds, extra_d=None, extra_h=None, extra_t=None, epochs=60):
        N, M = Xd.shape[0], Xd.shape[1]
        ex_ch = 0 if extra_d is None else extra_d.shape[1]
        Ad = np.concatenate([Xd, extra_d], axis=1) if extra_d is not None else Xd
        Ah = np.concatenate([Xh, extra_h], axis=1) if extra_h is not None else Xh
        At = np.concatenate([Xt, extra_t], axis=1) if extra_t is not None else Xt
        oof = np.zeros((N, 128, 128), np.float32); hs, ts = [], []
        Ad_t = torch.from_numpy(Ad).to(dev); yd_t = torch.from_numpy(dg.astype(np.float32)).to(dev)
        Ah_t = torch.from_numpy(Ah).to(dev); At_t = torch.from_numpy(At).to(dev)
        for k in range(4):
            vi = np.array(folds[k]); ti = np.array([i for i in range(N) if i not in set(folds[k])])
            wk, bk = opt_linear_w(Xd[ti], dg[ti])
            iw = np.log(np.maximum(wk / max(float(wk.sum()), 1e-6), 1e-2))
            torch.manual_seed(42 + k)
            net = MetaStackNet(M, iw, bk, extra_ch=ex_ch).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2e-2)
            bsc, bval, bst = -1.0, None, None
            for ep in range(epochs):
                net.train()
                perm = ti[np.random.RandomState(100*k + ep).permutation(len(ti))]
                for s in range(0, len(perm), 16):
                    bi = perm[s:s+16]; opt.zero_grad()
                    sym_miou_loss(net(Ad_t[bi]), yd_t[bi]).backward(); opt.step()
                if (ep + 1) % 5 == 0:
                    net.eval()
                    with torch.no_grad(): vl = net(Ad_t[vi]).cpu().numpy()
                    sc = float(fast_miou_vec(vl >= 0.0, dg[vi]).mean())
                    if sc > bsc:
                        bsc, bval = sc, vl
                        bst = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
            oof[vi] = bval
            net.load_state_dict(bst); net.eval()
            with torch.no_grad():
                hs.append(net(Ah_t).cpu().numpy()); ts.append(net(At_t).cpu().numpy())
        return oof, np.mean(hs, axis=0), np.mean(ts, axis=0)

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

        idx_L = np.where(dL)[0]; idx_S = np.where(~dL)[0]
        folds = [[] for _ in range(4)]
        for k, i in enumerate(idx_L): folds[k % 4].append(i)
        for k, i in enumerate(idx_S): folds[k % 4].append(i)

        m_sc = [float(fast_miou_vec(Xd[:, j] >= 0.0, dg).mean()) for j in range(len(mnames))]
        order = np.argsort(-np.array(m_sc))[:8]
        Xd, Xh, Xt = Xd[:, order], Xh[:, order], Xt[:, order]

        # Stage 0A: Base MetaStack over Top-8 JURECA clean models
        meta_d, meta_h, meta_t = run_stack_cv(Xd, Xh, Xt, dg, folds, epochs=60)

        # Stage 0B: Collect ALL Monster Refiners (V14, V15, V16, V17!) aligned to dn, hn, tn
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

        Bd = np.stack([meta_d] + ex_d, axis=1)
        Bh = np.stack([meta_h] + ex_h, axis=1)
        Bt = np.stack([meta_t] + ex_t, axis=1)
        b_names = ["META"] + ex_n
        print(f"\n================ {method.upper()} (Stage-0 Candidates: " + ", ".join(f"{b_names[j]}={fast_miou_vec(Bd[:,j]>=0, dg).mean():.4f}" for j in range(len(b_names))) + ") ================", flush=True)

        # Compare (a) Linear Nested-CV Blend vs (b) 2nd-Stage Spatial MetaStacker-V3 (with Forensics + DenseOcc!)
        lin_oof = np.zeros_like(meta_d); lin_hs, lin_ts = [], []
        for k in range(4):
            vi = np.array(folds[k]); ti = np.array([i for i in range(len(dn)) if i not in set(folds[k])])
            wk, bk = opt_linear_w(Bd[ti], dg[ti])
            lin_oof[vi] = np.tensordot(Bd[vi], wk, axes=([1], [0])) + bk
            lin_hs.append(np.tensordot(Bh, wk, axes=([1], [0])) + bk)
            lin_ts.append(np.tensordot(Bt, wk, axes=([1], [0])) + bk)
        lin_h = np.mean(lin_hs, axis=0); lin_t = np.mean(lin_ts, axis=0)
        wf, _ = opt_linear_w(Bd, dg)

        # Build forensic extra channels for Spatial MetaStacker-V3: [occ_f, occ_c, tf_m, tf_c, mm_m]
        def build_fex(names):
            arr = []
            for n in names:
                of, oc = occ_by_name.get(n, (np.zeros((128, 128), np.float32),)*2)
                tm_, tc_ = tf_by_name.get(n, (np.zeros((128, 128), np.float32),)*2)
                mm_ = mm_by_name.get(n, np.zeros((128, 128), np.float32))
                arr.append(np.stack([of, oc, tm_, tc_, mm_], axis=0))
            return np.stack(arr).astype(np.float32)

        sp_d, sp_h, sp_t = run_stack_cv(Bd, Bh, Bt, dg, folds, build_fex(dn), build_fex(hn), build_fex(tn), epochs=50)
        sc_lin = float(fast_miou_vec(lin_oof >= 0.0, dg).mean()); h_lin = float(fast_miou_vec(lin_h >= 0.0, hg).mean())
        sc_sp = float(fast_miou_vec(sp_d >= 0.0, dg).mean()); h_sp = float(fast_miou_vec(sp_h >= 0.0, hg).mean())
        print(f"  [0. BLEND] LINEAR-CV dev={sc_lin:.4f} hold={h_lin:.4f} w={np.round(wf,2).tolist()} | SPATIAL-V3 dev={sc_sp:.4f} hold={h_sp:.4f}", flush=True)
        if sc_sp > sc_lin + 0.001:
            d_lg, h_lg, t_lg = sp_d, sp_h, sp_t
            base_d, base_h = sc_sp, h_sp
        else:
            d_lg, h_lg, t_lg = lin_oof, lin_h, lin_t
            base_d, base_h = sc_lin, h_lin
        print(f"  >> SELECTED NEURAL STAGE: DEV={base_d:.4f} | HOLD={base_h:.4f}", flush=True)

        # Stage 1: Adaptive Split-Thr
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

        # Stage 2: rect_nnref override
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

        # Stage 3: JOINT SAM-1 ViT-H vs AXIS-RECT ARBITRATION!
        def precompute_sam_and_rect(names, lgs, cur_ms):
            isL_a = (lgs >= 0.0).mean(axis=(1, 2)) > 0.5
            sh, sb2, sb4 = cur_ms.copy(), cur_ms.copy(), cur_ms.copy()
            rm = cur_ms.copy()
            bi = np.zeros(len(names), np.float32); si = np.zeros(len(names), np.float32)
            rsi = np.zeros(len(names), np.float32)
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
                if not isL_a[i]:
                    mr, sir = best_axis_rect_from_prob(lgs[i])
                    rm[i] = mr; rsi[i] = sir
            return isL_a, {"hard": sh, "b2": sb2, "b4": sb4}, rm, bi, si, rsi, ct, ar, sol

        d_isL, d_sams, d_rm, d_bi, d_si, d_rsi, d_ct, d_ar, d_sol = precompute_sam_and_rect(dn, d_lg, d_m)
        h_isL, h_sams, h_rm, h_bi, h_si, h_rsi, h_ct, h_ar, h_sol = precompute_sam_and_rect(hn, h_lg, h_m)
        t_isL, t_sams, t_rm, t_bi, t_si, t_rsi, t_ct, t_ar, t_sol = precompute_sam_and_rect(tn, t_lg, t_m)
        d_sam_trig = np.zeros(len(dn), bool); h_sam_trig = np.zeros(len(hn), bool); t_sam_trig = np.zeros(len(tn), bool)

        # 3A: SAM_S_OBJ with RECT-COMPETITION GUARD (`d_si >= d_rsi - r_margin` so SAM never overwrites crisp rectangles!)
        for sub_name, is_L_target in [("SAM_S_OBJ", False), ("SAM_L_HOLE", True)]:
            base_v = fast_miou_vec(d_m, dg)
            sam_vs = {k: fast_miou_vec(d_sams[k], dg) for k in ("hard", "b2", "b4")}
            elig_d = (d_isL == is_L_target) & (~d_rect_trig)
            elig_h = (h_isL == is_L_target) & (~h_rect_trig)
            elig_t = (t_isL == is_L_target) & (~t_rect_trig)
            rm_opts = (-9.0, 0.0, 0.08) if not is_L_target else (-9.0,)
            sgrid = [(md, t_si, t_bi, s_max, rm_g)
                     for md in ("hard", "b2", "b4")
                     for t_si in (0.18, 0.28, 0.38, 0.48, 0.58)
                     for t_bi in (0.25, 0.40, 0.50, 0.60)
                     for s_max in (0.84, 0.90, 1.01)
                     for rm_g in rm_opts]
            cv_s = np.zeros_like(base_v)
            for fold in range(4):
                vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
                bsc, bc = float(base_v[tm].mean()), None
                for c in sgrid:
                    md, thr_si, thr_bi, sol_max, rm_g = c
                    tf = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_sol <= sol_max) & (d_ar >= 0.008) & ((d_si >= d_rsi - rm_g) if rm_g > -1 else True)
                    sc = float(np.where(tf, sam_vs[md], base_v)[tm].mean())
                    if sc > bsc: bsc, bc = sc, c
                if bc is None: cv_s[vm] = base_v[vm]
                else:
                    md, thr_si, thr_bi, sol_max, rm_g = bc
                    tf = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_sol <= sol_max) & (d_ar >= 0.008) & ((d_si >= d_rsi - rm_g) if rm_g > -1 else True)
                    cv_s[vm] = np.where(tf, sam_vs[md], base_v)[vm]
            bfull, bfc = float(base_v.mean()), None
            for c in sgrid:
                md, thr_si, thr_bi, sol_max, rm_g = c
                tf = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_sol <= sol_max) & (d_ar >= 0.008) & ((d_si >= d_rsi - rm_g) if rm_g > -1 else True)
                sc = float(np.where(tf, sam_vs[md], base_v).mean())
                if sc > bfull: bfull, bfc = sc, c
            gain_cv = float(cv_s.mean() - base_v.mean())
            print(f"  [3. {sub_name:10s}] nested-CV={cv_s.mean():.4f} ({gain_cv:+.4f}) full-dev={bfull:.4f} cfg={bfc}", flush=True)
            if bfc is not None and gain_cv > 0.001:
                md, thr_si, thr_bi, sol_max, rm_g = bfc
                td = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_sol <= sol_max) & (d_ar >= 0.008) & ((d_si >= d_rsi - rm_g) if rm_g > -1 else True)
                th = elig_h & (h_si >= thr_si) & (h_bi >= thr_bi) & (h_sol <= sol_max) & (h_ar >= 0.008) & ((h_si >= h_rsi - rm_g) if rm_g > -1 else True)
                tt = elig_t & (t_si >= thr_si) & (t_bi >= thr_bi) & (t_sol <= sol_max) & (t_ar >= 0.008) & ((t_si >= t_rsi - rm_g) if rm_g > -1 else True)
                d_m = np.where(td[:, None, None], d_sams[md], d_m)
                h_m = np.where(th[:, None, None], h_sams[md], h_m)
                t_m = np.where(tt[:, None, None], t_sams[md], t_m)
                d_sam_trig |= td; h_sam_trig |= th; t_sam_trig |= tt
                print(f"     >> ADOPTED {sub_name} ({md})! trig dev={td.sum()}/136 hold={th.sum()}/34 test={tt.sum()}/340 | hold_now={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        # 3B: AXIS-RECT SNAPPER on remaining S images!
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

        # Stage 4: Connected-Component Speck Cleaning (strict nested DEV CV!)
        base_v = fast_miou_vec(d_m, dg)
        sp_opts = (0, 8, 16, 32, 64)
        sp_vecs = {k: fast_miou_vec(clean_specks(d_m, d_isL, k), dg) for k in sp_opts}
        cv_sp = np.zeros_like(base_v)
        for fold in range(4):
            vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
            bk = max(sp_opts, key=lambda k: float(sp_vecs[k][tm].mean()))
            cv_sp[vm] = sp_vecs[bk][vm]
        best_k = max(sp_opts, key=lambda k: float(sp_vecs[k].mean()))
        if cv_sp.mean() > base_v.mean() + 0.0005 and best_k > 0:
            d_m = clean_specks(d_m, d_isL, best_k)
            h_m = clean_specks(h_m, h_isL, best_k)
            t_m = clean_specks(t_m, t_isL, best_k)
            print(f"  [5. SPECK_CLN ] nested-CV={cv_sp.mean():.4f} (+{cv_sp.mean()-base_v.mean():.4f}) min_px={best_k} | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        d_off = np.array([miou_image(d_m[i], dg[i]) for i in range(len(dg))])
        h_off = np.array([miou_image(h_m[i], hg[i]) for i in range(len(hg))])
        print(f"  *** FINAL {method.upper()} *** DEV={d_off.mean():.4f} (L {d_off[dL].mean():.4f} S {d_off[~dL].mean():.4f}) | HOLD={h_off.mean():.4f} (L {h_off[hL].mean():.4f} S {h_off[~hL].mean():.4f}) | test_fg={t_m.mean():.4f}", flush=True)
        summary[method] = {"dev": float(d_off.mean()), "hold": float(h_off.mean()), "hold_L": float(h_off[hL].mean()), "hold_S": float(h_off[~hL].mean())}
        for i, n in enumerate(tn): test_masks[n] = t_m[i].astype(np.uint8)

    m_dev = float(np.mean([summary[m]["dev"] for m in summary]))
    m_hld = float(np.mean([summary[m]["hold"] for m in summary]))
    print("\n============================================================", flush=True)
    print(f"FINAL V18 CHAMPION ({out_name}) 3-METH DEV  OOF : {m_dev:.4f}", flush=True)
    print(f"FINAL V18 CHAMPION ({out_name}) 3-METH HOLDOUT  : {m_hld:.4f}", flush=True)
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
def main(out_name: str = "sub_v18_champion.pt"):
    dev_s, hld_s = run_champion_fusion.remote(out_name)
    print(f"=== V18 CHAMPION ({out_name}) COMPLETE: DEV={dev_s:.4f} | HOLD={hld_s:.4f} ===")
