"""Modal GPU All-In-One Clean Nested-CV Fusion Engine (`V6-Fix01-SAM-H-V2`):
Takes `ens_top8_v4_fix01.npz` (our Top-8 clean 4-fold OOF models per method, including brand-new
TrustMark V11-Fix01 0.6935 + PixelSeal V7.2-Fix01 0.6809 + StegaStamp V11-Fix01 0.8711!):
  1. Fits Linear Ensemble (greedy forward + Dirichlet coordinate ascent, strictly on DEV)
  2. Fits Spatial `MetaStackNet` (strict 4-fold stratified DEV CV on GPU, ~3 seconds!)
  3. Picks best of {Linear, MetaStack} strictly by DEV OOF
  4. Applies Adaptive L/S Split-Threshold + Hole-Dip Recovery (4-fold nested DEV CV)
  5. Applies `rect_nnref` exact rectangle override (4-fold nested DEV CV)
  6. Applies `SAM-ViT-H V2` (1,530 cached shards + Top-2 union + Rectangularity Guard + Hard vs Soft-Blend, 4-fold nested DEV CV!)
  7. Evaluates ONCE on 102 untouched holdout images + writes verified `sub_v6_fix01_samh.pt` (<20 MB)!
"""
import modal

app = modal.App("task1-sam-h-v2-fuse")
vol = modal.Volume.from_name("task1-wm-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.5.1", "numpy", "scipy", "scikit-learn", "pillow", "tqdm")
)


@app.function(image=image, gpu=["A100-80GB", "L40S", "A10G", "H100"], volumes={"/vol": vol}, timeout=900)
def run_sam_h_v2_fuse(top8_file: str = "/vol/jureca_ens/ens_top8_v4_fix01.npz", out_name: str = "sub_v6_fix01_samh.pt"):
    import sys, json, time, pathlib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    t0 = time.time()
    dev = "cuda:0"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import miou_image, downsample_mask

    sam_packed = {}
    for r in range(4):
        z = np.load(f"/vol/outputs/sam_h/shard_{r}.npz")
        for k in z.files:
            sam_packed[k] = z[k]
    print(f"Loaded cached SAM-ViT-H shards for {len(sam_packed)}/1530 images in {time.time()-t0:.1f}s", flush=True)

    ez = np.load(top8_file, allow_pickle=True)
    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    tr_by_name = {ds_tr[i]["image_name"]: ds_tr[i] for i in range(len(ds_tr))}
    te_names_all = [ds_te[i]["image_name"] for i in range(len(ds_te))]

    def unpack_cands(nm):
        p = sam_packed.get(nm)
        if p is None or len(p) == 0:
            return np.zeros((0, 128, 128), dtype=bool)
        return np.unpackbits(p, axis=1).reshape(-1, 128, 128).astype(bool)

    def fast_miou_vec(pred, gt):
        p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
        i1 = (p & g).sum(1); u1 = (p | g).sum(1)
        i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
        return 0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0) + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0))

    def load_rect(p):
        z = np.load(p, allow_pickle=True)
        return {str(z["names"][i]): {
            "box": z["box"][i].astype(np.float32),
            "margin": float(z["margin"][i]), "acc": float(z["acc"][i]),
            "full_margin": float(z["full_margin"][i]), "full_acc": float(z["full_acc"][i]),
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

    class MetaStackNet(nn.Module):
        def __init__(self, M, init_w, init_b):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(float(init_b), dtype=torch.float32))
            in_ch = M + 4
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.GELU(),
                nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.GroupNorm(4, 24), nn.GELU(),
                nn.Conv2d(24, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
                nn.Conv2d(16, 1, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
            self.res_scale = nn.Parameter(torch.tensor(0.25))

        def forward(self, x):
            w = torch.softmax(self.w, dim=0)
            base = (x * w[None, :, None, None]).sum(1, keepdim=True) + self.b
            xm = x.mean(1, keepdim=True)
            xs = x.std(1, keepdim=True)
            gm = base.mean(dim=(2, 3), keepdim=True).expand_as(base)
            gs = base.std(dim=(2, 3), keepdim=True).expand_as(base)
            feat = torch.cat([x, xm, xs, gm, gs], dim=1)
            return (base + self.res_scale * self.net(feat)).squeeze(1)

    def sym_miou_loss(logits, target):
        p = torch.sigmoid(logits).flatten(1)
        y = target.flatten(1)
        i1 = (p * y).sum(1); u1 = (p + y - p * y).sum(1)
        i0 = ((1 - p) * (1 - y)).sum(1); u0 = ((1 - p) + (1 - y) - (1 - p) * (1 - y)).sum(1)
        miou = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0))
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return (1.0 - miou).mean() + 0.15 * bce

    def fit_linear_and_meta_cv(Xd, Xh, Xt, dg, dL):
        N, M = Xd.shape[0], Xd.shape[1]
        # 1) Linear ensemble strictly via 4-fold nested DEV CV
        idx_L = np.where(dL)[0]; idx_S = np.where(~dL)[0]
        folds = [[] for _ in range(4)]
        for k, i in enumerate(idx_L): folds[k % 4].append(i)
        for k, i in enumerate(idx_S): folds[k % 4].append(i)

        def opt_weights(Xtr, gtr):
            w = np.zeros(M, np.float32); w[0] = 1.0
            best = float(fast_miou_vec(Xtr[:, 0] >= 0.0, gtr).mean())
            for _ in range(3):
                for j in range(M):
                    for step in (-0.3, -0.15, 0.15, 0.3, 0.5, 1.0):
                        wc = w.copy(); wc[j] = max(0.0, wc[j] + step)
                        if wc.sum() <= 0: continue
                        wc /= wc.sum()
                        lg = np.tensordot(Xtr, wc, axes=([1], [0]))
                        sc = float(fast_miou_vec(lg >= 0.0, gtr).mean())
                        if sc > best + 1e-5: best, w = sc, wc
            # Fine bias
            lg = np.tensordot(Xtr, w, axes=([1], [0]))
            best_b, best_sb = 0.0, best
            for b in np.linspace(-0.6, 0.6, 25):
                sc = float(fast_miou_vec(lg + b >= 0.0, gtr).mean())
                if sc > best_sb: best_sb, best_b = sc, float(b)
            return w, best_b

        lin_oof = np.zeros((N, 128, 128), np.float32)
        lin_hs, lin_ts = [], []
        ws_all, bs_all = [], []
        for k in range(4):
            vi = np.array(folds[k]); ti = np.array([i for i in range(N) if i not in set(folds[k])])
            wk, bk = opt_weights(Xd[ti], dg[ti])
            ws_all.append(wk); bs_all.append(bk)
            lin_oof[vi] = np.tensordot(Xd[vi], wk, axes=([1], [0])) + bk
            lin_hs.append(np.tensordot(Xh, wk, axes=([1], [0])) + bk)
            lin_ts.append(np.tensordot(Xt, wk, axes=([1], [0])) + bk)
        lin_h = np.mean(lin_hs, axis=0)
        lin_t = np.mean(lin_ts, axis=0)
        w_full, b_full = opt_weights(Xd, dg)

        # 2) MetaStackNet (strict 4-fold nested DEV CV on GPU!)
        meta_oof = np.zeros((N, 128, 128), np.float32)
        meta_hs, meta_ts = [], []
        Xd_t = torch.from_numpy(Xd).to(dev)
        yd_t = torch.from_numpy(dg.astype(np.float32)).to(dev)
        Xh_t = torch.from_numpy(Xh).to(dev)
        Xt_t = torch.from_numpy(Xt).to(dev)
        for k in range(4):
            vi = np.array(folds[k]); ti = np.array([i for i in range(N) if i not in set(folds[k])])
            iw = np.log(np.maximum(ws_all[k] / max(float(ws_all[k].sum()), 1e-6), 1e-2))
            torch.manual_seed(42 + k)
            net = MetaStackNet(M, iw, bs_all[k]).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2e-2)
            best_sc, best_val, best_st = -1.0, None, None
            for ep in range(60):
                net.train()
                perm = ti[np.random.RandomState(100*k + ep).permutation(len(ti))]
                for s in range(0, len(perm), 16):
                    bi = perm[s:s+16]
                    opt.zero_grad()
                    loss = sym_miou_loss(net(Xd_t[bi]), yd_t[bi])
                    loss.backward(); opt.step()
                if (ep + 1) % 5 == 0:
                    net.eval()
                    with torch.no_grad():
                        vl = net(Xd_t[vi]).cpu().numpy()
                    sc = float(fast_miou_vec(vl >= 0.0, dg[vi]).mean())
                    if sc > best_sc:
                        best_sc, best_val = sc, vl
                        best_st = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
            meta_oof[vi] = best_val
            net.load_state_dict(best_st); net.eval()
            with torch.no_grad():
                meta_hs.append(net(Xh_t).cpu().numpy())
                meta_ts.append(net(Xt_t).cpu().numpy())
        meta_h = np.mean(meta_hs, axis=0)
        meta_t = np.mean(meta_ts, axis=0)
        return (lin_oof, lin_h, lin_t, w_full), (meta_oof, meta_h, meta_t)

    def best_sam_match_v2(nm, lg, isL_flag, cur_mask):
        cands = unpack_cands(nm)
        cur_min = (~cur_mask) if isL_flag else cur_mask
        sol = rect_solidity(cur_min)
        if len(cands) == 0:
            return None, None, 0.0, 0.0, -99.0, 0.0, sol
        if isL_flag:
            prob_min = 1.0 / (1.0 + np.exp(np.clip(lg, -20, 20)))
            sig = -lg
        else:
            prob_min = 1.0 / (1.0 + np.exp(np.clip(-lg, -20, 20)))
            sig = lg

        cf = cands.reshape(len(cands), -1).astype(np.float32)
        pf = prob_min.ravel()[None, :]
        sf = sig.ravel()[None, :]
        cmf = cur_min.ravel().astype(np.float32)[None, :]

        inter_soft = (cf * pf).sum(1)
        union_soft = (cf + pf - cf * pf).sum(1).clip(min=1e-4)
        soft_iou = inter_soft / union_soft

        top_idx = np.argsort(-soft_iou)[:6]
        best_m = cands[top_idx[0]]
        best_si = float(soft_iou[top_idx[0]])
        for j2 in top_idx[1:]:
            u = cands[top_idx[0]] | cands[j2]
            if u.mean() > 0.48: continue
            uf = u.ravel().astype(np.float32)
            si_u = float((uf * pf[0]).sum() / max((uf + pf[0] - uf * pf[0]).sum(), 1e-4))
            if si_u > best_si + 0.025:
                best_si, best_m = si_u, u

        bmf = best_m.ravel().astype(np.float32)
        bi = float((bmf * cmf[0]).sum() / max((bmf + cmf[0] - bmf * cmf[0]).sum(), 1e-4))
        s_in = float((bmf * sf[0]).sum() / max(bmf.sum(), 1.0))
        s_out = float(((1.0 - bmf) * sf[0]).sum() / max((1.0 - bmf).sum(), 1.0))
        ct = s_in - s_out
        ar = float(best_m.mean())
        full_m = (~best_m) if isL_flag else best_m
        return full_m, best_m, bi, best_si, ct, ar, sol

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

        # Rank all clean honest-OOF models strictly on DEV OOF and keep top-8!
        m_sc = [float(fast_miou_vec(Xd[:, j] >= 0.0, dg).mean()) for j in range(len(mnames))]
        order = np.argsort(-np.array(m_sc))[:8]
        mnames = [mnames[j] for j in order]
        Xd, Xh, Xt = Xd[:, order], Xh[:, order], Xt[:, order]
        print(f"\n================ {method.upper()} (Top-{len(mnames)}: " + ", ".join(f"{mnames[k].split('_')[2] if len(mnames[k].split('_'))>2 else mnames[k]}={m_sc[order[k]]:.4f}" for k in range(min(5, len(mnames)))) + ") ================", flush=True)
        (lin_d, lin_h, lin_t, wf), (meta_d, meta_h, meta_t) = fit_linear_and_meta_cv(Xd, Xh, Xt, dg, dL)
        lin_ds = float(fast_miou_vec(lin_d >= 0.0, dg).mean()); lin_hs = float(fast_miou_vec(lin_h >= 0.0, hg).mean())
        met_ds = float(fast_miou_vec(meta_d >= 0.0, dg).mean()); met_hs = float(fast_miou_vec(meta_h >= 0.0, hg).mean())
        print(f"  [0. ENS] LINEAR dev={lin_ds:.4f} hold={lin_hs:.4f} w={np.round(wf,2).tolist()} | META dev={met_ds:.4f} hold={met_hs:.4f}", flush=True)
        if met_ds > lin_ds:
            d_lg, h_lg, t_lg = meta_d, meta_h, meta_t
            base_d, base_h = met_ds, met_hs
            ens_tag = "META"
        else:
            d_lg, h_lg, t_lg = lin_d, lin_h, lin_t
            base_d, base_h = lin_ds, lin_hs
            ens_tag = "LINEAR"
        print(f"  >> SELECTED {ens_tag}: dev={base_d:.4f} hold={base_h:.4f}", flush=True)

        # 1. Adaptive Split-Threshold + Hole-Dip Recovery
        def apply_adapt(lg, tL, tS, zL, zS):
            isL = (lg >= 0.0).mean(axis=(1, 2)) > 0.5
            out = np.where(isL[:, None, None], lg >= tL, lg >= tS)
            for i in range(len(lg)):
                m = lg[i]; mu = float(m.mean()); sd = float(m.std()) + 1e-6
                if isL[i] and (~out[i]).mean() < 0.005 and zL < 0:
                    out[i] = (m - mu) / sd >= zL
                elif (not isL[i]) and out[i].mean() < 0.005 and zS > 0:
                    out[i] = (m - mu) / sd >= zS
            return out

        ad_grid = [(tL, tS, zL, zS)
                   for tL in (-0.2, 0.0, 0.3, 0.6, 1.0)
                   for tS in (0.0, -0.4, -0.8, -1.2)
                   for zL in (0.0, -2.0, -2.5, -3.0)
                   for zS in (0.0, 2.5, 3.0)]
        ad_scores = [fast_miou_vec(apply_adapt(d_lg, *c), dg) for c in ad_grid]
        cv_ad = np.zeros(len(dg), np.float32)
        for fold in range(4):
            vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
            bi = max(range(len(ad_grid)), key=lambda idx: float(ad_scores[idx][tm].mean()))
            cv_ad[vm] = ad_scores[bi][vm]
        best_ad_i = max(range(len(ad_grid)), key=lambda idx: float(ad_scores[idx].mean()))
        if cv_ad.mean() > base_d + 0.0005:
            ad_cfg = ad_grid[best_ad_i]
            d_m = apply_adapt(d_lg, *ad_cfg)
            h_m = apply_adapt(h_lg, *ad_cfg)
            t_m = apply_adapt(t_lg, *ad_cfg)
            print(f"  [1. ADAPT] nested-CV={cv_ad.mean():.4f} ({cv_ad.mean()-base_d:+.4f}) cfg={ad_cfg} | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)
        else:
            d_m, h_m, t_m = d_lg >= 0.0, h_lg >= 0.0, t_lg >= 0.0
            print(f"  [1. ADAPT] skipped (nested-CV={cv_ad.mean():.4f})", flush=True)

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
                best_c = max(rgrid, key=lambda c: float(np.where((fa_d < c[0]) & (ac_d >= c[1]) & (gn_d >= c[2]), rep_v, base_v)[tm].mean()))
                tr_f = (fa_d < best_c[0]) & (ac_d >= best_c[1]) & (gn_d >= best_c[2])
                cv_r[vm] = np.where(tr_f, rep_v, base_v)[vm]
            best_rc = max(rgrid, key=lambda c: float(np.where((fa_d < c[0]) & (ac_d >= c[1]) & (gn_d >= c[2]), rep_v, base_v).mean()))
            if cv_r.mean() > base_v.mean() + 0.001:
                fa, ac, gn = best_rc
                d_rect_trig = (fa_d < fa) & (ac_d >= ac) & (gn_d >= gn)
                h_rect_trig = np.array([r["full_acc"] < fa and r["acc"] >= ac and (r["margin"]-r["full_margin"]) >= gn for r in h_rec])
                t_rect_trig = np.array([r["full_acc"] < fa and r["acc"] >= ac and (r["margin"]-r["full_margin"]) >= gn for r in t_rec])
                d_m = np.where(d_rect_trig[:, None, None], d_bm, d_m)
                h_m = np.where(h_rect_trig[:, None, None], h_bm, h_m)
                t_m = np.where(t_rect_trig[:, None, None], t_bm, t_m)
                print(f"  [2. RECT_NNREF] nested-CV={cv_r.mean():.4f} ({cv_r.mean()-base_v.mean():+.4f}) cfg={best_rc} trig={d_rect_trig.sum()}/136 | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        # 3. SAM-ViT-H V2 Segment Snapping
        def precompute_sam_v2(names, lgs, cur_ms):
            isL_arr = (lgs >= 0.0).mean(axis=(1, 2)) > 0.5
            sm_hard = cur_ms.copy(); sm_b2 = cur_ms.copy(); sm_b4 = cur_ms.copy()
            biou = np.zeros(len(names), np.float32); siou = np.zeros(len(names), np.float32)
            cont = np.zeros(len(names), np.float32); area = np.zeros(len(names), np.float32)
            rsol = np.zeros(len(names), np.float32)
            for i, nm in enumerate(names):
                cm, bm, bi, si, ct, ar, sol = best_sam_match_v2(nm, lgs[i], bool(isL_arr[i]), cur_ms[i])
                rsol[i] = sol
                if cm is not None:
                    sm_hard[i] = cm
                    sgn = (-1.0) if isL_arr[i] else 1.0
                    sm_b2[i] = (lgs[i] + 2.0 * sgn * (2.0 * bm.astype(np.float32) - 1.0)) >= 0.0
                    sm_b4[i] = (lgs[i] + 4.0 * sgn * (2.0 * bm.astype(np.float32) - 1.0)) >= 0.0
                    biou[i], siou[i], cont[i], area[i] = bi, si, ct, ar
            return isL_arr, {"hard": sm_hard, "b2": sm_b2, "b4": sm_b4}, biou, siou, cont, area, rsol

        d_isL, d_sams, d_bi, d_si, d_ct, d_ar, d_sol = precompute_sam_v2(dn, d_lg, d_m)
        h_isL, h_sams, h_bi, h_si, h_ct, h_ar, h_sol = precompute_sam_v2(hn, h_lg, h_m)
        t_isL, t_sams, t_bi, t_si, t_ct, t_ar, t_sol = precompute_sam_v2(tn, t_lg, t_m)

        for sub_name, is_L_target in [("SAM_S_OBJ", False), ("SAM_L_HOLE", True)]:
            base_v = fast_miou_vec(d_m, dg)
            sam_vs = {k: fast_miou_vec(d_sams[k], dg) for k in ("hard", "b2", "b4")}
            elig_d = (d_isL == is_L_target) & (~d_rect_trig)
            elig_h = (h_isL == is_L_target) & (~h_rect_trig)
            elig_t = (t_isL == is_L_target) & (~t_rect_trig)

            sgrid = [(mode, t_si, t_bi, t_ct, s_max)
                     for mode in ("hard", "b2", "b4")
                     for t_si in (0.18, 0.28, 0.38, 0.48, 0.58)
                     for t_bi in (0.25, 0.40, 0.50, 0.60)
                     for t_ct in (0.0, 1.0, 2.0)
                     for s_max in (0.85, 0.92, 1.01)]
            cv_s = np.zeros_like(base_v)
            for fold in range(4):
                vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
                best_sc_tr, best_c = float(base_v[tm].mean()), None
                for c in sgrid:
                    mode, thr_si, thr_bi, thr_ct, sol_max = c
                    tr_f = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_ct >= thr_ct) & (d_sol <= sol_max) & (d_ar >= 0.008)
                    sc = float(np.where(tr_f, sam_vs[mode], base_v)[tm].mean())
                    if sc > best_sc_tr: best_sc_tr, best_c = sc, c
                if best_c is None:
                    cv_s[vm] = base_v[vm]
                else:
                    mode, thr_si, thr_bi, thr_ct, sol_max = best_c
                    tr_f = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_ct >= thr_ct) & (d_sol <= sol_max) & (d_ar >= 0.008)
                    cv_s[vm] = np.where(tr_f, sam_vs[mode], base_v)[vm]

            best_full, best_fc = float(base_v.mean()), None
            for c in sgrid:
                mode, thr_si, thr_bi, thr_ct, sol_max = c
                tr_f = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_ct >= thr_ct) & (d_sol <= sol_max) & (d_ar >= 0.008)
                sc = float(np.where(tr_f, sam_vs[mode], base_v).mean())
                if sc > best_full: best_full, best_fc = sc, c

            gain_cv = float(cv_s.mean() - base_v.mean())
            print(f"  [3. {sub_name:10s}] nested-CV={cv_s.mean():.4f} ({gain_cv:+.4f}) full-dev={best_full:.4f} cfg={best_fc}", flush=True)
            if best_fc is not None and gain_cv > 0.001:
                mode, thr_si, thr_bi, thr_ct, sol_max = best_fc
                td = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_ct >= thr_ct) & (d_sol <= sol_max) & (d_ar >= 0.008)
                th = elig_h & (h_si >= thr_si) & (h_bi >= thr_bi) & (h_ct >= thr_ct) & (h_sol <= sol_max) & (h_ar >= 0.008)
                tt = elig_t & (t_si >= thr_si) & (t_bi >= thr_bi) & (t_ct >= thr_ct) & (t_sol <= sol_max) & (t_ar >= 0.008)
                d_m = np.where(td[:, None, None], d_sams[mode], d_m)
                h_m = np.where(th[:, None, None], h_sams[mode], h_m)
                t_m = np.where(tt[:, None, None], t_sams[mode], t_m)
                print(f"     >> ADOPTED {sub_name} ({mode})! trig dev={td.sum()}/136 hold={th.sum()}/34 test={tt.sum()}/340 | hold_now={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        d_off = np.array([miou_image(d_m[i], dg[i]) for i in range(len(dg))])
        h_off = np.array([miou_image(h_m[i], hg[i]) for i in range(len(hg))])
        print(f"  *** FINAL {method.upper()} *** DEV={d_off.mean():.4f} (L {d_off[dL].mean():.4f} S {d_off[~dL].mean():.4f}) | HOLD={h_off.mean():.4f} (L {h_off[hL].mean():.4f} S {h_off[~hL].mean():.4f}) | test_fg={t_m.mean():.4f}", flush=True)
        summary[method] = {"dev": float(d_off.mean()), "hold": float(h_off.mean()), "hold_L": float(h_off[hL].mean()), "hold_S": float(h_off[~hL].mean())}
        for i, n in enumerate(tn):
            test_masks[n] = t_m[i].astype(np.uint8)

    m_dev = float(np.mean([summary[m]["dev"] for m in summary]))
    m_hld = float(np.mean([summary[m]["hold"] for m in summary]))
    print("\n============================================================", flush=True)
    print(f"FINAL V6-Fix01 + SAM-ViT-H V2 3-METH DEV  OOF : {m_dev:.4f}", flush=True)
    print(f"FINAL V6-Fix01 + SAM-ViT-H V2 3-METH HOLDOUT  : {m_hld:.4f}", flush=True)
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
def main():
    dev_s, hld_s = run_sam_h_v2_fuse.remote()
    print(f"=== V6-Fix01-SAM-H-V2 COMPLETE: DEV={dev_s:.4f} | HOLD={hld_s:.4f} ===")
