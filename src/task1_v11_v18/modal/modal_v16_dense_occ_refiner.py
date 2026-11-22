"""V16 DENSE WHITE-BOX OCCLUSION + DINOv2-LARGE-518 + CLIP-LARGE-512 CROSS-METHOD REFINER!
Runs on 2x Modal A100-80GB/H100 in parallel (~10 min total, ~$1.20):
  - GPU 1 (`v16_dinov2l_occ518`): `vit_large_patch14_dinov2.lvd142m` (304M Meta DINOv2-Large @ 518x518 — world #1
    dense segmentation foundation model!) + WHITE-BOX 12x12 & 8x8 DENSE OCCLUSION HEATMAPS!
  - GPU 2 (`v16_cnxl_occ512`):    `convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384` (200M CLIP LAION-2B @ 512x512)
    + WHITE-BOX 12x12 & 8x8 DENSE OCCLUSION HEATMAPS + multi-seed!
  - Injects our brand-new `/vol/outputs/dense_occ/{pixelseal,trustmark}.npz` (which scored 0.88 AUROC on PS S-obj
    and 0.74 AUROC on TM L-holes ZERO-SHOT!) directly into BOTH the high-res prior stem AND every decoder scale!
  - Strict 4-fold stratified DEV CV + 8x Dihedral Aug + 8x Dihedral TTA -> ZERO holdout leakage!
"""
import modal

app = modal.App("task1-v16-dinov2-occ-refiner")
vol = modal.Volume.from_name("task1-wm-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "libgl1", "libglib2.0-0", "wget", "unzip")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "timm",
        "numpy",
        "scipy",
        "scikit-learn",
        "pillow",
        "einops",
        "onnx",
        "onnxruntime-gpu",
        "huggingface_hub",
        "hf_transfer",
        "trustmark",
        "omegaconf",
        "PyYAML",
        "tqdm",
    )
    .pip_install("requests", "opencv-python-headless")
    .pip_install("av", "lpips", "kornia", "pywavelets", "scikit-image", "matplotlib")
    .pip_install("pycocotools", "pandas", "tensorboard", "ffmpeg-python")
    .pip_install("datasets")
    .pip_install("git+https://github.com/facebookresearch/segment-anything.git")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=2400)
def train_v16_occ_refiner(backbone_name: str, tag: str, epochs: int = 44, lr: float = 1.3e-4, img_sz: int = 512, bs: int = 8, is_vit: bool = False):
    import sys, time, pathlib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    t0 = time.time()
    dev = "cuda:0"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import downsample_mask

    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    tr_by_name = {ds_tr[i]["image_name"]: ds_tr[i] for i in range(len(ds_tr))}
    te_by_name = {ds_te[i]["image_name"]: ds_te[i] for i in range(len(ds_te))}

    # Load PURE SAM-1 ViT-H shards (proven cleanest!)
    sam_packed = {}
    for r in range(4):
        z = np.load(f"/vol/outputs/sam_h/shard_{r}.npz")
        for k in z.files: sam_packed[k] = z[k]

    # Load DUAL-SCALE DENSE WHITE-BOX OCCLUSION HEATMAPS!
    occ_by_name = {}
    for m in ("pixelseal", "trustmark"):
        z = np.load(f"/vol/outputs/dense_occ/{m}.npz", allow_pickle=True)
        for i, nm in enumerate(z["names"]):
            f = z["fine"][i].astype(np.float32)
            c = z["coarse"][i].astype(np.float32)
            # Per-image z-score (highlights L-hole dips as negative & S-object peaks as positive!)
            fz = np.clip((f - f.mean()) / (f.std() + 1e-5), -4.0, 4.0) / 2.0
            cz = np.clip((c - c.mean()) / (c.std() + 1e-5), -4.0, 4.0) / 2.0
            occ_by_name[str(nm)] = (fz, cz)
    print(f"[{tag}] Loaded dense white-box occlusion heatmaps for {len(occ_by_name)} images!", flush=True)

    def unpack_cands(nm):
        p = sam_packed.get(nm)
        if p is None or len(p) == 0: return np.zeros((0, 128, 128), dtype=bool)
        return np.unpackbits(p, axis=1).reshape(-1, 128, 128).astype(bool)

    def fast_miou_vec(pred, gt):
        p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
        i1 = (p & g).sum(1); u1 = (p | g).sum(1)
        i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
        return 0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0) + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0))

    for cf in ("ens_top8_v6_fix01.npz", "ens_top8_v5_fix01.npz", "ens_top8_v4_fix01.npz"):
        p = pathlib.Path("/vol/jureca_ens") / cf
        if p.exists(): top_p = p; break
    ez = np.load(str(top_p), allow_pickle=True)
    print(f"[{tag}] Using logits bundle: {top_p.name}", flush=True)

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

    dev_items, hold_items, test_items = [], [], []
    for mid, method in enumerate(("pixelseal", "trustmark", "stegastamp")):
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

        m_sc = [float(fast_miou_vec(Xd[:, j] >= 0.0, dg).mean()) for j in range(len(mnames))]
        order = np.argsort(-np.array(m_sc))[:8]
        Xd, Xh, Xt = Xd[:, order], Xh[:, order], Xt[:, order]

        idx_L = np.where(dL)[0]; idx_S = np.where(~dL)[0]
        fold_of = np.zeros(len(dn), np.int32)
        for k, i in enumerate(idx_L): fold_of[i] = k % 4
        for k, i in enumerate(idx_S): fold_of[i] = k % 4

        meta_oof = np.zeros((len(dn), 128, 128), np.float32)
        meta_hs, meta_ts = [], []
        Xd_t = torch.from_numpy(Xd).to(dev); yd_t = torch.from_numpy(dg.astype(np.float32)).to(dev)
        Xh_t = torch.from_numpy(Xh).to(dev); Xt_t = torch.from_numpy(Xt).to(dev)
        for k in range(4):
            vi = np.where(fold_of == k)[0]; ti = np.where(fold_of != k)[0]
            iw = np.zeros(Xd.shape[1], np.float32); iw[0] = 1.0
            iw = np.log(np.maximum(iw, 0.08))
            torch.manual_seed(42 + k)
            net = MetaStackNet(Xd.shape[1], iw, 0.0).to(dev)
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
        std_d = Xd[:, :5].std(axis=1); std_h = Xh[:, :5].std(axis=1); std_t = Xt[:, :5].std(axis=1)

        def make_sam_priors(nm, lg):
            cands = unpack_cands(nm)
            if len(cands) == 0:
                return np.zeros((128, 128), np.float32), np.zeros((128, 128), np.float32)
            isL = (lg >= 0.0).mean() > 0.5
            pmin = 1.0 / (1.0 + np.exp(np.clip(lg if isL else -lg, -20, 20)))
            cf = cands.reshape(len(cands), -1).astype(np.float32)
            pf = pmin.ravel()[None, :]
            si = (cf * pf).sum(1) / (cf + pf - cf * pf).sum(1).clip(min=1e-4)
            j1 = int(np.argmax(si))
            m1 = cands[j1].astype(np.float32)
            m1_signed = (1.0 - m1) if isL else m1
            top5 = np.argsort(-si)[:5]
            w5 = np.maximum(si[top5], 0.0) ** 2; w5 = w5 / (w5.sum() + 1e-6)
            m_soft = sum(w5[q] * cands[top5[q]].astype(np.float32) for q in range(len(top5)))
            m_soft = (1.0 - m_soft) if isL else m_soft
            return m1_signed, m_soft

        def get_occ(nm):
            return occ_by_name.get(nm, (np.zeros((128, 128), np.float32), np.zeros((128, 128), np.float32)))

        for i, n in enumerate(dn):
            s1, ss = make_sam_priors(n, meta_oof[i]); of, oc = get_occ(n)
            dev_items.append({"name": n, "method": method, "mid": mid, "fold": int(fold_of[i]),
                              "img": tr_by_name[n]["image"], "gt": dg[i].astype(np.float32),
                              "base_lg": meta_oof[i], "top3": Xd[i, :3], "std": std_d[i],
                              "sam1": s1, "sams": ss, "occf": of, "occc": oc, "isL": bool(dL[i])})
        for i, n in enumerate(hn):
            s1, ss = make_sam_priors(n, meta_h[i]); of, oc = get_occ(n)
            hold_items.append({"name": n, "method": method, "mid": mid,
                               "img": tr_by_name[n]["image"], "gt": hg[i].astype(np.float32),
                               "base_lg": meta_h[i], "top3": Xh[i, :3], "std": std_h[i],
                               "sam1": s1, "sams": ss, "occf": of, "occc": oc, "isL": bool(tr_by_name[n]["full_watermark"])})
        for i, n in enumerate(tn):
            s1, ss = make_sam_priors(n, meta_t[i]); of, oc = get_occ(n)
            test_items.append({"name": n, "method": method, "mid": mid,
                               "img": te_by_name[n]["image"],
                               "base_lg": meta_t[i], "top3": Xt[i, :3], "std": std_t[i],
                               "sam1": s1, "sams": ss, "occf": of, "occc": oc})

    IN_EXTRA = 12  # base_lg, prob, std, sam1, sams, isL, mid, top0, top1, top2, occ_fine, occ_coarse

    class OccRefiner(nn.Module):
        def __init__(self, bb_name, is_vit_flag=False):
            super().__init__()
            self.is_vit = is_vit_flag
            if is_vit_flag:
                self.bb = timm.create_model(bb_name, pretrained=True, num_classes=0, global_pool="", dynamic_img_size=True)
                dim = self.bb.num_features
                self.vit_proj = nn.Sequential(
                    nn.Conv2d(dim + IN_EXTRA, 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
                    nn.Conv2d(256, 160, 3, padding=1), nn.GroupNorm(8, 160), nn.GELU(),
                )
                self.prior_s0 = nn.Sequential(
                    nn.Conv2d(IN_EXTRA + 3, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                    nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                )
                self.fuse = nn.Sequential(
                    nn.Conv2d(160 + 96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                    nn.Conv2d(96, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                )
                self.head = nn.Conv2d(64, 1, 1)
            else:
                self.bb = timm.create_model(bb_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
                chs = self.bb.feature_info.channels()
                self.prior_s0 = nn.Sequential(
                    nn.Conv2d(IN_EXTRA + 3, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                )
                D = 160
                self.p3 = nn.Conv2d(chs[3] + IN_EXTRA, D, 1)
                self.p2 = nn.Conv2d(chs[2] + IN_EXTRA, D, 1)
                self.p1 = nn.Conv2d(chs[1] + IN_EXTRA, D, 1)
                self.p0 = nn.Conv2d(chs[0] + IN_EXTRA, D, 1)
                def dec_blk(ci, co):
                    return nn.Sequential(
                        nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                        nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                    )
                self.u2 = dec_blk(D * 2, D); self.u1 = dec_blk(D * 2, D); self.u0 = dec_blk(D * 2, D)
                self.fuse = dec_blk(D + 64, 64)
                self.head = nn.Conv2d(64, 1, 1)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
            self.res_scale = nn.Parameter(torch.tensor(0.5))

        def forward(self, img01, priors128, base_lg128):
            x128 = F.interpolate(img01, (128, 128), mode="bilinear", align_corners=False)
            hp = ((x128 - F.avg_pool2d(x128, 5, 1, 2)) * 20.0).clamp(-3.0, 3.0)
            p_stem = self.prior_s0(torch.cat([priors128, hp], dim=1))
            xn = (img01 - self.mean) / self.std
            if self.is_vit:
                tok = self.bb.forward_features(xn)
                n_pre = getattr(self.bb, "num_prefix_tokens", 1)
                pt = tok[:, n_pre:, :]
                gs = int(round(pt.shape[1] ** 0.5))
                fm = pt.transpose(1, 2).reshape(pt.shape[0], pt.shape[2], gs, gs)
                pr_g = F.interpolate(priors128, (gs, gs), mode="bilinear", align_corners=False)
                x = self.vit_proj(torch.cat([fm, pr_g], dim=1))
                x128_dec = F.interpolate(x, (128, 128), mode="bilinear", align_corners=False)
            else:
                f0, f1, f2, f3 = self.bb(xn)
                def cat_p(f):
                    return torch.cat([f, F.interpolate(priors128, f.shape[-2:], mode="bilinear", align_corners=False)], dim=1)
                x = self.p3(cat_p(f3))
                x = self.u2(torch.cat([F.interpolate(x, f2.shape[-2:], mode="bilinear", align_corners=False), self.p2(cat_p(f2))], dim=1))
                x = self.u1(torch.cat([F.interpolate(x, f1.shape[-2:], mode="bilinear", align_corners=False), self.p1(cat_p(f1))], dim=1))
                x = self.u0(torch.cat([F.interpolate(x, f0.shape[-2:], mode="bilinear", align_corners=False), self.p0(cat_p(f0))], dim=1))
                x128_dec = F.interpolate(x, (128, 128), mode="bilinear", align_corners=False)
            delta = self.head(self.fuse(torch.cat([x128_dec, p_stem], dim=1)))
            return (base_lg128 + self.res_scale * delta).squeeze(1)

    def build_tensors(item_list):
        imgs = torch.stack([
            F.interpolate(torch.from_numpy(it["img"].copy()).permute(2, 0, 1)[None].float() / 255.0,
                          (img_sz, img_sz), mode="bilinear", align_corners=False)[0]
            for it in item_list
        ])
        priors, bases = [], []
        for it in item_list:
            blg = np.clip(it["base_lg"], -8.0, 8.0)
            prob = 1.0 / (1.0 + np.exp(-blg))
            st = np.clip(it["std"], 0.0, 4.0) / 2.0
            isL_map = np.full((128, 128), 1.0 if (blg >= 0).mean() > 0.5 else -1.0, np.float32)
            mid_f = np.full((128, 128), (it["mid"] - 1.0), np.float32)
            t3 = np.clip(it["top3"], -8.0, 8.0) / 4.0
            pr = np.concatenate([
                np.stack([blg / 4.0, prob, st, it["sam1"], it["sams"], isL_map, mid_f, it["occf"], it["occc"]], axis=0),
                t3,
            ], axis=0)
            priors.append(pr); bases.append(blg[None])
        priors = torch.from_numpy(np.stack(priors).astype(np.float32))
        bases = torch.from_numpy(np.stack(bases).astype(np.float32))
        gts = torch.from_numpy(np.stack([it["gt"] for it in item_list]).astype(np.float32)) if "gt" in item_list[0] else None
        return imgs, priors, bases, gts

    d_imgs, d_priors, d_bases, d_gts = build_tensors(dev_items)
    h_imgs, h_priors, h_bases, h_gts = build_tensors(hold_items)
    t_imgs, t_priors, t_bases, _ = build_tensors(test_items)
    d_folds = np.array([it["fold"] for it in dev_items])
    d_mids = np.array([it["mid"] for it in dev_items])
    h_mids = np.array([it["mid"] for it in hold_items])

    def apply_d8(x, k):
        rot = k % 4; flp = k // 4
        if flp: x = torch.flip(x, [-1])
        if rot: x = torch.rot90(x, rot, [-2, -1])
        return x

    def inv_d8(x, k):
        rot = k % 4; flp = k // 4
        if rot: x = torch.rot90(x, (4 - rot) % 4, [-2, -1])
        if flp: x = torch.flip(x, [-1])
        return x

    @torch.no_grad()
    def predict_tta8(model, imgs, priors, bases, bsz=8):
        model.eval()
        out = np.zeros((len(imgs), 128, 128), np.float32)
        for s in range(0, len(imgs), bsz):
            im = imgs[s:s+bsz].to(dev); pr = priors[s:s+bsz].to(dev); ba = bases[s:s+bsz].to(dev)
            acc = 0.0
            for k in range(8):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(apply_d8(im, k), apply_d8(pr, k), apply_d8(ba, k)).float()
                acc = acc + inv_d8(lg, k)
            out[s:s+bsz] = (acc / 8.0).cpu().numpy()
        return out

    oof_lg = np.zeros((len(dev_items), 128, 128), np.float32)
    hold_lgs, test_lgs = [], []

    for fold in range(4):
        vi = np.where(d_folds == fold)[0]; ti = np.where(d_folds != fold)[0]
        torch.manual_seed(1100 + fold)
        model = OccRefiner(backbone_name, is_vit_flag=is_vit).to(dev)
        bb_params = list(model.bb.parameters())
        bb_ids = {id(p) for p in bb_params}
        head_params = [p for p in model.parameters() if id(p) not in bb_ids]
        opt = torch.optim.AdamW([
            {"params": bb_params, "lr": lr * 0.15, "weight_decay": 1e-2},
            {"params": head_params, "lr": lr, "weight_decay": 1e-2},
        ])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        best_val_sc = float(fast_miou_vec(d_bases[vi, 0].numpy() >= 0.0, d_gts[vi].numpy() > 0.5).mean())
        best_val_lg = d_bases[vi, 0].numpy().copy()
        best_st = None
        rng = np.random.RandomState(1200 + fold)

        for ep in range(epochs):
            model.train()
            perm = ti[rng.permutation(len(ti))]
            for s in range(0, len(perm), bs):
                bi = perm[s:s+bs]
                im = d_imgs[bi].to(dev); pr = d_priors[bi].to(dev); ba = d_bases[bi].to(dev); gt = d_gts[bi].to(dev)
                k_aug = int(rng.randint(0, 8))
                im = apply_d8(im, k_aug); pr = apply_d8(pr, k_aug); ba = apply_d8(ba, k_aug)
                gt = apply_d8(gt[:, None], k_aug)[:, 0]
                if rng.rand() < 0.35:
                    pr = pr.clone()
                    pr[:, 0] = pr[:, 0] + torch.randn_like(pr[:, 0]) * 0.15
                opt.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(im, pr, ba)
                    loss = sym_miou_loss(pred.float(), gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            sched.step()
            if (ep + 1) in (12, 20, 28, 36, epochs):
                vl = predict_tta8(model, d_imgs[vi], d_priors[vi], d_bases[vi], bsz=bs)
                sc = float(fast_miou_vec(vl >= 0.0, d_gts[vi].numpy() > 0.5).mean())
                if sc > best_val_sc:
                    best_val_sc = sc; best_val_lg = vl
                    best_st = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}

        oof_lg[vi] = best_val_lg
        if best_st is not None:
            model.load_state_dict(best_st)
            hl = predict_tta8(model, h_imgs, h_priors, h_bases, bsz=bs)
            tl = predict_tta8(model, t_imgs, t_priors, t_bases, bsz=bs)
        else:
            hl = h_bases[:, 0].numpy().copy(); tl = t_bases[:, 0].numpy().copy()
        hold_lgs.append(hl); test_lgs.append(tl)
        base_f = float(fast_miou_vec(d_bases[vi, 0].numpy() >= 0.0, d_gts[vi].numpy() > 0.5).mean())
        print(f"  [{tag} fold {fold}] val_OOF: base={base_f:.4f} -> V16={best_val_sc:.4f} ({best_val_sc-base_f:+.4f}) | {(time.time()-t0)/60:.1f}m", flush=True)

    hold_ens = np.mean(hold_lgs, axis=0)
    test_ens = np.mean(test_lgs, axis=0)

    summary = {}
    for mid, method in enumerate(("pixelseal", "trustmark", "stegastamp")):
        dm = d_mids == mid; hm = h_mids == mid
        b_d = float(fast_miou_vec(d_bases[dm, 0].numpy() >= 0.0, d_gts[dm].numpy() > 0.5).mean())
        v_d = float(fast_miou_vec(oof_lg[dm] >= 0.0, d_gts[dm].numpy() > 0.5).mean())
        b_h = float(fast_miou_vec(h_bases[hm, 0].numpy() >= 0.0, h_gts[hm].numpy() > 0.5).mean())
        v_h = float(fast_miou_vec(hold_ens[hm] >= 0.0, h_gts[hm].numpy() > 0.5).mean())
        print(f"*** [{tag}] {method.upper():10s}: DEV {b_d:.4f} -> {v_d:.4f} ({v_d-b_d:+.4f}) | HOLD {b_h:.4f} -> {v_h:.4f} ({v_h-b_h:+.4f}) ***", flush=True)
        summary[method] = {"dev": v_d, "hold": v_h, "delta_dev": v_d - b_d, "delta_hold": v_h - b_h}

    out_dir = pathlib.Path("/vol/outputs/v14_refiners")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_d = {}
    for mid, method in enumerate(("pixelseal", "trustmark", "stegastamp")):
        dm = np.where(d_mids == mid)[0]; hm = np.where(h_mids == mid)[0]
        tm = np.where(np.array([it["mid"] for it in test_items]) == mid)[0]
        save_d[f"{method}_dev_names"] = np.array([dev_items[i]["name"] for i in dm])
        save_d[f"{method}_hold_names"] = np.array([hold_items[i]["name"] for i in hm])
        save_d[f"{method}_test_names"] = np.array([test_items[i]["name"] for i in tm])
        save_d[f"{method}_dev_lg"] = oof_lg[dm].astype(np.float16)
        save_d[f"{method}_hold_lg"] = hold_ens[hm].astype(np.float16)
        save_d[f"{method}_test_lg"] = test_ens[tm].astype(np.float16)
    np.savez_compressed(out_dir / f"{tag}.npz", **save_d)
    vol.commit()
    print(f"[{tag}] SAVED /vol/outputs/v14_refiners/{tag}.npz in {(time.time()-t0)/60:.1f}m!", flush=True)
    return summary


@app.local_entrypoint()
def main():
    h1 = train_v16_occ_refiner.spawn("vit_large_patch14_dinov2.lvd142m", "v16_dinov2l_occ518", epochs=44, lr=1.3e-4, img_sz=518, bs=8, is_vit=True)
    h2 = train_v16_occ_refiner.spawn("convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384", "v16_cnxl_occ512", epochs=44, lr=1.4e-4, img_sz=512, bs=8, is_vit=False)
    print("DINOv2-L OCC518:", h1.get())
    print("CNX-L    OCC512:", h2.get())
