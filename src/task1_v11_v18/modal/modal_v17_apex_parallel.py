"""V17 APEX 8-GPU PARALLEL (1 GPU PER FOLD = ZERO TIMEOUTS, ~10 MINUTES TOTAL!!):
Combines EVERY breakthrough from the entire hackathon into a 17-channel Multi-Modal Forensic + Watermark Super-Refiner:
  * 3ch RGB [0,1] + 3ch HighPass(40) residual
  * 1ch MetaStack OOF logit + 1ch prob + 1ch ensemble std + 3ch Top-3 individual clean models
  * 2ch PURE SAM-1 ViT-H priors (`sam1`, `sams`)
  * 2ch WHITE-BOX DUAL-SCALE DENSE OCCLUSION HEATMAPS (`occ_fine 12x12`, `occ_coarse 8x8`)
  * 5ch ACADEMIC SOTA FORENSICS: `TruFor map`, `TruFor conf`, `Noiseprint++ (npp)`, `MMFusion map`, `MMFusion conf`!
  * 2ch Global `isL` + method indicator
Spawns 8 GPUs simultaneously:
  - GPUs 0..3: `v17_dinov2l_ms_apex518` (304M Meta DINOv2-Large @ 518px with 4-STAGE MULTI-SCALE ViT PYRAMID blocks [5,11,17,23]!)
  - GPUs 4..7: `v17_cnxxl_846m_apex512` (846M CLIP LAION-2B ConvNeXt-XXLarge @ 512px!)
Strict 4-fold stratified DEV CV + 8x Dihedral Aug + 8x Dihedral TTA -> ZERO holdout leakage!
Assembles both into `/vol/outputs/v14_refiners/{v17_dinov2l_ms_apex518, v17_cnxxl_846m_apex512}.npz`!
"""
import modal

app = modal.App("task1-v17-apex-8gpu")
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


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1800)
def train_v17_fold(backbone_name: str, tag: str, fold: int, epochs: int = 44, lr: float = 1.3e-4, img_sz: int = 518, bs: int = 8, is_dinov2: bool = True):
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

    # 1. PURE SAM-1 ViT-H shards
    sam_packed = {}
    for r in range(4):
        z = np.load(f"/vol/outputs/sam_h/shard_{r}.npz")
        for k in z.files: sam_packed[k] = z[k]

    # 2. WHITE-BOX DENSE OCCLUSION HEATMAPS (load arrays ONCE — never index inside loop!)
    occ_by_name = {}
    for m in ("pixelseal", "trustmark"):
        z = np.load(f"/vol/outputs/dense_occ/{m}.npz", allow_pickle=True)
        z_names = z["names"]; z_fine = z["fine"].astype(np.float32); z_coarse = z["coarse"].astype(np.float32)
        for i, nm in enumerate(z_names):
            f = z_fine[i]; c = z_coarse[i]
            fz = np.clip((f - f.mean()) / (f.std() + 1e-5), -4.0, 4.0) / 2.0
            cz = np.clip((c - c.mean()) / (c.std() + 1e-5), -4.0, 4.0) / 2.0
            occ_by_name[str(nm)] = (fz, cz)

    # 3. ACADEMIC SOTA FORENSICS: TruFor (map, conf, Noiseprint++) + MMFusion (map, conf)!
    tf_z = np.load("/vol/outputs/forensics/trufor_1530.npz", allow_pickle=True)
    tf_names = tf_z["names"]; tf_map = tf_z["map"].astype(np.float32); tf_conf = tf_z["conf"].astype(np.float32); tf_npp = tf_z["npp"].astype(np.float32)
    tf_by_name = {}
    for i, nm in enumerate(tf_names):
        mp = tf_map[i]; cf = tf_conf[i]; np_ = tf_npp[i]
        np_z = np.clip((np_ - np_.mean()) / (np_.std() + 1e-5), -4.0, 4.0) / 2.0
        tf_by_name[str(nm)] = (mp * 2.0 - 1.0, cf * 2.0 - 1.0, np_z)

    mm_z = np.load("/vol/outputs/forensics/mmfusion_1530.npz", allow_pickle=True)
    mm_names = mm_z["names"]; mm_map = mm_z["map"].astype(np.float32); mm_conf = mm_z["conf"].astype(np.float32)
    mm_by_name = {}
    for i, nm in enumerate(mm_names):
        mm_by_name[str(nm)] = (mm_map[i] * 2.0 - 1.0, mm_conf[i] * 2.0 - 1.0)

    def unpack_cands(nm):
        p = sam_packed.get(nm)
        if p is None or len(p) == 0: return np.zeros((0, 128, 128), dtype=bool)
        return np.unpackbits(p, axis=1).reshape(-1, 128, 128).astype(bool)

    def fast_miou_vec(pred, gt):
        p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
        i1 = (p & g).sum(1); u1 = (p | g).sum(1)
        i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
        return 0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0) + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0))

    # Load latest JURECA bundle (prefer v6 -> v5 -> v4)
    for cf in ("ens_top8_v6_fix01.npz", "ens_top8_v5_fix01.npz", "ens_top8_v4_fix01.npz"):
        p = pathlib.Path("/vol/jureca_ens") / cf
        if p.exists(): top_p = p; break
    ez = np.load(str(top_p), allow_pickle=True)
    print(f"[{tag} f{fold}] Bundle={top_p.name} | TruFor={len(tf_by_name)} | MMFusion={len(mm_by_name)}", flush=True)

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
            m1 = cands[j1].astype(np.float32); m1_s = (1.0 - m1) if isL else m1
            top5 = np.argsort(-si)[:5]
            w5 = np.maximum(si[top5], 0.0) ** 2; w5 = w5 / (w5.sum() + 1e-6)
            m_soft = sum(w5[q] * cands[top5[q]].astype(np.float32) for q in range(len(top5)))
            m_soft = (1.0 - m_soft) if isL else m_soft
            return m1_s, m_soft

        def get_extra(nm):
            of, oc = occ_by_name.get(nm, (np.zeros((128, 128), np.float32), np.zeros((128, 128), np.float32)))
            tm_, tc_, tn_ = tf_by_name.get(nm, (np.zeros((128, 128), np.float32),)*3)
            mm_, mc_ = mm_by_name.get(nm, (np.zeros((128, 128), np.float32),)*2)
            return of, oc, tm_, tc_, tn_, mm_, mc_

        for i, n in enumerate(dn):
            s1, ss = make_sam_priors(n, meta_oof[i])
            dev_items.append({"name": n, "method": method, "mid": mid, "fold": int(fold_of[i]),
                              "img": tr_by_name[n]["image"], "gt": dg[i].astype(np.float32),
                              "base_lg": meta_oof[i], "top3": Xd[i, :3], "std": std_d[i],
                              "sam1": s1, "sams": ss, "ex": get_extra(n), "isL": bool(dL[i])})
        for i, n in enumerate(hn):
            s1, ss = make_sam_priors(n, meta_h[i])
            hold_items.append({"name": n, "method": method, "mid": mid,
                               "img": tr_by_name[n]["image"], "gt": hg[i].astype(np.float32),
                               "base_lg": meta_h[i], "top3": Xh[i, :3], "std": std_h[i],
                               "sam1": s1, "sams": ss, "ex": get_extra(n), "isL": bool(tr_by_name[n]["full_watermark"])})
        for i, n in enumerate(tn):
            s1, ss = make_sam_priors(n, meta_t[i])
            test_items.append({"name": n, "method": method, "mid": mid,
                               "img": te_by_name[n]["image"],
                               "base_lg": meta_t[i], "top3": Xt[i, :3], "std": std_t[i],
                               "sam1": s1, "sams": ss, "ex": get_extra(n)})

    # 17 prior channels: [blg, prob, std, sam1, sams, isL, mid, top0, top1, top2, occf, occc, tf_m, tf_c, tf_npp, mm_m, mm_c]
    IN_EXTRA = 17

    class ApexForensicRefiner(nn.Module):
        def __init__(self, bb_name, dinov2_ms=True):
            super().__init__()
            self.dinov2_ms = dinov2_ms
            if dinov2_ms:
                # 4-STAGE MULTI-SCALE DINOv2-Large Feature Pyramid (blocks 5, 11, 17, 23)!
                self.bb = timm.create_model(bb_name, pretrained=True, num_classes=0, global_pool="", dynamic_img_size=True)
                dim = self.bb.num_features
                self.ms_proj = nn.Sequential(
                    nn.Conv2d(dim * 4 + IN_EXTRA, 320, 1), nn.GroupNorm(16, 320), nn.GELU(),
                    nn.Conv2d(320, 192, 3, padding=1), nn.GroupNorm(16, 192), nn.GELU(),
                )
                self.prior_s0 = nn.Sequential(
                    nn.Conv2d(IN_EXTRA + 3, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                    nn.Conv2d(96, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                )
                self.fuse = nn.Sequential(
                    nn.Conv2d(192 + 96, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
                    nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
                )
                self.head = nn.Conv2d(64, 1, 1)
            else:
                self.bb = timm.create_model(bb_name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3))
                chs = self.bb.feature_info.channels()
                self.prior_s0 = nn.Sequential(
                    nn.Conv2d(IN_EXTRA + 3, 80, 3, padding=1), nn.GroupNorm(8, 80), nn.GELU(),
                    nn.Conv2d(80, 80, 3, padding=1), nn.GroupNorm(8, 80), nn.GELU(),
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
                self.fuse = dec_blk(D + 80, 64)
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
            if self.dinov2_ms:
                # Extract 4 multi-scale blocks [5, 11, 17, 23] (low-level edges + mid-level textures + high-level objects!)
                n_blk = len(self.bb.blocks)
                idx_b = [n_blk // 4 - 1, n_blk // 2 - 1, 3 * n_blk // 4 - 1, n_blk - 1]
                fms = self.bb.get_intermediate_layers(xn, n=idx_b, reshape=True)  # list of 4x [B, C, gs, gs]
                gs = fms[0].shape[-1]
                pr_g = F.interpolate(priors128, (gs, gs), mode="bilinear", align_corners=False)
                x = self.ms_proj(torch.cat(list(fms) + [pr_g], dim=1))
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
            of, oc, tm_, tc_, tn_, mm_, mc_ = it["ex"]
            pr = np.concatenate([
                np.stack([blg / 4.0, prob, st, it["sam1"], it["sams"], isL_map, mid_f,
                          of, oc, tm_, tc_, tn_, mm_, mc_], axis=0),
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

    vi = np.where(d_folds == fold)[0]; ti = np.where(d_folds != fold)[0]
    torch.manual_seed(1500 + fold * 17)
    model = ApexForensicRefiner(backbone_name, dinov2_ms=is_dinov2).to(dev)
    bb_params = list(model.bb.parameters())
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if id(p) not in bb_ids]
    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": lr * 0.12, "weight_decay": 1e-2},
        {"params": head_params, "lr": lr, "weight_decay": 1e-2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    base_val_sc = float(fast_miou_vec(d_bases[vi, 0].numpy() >= 0.0, d_gts[vi].numpy() > 0.5).mean())
    best_val_sc = base_val_sc
    best_val_lg = d_bases[vi, 0].numpy().copy()
    best_st = None
    rng = np.random.RandomState(1600 + fold * 17)

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
        if (ep + 1) in (10, 18, 26, 32, epochs):
            vl = predict_tta8(model, d_imgs[vi], d_priors[vi], d_bases[vi], bsz=bs)
            sc = float(fast_miou_vec(vl >= 0.0, d_gts[vi].numpy() > 0.5).mean())
            if sc > best_val_sc:
                best_val_sc = sc; best_val_lg = vl
                best_st = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
            print(f"  [{tag} f{fold} ep {ep+1:2d}/{epochs}] val_OOF={sc:.4f} (best={best_val_sc:.4f}, base={base_val_sc:.4f}) | {(time.time()-t0)/60:.1f}m", flush=True)

    if best_st is not None:
        model.load_state_dict(best_st)
        hl = predict_tta8(model, h_imgs, h_priors, h_bases, bsz=bs)
        tl = predict_tta8(model, t_imgs, t_priors, t_bases, bsz=bs)
    else:
        hl = h_bases[:, 0].numpy().copy(); tl = t_bases[:, 0].numpy().copy()

    out_dir = pathlib.Path(f"/vol/outputs/v17_folds/{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"fold_{fold}.npz",
        val_idx=vi,
        val_names=np.array([dev_items[i]["name"] for i in vi]),
        val_mids=np.array([dev_items[i]["mid"] for i in vi]),
        val_lg=best_val_lg.astype(np.float16),
        hold_names=np.array([it["name"] for it in hold_items]),
        hold_mids=np.array([it["mid"] for it in hold_items]),
        hold_lg=hl.astype(np.float16),
        test_names=np.array([it["name"] for it in test_items]),
        test_mids=np.array([it["mid"] for it in test_items]),
        test_lg=tl.astype(np.float16),
    )
    vol.commit()
    print(f"*** [{tag} f{fold}] SAVED! base={base_val_sc:.4f} -> V17={best_val_sc:.4f} ({best_val_sc-base_val_sc:+.4f}) in {(time.time()-t0)/60:.1f}m ***", flush=True)
    return float(best_val_sc - base_val_sc)


@app.function(image=image, gpu=None, cpu=4, memory=8192, volumes={"/vol": vol}, timeout=300)
def assemble_v17(tag: str):
    import sys, pathlib, numpy as np
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import downsample_mask

    ds_tr = LocalisationDataset(str(repo_dir), "train")
    tr_by_name = {ds_tr[i]["image_name"]: ds_tr[i] for i in range(len(ds_tr))}
    fdir = pathlib.Path(f"/vol/outputs/v17_folds/{tag}")
    fs = [np.load(fdir / f"fold_{k}.npz", allow_pickle=True) for k in range(4)]

    # Assemble OOF dev, mean hold, mean test per method!
    all_vn = np.concatenate([f["val_names"] for f in fs])
    all_vm = np.concatenate([f["val_mids"] for f in fs])
    all_vl = np.concatenate([f["val_lg"].astype(np.float32) for f in fs], axis=0)
    hn_all = fs[0]["hold_names"]; hm_all = fs[0]["hold_mids"]
    hl_all = np.mean([f["hold_lg"].astype(np.float32) for f in fs], axis=0)
    tn_all = fs[0]["test_names"]; tm_all = fs[0]["test_mids"]
    tl_all = np.mean([f["test_lg"].astype(np.float32) for f in fs], axis=0)

    def fast_miou(pred, gt):
        p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
        i1 = (p & g).sum(1); u1 = (p | g).sum(1)
        i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
        return float((0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0) + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0))).mean())

    save_d = {}
    for mid, method in enumerate(("pixelseal", "trustmark", "stegastamp")):
        dm = all_vm == mid; hm = hm_all == mid; tm = tm_all == mid
        dn = all_vn[dm]; dl = all_vl[dm]
        hn = hn_all[hm]; hl = hl_all[hm]
        tn = tn_all[tm]; tl = tl_all[tm]
        dg = np.stack([downsample_mask(tr_by_name[str(n)]["mask"] > 0) > 0 for n in dn])
        hg = np.stack([downsample_mask(tr_by_name[str(n)]["mask"] > 0) > 0 for n in hn])
        print(f"=== ASSEMBLED [{tag}] {method.upper():10s}: DEV={fast_miou(dl>=0, dg):.4f} | HOLD={fast_miou(hl>=0, hg):.4f} ===", flush=True)
        save_d[f"{method}_dev_names"] = dn
        save_d[f"{method}_hold_names"] = hn
        save_d[f"{method}_test_names"] = tn
        save_d[f"{method}_dev_lg"] = dl.astype(np.float16)
        save_d[f"{method}_hold_lg"] = hl.astype(np.float16)
        save_d[f"{method}_test_lg"] = tl.astype(np.float16)

    out_p = pathlib.Path(f"/vol/outputs/v14_refiners/{tag}.npz")
    np.savez_compressed(out_p, **save_d)
    vol.commit()
    print(f"SAVED {out_p}!", flush=True)


@app.local_entrypoint()
def main():
    # Spawn ALL 8 GPUs simultaneously (4 folds DINOv2-Large Multi-Scale + 4 folds ConvNeXt-XXLarge-CLIP 846M!)
    hs_dino = [train_v17_fold.spawn("vit_large_patch14_dinov2.lvd142m", "v17_dinov2l_ms_apex518", f, 36, 1.4e-4, 518, 8, True) for f in range(4)]
    hs_xxl = [train_v17_fold.spawn("convnext_xxlarge.clip_laion2b_soup_ft_in1k", "v17_cnxxl_846m_apex512", f, 34, 1.3e-4, 512, 6, False) for f in range(4)]
    for f, h in enumerate(hs_dino):
        print(f"DINOv2-L-MS f{f} delta={h.get():+.4f}")
    assemble_v17.remote("v17_dinov2l_ms_apex518")
    for f, h in enumerate(hs_xxl):
        print(f"CNX-XXL-846M f{f} delta={h.get():+.4f}")
    assemble_v17.remote("v17_cnxxl_846m_apex512")
    print("=== ALL V17 APEX MODELS ASSEMBLED ===")
