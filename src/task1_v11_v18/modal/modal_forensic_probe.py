"""90-second Modal H100 Breakthrough Test:
Cross-Method Real-Image Forensic Edit-Region Segmentation (ZERO synth3!)
Tests TWO huge hypotheses in 90 seconds ($0.10):
  1. How accurately do (full_acc, full_margin) classify L (full_watermark=True) vs S (False)?
  2. If we pool ALL 3 methods' real train images (properly normalized [0,1]!) + SRM forensic noise filters
     and train ConvNeXt-Base + FPN to predict the MINORITY/EDITED region (hole on L, watermarked patch on S),
     then flip using predicted L/S -> what honest OOF mIoU do we get on PixelSeal, TrustMark, StegaStamp?!
     Compare BOTH:
       (A) Predicting y_minor (the edited region!) + flipping by decoder L/S classifier
       (B) Directly predicting y_mask conditioned on (full_acc, full_margin, method_onehot)
"""
import modal

app = modal.App("task1-forensic-edit-probe")
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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=1200)
def run_forensic_probe():
    import os, sys, time, pathlib, math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm
    import onnx
    from onnx import numpy_helper

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import miou_image, downsample_mask

    dev = "cuda:0"
    torch.backends.cudnn.benchmark = True

    # 1. Load ALL 510 train images + compute exact decoder full_acc & full_margin (cached to /vol/outputs/train_dec_stats.npz)
    ds_all = LocalisationDataset(str(repo_dir), "train")
    N = len(ds_all)
    print(f"Loaded {N} real train images", flush=True)

    imgs = np.empty((N, 512, 512, 3), dtype=np.uint8)
    masks128 = np.empty((N, 128, 128), dtype=np.uint8)
    methods = []
    names = []
    full_wm = np.empty(N, dtype=bool)
    shapes = []
    for i in range(N):
        it = ds_all[i]
        imgs[i] = it["image"]
        m512 = it["mask"] > 0
        masks128[i] = (downsample_mask(m512) > 0).astype(np.uint8)
        methods.append(it["method"])
        names.append(it["image_name"])
        full_wm[i] = bool(it["full_watermark"])
        if full_wm[i]:
            shapes.append("large")
        else:
            ys, xs = np.where(m512)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            shapes.append("rect" if m512.sum() / max((y1 - y0) * (x1 - x0), 1) > 0.98 else "object")

    methods = np.array(methods)
    shapes = np.array(shapes)

    # Print exact mask area stats across L / rect / object!
    print("\n=== DATASET MASK STATS ===", flush=True)
    for meth in ("pixelseal", "trustmark", "stegastamp"):
        for sh in ("large", "rect", "object"):
            idx = np.where((methods == meth) & (shapes == sh))[0]
            areas = masks128[idx].mean(axis=(1, 2))
            minor = np.where(sh == "large", 1.0 - areas, areas)
            print(f"  {meth:10s} {sh:6s} n={len(idx):2d} | minor_area% med={np.median(minor)*100:5.2f}% p10={np.percentile(minor,10)*100:4.2f}% p90={np.percentile(minor,90)*100:5.2f}%", flush=True)

    # Compute fast full-image decoder stats (full_acc, full_margin) for ALL 510 images
    stats_path = pathlib.Path("/vol/outputs/train_dec_stats.npz")
    if stats_path.exists():
        z = np.load(stats_path)
        full_acc, full_margin = z["full_acc"], z["full_margin"]
    else:
        full_acc = np.zeros(N, dtype=np.float32)
        full_margin = np.zeros(N, dtype=np.float32)
        for meth in ("pixelseal", "trustmark", "stegastamp"):
            wm = load_watermark(meth, device=dev)
            idxs = np.where(methods == meth)[0]
            with torch.no_grad():
                for i in idxs:
                    it = ds_all[int(i)]
                    img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
                    sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
                    if meth == "pixelseal":
                        lg = wm.det(F.interpolate(img01, (256, 256), mode="bilinear", align_corners=False))[:, 1:].float()
                    elif meth == "trustmark":
                        x256 = F.interpolate(img01, (wm.resolution, wm.resolution), mode="bilinear", align_corners=False)
                        lg = wm.net(x256 * 2.0 - 1.0).float()
                    else:
                        lg = wm.logits(img01.cpu()).to(dev).float()
                    sl = lg * sign
                    full_acc[i] = float((sl > 0).float().mean().item())
                    full_margin[i] = float(sl.mean().item())
        np.savez(stats_path, full_acc=full_acc, full_margin=full_margin)
        vol.commit()

    print("\n=== STEP 1: L vs S CLASSIFICATION ACCURACY FROM full_acc ALONE ===", flush=True)
    pred_L = np.zeros(N, dtype=bool)
    for meth in ("pixelseal", "trustmark", "stegastamp"):
        idx = np.where(methods == meth)[0]
        best_a, best_t = 0.0, 0.7
        for t in np.linspace(0.55, 0.98, 44):
            a = float(((full_acc[idx] >= t) == full_wm[idx]).mean())
            if a > best_a: best_a, best_t = a, t
        pred_L[idx] = full_acc[idx] >= best_t
        print(f"  {meth:10s}: best thr={best_t:.2f} -> L/S accuracy = {best_a*100:.2f}% (L med_acc={np.median(full_acc[idx][full_wm[idx]]):.3f}, S med_acc={np.median(full_acc[idx][~full_wm[idx]]):.3f})", flush=True)

    # 2. Stratified 5-fold split (fold 0 = 20% = 102 images, exact same size as holdout!)
    val_mask = np.zeros(N, dtype=bool)
    for meth in ("pixelseal", "trustmark", "stegastamp"):
        for sh in ("large", "rect", "object"):
            idx = np.where((methods == meth) & (shapes == sh))[0]
            # deterministic sort by image_name then take every 5th as val (exact 20% = 102 imgs!)
            idx = idx[np.argsort([names[j] for j in idx])]
            val_mask[idx[::5]] = True
    tr_idx = np.where(~val_mask)[0]
    va_idx = np.where(val_mask)[0]
    print(f"\nSplit: train={len(tr_idx)} real imgs, val={len(va_idx)} real imgs (stratified across all 3 methods x 3 edit types)", flush=True)

    # Build SRM 3-filter forensic noise kernel (fixed high-pass steganalysis filters!)
    srm_k = np.zeros((3, 1, 5, 5), dtype=np.float32)
    srm_k[0, 0, 1:4, 1:4] = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float32)  # 1st order
    srm_k[1, 0, 1:4, 1:4] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32) / 4.0  # 2nd order (KV-like)
    srm_k[2, 0] = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], np.float32) / 12.0  # 3rd order SRM 5x5!

    class ForensicEditNet(nn.Module):
        def __init__(self, backbone="convnext_base.fb_in22k_ft_in1k_384", cond_dim=5):
            super().__init__()
            self.vis = timm.create_model(backbone, pretrained=True, features_only=True)
            # Expand stem from 3ch -> 6ch (RGB + 3ch SRM forensic noise residual)
            for p_name, parent in self.vis.named_modules():
                for c_name, child in parent.named_children():
                    if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                        nc = nn.Conv2d(6, child.out_channels, child.kernel_size, stride=child.stride,
                                       padding=child.padding, dilation=child.dilation, groups=child.groups,
                                       bias=(child.bias is not None))
                        with torch.no_grad():
                            nc.weight[:, :3].copy_(child.weight)
                            nc.weight[:, 3:].copy_(child.weight * 0.25)
                            if child.bias is not None: nc.bias.copy_(child.bias)
                        setattr(parent, c_name, nc)
                        break
                else:
                    continue
                break

            chs = list(self.vis.feature_info.channels())
            self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.register_buffer("srm", torch.from_numpy(srm_k))  # [3, 1, 5, 5]

            d = 128
            self.lats = nn.ModuleList([nn.Conv2d(c, d, 1) for c in chs])
            self.smooth = nn.ModuleList([
                nn.Sequential(nn.Conv2d(d, d, 3, padding=1, bias=False), nn.GroupNorm(8, d), nn.SiLU(True))
                for _ in range(len(chs) - 1)
            ])
            self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, 64), nn.SiLU(True), nn.Linear(64, d))
            self.head = nn.Sequential(
                nn.Conv2d(d, 64, 3, padding=1, bias=False), nn.GroupNorm(8, 64), nn.SiLU(True),
                nn.Conv2d(64, 1, 1),
            )
            self.cls_head = nn.Linear(chs[-1] + cond_dim, 1)

        def forward(self, x01, cond):
            # x01 STRICTLY in [0, 1]!
            rgb = (x01 - self.im_mean) / self.im_std
            gray = x01.mean(dim=1, keepdim=True)
            srm_f = F.conv2d(F.pad(gray, (2, 2, 2, 2), mode="reflect"), self.srm)
            srm_f = (srm_f * 20.0).clamp(-3.0, 3.0)
            feats = self.vis(torch.cat([rgb, srm_f], dim=1))
            ps = [lat(f) for lat, f in zip(self.lats, feats)]
            p = ps[-1]
            for i in range(len(ps) - 2, -1, -1):
                p = self.smooth[i](ps[i] + F.interpolate(p, ps[i].shape[-2:], mode="bilinear", align_corners=False))
            p = p + self.cond_mlp(cond)[:, :, None, None]
            seg = self.head(p).squeeze(1)  # [B, 128, 128]
            if seg.shape[-1] != 128:
                seg = F.interpolate(seg.unsqueeze(1), (128, 128), mode="bilinear", align_corners=False).squeeze(1)
            cls = self.cls_head(torch.cat([feats[-1].mean(dim=(2, 3)), cond], dim=1)).squeeze(1)
            return seg, cls

    def sym_miou_loss(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
        i1 = (p * t).sum(1); u1 = (p + t - p * t).sum(1)
        i0 = ((1 - p) * (1 - t)).sum(1); u0 = ((1 - p) + (1 - t) - (1 - p) * (1 - t)).sum(1)
        sym = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0)).mean()
        return 0.4 * bce + 0.6 * (1.0 - sym)

    # Conditioning vector per image: [full_acc, full_margin/10, is_ps, is_tm, is_ss]
    cond_all = np.stack([
        full_acc,
        np.clip(full_margin / 10.0, -2.0, 2.0),
        (methods == "pixelseal").astype(np.float32),
        (methods == "trustmark").astype(np.float32),
        (methods == "stegastamp").astype(np.float32),
    ], axis=1).astype(np.float32)

    # Train TWO modes:
    #   Mode A: Predict y_minor (the EDITED/MINORITY region!), flip by learned cls head (L/S)!
    #   Mode B: Directly predict y_mask conditioned on [full_acc, full_margin, method_onehot]!
    for mode in ("A_MINOR_EDIT_REGION", "B_DIRECT_MASK"):
        print(f"\n================ MODE {mode} (PURE 408 REAL DEV, ZERO SYNTH3!) ================", flush=True)
        torch.manual_seed(42); np.random.seed(42)
        net = ForensicEditNet("convnext_base.fb_in22k_ft_in1k_384", cond_dim=5).to(dev)
        opt = torch.optim.AdamW([
            {"params": net.vis.parameters(), "lr": 3e-5},
            {"params": [p for n, p in net.named_parameters() if not n.startswith("vis.")], "lr": 5e-4},
        ], weight_decay=1e-2)
        steps = 600
        bs = 16
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)
        t0 = time.time()
        best_val = 0.0
        best_detail = ""
        for step in range(1, steps + 1):
            net.train()
            bi = np.random.choice(tr_idx, size=bs, replace=False)
            xb = torch.from_numpy(imgs[bi]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            yb_raw = torch.from_numpy(masks128[bi]).to(dev).float()
            fb = torch.from_numpy(full_wm[bi].astype(np.float32)).to(dev)
            cb = torch.from_numpy(cond_all[bi]).to(dev)

            # Random D4 dihedral augmentations (exact symmetries!)
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, [-1]); yb_raw = torch.flip(yb_raw, [-1])
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, [-2]); yb_raw = torch.flip(yb_raw, [-2])
            k_rot = np.random.randint(0, 4)
            if k_rot > 0:
                xb = torch.rot90(xb, k_rot, [-2, -1]); yb_raw = torch.rot90(yb_raw, k_rot, [-2, -1])

            if mode == "A_MINOR_EDIT_REGION":
                # Target = EDITED / MINORITY region (1 - mask for L, mask for S)
                yb = torch.where(fb[:, None, None] > 0.5, 1.0 - yb_raw, yb_raw)
            else:
                yb = yb_raw

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                seg, cls = net(xb, cb)
                loss = sym_miou_loss(seg.float(), yb) + 0.3 * F.binary_cross_entropy_with_logits(cls.float(), fb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            sched.step()

            if step % 150 == 0 or step == steps:
                net.eval()
                preds_mask = np.zeros((len(va_idx), 128, 128), dtype=bool)
                preds_L = np.zeros(len(va_idx), dtype=bool)
                with torch.no_grad():
                    for s in range(0, len(va_idx), 16):
                        vi = va_idx[s:s+16]
                        xv = torch.from_numpy(imgs[vi]).to(dev).permute(0, 3, 1, 2).float() / 255.0
                        cv = torch.from_numpy(cond_all[vi]).to(dev)
                        # 4x TTA (identity + hflip + vflip + hvflip)
                        seg_acc = 0.0; cls_acc = 0.0
                        for hf, vf in [(False, False), (True, False), (False, True), (True, True)]:
                            x_t = xv
                            if hf: x_t = torch.flip(x_t, [-1])
                            if vf: x_t = torch.flip(x_t, [-2])
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                sg, cl = net(x_t, cv)
                            if vf: sg = torch.flip(sg, [-2])
                            if hf: sg = torch.flip(sg, [-1])
                            seg_acc = seg_acc + torch.sigmoid(sg.float())
                            cls_acc = cls_acc + torch.sigmoid(cl.float())
                        prob = (seg_acc / 4.0).cpu().numpy()
                        pL = (cls_acc / 4.0).cpu().numpy() >= 0.5
                        preds_L[s:s+len(vi)] = pL
                        if mode == "A_MINOR_EDIT_REGION":
                            m_ed = prob >= 0.5
                            preds_mask[s:s+len(vi)] = np.where(pL[:, None, None], ~m_ed, m_ed)
                        else:
                            preds_mask[s:s+len(vi)] = prob >= 0.5

                ious = np.array([miou_image(preds_mask[k], masks128[va_idx[k]] > 0) for k in range(len(va_idx))])
                vm = methods[va_idx]; vs = shapes[va_idx]; vf_gt = full_wm[va_idx]
                cls_acc_v = float((preds_L == vf_gt).mean())
                ps_m = ious[vm == "pixelseal"].mean()
                tm_m = ious[vm == "trustmark"].mean()
                ss_m = ious[vm == "stegastamp"].mean()
                avg_m = (ps_m + tm_m + ss_m) / 3.0
                ps_L, ps_r, ps_o = ious[(vm=="pixelseal")&(vs=="large")].mean(), ious[(vm=="pixelseal")&(vs=="rect")].mean(), ious[(vm=="pixelseal")&(vs=="object")].mean()
                tm_L, tm_r, tm_o = ious[(vm=="trustmark")&(vs=="large")].mean(), ious[(vm=="trustmark")&(vs=="rect")].mean(), ious[(vm=="trustmark")&(vs=="object")].mean()
                line = (f"  step {step:3d} ({time.time()-t0:4.1f}s) | L/S={cls_acc_v*100:.1f}% | "
                        f"3-METH={avg_m:.4f} | PS={ps_m:.4f} (L {ps_L:.3f} r {ps_r:.3f} o {ps_o:.3f}) | "
                        f"TM={tm_m:.4f} (L {tm_L:.3f} r {tm_r:.3f} o {tm_o:.3f}) | SS={ss_m:.4f}")
                print(line, flush=True)
                if avg_m > best_val:
                    best_val = avg_m; best_detail = line
        print(f"  >>> BEST {mode}: {best_detail}", flush=True)


@app.local_entrypoint()
def main():
    run_forensic_probe.remote()
