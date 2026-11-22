"""Modal H100 5-in-1 White-Box Known-Message Physics Discovery for Task 1.
Reuses cached Modal Image + cached Modal Volume `task1-wm-vol`.
"""
import modal

app = modal.App("task1-wb-physics-probe2")
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


@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": vol},
    timeout=1800,
)
def run_probe():
    import os, sys, time, subprocess, pathlib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.ops import roi_align
    from sklearn.metrics import roc_auc_score
    import onnx
    from onnx import numpy_helper

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    # 1. Ensure official models downloaded into /vol/models
    if not (models_dir / "pixelseal" / "checkpoint.pth").exists() or not (models_dir / "stegastamp" / "stega_stamp.onnx").exists():
        print("=== Downloading official watermark models into /vol/models ===", flush=True)
        subprocess.run(["python", str(repo_dir / "download_models.py"), "--models_dir", str(models_dir)], check=True)
        vol.commit()

    # Inspect wmloc/watermarks.py to be 100% sure of API
    wm_py = (repo_dir / "wmloc" / "watermarks.py").read_text()
    print("=== wmloc/watermarks.py loaded ===", len(wm_py), "chars", flush=True)

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import miou_image, downsample_mask

    dev = "cuda:0"

    # Helper: exact PyTorch StegaStamp decoder (from ONNX weights, differentiable!)
    class StegaStampDecoderPT(nn.Module):
        def __init__(self, onnx_path):
            super().__init__()
            m = onnx.load(str(onnx_path))
            inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
            def mkc(wn, bn, s, p=0):
                w, b = inits[wn], inits[bn]
                c = nn.Conv2d(w.shape[1], w.shape[0], (w.shape[2], w.shape[3]), stride=s, padding=p)
                with torch.no_grad():
                    c.weight.copy_(torch.from_numpy(w.copy()))
                    c.bias.copy_(torch.from_numpy(b.copy()))
                return c
            def mkl(wn, bn):
                w, b = inits[wn], inits[bn]
                l = nn.Linear(w.shape[0], w.shape[1])
                with torch.no_grad():
                    l.weight.copy_(torch.from_numpy(w.T.copy()))
                    l.bias.copy_(torch.from_numpy(b.copy()))
                return l
            self.stn_conv_1 = mkc("gen_stn/stn_conv_1/Conv2D/ReadVariableOp__42", "gen_stn/stn_conv_1/BiasAdd/ReadVariableOp__43", 2)
            self.stn_conv_2 = mkc("gen_stn/stn_conv_2/Conv2D/ReadVariableOp__44", "gen_stn/stn_conv_2/BiasAdd/ReadVariableOp__45", 2)
            self.stn_conv_3 = mkc("gen_stn/stn_conv_3/Conv2D/ReadVariableOp__46", "gen_stn/stn_conv_3/BiasAdd/ReadVariableOp__47", 2)
            self.stn_fc_1 = mkl("gen_stn/dense_1/MatMul/ReadVariableOp__48", "gen_stn/dense_1/BiasAdd/ReadVariableOp__49")
            self.stn_fc_2 = mkl("gen_stn/W_fc1/read__50", "gen_stn/b_fc1/read__51")
            self.register_buffer("stn_grid", torch.from_numpy(inits["gen_stn/ExpandDims__52"].astype(np.float32).copy()))
            self.dec_conv_1 = mkc("gen_decoder/dec_conv_1/Conv2D/ReadVariableOp__61", "gen_decoder/dec_conv_1/BiasAdd/ReadVariableOp__62", 2)
            self.dec_conv_2 = mkc("gen_decoder/dec_conv_2/Conv2D/ReadVariableOp__63", "gen_decoder/dec_conv_2/BiasAdd/ReadVariableOp__64", 1, 1)
            self.dec_conv_3 = mkc("gen_decoder/dec_conv_3/Conv2D/ReadVariableOp__65", "gen_decoder/dec_conv_3/BiasAdd/ReadVariableOp__66", 2)
            self.dec_conv_4 = mkc("gen_decoder/dec_conv_4/Conv2D/ReadVariableOp__67", "gen_decoder/dec_conv_4/BiasAdd/ReadVariableOp__68", 1, 1)
            self.dec_conv_5 = mkc("gen_decoder/dec_conv_5/Conv2D/ReadVariableOp__69", "gen_decoder/dec_conv_5/BiasAdd/ReadVariableOp__70", 2)
            self.dec_conv_6 = mkc("gen_decoder/dec_conv_6/Conv2D/ReadVariableOp__71", "gen_decoder/dec_conv_6/BiasAdd/ReadVariableOp__72", 2)
            self.dec_conv_7 = mkc("gen_decoder/dec_conv_7/Conv2D/ReadVariableOp__73", "gen_decoder/dec_conv_7/BiasAdd/ReadVariableOp__74", 2, 1)
            self.dec_dense = mkl("gen_decoder/dec_dense/MatMul/ReadVariableOp__75", "gen_decoder/dec_dense/BiasAdd/ReadVariableOp__76")
            self.decoder_out = mkl("gen_decoder/decoder_out/MatMul/ReadVariableOp__77", "gen_decoder/decoder_out/BiasAdd/ReadVariableOp__78")

        def stn_warp(self, x):
            h = F.relu(self.stn_conv_1(F.pad(x - 0.5, (0, 1, 0, 1))))
            h = F.relu(self.stn_conv_2(F.pad(h, (0, 1, 0, 1))))
            h = F.relu(self.stn_conv_3(F.pad(h, (0, 1, 0, 1))))
            h = h.permute(0, 2, 3, 1).contiguous().flatten(1)
            theta = self.stn_fc_2(F.relu(self.stn_fc_1(h))).view(-1, 2, 3)
            B = x.shape[0]
            coords = torch.bmm(theta, self.stn_grid.expand(B, -1, -1)).view(B, 2, 400, 400)
            x_f = (coords[:, 0] + 1.0) * 199.0
            y_f = (coords[:, 1] + 1.0) * 199.0
            x0 = torch.floor(x_f).long().clamp(0, 399)
            x1 = (x0 + 1).clamp(0, 399)
            y0 = torch.floor(y_f).long().clamp(0, 399)
            y1 = (y0 + 1).clamp(0, 399)
            wx1 = (x_f - x0.float()).unsqueeze(1); wx0 = (x1.float() - x_f).unsqueeze(1)
            wy1 = (y_f - y0.float()).unsqueeze(1); wy0 = (y1.float() - y_f).unsqueeze(1)
            flat = x.view(B, 3, 160000)
            def g(yy, xx):
                idx = (yy * 400 + xx).view(B, 1, 160000).expand(B, 3, 160000)
                return torch.gather(flat, 2, idx).view(B, 3, 400, 400)
            return g(y0, x0) * (wx0 * wy0) + g(y1, x0) * (wx0 * wy1) + g(y0, x1) * (wx1 * wy0) + g(y1, x1) * (wx1 * wy1)

        def forward(self, x, use_stn=True):
            w = self.stn_warp(x) if use_stn else x
            h = F.relu(self.dec_conv_1(F.pad(w - 0.5, (0, 1, 0, 1))))
            h = F.relu(self.dec_conv_2(h))
            h = F.relu(self.dec_conv_3(F.pad(h, (0, 1, 0, 1))))
            h = F.relu(self.dec_conv_4(h))
            h = F.relu(self.dec_conv_5(F.pad(h, (0, 1, 0, 1))))
            h = F.relu(self.dec_conv_6(F.pad(h, (0, 1, 0, 1))))
            h = F.relu(self.dec_conv_7(h))
            h = h.permute(0, 2, 3, 1).contiguous().flatten(1)
            return self.decoder_out(F.relu(self.dec_dense(h)))

    def blur2d(x, k=9, sigma=2.5):
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        ker = (g[:, None] * g[None, :]).view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), ker, groups=x.shape[1])

    def smooth128(m, k=9, sigma=2.5):
        t = torch.from_numpy(m).float().to(dev)[None, None]
        return blur2d(t, k=k, sigma=sigma)[0, 0].cpu().numpy()

    def classify_item(it):
        mask = it["mask"] > 0
        if it["full_watermark"]:
            return "large"
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return "rect" if mask.sum() / max((y1 - y0) * (x1 - x0), 1) > 0.98 else "object"

    def eval_maps(tag, maps, items):
        print(f"  [{tag}]", flush=True)
        for grp in ("ALL", "large", "rect", "object"):
            sub = [it for it in items if (grp == "ALL" or classify_item(it) == grp)]
            if not sub: continue
            aucs, gts, scs = [], [], []
            for it in sub:
                gt128 = downsample_mask(it["mask"] > 0) > 0
                sc = maps[it["image_name"]]
                if gt128.min() != gt128.max():
                    aucs.append(roc_auc_score(gt128.ravel(), sc.ravel()))
                gts.append(gt128); scs.append(sc)
            best_iou = 0.0
            for mode in ("raw", "z"):
                vals = scs if mode == "raw" else [(s - s.mean()) / (s.std() + 1e-6) for s in scs]
                flat = np.concatenate([v.ravel() for v in vals])
                for thr in np.linspace(np.percentile(flat, 5), np.percentile(flat, 95), 25):
                    m = float(np.mean([miou_image(v > thr, g) for v, g in zip(vals, gts)]))
                    if m > best_iou: best_iou = m
            print(f"    {grp:6s} (n={len(sub):2d}): pixel-AUROC={np.mean(aucs):.4f} (med={np.median(aucs):.4f}) | raw-mIoU={best_iou:.4f}", flush=True)

    # ============================================================
    # PART A: PIXELSEAL GT-RECT PHYSICS DIAGNOSTIC (WHY 11/43 vs 32/43?!)
    # ============================================================
    print("\n================ PART A: PIXELSEAL GT-RECT PHYSICS ================", flush=True)
    wm_ps = load_watermark("pixelseal", device=dev)
    ds_ps = LocalisationDataset(str(repo_dir), "train", methods=["pixelseal"])
    ps_all = [ds_ps[i] for i in range(len(ds_ps))]
    ps_rects = [it for it in ps_all if classify_item(it) == "rect"]

    with torch.no_grad():
        stats = {k: [] for k in ("crop256", "crop512", "native", "pad512_zero", "full512_blur_out", "pxdec_full_auc", "area", "hw", "ar")}
        for it in ps_rects:
            img01 = torch.from_numpy(it["image"]).to(dev).permute(2, 0, 1)[None].float() / 255.0
            sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
            mask = it["mask"] > 0
            ys, xs = np.where(mask)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            h, w = y1 - y0, x1 - x0
            stats["area"].append((h * w) / (512 * 512))
            stats["hw"].append((h, w))
            stats["ar"].append(max(h, w) / max(min(h, w), 1))
            crop = img01[:, :, y0:y1, x0:x1]
            # 1) resize to 256
            lg = wm_ps.det(F.interpolate(crop, (256, 256), mode="bilinear", align_corners=False))[:, 1:].float() * sign
            stats["crop256"].append((float(lg.mean()), float((lg > 0).float().mean())))
            # 2) resize to 512
            lg = wm_ps.det(F.interpolate(crop, (512, 512), mode="bilinear", align_corners=False))[:, 1:].float() * sign
            stats["crop512"].append((float(lg.mean()), float((lg > 0).float().mean())))
            # 3) native (pad to mult of 32)
            nh, nw = max(64, ((h + 31) // 32) * 32), max(64, ((w + 31) // 32) * 32)
            lg = wm_ps.det(F.pad(crop, (0, nw - w, 0, nh - h), mode="replicate"))[:, 1:].float() * sign
            stats["native"].append((float(lg.mean()), float((lg > 0).float().mean())))
            # 4) full 512 zero outside
            m_t = torch.from_numpy(mask.astype(np.float32)).to(dev)[None, None]
            lg = wm_ps.det(img01 * m_t + 0.5 * (1.0 - m_t))[:, 1:].float() * sign
            stats["pad512_zero"].append((float(lg.mean()), float((lg > 0).float().mean())))
            # 5) full 512 heavy blur outside
            lg = wm_ps.det(img01 * m_t + blur2d(img01, 21, 7.0) * (1.0 - m_t))[:, 1:].float() * sign
            stats["full512_blur_out"].append((float(lg.mean()), float((lg > 0).float().mean())))
            # 6) full 512 native pixel_decoder AUROC
            det = wm_ps.det
            lat = det.convnext(det.preprocess(img01))
            up = det.pixel_decoder.output_upscaling(lat)
            sp = det.pixel_decoder.linear(up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float()
            sl = (sp[:, 1:] * sign[:, :, None, None]).mean(1, keepdim=True)
            sl128 = F.interpolate(sl, (128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            gt128 = downsample_mask(mask) > 0
            stats["pxdec_full_auc"].append(roc_auc_score(gt128.ravel(), sl128.ravel()))

        for k in ("crop256", "crop512", "native", "pad512_zero", "full512_blur_out"):
            ms = [x[0] for x in stats[k]]; acs = [x[1] for x in stats[k]]
            print(f"  PS GT-rect {k:17s}: med_m={np.median(ms):+.3f} med_acc={np.median(acs):.3f} | >0.70:{np.mean([a>0.7 for a in acs]):.2f} >0.85:{np.mean([a>0.85 for a in acs]):.2f} >0.95:{np.mean([a>0.95 for a in acs]):.2f}", flush=True)
        print(f"  PS GT-rect full512 pixel_decoder AUROC: mean={np.mean(stats['pxdec_full_auc']):.4f} med={np.median(stats['pxdec_full_auc']):.4f}", flush=True)
        order = np.argsort(stats["area"])
        print("  Detail across areas (area%, HxW | c256_acc, c512_acc, nat_acc, f512blur_acc, pxdec_auc):", flush=True)
        for idx in order[::4]:
            h, w = stats["hw"][idx]
            print(f"    {stats['area'][idx]*100:5.2f}% ({h:3d}x{w:3d}) | c256={stats['crop256'][idx][1]:.3f} c512={stats['crop512'][idx][1]:.3f} nat={stats['native'][idx][1]:.3f} f512b={stats['full512_blur_out'][idx][1]:.3f} pxauc={stats['pxdec_full_auc'][idx]:.3f}", flush=True)

    # ============================================================
    # PARTS B, C, D, E: WHITE-BOX PHYSICS PROBE ACROSS ALL 3 METHODS!
    # ============================================================
    # Precompute 16x16 occlusion / keep-only soft masks at 512x512 (or res)
    def make_grid_masks(grid=16, patch=64, res=512, dev="cuda:0"):
        ys = np.linspace(0, res - patch, grid).round().astype(int)
        xs = np.linspace(0, res - patch, grid).round().astype(int)
        # Build [G*G, 1, res, res] binary masks + [G*G, 128, 128] accumulation weights
        ms = torch.zeros((grid * grid, 1, res, res), device=dev, dtype=torch.float32)
        coords128 = []
        idx = 0
        for y in ys:
            for x in xs:
                ms[idx, 0, y:y + patch, x:x + patch] = 1.0
                y0 = int(round(y * 128 / res)); y1 = int(round((y + patch) * 128 / res))
                x0 = int(round(x * 128 / res)); x1 = int(round((x + patch) * 128 / res))
                coords128.append((y0, y1, x0, x1))
                idx += 1
        return ms, coords128

    def probe_method(method, n_per_grp=18):
        print(f"\n================ WHITE-BOX PHYSICS PROBE: {method.upper()} ================", flush=True)
        t0 = time.time()
        ds = LocalisationDataset(str(repo_dir), "train", methods=[method])
        all_items = [ds[i] for i in range(len(ds))]
        sub = (
            [it for it in all_items if classify_item(it) == "large"][:n_per_grp] +
            [it for it in all_items if classify_item(it) == "rect"][:n_per_grp] +
            [it for it in all_items if classify_item(it) == "object"][:n_per_grp]
        )
        print(f"  Loaded {len(sub)} balanced images (L/rect/obj)", flush=True)

        if method == "stegastamp":
            ss_pt = StegaStampDecoderPT(models_dir / "stegastamp" / "stega_stamp.onnx").to(dev).eval()
            for p in ss_pt.parameters(): p.requires_grad_(False)
            res = 400
            def decode_logits(x01):
                if x01.shape[-1] != 400:
                    x01 = F.interpolate(x01, (400, 400), mode="bilinear", align_corners=False)
                return ss_pt(x01, use_stn=True).float()
        elif method == "pixelseal":
            wm = wm_ps
            for p in wm.det.parameters(): p.requires_grad_(False)
            res = 256
            def decode_logits(x01):
                return wm.det(x01)[:, 1:].float()
        else:
            wm = load_watermark("trustmark", device=dev)
            for p in wm.net.parameters(): p.requires_grad_(False)
            res = wm.resolution
            def decode_logits(x01):
                if x01.shape[-1] != res:
                    x01 = F.interpolate(x01, (res, res), mode="bilinear", align_corners=False)
                return wm.net(x01 * 2.0 - 1.0).float()

        # Multi-crop tiles: 1x512 + 2x2@320 + 3x3@224 + 4x4@160 = 30 crops
        tile_specs = []
        for g, c in [(1, 512), (2, 320), (3, 224), (4, 160)]:
            xs = [0] if g == 1 else [int(round(i * (512 - c) / (g - 1))) for i in range(g)]
            for y in xs:
                for x in xs:
                    tile_specs.append((y, x, y + c, x + c))
        boxes_t = torch.tensor([[0, x0, y0, x1, y1] for (y0, x0, y1, x1) in tile_specs], device=dev, dtype=torch.float32)

        # Precompute occlusion grids at decoder native res (fast! 256 patches per batch on H100!)
        occ_res = 256 if method != "stegastamp" else 400
        occ_patch_sm = 32 if occ_res == 256 else 48
        occ_patch_lg = 64 if occ_res == 256 else 96
        ms_sm, c128_sm = make_grid_masks(16, occ_patch_sm, occ_res, dev)
        ms_lg, c128_lg = make_grid_masks(14, occ_patch_lg, occ_res, dev)

        maps_mf_full = {}
        maps_mf_multi = {}
        maps_grad_abs = {}
        maps_occ_drop_sm = {}
        maps_occ_drop_lg = {}
        maps_keep_only = {}
        maps_pxdec_ms = {}

        for it in sub:
            nm = it["image_name"]
            img01 = torch.from_numpy(it["image"]).to(dev).permute(2, 0, 1)[None].float() / 255.0
            sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
            hp = img01 - blur2d(img01, k=7, sigma=2.0)

            # B1) Full-image Matched-Filter Gradient: <grad_I L_msg, I - blur(I)>
            x = img01.detach().clone().requires_grad_(True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = decode_logits(x)
                loss = (lg * sign).mean()
            g = torch.autograd.grad(loss, x)[0].float()
            mf1 = (g * hp).sum(1, keepdim=True)
            ab1 = g.abs().sum(1, keepdim=True)
            maps_mf_full[nm] = smooth128(F.avg_pool2d(mf1, 4)[0, 0].cpu().numpy(), 9, 2.5)
            maps_grad_abs[nm] = smooth128(F.avg_pool2d(ab1, 4)[0, 0].cpu().numpy(), 9, 2.5)

            # B2) 30-Crop Softplus-Weighted Matched-Filter Gradient
            x2 = img01.detach().clone().requires_grad_(True)
            crops = roi_align(x2, boxes_t, output_size=res, sampling_ratio=2, aligned=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg_c = decode_logits(crops)
                crop_m = (lg_c * sign).mean(1)
                loss_c = F.softplus(crop_m).sum()
            g2 = torch.autograd.grad(loss_c, x2)[0].float()
            mf2 = (g2 * hp).sum(1, keepdim=True)
            maps_mf_multi[nm] = smooth128(F.avg_pool2d(mf2, 4)[0, 0].cpu().numpy(), 9, 2.5)

            # C) Dense Patch Occlusion Drop (replace patch with heavy blur -> drop = base - occ!)
            with torch.no_grad():
                x_r = F.interpolate(img01, (occ_res, occ_res), mode="bilinear", align_corners=False)
                x_blur = blur2d(x_r, k=15, sigma=5.0)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    base_m = float((decode_logits(x_r) * sign).mean().item())

                for ms_g, c128_g, out_dict in [(ms_sm, c128_sm, maps_occ_drop_sm), (ms_lg, c128_lg, maps_occ_drop_lg)]:
                    # x_occ = x_r*(1-M) + x_blur*M
                    x_occ = x_r * (1.0 - ms_g) + x_blur * ms_g
                    drops = []
                    bs = 128
                    for s in range(0, len(ms_g), bs):
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            lm = (decode_logits(x_occ[s:s+bs]) * sign).mean(1).float()
                        drops.append(base_m - lm)
                    drops = torch.cat(drops).cpu().numpy()
                    acc = np.zeros((128, 128), np.float32); cnt = np.zeros((128, 128), np.float32)
                    for val, (y0, y1, x0, x1) in zip(drops, c128_g):
                        acc[y0:y1, x0:x1] += val
                        cnt[y0:y1, x0:x1] += 1.0
                    out_dict[nm] = acc / np.maximum(cnt, 1.0)

                # D) Keep-Only Patch Isolation (keep ONLY patch, blur everything outside!)
                x_keep = x_r * ms_lg + x_blur * (1.0 - ms_lg)
                kvals = []
                for s in range(0, len(ms_lg), 128):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lm = (decode_logits(x_keep[s:s+128]) * sign).mean(1).float()
                    kvals.append(lm)
                kvals = torch.cat(kvals).cpu().numpy()
                acc = np.zeros((128, 128), np.float32); cnt = np.zeros((128, 128), np.float32)
                for val, (y0, y1, x0, x1) in zip(kvals, c128_lg):
                    acc[y0:y1, x0:x1] += val
                    cnt[y0:y1, x0:x1] += 1.0
                maps_keep_only[nm] = acc / np.maximum(cnt, 1.0)

                # E) PixelSeal Native Multi-Scale pixel_decoder.linear!
                if method == "pixelseal":
                    det = wm.det
                    acc_p = np.zeros((128, 128), np.float32)
                    for g_n, c_s in [(1, 512), (2, 320), (3, 256)]:
                        xs = [0] if g_n == 1 else [int(round(i * (512 - c_s) / (g_n - 1))) for i in range(g_n)]
                        tiles = torch.cat([img01[:, :, y:y+c_s, x:x+c_s] for y in xs for x in xs], dim=0)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            lat = det.convnext(det.preprocess(tiles))
                            up = det.pixel_decoder.output_upscaling(lat)
                            sp = det.pixel_decoder.linear(up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float()
                        sl = (sp[:, 1:] * sign[:, :, None, None]).mean(1, keepdim=True)
                        c128 = c_s // 4
                        sl_up = F.interpolate(sl, (c128, c128), mode="bilinear", align_corners=False).cpu().numpy()
                        a2 = np.zeros((128, 128), np.float32); w2 = np.zeros((128, 128), np.float32)
                        ti = 0
                        for y in xs:
                            for x in xs:
                                y4, x4 = y // 4, x // 4
                                a2[y4:y4+c128, x4:x4+c128] += sl_up[ti, 0]
                                w2[y4:y4+c128, x4:x4+c128] += 1.0
                                ti += 1
                        acc_p += a2 / np.maximum(w2, 1.0)
                    maps_pxdec_ms[nm] = acc_p / 3.0

        print(f"  [{method.upper()}] DONE in {time.time()-t0:.1f}s ({(time.time()-t0)/len(sub):.2f}s/img)", flush=True)
        eval_maps("B1: Full-Image Matched-Filter (Grad x HP)", maps_mf_full, sub)
        eval_maps("B2: 30-Crop Softplus Matched-Filter (Grad x HP)", maps_mf_multi, sub)
        eval_maps("B3: Full-Image |Grad|", maps_grad_abs, sub)
        eval_maps("C1: Dense 16x16 SMALL Patch Occlusion Drop", maps_occ_drop_sm, sub)
        eval_maps("C2: Dense 14x14 LARGE Patch Occlusion Drop", maps_occ_drop_lg, sub)
        eval_maps("D : Dense 14x14 Keep-Only Patch Isolation", maps_keep_only, sub)
        if method == "pixelseal":
            eval_maps("E : PixelSeal Native 3-Scale pixel_decoder", maps_pxdec_ms, sub)

    probe_method("pixelseal", n_per_grp=18)
    probe_method("trustmark", n_per_grp=18)
    probe_method("stegastamp", n_per_grp=15)


@app.local_entrypoint()
def main():
    run_probe.remote()
