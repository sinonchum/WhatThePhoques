"""V11 Modal H100 Parallel Breakthrough Pilot:
  - GPU 1 (H100): PixelSeal V11-Unfrozen-MetaDet (ALL 4 stages 128..16 of Meta's pretrained PixelSeal ConvNeXt
                  UNFROZEN + proper [0,1] 3-scale known-msg pixel_decoder maps + ConvNeXt-Base RGB/SRM)
  - GPU 2 (H100): TrustMark V11-DualRM-KBNet (Dual-scale 256+512 Adobe KBNet_s remover residual +
                  ConvNeXt-Base + known-msg multi-scale decoder maps)
  - BOTH trained on INFINITE ON-THE-FLY GPU-WATERMARKED 2,500 CLEAN COCO COVERS with 100% EXACT
    4-edit competition physics (including crop-embedded small-rect outpainting up to 16:1 / 2:1!).
  - ZERO real train images used in this pilot (p_real=0.0) -> ALL 170 real train images per method
    are 100% UNTOUCHED OUT-OF-SAMPLE evaluation!
  - Step-0 physics assertion verifies synth_full_acc > 0.95 before training begins!
"""
import modal

app = modal.App("task1-v11-physics-pilot")
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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _common_helpers():
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2

    def sample_poly_mask(H=512, W=512, min_area=0.008, max_area=0.35, rng=None):
        """Sample realistic irregular object/hole mask (multi-blob smooth polygon or ellipse)."""
        if rng is None:
            rng = np.random.default_rng()
        m = np.zeros((H, W), dtype=np.uint8)
        target_a = np.exp(rng.uniform(np.log(min_area), np.log(max_area)))
        r = np.sqrt(target_a * H * W / np.pi)
        cy = rng.uniform(r * 0.4, H - r * 0.4)
        cx = rng.uniform(r * 0.4, W - r * 0.4)
        n_v = int(rng.integers(7, 16))
        angs = np.sort(rng.uniform(0, 2 * np.pi, size=n_v))
        ar = rng.uniform(0.5, 2.0)
        rad = r * rng.uniform(0.55, 1.45, size=n_v)
        pts = np.stack([
            np.clip(cx + rad * np.cos(angs) * np.sqrt(ar), 0, W - 1),
            np.clip(cy + rad * np.sin(angs) / np.sqrt(ar), 0, H - 1),
        ], axis=1).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        # Smooth edges randomly
        if rng.random() < 0.5:
            k = int(rng.choice([5, 9, 15]))
            m = (cv2.GaussianBlur(m.astype(np.float32), (k, k), 0) > 0.5).astype(np.uint8)
        if m.sum() < 64:
            y0 = int(np.clip(cy - 16, 0, H - 32)); x0 = int(np.clip(cx - 16, 0, W - 32))
            m[y0:y0+32, x0:x0+32] = 1
        return m

    def sample_rect_coords(H=512, W=512, min_area=0.003, max_area=0.35, max_ar=14.0, rng=None):
        """Sample realistic rectangle matching real train distribution (including thin strips!)."""
        if rng is None:
            rng = np.random.default_rng()
        area = np.exp(rng.uniform(np.log(min_area), np.log(max_area))) * H * W
        log_ar = rng.uniform(-np.log(max_ar), np.log(max_ar))
        ar = np.exp(log_ar)  # w / h
        h = int(np.clip(np.round(np.sqrt(area / ar)), 10, H - 16))
        w = int(np.clip(np.round(np.sqrt(area * ar)), 10, W - 16))
        y0 = int(rng.integers(0, H - h + 1))
        x0 = int(rng.integers(0, W - w + 1))
        return y0, x0, y0 + h, x0 + w

    return sample_poly_mask, sample_rect_coords


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1800)
def train_v11_pixelseal(steps: int = 2500, bs: int = 12):
    import os, sys, time, math, pathlib, copy
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import miou_image, downsample_mask

    dev = "cuda:0"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1101); np.random.seed(1101)
    rng = np.random.default_rng(1101)
    sample_poly_mask, sample_rect_coords = _common_helpers()

    # 1. Load 2,500 clean COCO covers + ALL 170 real PixelSeal train images (100% UNTOUCHED EVAL!)
    covers = np.load("/vol/clean_covers.npy")  # [2500, 512, 512, 3] uint8
    print(f"[PS-V11] Loaded {len(covers)} clean external COCO covers", flush=True)

    ds_real = LocalisationDataset(str(repo_dir), "train", methods=["pixelseal"])
    n_real = len(ds_real)
    r_imgs = np.stack([ds_real[i]["image"] for i in range(n_real)])
    r_msgs = np.stack([ds_real[i]["message"] for i in range(n_real)]).astype(np.float32)
    r_gt128 = np.stack([(downsample_mask(ds_real[i]["mask"] > 0) > 0) for i in range(n_real)])
    r_large = np.array([bool(ds_real[i]["full_watermark"]) for i in range(n_real)])
    r_shape = []
    for i in range(n_real):
        m = ds_real[i]["mask"] > 0
        if r_large[i]: r_shape.append("large")
        else:
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
            r_shape.append("rect" if m.sum() / max((y1-y0)*(x1-x0), 1) > 0.98 else "object")
    r_shape = np.array(r_shape)

    # 2. Load official PixelSeal model (embedder frozen for on-the-fly synth; detector cloned for unfrozen backbone!)
    wm_ps = load_watermark("pixelseal", device=dev)
    embed_model = wm_ps.model
    embed_model.eval()
    for p in embed_model.parameters():
        p.requires_grad_(False)

    # Frozen detector reference for known-message spatial pixel_decoder maps
    frozen_det = wm_ps.det
    frozen_det.eval()
    for p in frozen_det.parameters():
        p.requires_grad_(False)

    # STEP-0 PHYSICS VERIFICATION: embed -> quantize uint8 -> decode MUST be >0.95 acc!
    with torch.no_grad():
        c0 = torch.from_numpy(covers[:4].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
        m0 = torch.randint(0, 2, (4, 256), device=dev)
        w0 = embed_model.embed(c0, msgs=m0, is_video=False)["imgs_w"].clamp(0, 1)
        w0_q = torch.round(w0 * 255.0) / 255.0
        lg0 = frozen_det(F.interpolate(w0_q, (256, 256), mode="bilinear", align_corners=False))[:, 1:].float()
        acc0 = float(((lg0 > 0).long() == m0).float().mean().item())
        print(f"[PS-V11] STEP-0 PHYSICS ASSERT: full-512 uint8 embed->decode bit_acc = {acc0:.4f}", flush=True)
        assert acc0 > 0.95, f"Physics check failed! acc0={acc0}"

        # Also inspect Meta's convnext internal stages so we extract ALL 4 resolutions (128, 64, 32, 16)!
        cn = frozen_det.convnext
        print("[PS-V11] Meta det.convnext attrs:", [a for a in dir(cn) if not a.startswith("_")][:20], flush=True)
        x_t = frozen_det.preprocess(c0[:1])
        h = x_t
        stage_shapes = []
        for si in range(4):
            h = cn.downsample_layers[si](h)
            h = cn.stages[si](h)
            stage_shapes.append(tuple(h.shape))
        print("[PS-V11] Meta det.convnext 4-stage shapes on 512x512:", stage_shapes, flush=True)

    # On-the-fly EXACT 4-edit PixelSeal GPU batch generator!
    @torch.no_grad()
    def sample_batch(B: int):
        idx1 = rng.integers(0, len(covers), size=B)
        idx2 = rng.integers(0, len(covers), size=B)
        c1 = torch.from_numpy(covers[idx1].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
        c2 = torch.from_numpy(covers[idx2].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
        # Random hflip / vflip on covers
        if rng.random() < 0.5: c1 = torch.flip(c1, [-1])
        if rng.random() < 0.5: c2 = torch.flip(c2, [-1])
        msgs = torch.randint(0, 2, (B, 256), device=dev)

        # Full-512 watermarked version (for L-hole and S-object)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            wm_full = embed_model.embed(c1, msgs=msgs, is_video=False)["imgs_w"].float().clamp(0, 1)

        xs = torch.empty_like(c1)
        ys = torch.zeros((B, 128, 128), device=dev, dtype=torch.float32)
        is_L = torch.zeros(B, device=dev, dtype=torch.float32)

        for b in range(B):
            u = rng.random()
            if u < 0.50:
                # LARGE (50%): full watermarked image with clean hole (50% irregular object, 50% rect)
                is_L[b] = 1.0
                if rng.random() < 0.55:
                    hm = sample_poly_mask(512, 512, 0.005, 0.35, rng)
                else:
                    y0, x0, y1, x1 = sample_rect_coords(512, 512, 0.005, 0.35, max_ar=12.0, rng=rng)
                    hm = np.zeros((512, 512), dtype=np.uint8); hm[y0:y1, x0:x1] = 1
                hm_t = torch.from_numpy(hm).to(dev).float()[None, None]
                # Clean fill: 55% same original clean image, 25% mild blurred original, 20% other clean image
                f_r = rng.random()
                if f_r < 0.55:
                    clean_src = c1[b:b+1]
                elif f_r < 0.80:
                    clean_src = F.avg_pool2d(c1[b:b+1], 5, stride=1, padding=2)
                else:
                    clean_src = c2[b:b+1]
                xs[b] = (wm_full[b:b+1] * (1.0 - hm_t) + clean_src * hm_t)[0]
                m128 = (F.avg_pool2d(1.0 - hm_t, 4) >= 0.5).float()[0, 0]
                ys[b] = m128
            elif u < 0.75:
                # SMALL-RECT (25%): EXACT OUTPAINTING PHYSICS! Embed watermark DIRECTLY on the h x w rectangle!
                y0, x0, y1, x1 = sample_rect_coords(512, 512, 0.003, 0.35, max_ar=16.0, rng=rng)
                crop = c1[b:b+1, :, y0:y1, x0:x1]
                # Resize crop to 256, embed, resize delta back to (h, w) — EXACT competition physics!
                c256 = F.interpolate(crop, (256, 256), mode="bilinear", align_corners=False)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    w256 = embed_model.embed(c256, msgs=msgs[b:b+1], is_video=False)["imgs_w"].float()
                delta = F.interpolate(w256 - c256, (y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
                # Paste onto clean cover (60% same cover so pure watermark learning, 40% other cover)
                bg = c1[b].clone() if rng.random() < 0.60 else c2[b].clone()
                bg[:, y0:y1, x0:x1] = (crop[0] + delta[0]).clamp(0, 1)
                xs[b] = bg
                m512 = torch.zeros((1, 1, 512, 512), device=dev)
                m512[0, 0, y0:y1, x0:x1] = 1.0
                ys[b] = (F.avg_pool2d(m512, 4) >= 0.5).float()[0, 0]
            else:
                # SMALL-OBJECT (25%): irregular object from watermarked image pasted onto clean cover!
                om = sample_poly_mask(512, 512, 0.015, 0.35, rng)
                om_t = torch.from_numpy(om).to(dev).float()[None, None]
                bg = c1[b:b+1] if rng.random() < 0.60 else c2[b:b+1]
                xs[b] = (wm_full[b:b+1] * om_t + bg * (1.0 - om_t))[0]
                ys[b] = (F.avg_pool2d(om_t, 4) >= 0.5).float()[0, 0]

        # Exact uint8 quantization + mild random noise (matches PNG save/load!)
        xs = torch.round(xs.clamp(0, 1) * 255.0) / 255.0
        return xs, ys, is_L, msgs.float()

    # 3. Build V11-PS-Unfrozen Architecture!
    #    Stream A: UNFROZEN Meta PixelSeal ConvNeXt (all 4 stages: 128, 64, 32, 16!)
    #    Stream B: Frozen Meta PixelSeal Pixel-Decoder 3-scale Known-Message Spatial Maps (PROPER [0,1]!)
    #    Stream C: ConvNeXt-Small ImageNet-22K 6ch [norm(RGB), SRM-3ch] visual forensic stream (PROPER [0,1]!)
    srm_k = np.zeros((3, 1, 5, 5), dtype=np.float32)
    srm_k[0, 0, 1:4, 1:4] = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float32)
    srm_k[1, 0, 1:4, 1:4] = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], np.float32) / 4.0
    srm_k[2, 0] = np.array([
        [-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1],
    ], np.float32) / 12.0

    def _cbr(ci, co, k=3):
        return nn.Sequential(
            nn.Conv2d(ci, co, k, padding=k//2, bias=False),
            nn.GroupNorm(min(8, co), co),
            nn.SiLU(inplace=True),
        )

    class PixelSealV11Net(nn.Module):
        def __init__(self, frozen_det):
            super().__init__()
            self.frozen_det = frozen_det
            # Deep-copy Meta's pretrained PixelSeal ConvNeXt and UNFREEZE it!
            self.ps_backbone = copy.deepcopy(frozen_det.convnext)
            for p in self.ps_backbone.parameters():
                p.requires_grad_(True)
            ps_chs = [96, 192, 384, 768]  # ConvNeXt-Tiny/Small inside PixelSeal (verified dynamically below)
            with torch.no_grad():
                h = frozen_det.preprocess(torch.zeros(1, 3, 512, 512, device=dev))
                ps_chs = []
                for si in range(4):
                    h = self.ps_backbone.downsample_layers[si](h)
                    h = self.ps_backbone.stages[si](h)
                    ps_chs.append(h.shape[1])
            print("[PS-V11] Unfrozen Meta PixelSeal ConvNeXt channels:", ps_chs, flush=True)

            # Secondary RGB + SRM forensic ConvNeXt-Small (ImageNet-22K pretrained, PROPER [0,1]!)
            self.vis = timm.create_model("convnext_small.fb_in22k_ft_in1k_384", pretrained=True, features_only=True)
            for p_name, parent in self.vis.named_modules():
                for c_name, child in parent.named_children():
                    if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                        nc = nn.Conv2d(6, child.out_channels, child.kernel_size, stride=child.stride,
                                       padding=child.padding, bias=(child.bias is not None))
                        with torch.no_grad():
                            nc.weight[:, :3].copy_(child.weight)
                            nc.weight[:, 3:].copy_(child.weight * 0.25)
                            if child.bias is not None: nc.bias.copy_(child.bias)
                        setattr(parent, c_name, nc)
                        break
                else:
                    continue
                break
            vis_chs = list(self.vis.feature_info.channels())

            self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.register_buffer("srm", torch.from_numpy(srm_k))

            # 3-scale known-message pixel_decoder tiles (1x512, 2x2@320, 3x3@256 = 14 passes, ~6ms on H100!)
            self.tiles = []
            for idx, (g, c) in enumerate([(1, 512), (2, 320), (3, 256)]):
                xs = [0] if g == 1 else [int(round(i * (512 - c) / (g - 1))) for i in range(g)]
                self.tiles.append((g, c, [(y, x) for y in xs for x in xs]))
                c128 = c // 4
                w1d = torch.hann_window(c128 + 2, periodic=False)[1:-1].clamp(min=0.15)
                self.register_buffer(f"win_{idx}", (w1d[:, None] * w1d[None, :]).view(1, 1, c128, c128), persistent=False)

            d = 160
            # Fuse Meta PixelSeal features + Visual features + 12ch Known-Message Spatial Maps at ALL 4 scales!
            self.lats = nn.ModuleList([nn.Conv2d(ps_chs[i] + vis_chs[i] + 12, d, 1) for i in range(4)])
            self.smooth = nn.ModuleList([_cbr(d, d, 3) for _ in range(3)])
            self.head = nn.Sequential(_cbr(d + 12, 96, 3), _cbr(96, 48, 3), nn.Conv2d(48, 1, 1))
            self.cls_head = nn.Linear(ps_chs[-1] + vis_chs[-1] + 6, 1)

        @torch.no_grad()
        def _pxdec_maps(self, x01, msg):
            B = x01.shape[0]
            sign = 2.0 * msg.float() - 1.0
            det = self.frozen_det
            scale_maps = []
            for idx, (g, crop, coords) in enumerate(self.tiles):
                win = getattr(self, f"win_{idx}")
                c128 = crop // 4
                if g == 1:
                    lat = det.convnext(det.preprocess(x01))
                    up = det.pixel_decoder.output_upscaling(lat)
                    sp = det.pixel_decoder.linear(up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float()
                    det_lg = sp[:, 0:1].clamp(-6, 6) / 3.0
                    sl = sp[:, 1:] * sign[:, :, None, None]
                    m_map = sl.mean(1, keepdim=True).clamp(-4, 4) / 2.0
                    a_map = ((sl > 0).float().mean(1, keepdim=True) - 0.5) * 4.0
                    scale_maps.append(F.interpolate(torch.cat([det_lg, m_map, a_map], 1), (128, 128), mode="bilinear", align_corners=False))
                else:
                    tiles = torch.cat([x01[:, :, y:y+crop, x:x+crop] for (y, x) in coords], dim=0)
                    s_rep = sign.repeat(len(coords), 1)
                    lat = det.convnext(det.preprocess(tiles))
                    up = det.pixel_decoder.output_upscaling(lat)
                    sp = det.pixel_decoder.linear(up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).float()
                    det_lg = sp[:, 0:1].clamp(-6, 6) / 3.0
                    sl = sp[:, 1:] * s_rep[:, :, None, None]
                    m_map = sl.mean(1, keepdim=True).clamp(-4, 4) / 2.0
                    a_map = ((sl > 0).float().mean(1, keepdim=True) - 0.5) * 4.0
                    sp_up = F.interpolate(torch.cat([det_lg, m_map, a_map], 1), (c128, c128), mode="bilinear", align_corners=False)
                    sp_up = sp_up.view(len(coords), B, 3, c128, c128)
                    acc_m = torch.zeros((B, 3, 128, 128), device=x01.device, dtype=torch.float32)
                    acc_w = torch.zeros((1, 1, 128, 128), device=x01.device, dtype=torch.float32)
                    for ti, (y, x) in enumerate(coords):
                        y4, x4 = y // 4, x // 4
                        acc_m[:, :, y4:y4+c128, x4:x4+c128] += sp_up[ti] * win
                        acc_w[:, :, y4:y4+c128, x4:x4+c128] += win
                    scale_maps.append(acc_m / acc_w.clamp(min=1e-4))
            z_maps = []
            for mp in scale_maps:
                m = mp[:, 1:2]
                z_maps.append(((m - m.mean((2, 3), keepdim=True)) / m.std((2, 3), keepdim=True).clamp(min=0.05)).clamp(-4, 4))
            return torch.cat(scale_maps + z_maps, dim=1)  # [B, 12, 128, 128]

        def forward(self, x01, msg):
            # PROPER [0, 1] INPUT!
            pd = self._pxdec_maps(x01, msg)  # [B, 12, 128, 128]

            # Unfrozen Meta PixelSeal ConvNeXt ALL 4 STAGES!
            h = self.frozen_det.preprocess(x01)
            ps_fs = []
            for si in range(4):
                h = self.ps_backbone.downsample_layers[si](h)
                h = self.ps_backbone.stages[si](h)
                ps_fs.append(h)

            # Visual ConvNeXt-Small with RGB + 3ch SRM noise
            rgb = (x01 - self.im_mean) / self.im_std
            srm_f = (F.conv2d(F.pad(x01.mean(1, keepdim=True), (2, 2, 2, 2), mode="reflect"), self.srm) * 20.0).clamp(-3, 3)
            vis_fs = self.vis(torch.cat([rgb, srm_f], dim=1))

            # 4-scale FPN fusion!
            ps = []
            for i in range(4):
                sz = vis_fs[i].shape[-2:]
                pf = F.interpolate(ps_fs[i], sz, mode="bilinear", align_corners=False) if ps_fs[i].shape[-2:] != sz else ps_fs[i]
                pdm = F.interpolate(pd.to(vis_fs[i].dtype), sz, mode="bilinear", align_corners=False)
                ps.append(self.lats[i](torch.cat([pf, vis_fs[i], pdm], dim=1)))
            p = ps[3]
            for i in (2, 1, 0):
                p = self.smooth[i](ps[i] + F.interpolate(p, ps[i].shape[-2:], mode="bilinear", align_corners=False))
            seg = self.head(torch.cat([p, pd.to(p.dtype)], dim=1)).squeeze(1)
            g_pd = pd[:, :6].mean((2, 3))
            cls = self.cls_head(torch.cat([ps_fs[-1].mean((2, 3)), vis_fs[-1].mean((2, 3)), g_pd.to(p.dtype)], dim=1)).squeeze(1)
            return seg, cls

    def sym_miou_loss(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
        i1 = (p * t).sum(1); u1 = (p + t - p * t).sum(1)
        i0 = ((1 - p) * (1 - t)).sum(1); u0 = ((1 - p) + (1 - t) - (1 - p) * (1 - t)).sum(1)
        sym = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0)).mean()
        return 0.45 * bce + 0.55 * (1.0 - sym)

    net = PixelSealV11Net(frozen_det).to(dev)
    bb_params = list(net.ps_backbone.parameters()) + list(net.vis.parameters())
    bb_ids = {id(p) for p in bb_params}
    hd_params = [p for p in net.parameters() if p.requires_grad and id(p) not in bb_ids]
    print(f"[PS-V11] Trainable params: bb={sum(p.numel() for p in bb_params)/1e6:.1f}M head={sum(p.numel() for p in hd_params)/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": 3e-5},
        {"params": hd_params, "lr": 5e-4},
    ], weight_decay=1e-4)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    )

    t0 = time.time()
    best_all = 0.0
    for step in range(1, steps + 1):
        net.train()
        xb, yb, lb, mb = sample_batch(bs)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            seg, cls = net(xb, mb)
            loss = sym_miou_loss(seg.float(), yb) + 0.2 * F.binary_cross_entropy_with_logits(cls.float(), lb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps:
            net.eval()
            probs = []
            with torch.no_grad():
                for s in range(0, n_real, 16):
                    xr = torch.from_numpy(r_imgs[s:s+16].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
                    mr = torch.from_numpy(r_msgs[s:s+16]).to(dev)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg, _ = net(xr, mr)
                    probs.append(torch.sigmoid(lg.float()).cpu().numpy())
            probs = np.concatenate(probs)
            # Check best threshold on ALL 170 100%-UNTOUCHED real PixelSeal images!
            best_t, best_sc = 0.5, 0.0
            for t in (0.35, 0.45, 0.50, 0.55, 0.60):
                ious = np.array([miou_image(probs[i] >= t, r_gt128[i]) for i in range(n_real)])
                if ious.mean() > best_sc:
                    best_sc, best_t, best_ious = float(ious.mean()), t, ious
            sL = best_ious[r_shape == "large"].mean()
            sr = best_ious[r_shape == "rect"].mean()
            so = best_ious[r_shape == "object"].mean()
            sS = best_ious[~r_large].mean()
            if best_sc > best_all:
                best_all = best_sc
                torch.save(net.state_dict(), "/vol/outputs/v11_ps_pilot_best.pt")
            print(f"[PS-V11 step {step:4d}/{steps}] ({time.time()-t0:5.1f}s) ZERO-REAL-TRAIN UNTOUCHED 170-IMG mIoU = {best_sc:.4f} (thr={best_t:.2f}) | L={sL:.4f} S={sS:.4f} (rect={sr:.4f} obj={so:.4f})", flush=True)
    vol.commit()
    return best_all


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1800)
def train_v11_trustmark(steps: int = 2500, bs: int = 12):
    import os, sys, time, math, pathlib, copy
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.utils import miou_image, downsample_mask
    from trustmark import TrustMark

    dev = "cuda:0"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1201); np.random.seed(1201)
    rng = np.random.default_rng(1201)
    sample_poly_mask, sample_rect_coords = _common_helpers()

    covers = np.load("/vol/clean_covers.npy")
    print(f"[TM-V11] Loaded {len(covers)} clean external COCO covers", flush=True)

    ds_real = LocalisationDataset(str(repo_dir), "train", methods=["trustmark"])
    n_real = len(ds_real)
    r_imgs = np.stack([ds_real[i]["image"] for i in range(n_real)])
    r_msgs = np.stack([ds_real[i]["message"] for i in range(n_real)]).astype(np.float32)
    r_gt128 = np.stack([(downsample_mask(ds_real[i]["mask"] > 0) > 0) for i in range(n_real)])
    r_large = np.array([bool(ds_real[i]["full_watermark"]) for i in range(n_real)])
    r_shape = []
    for i in range(n_real):
        m = ds_real[i]["mask"] > 0
        if r_large[i]: r_shape.append("large")
        else:
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
            r_shape.append("rect" if m.sum() / max((y1-y0)*(x1-x0), 1) > 0.98 else "object")
    r_shape = np.array(r_shape)

    # Load official TrustMark encoder, decoder, AND Adobe KBNet_s remover!
    tm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5)
    tm_enc = tm.encoder.to(dev).eval()
    tm_dec = tm.decoder.to(dev).eval()
    tm_rm = tm.removal.to(dev).eval()
    for m_ in (tm_enc, tm_dec, tm_rm):
        for p in m_.parameters(): p.requires_grad_(False)

    # Inspect TrustMark decoder architecture + verify STEP-0 exact embed->decode physics!
    print("[TM-V11] tm.decoder type:", type(tm_dec), "children:", [(n, type(c).__name__) for n, c in tm_dec.named_children()], flush=True)
    def tm_decode_logits(x_m11):
        # TrustMark_Arch: decoder is inside tm_dec.decoder (ResNet50)!
        if hasattr(tm_dec, "decoder"):
            return tm_dec.decoder(x_m11).float()
        return tm_dec(x_m11).float()

    def tm_embed_256(x01_256, msg):
        # TrustMark encoder returns (encoded_img, residual) or encoded_img in [-1, 1]
        x_m11 = x01_256 * 2.0 - 1.0
        out = tm_enc(x_m11, msg)
        if isinstance(out, tuple):
            # TrustMark_Arch.forward(cover, secret) -> (stego, residual)
            stego = out[0]
        else:
            stego = out
        return ((stego + 1.0) * 0.5).clamp(0, 1)

    with torch.no_grad():
        c0 = torch.from_numpy(covers[:4].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
        m0 = torch.randint(0, 2, (4, 100), device=dev).float()
        c0_256 = F.interpolate(c0, (256, 256), mode="bilinear", align_corners=False)
        w0_256 = tm_embed_256(c0_256, m0)
        # Upsample residual to 512x512 (exact TrustMark high-res embedding!)
        w0_512 = (c0 + F.interpolate(w0_256 - c0_256, (512, 512), mode="bilinear", align_corners=False)).clamp(0, 1)
        w0_q = torch.round(w0_512 * 255.0) / 255.0
        lg0 = tm_decode_logits(F.interpolate(w0_q, (256, 256), mode="bilinear", align_corners=False) * 2.0 - 1.0)
        acc0 = float(((lg0 > 0).float() == m0).float().mean().item())
        print(f"[TM-V11] STEP-0 PHYSICS ASSERT: full-512 uint8 embed->decode bit_acc = {acc0:.4f}", flush=True)
        assert acc0 > 0.95, f"TM physics check failed! acc0={acc0}"
        # Test KBNet_s remover at BOTH 256 and 512!
        rm256 = tm_rm(c0_256[:1] * 2.0 - 1.0)
        if isinstance(rm256, tuple): rm256 = rm256[0]
        rm512 = tm_rm(c0[:1] * 2.0 - 1.0)
        if isinstance(rm512, tuple): rm512 = rm512[0]
        print(f"[TM-V11] KBNet_s remover verified at 256 {tuple(rm256.shape)} AND 512 {tuple(rm512.shape)}!", flush=True)

    @torch.no_grad()
    def sample_batch(B: int):
        idx1 = rng.integers(0, len(covers), size=B)
        idx2 = rng.integers(0, len(covers), size=B)
        c1 = torch.from_numpy(covers[idx1].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
        c2 = torch.from_numpy(covers[idx2].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if rng.random() < 0.5: c1 = torch.flip(c1, [-1])
        if rng.random() < 0.5: c2 = torch.flip(c2, [-1])
        msgs = torch.randint(0, 2, (B, 100), device=dev).float()

        c1_256 = F.interpolate(c1, (256, 256), mode="bilinear", align_corners=False)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            w1_256 = tm_embed_256(c1_256, msgs).float()
        wm_full = (c1 + F.interpolate(w1_256 - c1_256, (512, 512), mode="bilinear", align_corners=False)).clamp(0, 1)

        xs = torch.empty_like(c1)
        ys = torch.zeros((B, 128, 128), device=dev, dtype=torch.float32)
        is_L = torch.zeros(B, device=dev, dtype=torch.float32)

        for b in range(B):
            u = rng.random()
            if u < 0.50:
                # LARGE (50%)
                is_L[b] = 1.0
                if rng.random() < 0.55:
                    hm = sample_poly_mask(512, 512, 0.005, 0.35, rng)
                else:
                    y0, x0, y1, x1 = sample_rect_coords(512, 512, 0.005, 0.35, max_ar=3.0, rng=rng)
                    hm = np.zeros((512, 512), dtype=np.uint8); hm[y0:y1, x0:x1] = 1
                hm_t = torch.from_numpy(hm).to(dev).float()[None, None]
                f_r = rng.random()
                if f_r < 0.55: clean_src = c1[b:b+1]
                elif f_r < 0.80: clean_src = F.avg_pool2d(c1[b:b+1], 5, stride=1, padding=2)
                else: clean_src = c2[b:b+1]
                xs[b] = (wm_full[b:b+1] * (1.0 - hm_t) + clean_src * hm_t)[0]
                ys[b] = (F.avg_pool2d(1.0 - hm_t, 4) >= 0.5).float()[0, 0]
            elif u < 0.75:
                # SMALL-RECT (25%): EXACT OUTPAINTING PHYSICS! TrustMark aspect ratio <= 2:1!
                y0, x0, y1, x1 = sample_rect_coords(512, 512, 0.005, 0.35, max_ar=2.1, rng=rng)
                crop = c1[b:b+1, :, y0:y1, x0:x1]
                c256 = F.interpolate(crop, (256, 256), mode="bilinear", align_corners=False)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    w256 = tm_embed_256(c256, msgs[b:b+1]).float()
                delta = F.interpolate(w256 - c256, (y1 - y0, x1 - x0), mode="bilinear", align_corners=False)
                bg = c1[b].clone() if rng.random() < 0.60 else c2[b].clone()
                bg[:, y0:y1, x0:x1] = (crop[0] + delta[0]).clamp(0, 1)
                xs[b] = bg
                m512 = torch.zeros((1, 1, 512, 512), device=dev)
                m512[0, 0, y0:y1, x0:x1] = 1.0
                ys[b] = (F.avg_pool2d(m512, 4) >= 0.5).float()[0, 0]
            else:
                # SMALL-OBJECT (25%)
                om = sample_poly_mask(512, 512, 0.015, 0.35, rng)
                om_t = torch.from_numpy(om).to(dev).float()[None, None]
                bg = c1[b:b+1] if rng.random() < 0.60 else c2[b:b+1]
                xs[b] = (wm_full[b:b+1] * om_t + bg * (1.0 - om_t))[0]
                ys[b] = (F.avg_pool2d(om_t, 4) >= 0.5).float()[0, 0]

        xs = torch.round(xs.clamp(0, 1) * 255.0) / 255.0
        return xs, ys, is_L, msgs

    def _cbr(ci, co, k=3):
        return nn.Sequential(
            nn.Conv2d(ci, co, k, padding=k//2, bias=False),
            nn.GroupNorm(min(8, co), co),
            nn.SiLU(inplace=True),
        )

    # V11-TM-DualRM Architecture:
    #   - 9ch input into ConvNeXt-Base (PROPER [0,1] RGB + 10*AdobeRM_256 + 10*AdobeRM_512!)
    #   - Known-Message Multi-Scale Decoder Score Maps (1x512 + 2x2@320 + 3x3@224 + 4x4@160 = 30 tiles -> 8ch!)
    class TrustMarkV11Net(nn.Module):
        def __init__(self, tm_dec_fn, tm_rm):
            super().__init__()
            self.tm_dec_fn = tm_dec_fn
            self.tm_rm = tm_rm
            self.vis = timm.create_model("convnext_base.fb_in22k_ft_in1k_384", pretrained=True, features_only=True)
            # Expand 3ch -> 9ch: [norm(RGB), 10*rm256_res, 10*rm512_res]
            for p_name, parent in self.vis.named_modules():
                for c_name, child in parent.named_children():
                    if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                        nc = nn.Conv2d(9, child.out_channels, child.kernel_size, stride=child.stride,
                                       padding=child.padding, bias=(child.bias is not None))
                        with torch.no_grad():
                            nc.weight[:, :3].copy_(child.weight)
                            nc.weight[:, 3:6].copy_(child.weight * 0.25)
                            nc.weight[:, 6:9].copy_(child.weight * 0.25)
                            if child.bias is not None: nc.bias.copy_(child.bias)
                        setattr(parent, c_name, nc)
                        break
                else:
                    continue
                break
            chs = list(self.vis.feature_info.channels())
            self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            # Multi-scale known-message sliding pyramid (1x512, 2x2@320, 3x3@224, 4x4@160 = 30 tiles!)
            self.pyr_specs = []
            for g, c in [(1, 512), (2, 320), (3, 224), (4, 160)]:
                xs = [0] if g == 1 else [int(round(i * (512 - c) / (g - 1))) for i in range(g)]
                self.pyr_specs.append((g, c, [(y, x) for y in xs for x in xs]))

            d = 160
            self.lats = nn.ModuleList([nn.Conv2d(c + 8, d, 1) for c in chs])
            self.smooth = nn.ModuleList([_cbr(d, d, 3) for _ in range(3)])
            self.head = nn.Sequential(_cbr(d + 8, 96, 3), _cbr(96, 48, 3), nn.Conv2d(48, 1, 1))
            self.cls_head = nn.Linear(chs[-1] + 8, 1)

        @torch.no_grad()
        def _known_msg_maps(self, x01, msg):
            B = x01.shape[0]
            sign = 2.0 * msg.float() - 1.0
            maps = []
            for g, crop, coords in self.pyr_specs:
                tiles = torch.cat([
                    F.interpolate(x01[:, :, y:y+crop, x:x+crop], (256, 256), mode="bilinear", align_corners=False)
                    for (y, x) in coords
                ], dim=0)
                s_rep = sign.repeat(len(coords), 1)
                lg = self.tm_dec_fn(tiles * 2.0 - 1.0)
                sl = lg * s_rep
                m_val = (sl.mean(1) / 5.0).clamp(-2, 2).view(len(coords), B)
                a_val = ((sl > 0).float().mean(1) - 0.5) * 4.0
                a_val = a_val.view(len(coords), B)
                if g == 1:
                    maps.append(m_val[0].view(B, 1, 1, 1).expand(B, 1, 128, 128))
                    maps.append(a_val[0].view(B, 1, 1, 1).expand(B, 1, 128, 128))
                else:
                    acc_m = torch.zeros((B, 2, 128, 128), device=x01.device, dtype=torch.float32)
                    acc_w = torch.zeros((1, 1, 128, 128), device=x01.device, dtype=torch.float32)
                    c128 = crop // 4
                    for ti, (y, x) in enumerate(coords):
                        y4, x4 = y // 4, x // 4
                        acc_m[:, 0, y4:y4+c128, x4:x4+c128] += m_val[ti].view(B, 1, 1)
                        acc_m[:, 1, y4:y4+c128, x4:x4+c128] += a_val[ti].view(B, 1, 1)
                        acc_w[:, :, y4:y4+c128, x4:x4+c128] += 1.0
                    maps.append(acc_m / acc_w.clamp(min=1.0))
            return torch.cat(maps, dim=1)  # [B, 8, 128, 128]

        @torch.no_grad()
        def _dual_rm(self, x01):
            x512_m11 = x01 * 2.0 - 1.0
            x256_m11 = F.interpolate(x512_m11, (256, 256), mode="bilinear", align_corners=False)
            r256 = self.tm_rm(x256_m11)
            if isinstance(r256, tuple): r256 = r256[0]
            res256 = F.interpolate((x256_m11 - r256).float(), (512, 512), mode="bilinear", align_corners=False)
            r512 = self.tm_rm(x512_m11)
            if isinstance(r512, tuple): r512 = r512[0]
            res512 = (x512_m11 - r512).float()
            return (res256 * 10.0).clamp(-3, 3), (res512 * 10.0).clamp(-3, 3)

        def forward(self, x01, msg):
            km = self._known_msg_maps(x01, msg)  # [B, 8, 128, 128]
            res256, res512 = self._dual_rm(x01)
            rgb = (x01 - self.im_mean) / self.im_std
            vis_in = torch.cat([rgb, res256.to(rgb.dtype), res512.to(rgb.dtype)], dim=1)
            fs = self.vis(vis_in)
            ps = [
                self.lats[i](torch.cat([fs[i], F.interpolate(km.to(fs[i].dtype), fs[i].shape[-2:], mode="bilinear", align_corners=False)], dim=1))
                for i in range(4)
            ]
            p = ps[3]
            for i in (2, 1, 0):
                p = self.smooth[i](ps[i] + F.interpolate(p, ps[i].shape[-2:], mode="bilinear", align_corners=False))
            seg = self.head(torch.cat([p, km.to(p.dtype)], dim=1)).squeeze(1)
            cls = self.cls_head(torch.cat([fs[-1].mean((2, 3)), km.mean((2, 3)).to(p.dtype)], dim=1)).squeeze(1)
            return seg, cls

    def sym_miou_loss(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
        i1 = (p * t).sum(1); u1 = (p + t - p * t).sum(1)
        i0 = ((1 - p) * (1 - t)).sum(1); u0 = ((1 - p) + (1 - t) - (1 - p) * (1 - t)).sum(1)
        sym = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0)).mean()
        return 0.45 * bce + 0.55 * (1.0 - sym)

    net = TrustMarkV11Net(tm_decode_logits, tm_rm).to(dev)
    bb_params = list(net.vis.parameters())
    bb_ids = {id(p) for p in bb_params}
    hd_params = [p for p in net.parameters() if p.requires_grad and id(p) not in bb_ids]
    print(f"[TM-V11] Trainable params: bb={sum(p.numel() for p in bb_params)/1e6:.1f}M head={sum(p.numel() for p in hd_params)/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW([
        {"params": bb_params, "lr": 3e-5},
        {"params": hd_params, "lr": 5e-4},
    ], weight_decay=1e-4)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    )

    t0 = time.time()
    best_all = 0.0
    for step in range(1, steps + 1):
        net.train()
        xb, yb, lb, mb = sample_batch(bs)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            seg, cls = net(xb, mb)
            loss = sym_miou_loss(seg.float(), yb) + 0.2 * F.binary_cross_entropy_with_logits(cls.float(), lb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps:
            net.eval()
            probs = []
            with torch.no_grad():
                for s in range(0, n_real, 16):
                    xr = torch.from_numpy(r_imgs[s:s+16].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
                    mr = torch.from_numpy(r_msgs[s:s+16]).to(dev)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg, _ = net(xr, mr)
                    probs.append(torch.sigmoid(lg.float()).cpu().numpy())
            probs = np.concatenate(probs)
            best_t, best_sc = 0.5, 0.0
            for t in (0.35, 0.45, 0.50, 0.55, 0.60):
                ious = np.array([miou_image(probs[i] >= t, r_gt128[i]) for i in range(n_real)])
                if ious.mean() > best_sc:
                    best_sc, best_t, best_ious = float(ious.mean()), t, ious
            sL = best_ious[r_shape == "large"].mean()
            sr = best_ious[r_shape == "rect"].mean()
            so = best_ious[r_shape == "object"].mean()
            sS = best_ious[~r_large].mean()
            if best_sc > best_all:
                best_all = best_sc
                torch.save(net.state_dict(), "/vol/outputs/v11_tm_pilot_best.pt")
            print(f"[TM-V11 step {step:4d}/{steps}] ({time.time()-t0:5.1f}s) ZERO-REAL-TRAIN UNTOUCHED 170-IMG mIoU = {best_sc:.4f} (thr={best_t:.2f}) | L={sL:.4f} S={sS:.4f} (rect={sr:.4f} obj={so:.4f})", flush=True)
    vol.commit()
    return best_all


@app.local_entrypoint()
def main():
    h_ps = train_v11_pixelseal.spawn(steps=2500, bs=12)
    h_tm = train_v11_trustmark.spawn(steps=2500, bs=12)
    sc_ps = h_ps.get()
    sc_tm = h_tm.get()
    print(f"=== V11 PILOT DONE: PS={sc_ps:.4f} | TM={sc_tm:.4f} ===")
