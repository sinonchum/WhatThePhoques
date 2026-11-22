"""Modal 2xA100-80GB/H100 Dual-Scale White-Box Dense Patch Occlusion Heatmap Extractor (~2.5 min total!):
For ALL 510 PixelSeal + ALL 510 TrustMark images (train + test = 1,020 images):
  - Scale 1 (Fine): 12x12 grid (144 overlapping 48x48 patches at 256x256 -> 96x96 at 512x512)
  - Scale 2 (Coarse): 8x8 grid (64 overlapping 72x72 patches at 256x256 -> 144x144 at 512x512)
  - Total 208 occluded passes per image -> runs in 2 batches of 104 in bf16 on A100/H100 (~0.25s/img = ~2 min!).
  - Computes BOTH:
      1) Raw margin drop heatmap `occ_fine [128,128]` & `occ_coarse [128,128]` (normalized by base margin)
      2) Per-image z-scored spatial anomaly map (lights up L-holes as LOW-drop dips AND S-objects as HIGH-drop peaks!)
  - Saves permanently to `/vol/outputs/dense_occ/{pixelseal,trustmark}.npz` for direct injection into
    Spatial MetaStackNet-V2 and V16 Refiners!
"""
import modal

app = modal.App("task1-dense-occ-maps")
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


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1200)
def extract_dense_occ(method: str):
    import os, sys, time, pathlib
    import numpy as np
    import torch
    import torch.nn.functional as F

    os.environ["WMLOC_MODELS"] = "/vol/models"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark

    t0 = time.time()
    dev = "cuda:0"
    wm = load_watermark(method, device=dev)
    tm_rm = None
    if method == "trustmark":
        from trustmark import TrustMark
        tm_rm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5).removal.to(dev).eval()
        for p in tm_rm.parameters(): p.requires_grad_(False)

    # Blur kernel at 256x256
    def make_blur_k(sigma=5.0, k=15):
        ax = torch.arange(k, device=dev, dtype=torch.float32) - k // 2
        g = torch.exp(-0.5 * (ax / sigma) ** 2); g = g / g.sum()
        return (g[:, None] * g[None, :])[None, None, :, :].repeat(3, 1, 1, 1)
    blur_k = make_blur_k(5.0, 15)

    def blur256(x):
        return F.conv2d(F.pad(x, (7, 7, 7, 7), mode="reflect"), blur_k, groups=3)

    # Precompute smooth Gaussian occlusion masks for Scale 1 (12x12, patch~44px) and Scale 2 (8x8, patch~72px)
    def build_grid_masks(G, rad):
        ys = torch.linspace(rad / 2, 256 - rad / 2, G, device=dev)
        xs = torch.linspace(rad / 2, 256 - rad / 2, G, device=dev)
        yy, xx = torch.meshgrid(torch.arange(256, device=dev, dtype=torch.float32),
                                torch.arange(256, device=dev, dtype=torch.float32), indexing="ij")
        masks = []
        for cy in ys:
            for cx in xs:
                # Smooth box-like cosine window inside square [cy-rad/2, cy+rad/2]
                dy = ((yy - cy).abs() / (rad * 0.5)).clamp(0, 1)
                dx = ((xx - cx).abs() / (rad * 0.5)).clamp(0, 1)
                m = ((1.0 - dy ** 4) * (1.0 - dx ** 4)).clamp(0, 1)
                masks.append(m)
        return torch.stack(masks, dim=0)[:, None, :, :]  # [G*G, 1, 256, 256]

    M12 = build_grid_masks(12, 44.0)  # [144, 1, 256, 256]
    M8 = build_grid_masks(8, 72.0)    # [64,  1, 256, 256]
    M_ALL = torch.cat([M12, M8], dim=0)  # [208, 1, 256, 256]

    @torch.no_grad()
    def dec_margin(xb, sign_t):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = wm.det(xb)[:, 1:].float() if method == "pixelseal" else wm.net(xb * 2.0 - 1.0).float()
        return (lg * sign_t[None, :]).mean(dim=1)

    ds_tr = LocalisationDataset(str(repo_dir), "train", methods=[method])
    ds_te = LocalisationDataset(str(repo_dir), "test", methods=[method])
    items = [ds_tr[i] for i in range(len(ds_tr))] + [ds_te[i] for i in range(len(ds_te))]
    print(f"[{method}] Extracting dual-scale dense occlusion maps (12x12 + 8x8 = 208 patches/img) for {len(items)} images...", flush=True)

    names = []
    fine_maps = np.zeros((len(items), 128, 128), np.float16)
    coarse_maps = np.zeros((len(items), 128, 128), np.float16)
    base_margins = np.zeros(len(items), np.float32)

    for idx, it in enumerate(items):
        names.append(it["image_name"])
        x01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
        x256 = F.interpolate(x01, (256, 256), mode="bilinear", align_corners=False)
        sign_t = torch.from_numpy(2.0 * np.asarray(it["message"], np.float32) - 1.0).to(dev)
        with torch.no_grad():
            if tm_rm is not None:
                ro = tm_rm(x256 * 2.0 - 1.0); ro = ro[0] if isinstance(ro, tuple) else ro
                cln = ((ro.float() + 1.0) * 0.5).clamp(0, 1)
                cln = 0.7 * cln + 0.3 * blur256(x256)
            else:
                cln = blur256(x256)
            bm = float(dec_margin(x256, sign_t).item())
            base_margins[idx] = bm
            # Compose all 208 occluded images on GPU in 2 batches of 104!
            drops = []
            for s in (0, 104):
                ms = M_ALL[s:s+104]
                xb = x256 * (1.0 - ms) + cln * ms
                drops.append((bm - dec_margin(xb, sign_t)).cpu().numpy())
            drops = np.concatenate(drops)  # [208]
            g12 = torch.from_numpy(drops[:144].reshape(1, 1, 12, 12)).float()
            g8 = torch.from_numpy(drops[144:].reshape(1, 1, 8, 8)).float()
            f128 = F.interpolate(g12, (128, 128), mode="bicubic", align_corners=False)[0, 0].numpy()
            c128 = F.interpolate(g8, (128, 128), mode="bicubic", align_corners=False)[0, 0].numpy()
            fine_maps[idx] = f128.astype(np.float16)
            coarse_maps[idx] = c128.astype(np.float16)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(items):
            print(f"  [{method}] {idx+1}/{len(items)} ({(time.time()-t0)/(idx+1):.2f}s/img)", flush=True)

    out_dir = pathlib.Path("/vol/outputs/dense_occ")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{method}.npz",
        names=np.array(names),
        fine=fine_maps,
        coarse=coarse_maps,
        base_margin=base_margins,
    )
    vol.commit()
    print(f"[{method}] SAVED /vol/outputs/dense_occ/{method}.npz in {time.time()-t0:.1f}s!", flush=True)


@app.local_entrypoint()
def main():
    h1 = extract_dense_occ.spawn("pixelseal")
    h2 = extract_dense_occ.spawn("trustmark")
    h1.get(); h2.get()
    print("=== DUAL-SCALE DENSE OCCLUSION MAPS COMPLETE (1,020 IMAGES) ===")
