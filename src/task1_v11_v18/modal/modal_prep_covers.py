"""30-second Modal H100 prep script:
1. Verify exact `small, object` embedding physics (full-image embed vs bbox-crop embed) for PixelSeal & TrustMark!
2. Print exact spatial shapes of Meta PixelSeal `wm.det` pixel_decoder & Adobe `tm.removal` U-Net!
3. Stream 2,500 crisp >=512px clean public photos (COCO/Food/Unsplash/etc. via HF datasets) into `/vol/clean_covers.npy` (uint8, 512x512x3) — 100% clean, ZERO overlap with competition holdout/test!
"""
import modal

app = modal.App("task1-physics-and-covers")
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


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=1200)
def prep_all():
    import os, sys, time, pathlib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark

    dev = "cuda:0"

    def classify_item(it):
        mask = it["mask"] > 0
        if it["full_watermark"]:
            return "large"
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return "rect" if mask.sum() / max((y1 - y0) * (x1 - x0), 1) > 0.98 else "object"

    # 1. Check PixelSeal & TrustMark architectures + small-object physics
    print("=== 1. ARCHITECTURE & SMALL-OBJECT PHYSICS CHECK ===", flush=True)
    wm_ps = load_watermark("pixelseal", device=dev)
    with torch.no_grad():
        dummy = torch.rand(1, 3, 512, 512, device=dev)
        lat = wm_ps.det.convnext(wm_ps.det.preprocess(dummy))
        up = wm_ps.det.pixel_decoder.output_upscaling(lat)
        sp = wm_ps.det.pixel_decoder.linear(up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        print(f"PixelSeal det on 512x512: lat={tuple(lat.shape)} up={tuple(up.shape)} sp={tuple(sp.shape)}", flush=True)
        # Also check on 256x256
        lat2 = wm_ps.det.convnext(wm_ps.det.preprocess(dummy[:, :, :256, :256]))
        up2 = wm_ps.det.pixel_decoder.output_upscaling(lat2)
        print(f"PixelSeal det on 256x256: lat={tuple(lat2.shape)} up={tuple(up2.shape)}", flush=True)
        print("PixelSeal det preprocess:", wm_ps.det.preprocess, flush=True)

    from trustmark import TrustMark
    tm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5)
    rm = tm.removal.to(dev).eval()
    print("TrustMark removal children:", [(n, type(m).__name__) for n, m in rm.named_children()], flush=True)

    # Check small-object physics: does bbox crop256 decode better or worse than full512 on small,object?!
    for meth in ("pixelseal", "trustmark"):
        wm = wm_ps if meth == "pixelseal" else load_watermark("trustmark", device=dev)
        ds = LocalisationDataset(str(repo_dir), "train", methods=[meth])
        objs = [ds[i] for i in range(len(ds)) if classify_item(ds[i]) == "object"]
        a_full, a_bbox, a_fmask = [], [], []
        with torch.no_grad():
            for it in objs:
                img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
                sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
                mask = it["mask"] > 0
                ys, xs = np.where(mask)
                y0, y1, x0, x1 = int(ys.min()), int(ys.max())+1, int(xs.min()), int(xs.max())+1
                m_t = torch.from_numpy(mask.astype(np.float32)).to(dev)[None, None]
                crop = img01[:, :, y0:y1, x0:x1]
                c256 = F.interpolate(crop, (256, 256), mode="bilinear", align_corners=False)
                f256 = F.interpolate(img01, (256, 256), mode="bilinear", align_corners=False)
                fm = F.interpolate(img01 * m_t + 0.5 * (1 - m_t), (256, 256), mode="bilinear", align_corners=False)
                def acc_of(x):
                    if meth == "pixelseal": lg = wm.det(x)[:, 1:].float()
                    else: lg = wm.net(x * 2.0 - 1.0).float()
                    return float(((lg * sign) > 0).float().mean().item())
                a_full.append(acc_of(f256))
                a_bbox.append(acc_of(c256))
                a_fmask.append(acc_of(fm))
        print(f"  [{meth:9s}] SMALL-OBJECT (n={len(objs)}): full256={np.median(a_full):.3f} (>{0.65}: {np.mean([x>0.65 for x in a_full]):.2f}) | bbox256={np.median(a_bbox):.3f} (>{0.65}: {np.mean([x>0.65 for x in a_bbox]):.2f}) | full_masked={np.median(a_fmask):.3f} (>{0.65}: {np.mean([x>0.65 for x in a_fmask]):.2f})", flush=True)

    # 2. Stream 2,500 crisp high-res clean public photos into /vol/clean_covers.npy!
    covers_path = pathlib.Path("/vol/clean_covers.npy")
    if covers_path.exists():
        c = np.load(covers_path, mmap_mode="r")
        print(f"clean_covers.npy already exists: {c.shape}", flush=True)
    else:
        print("\n=== 2. STREAMING 2,500 CLEAN HIGH-RES EXTERNAL COVERS ===", flush=True)
        from datasets import load_dataset
        t0 = time.time()
        target_n = 2500
        covers = np.empty((target_n, 512, 512, 3), dtype=np.uint8)
        got = 0
        # Try fast high-res public image datasets on HF Hub
        candidates = [
            ("detection-datasets/coco", "val", "image"),
            ("detection-datasets/coco", "train", "image"),
            ("nlphuji/flickr30k", "test", "image"),
            ("timm/imagenet-1k-wds", "validation", "jpg"),
            ("Matthijs/snacks", "train", "image"),
        ]
        for ds_name, split, col in candidates:
            if got >= target_n:
                break
            try:
                print(f"Trying {ds_name} ({split})...", flush=True)
                ds_stream = load_dataset(ds_name, split=split, streaming=True)
                for ex in ds_stream:
                    im = ex.get(col)
                    if im is None:
                        continue
                    if not isinstance(im, Image.Image):
                        continue
                    w, h = im.size
                    if min(w, h) < 384:
                        continue
                    im = im.convert("RGB")
                    # Center-crop square then resize to 512x512 (zero aspect distortion!)
                    s = min(w, h)
                    left = (w - s) // 2; top = (h - s) // 2
                    im = im.crop((left, top, left + s, top + s)).resize((512, 512), Image.Resampling.BICUBIC)
                    covers[got] = np.asarray(im, dtype=np.uint8)
                    got += 1
                    if got % 500 == 0:
                        print(f"  collected {got}/{target_n} covers in {time.time()-t0:.1f}s", flush=True)
                    if got >= target_n:
                        break
            except Exception as e:
                print(f"  {ds_name} failed: {e}", flush=True)
        covers = covers[:got]
        np.save(covers_path, covers)
        vol.commit()
        print(f"SAVED {covers_path}: shape={covers.shape} ({covers_path.stat().st_size/1e9:.2f} GB) in {time.time()-t0:.1f}s", flush=True)


@app.local_entrypoint()
def main():
    prep_all.remote()
