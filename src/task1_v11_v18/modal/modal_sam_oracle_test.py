"""60-second Modal H100 Breakthrough Test using native Meta `segment-anything` (works on PyTorch 2.5.1!):
SAM Object-Segment Proposals + Official Watermark Decoder Segment-Occlusion Scoring!
"""
import modal

app = modal.App("task1-sam-native-probe")
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


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=900)
def test_sam_native():
    import os, sys, time, pathlib, urllib.request
    import numpy as np
    import torch
    import torch.nn.functional as F
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import miou_image, downsample_mask

    dev = "cuda:0"
    sam_ckpt = models_dir / "sam_vit_b_01ec64.pth"
    if not sam_ckpt.exists():
        print("=== Downloading sam_vit_b_01ec64.pth (375MB) ===", flush=True)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            str(sam_ckpt),
        )
        vol.commit()

    t0 = time.time()
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt)).to(dev).eval()
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=24,
        points_per_batch=128,
        pred_iou_thresh=0.82,
        stability_score_thresh=0.86,
        min_mask_region_area=80,
    )
    print(f"Native SAM-ViT-B loaded in {time.time()-t0:.1f}s", flush=True)

    def blur2d(x, k=15, sigma=5.0):
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        ker = (g[:, None] * g[None, :]).view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), ker, groups=x.shape[1])

    def classify_item(it):
        mask = it["mask"] > 0
        if it["full_watermark"]: return "large"
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        return "rect" if mask.sum() / max((y1-y0)*(x1-x0), 1) > 0.98 else "object"

    for method in ("pixelseal", "trustmark"):
        print(f"\n================ SAM SEGMENT PROBE: {method.upper()} ================", flush=True)
        wm = load_watermark(method, device=dev)
        tm_rm = None
        if method == "trustmark":
            from trustmark import TrustMark
            tm_rm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5).removal.to(dev).eval()
            for p in tm_rm.parameters(): p.requires_grad_(False)

        ds = LocalisationDataset(str(repo_dir), "train", methods=[method])
        all_it = [ds[i] for i in range(len(ds))]
        sub = (
            [it for it in all_it if classify_item(it) == "large"][:20] +
            [it for it in all_it if classify_item(it) == "object"][:20] +
            [it for it in all_it if classify_item(it) == "rect"][:12]
        )

        oracle_by_grp = {"large": [], "object": [], "rect": []}
        zero_occ_by_grp = {"large": [], "object": [], "rect": []}
        n_segs_list = []

        t1 = time.time()
        for it in sub:
            grp = classify_item(it)
            gt128 = downsample_mask(it["mask"] > 0) > 0
            anns = mask_gen.generate(it["image"])
            segs = []
            for a in anns:
                m_bool = a["segmentation"]
                area = m_bool.mean()
                if 0.003 <= area <= 0.48:
                    segs.append(m_bool)
            n_segs_list.append(len(segs))
            if not segs:
                oracle_by_grp[grp].append(0.49)
                zero_occ_by_grp[grp].append(0.49)
                continue

            segs128 = [downsample_mask(s) > 0 for s in segs]
            if grp == "large":
                best_o = max(miou_image(~s128, gt128) for s128 in segs128)
            else:
                best_o = max(miou_image(s128, gt128) for s128 in segs128)
            oracle_by_grp[grp].append(best_o)

            with torch.no_grad():
                img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
                sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
                x256 = F.interpolate(img01, (256, 256), mode="bilinear", align_corners=False)
                if tm_rm is not None:
                    ro = tm_rm(x256 * 2.0 - 1.0); ro = ro[0] if isinstance(ro, tuple) else ro
                    clean256 = ((ro.float() + 1.0) * 0.5).clamp(0, 1)
                else:
                    clean256 = blur2d(x256, 15, 5.0)
                def dec_m(xb):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        lg = wm.det(xb)[:, 1:].float() if method == "pixelseal" else wm.net(xb * 2.0 - 1.0).float()
                    return (lg * sign).mean(1)
                base_m = float(dec_m(x256).item())
                seg_t = torch.from_numpy(np.stack(segs).astype(np.float32)).to(dev)[:, None]
                seg256 = F.interpolate(seg_t, (256, 256), mode="bilinear", align_corners=False)
                x_occ = x256 * (1.0 - seg256) + clean256 * seg256
                drops = []
                for s in range(0, len(segs), 64):
                    drops.append((base_m - dec_m(x_occ[s:s+64])).cpu().numpy())
                drops = np.concatenate(drops)
                areas = np.array([s.mean() for s in segs], np.float32)
                if grp == "large":
                    j = int(np.argmin(drops / np.maximum(areas, 0.01)))
                    zero_occ_by_grp[grp].append(miou_image(~segs128[j], gt128))
                else:
                    j = int(np.argmax(drops))
                    zero_occ_by_grp[grp].append(miou_image(segs128[j], gt128))

        print(f"  Done {len(sub)} imgs in {time.time()-t1:.1f}s (avg {np.mean(n_segs_list):.1f} SAM segments/img)", flush=True)
        for grp in ("large", "object", "rect"):
            print(f"  [{method:9s} {grp:6s} n={len(oracle_by_grp[grp]):2d}] SAM_ORACLE_mIoU = {np.mean(oracle_by_grp[grp]):.4f} (>0.75: {np.mean([x>0.75 for x in oracle_by_grp[grp]]):.2f}, >0.85: {np.mean([x>0.85 for x in oracle_by_grp[grp]]):.2f}) | ZERO_SHOT_OCC_mIoU = {np.mean(zero_occ_by_grp[grp]):.4f}", flush=True)


@app.local_entrypoint()
def main():
    test_sam_native.remote()
