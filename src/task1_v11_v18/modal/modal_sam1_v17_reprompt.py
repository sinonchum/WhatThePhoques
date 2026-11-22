"""90-Second 4xGPU `SAM-1 ViT-H` `SamPredictor` V17 Re-Prompting Engine (WITH `mask_input` LOGIT PROMPTING!):
Previously, SAM-1 ViT-H was prompted by old broken-norm `v3` logits (`TrustMark 0.6389`).
Now we re-prompt ONLY `SamPredictor` (NO slow AutomaticMaskGenerator -> ~90s across 4 GPUs!) using our
brand-new V15+V16+V17 Super-Ensemble logits (+ direct 256x256 `mask_input` low-res logit prompting!),
saving crisp neural-guided masks to `/vol/outputs/sam_h_v17/shard_{0..3}.npz`!
"""
import modal

app = modal.App("task1-sam1-v17-reprompt")
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
def reprompt_sam1_shard(rank: int, world: int = 4):
    import sys, time, pathlib
    import numpy as np
    from scipy import ndimage
    import torch
    import torch.nn.functional as F
    from segment_anything import sam_model_registry, SamPredictor

    t0 = time.time()
    dev = "cuda:0"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import downsample_mask

    # Load ALL available V14/V15/V16/V17 refiners and average their honest logits per image!
    ref_dir = pathlib.Path("/vol/outputs/v14_refiners")
    ref_files = sorted([f for f in ref_dir.glob("*.npz") if "swinl" not in f.stem])
    print(f"[SAM-REPROMPT r{rank}] Using {len(ref_files)} refiners: {[f.stem for f in ref_files]}", flush=True)
    acc_by_name = {}
    cnt_by_name = {}
    for rf in ref_files:
        rz = np.load(str(rf), allow_pickle=True)
        for meth in ("pixelseal", "trustmark", "stegastamp"):
            for sp in ("dev", "hold", "test"):
                if f"{meth}_{sp}_names" not in rz: continue
                nms = rz[f"{meth}_{sp}_names"]
                lgs = rz[f"{meth}_{sp}_lg"].astype(np.float32)
                for nm, lg in zip(nms, lgs):
                    k = str(nm)
                    if k not in acc_by_name:
                        acc_by_name[k] = lg.copy(); cnt_by_name[k] = 1
                    else:
                        acc_by_name[k] += lg; cnt_by_name[k] += 1
    lg_by_name = {k: acc_by_name[k] / cnt_by_name[k] for k in acc_by_name}

    sam = sam_model_registry["vit_h"](checkpoint="/vol/models/sam_vit_h_4b8939.pth").to(dev).eval()
    predictor = SamPredictor(sam)

    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    tasks = [("train", i) for i in range(len(ds_tr))] + [("test", i) for i in range(len(ds_te))]
    my_tasks = tasks[rank::world]

    out_dir = pathlib.Path("/vol/outputs/sam_h_v17")
    out_dir.mkdir(parents=True, exist_ok=True)
    packed_all = {}

    for ki, (sp, idx) in enumerate(my_tasks):
        it = ds_tr[idx] if sp == "train" else ds_te[idx]
        nm = it["image_name"]
        img = it["image"]
        lg = lg_by_name[nm]

        predictor.set_image(img)
        isL = (lg >= 0.0).mean() > 0.5
        sig = (-lg) if isL else lg
        sig_sm = ndimage.gaussian_filter(sig, sigma=1.2)
        cands = []

        # 1) Peak point + negative point (both raw and with 256x256 mask_input!)
        py, px = np.unravel_index(np.argmax(sig_sm), (128, 128))
        ny, nx = np.unravel_index(np.argmin(sig_sm), (128, 128))
        pts = np.array([[px*4+2, py*4+2], [nx*4+2, ny*4+2]], np.float32)
        lbs = np.array([1, 0], np.int32)
        # Upsample sig to 256x256 for SAM's mask_input!
        m_in = F.interpolate(torch.from_numpy(sig)[None, None].float(), (256, 256), mode="bilinear", align_corners=False)[0].numpy()
        m_in = np.clip(m_in * 2.5, -15.0, 15.0)
        for use_mi in (False, True):
            ms, _, _ = predictor.predict(point_coords=pts, point_labels=lbs, mask_input=(m_in if use_mi else None), multimask_output=True)
            for m in ms:
                if 0.002 <= m.mean() <= 0.49: cands.append(downsample_mask(m) > 0)

        # 2) Multi-threshold component bbox + peak point (with & without mask_input!)
        for thr_q in (-0.8, -0.2, 0.3, 0.8):
            bw = sig >= thr_q
            if bw.sum() < 8 or bw.mean() > 0.48: continue
            lab, nl = ndimage.label(bw)
            if nl == 0: continue
            bc = max(range(1, nl + 1), key=lambda c: float(sig[lab == c].clip(0).sum()))
            comp = lab == bc
            if comp.sum() < 8: continue
            ys, xs = np.where(comp)
            box = np.array([max(0, xs.min()*4 - 4), max(0, ys.min()*4 - 4),
                            min(512, (xs.max()+1)*4 + 4), min(512, (ys.max()+1)*4 + 4)], np.float32)
            cy, cx = np.unravel_index(np.argmax(np.where(comp, sig_sm, -1e9)), (128, 128))
            for use_mi in (False, True):
                ms, _, _ = predictor.predict(
                    point_coords=np.array([[cx*4+2, cy*4+2]], np.float32),
                    point_labels=np.array([1], np.int32),
                    box=box,
                    mask_input=(m_in if use_mi else None),
                    multimask_output=True,
                )
                for m in ms:
                    if 0.002 <= m.mean() <= 0.49: cands.append(downsample_mask(m) > 0)

        uniq = []
        for m in cands:
            mf = m.ravel()
            dup = False
            for u in uniq:
                inter = np.logical_and(mf, u).sum(); union = np.logical_or(mf, u).sum()
                if union > 0 and inter / union > 0.94: dup = True; break
            if not dup: uniq.append(mf)
        packed_all[nm] = np.packbits(np.stack(uniq, axis=0), axis=1) if uniq else np.zeros((0, 2048), dtype=np.uint8)
        if (ki + 1) % 100 == 0 or (ki + 1) == len(my_tasks):
            print(f"  [SAM-REPROMPT r{rank}] {ki+1}/{len(my_tasks)} ({(time.time()-t0)/(ki+1):.2f}s/img, last_M={len(uniq)})", flush=True)

    np.savez(out_dir / f"shard_{rank}.npz", **packed_all)
    vol.commit()
    print(f"[SAM-REPROMPT r{rank}] SAVED in {time.time()-t0:.1f}s!", flush=True)


@app.local_entrypoint()
def main():
    hs = [reprompt_sam1_shard.spawn(r, 4) for r in range(4)]
    for h in hs: h.get()
    print("=== SAM-1 ViT-H V17 RE-PROMPTING COMPLETE ===")
