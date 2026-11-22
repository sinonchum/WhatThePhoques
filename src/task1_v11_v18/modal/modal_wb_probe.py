"""Modal H100 4-in-1 White-Box Known-Message Physics Probe for Task 1.
Caches HF dataset + official watermark weights in persistent Modal Volume `task1-wm-vol`.
Tests on dev (never touching holdout for tuning):
  A) PixelSeal GT-rect embedding scale physics (why 11/43 worked and 32/43 failed in crop256)
  B) White-Box Matched-Filter Gradient: <grad_I L_msg, I - lowpass(I)> (full + multi-crop)
  C) Dense Multi-Scale Patch Occlusion Drop (16x16 grid, blur / AdobeRM / mean replace) -> L-holes & S-objects!
  D) Dense Multi-Scale Keep-Only Patch Isolation -> S-objects & S-rects!
"""
import modal

app = modal.App("task1-wb-physics-probe")
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
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/vol": vol},
    timeout=1800,
)
def setup_and_probe():
    import os, sys, time, subprocess, pathlib, json, zipfile, tarfile
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align
    from sklearn.metrics import roc_auc_score

    os.makedirs("/vol/repo", exist_ok=True)
    os.makedirs("/vol/models", exist_ok=True)
    os.makedirs("/vol/outputs", exist_ok=True)

    # 1. Download starter kit + dataset from HuggingFace if not already cached
    from huggingface_hub import snapshot_download
    repo_dir = pathlib.Path("/vol/repo/starter")
    if not (repo_dir / "release").exists():
        print("=== Downloading SprintML/Hackathon_Watermark_Localisation_Task_Paris ===", flush=True)
        t0 = time.time()
        snapshot_download(
            repo_id="SprintML/Hackathon_Watermark_Localisation_Task_Paris",
            repo_type="dataset",
            local_dir=str(repo_dir),
            max_workers=16,
        )
        print(f"Downloaded in {time.time()-t0:.1f}s. Contents:", [p.name for p in repo_dir.iterdir()], flush=True)
        # Extract any zip/tar archives if release/ doesn't exist yet
        for arc in list(repo_dir.glob("*.zip")) + list(repo_dir.glob("*.tar*")):
            print(f"Extracting {arc.name}...", flush=True)
            if arc.suffix == ".zip":
                with zipfile.ZipFile(arc) as zf:
                    zf.extractall(repo_dir)
            else:
                with tarfile.open(arc) as tf:
                    tf.extractall(repo_dir)
        vol.commit()

    print("Starter dir tree:", [str(p.relative_to(repo_dir)) for p in sorted(repo_dir.rglob("*")) if p.is_dir()][:30], flush=True)
    print("Starter py files:", [str(p.relative_to(repo_dir)) for p in sorted(repo_dir.rglob("*.py"))][:30], flush=True)
    # Check where data and models live
    for p in sorted(repo_dir.iterdir()):
        print("  top:", p.name, "dir" if p.is_dir() else f"{p.stat().st_size/1e6:.1f}MB", flush=True)

    # Also check README / setup script to see how official models are downloaded
    for md in list(repo_dir.glob("*.md")) + list(repo_dir.glob("*.sh")) + list(repo_dir.glob("*.py")):
        if md.stat().st_size < 40000:
            print(f"\n--- FILE: {md.name} (first 60 lines) ---", flush=True)
            print("\n".join(md.read_text(errors="ignore").splitlines()[:60]), flush=True)


@app.local_entrypoint()
def main():
    setup_and_probe.remote()
