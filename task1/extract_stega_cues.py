"""Extract the teammate's fixed 69 StegaStamp cues with our DEV/holdout split.

No fitting or labels. `dev` opens only the 136 DEV RGB images; `final` opens
only the fixed 34 holdout and 340 test RGB images. The original MPS Writer is
never imported or instantiated: only inspected pure feature functions are used.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import uniform_filter
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

from stega_template_probe import build_encoder


GROUPS = [
    {"name": "forensic_features.features(im)[::3]", "start": 0, "end": 29},
    {"name": "stega_pair.evidence (gray known/opposite)", "start": 29, "end": 35},
    {"name": "semantic_prior LRASPP", "start": 35, "end": 39},
    {"name": "rewrite_evidence.residual_maps (actual-image known/opposite)", "start": 39, "end": 51},
    {"name": "tile_evidence.texture", "start": 51, "end": 69},
]


def pure_functions(path, names):
    """Load named definitions only, excluding reference imports/Writer/run()."""
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in definitions} != set(names):
        raise ValueError(f"Missing requested pure functions in {path}")
    scope = {"np": np, "Image": Image, "uniform_filter": uniform_filter, "dctn": dctn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    return {name: scope[name] for name in names}


def native_image(image):
    # Exact reference Writer preprocessing: direct 512->400, no pilot 256 step.
    return np.asarray(Image.fromarray(image).resize((400, 400), Image.Resampling.BICUBIC), dtype=np.float32) / 255


@torch.inference_mode()
def semantics(model, rgb, device):
    x = torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float() / 255
    mu = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    sd = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    logits = model((x - mu) / sd)["out"]
    u = F.interpolate(logits, size=(64, 64), mode="bilinear", align_corners=False)
    p = u.softmax(1)
    return torch.cat([
        1 - p[:, :1],
        u[:, 1:].amax(1, keepdim=True) - u[:, :1],
        p[:, 1:].amax(1, keepdim=True),
        -(p * p.clamp_min(1e-8).log()).sum(1, keepdim=True),
    ], 1).cpu().numpy().astype(np.float16)


def read_rgb(data, source_split, name):
    with Image.open(data / source_split / "images" / f"{name}.png") as image:
        return np.asarray(image.convert("RGB"))


def extract_split(args, name, rows, encoder, model, functions):
    destination = args.out / f"{name}.npz"
    names = [row["image_name"] for row in rows]
    if destination.exists():
        with np.load(destination) as old:
            if old["image_names"].tolist() != names:
                raise ValueError(f"Existing {name} cache has different image names")
        print(f"{name}: using existing cache {destination}", flush=True)
        return len(names)
    source_split = "test" if name == "test" else "train"
    features = np.empty((len(rows), 69, 64, 64), dtype=np.float16)
    started = time.monotonic()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        rgb = np.stack([read_rgb(args.data, source_split, row["image_name"]) for row in batch])
        semantic = semantics(model, rgb, args.device)
        for j, (row, image) in enumerate(zip(batch, rgb)):
            bits = np.asarray([int(c) for c in row["message"]], dtype=np.float32)
            pair = functions["evidence"](image, bits, encoder)
            raw = native_image(image)
            residual = encoder.run(["residual"], {
                "image": np.stack([raw, raw]),
                "secret": np.stack([bits, 1 - bits]).astype(np.float32),
            })[0]
            rewrite = functions["residual_maps"](raw, residual[0], residual[1])
            value = np.concatenate((functions["features"](image)[::3], pair, semantic[j],
                                    rewrite, functions["texture"](image)), axis=0).astype(np.float16)
            if value.shape != (69, 64, 64) or not np.isfinite(value).all():
                raise ValueError(f"Invalid cue tensor for {row['image_name']}: {value.shape}")
            features[start + j] = value
        print(f"{name}: {min(start + len(batch), len(rows))}/{len(rows)} images, elapsed={time.monotonic() - started:.1f}s", flush=True)
    temporary = destination.with_suffix(".partial.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, image_names=np.asarray(names, dtype=str), features=features)
    temporary.replace(destination)
    print(f"Saved {destination}: {features.shape} float16", flush=True)
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output/stega_cues")
    parser.add_argument("--stage", choices=["dev", "final"], default="dev")
    parser.add_argument("--reference", type=Path, default=Path(__file__).parent / "reference_xuanhao")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but PyTorch CUDA is unavailable")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but PyTorch MPS is unavailable")
    else:
        raise ValueError("Use --device cuda or --device mps")
    args.out.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text())
    dev_names, holdout_names = split["train"], split["val"]
    if len(dev_names) != 136 or len(holdout_names) != 34 or not set(dev_names).isdisjoint(holdout_names):
        raise ValueError("Expected the original 136 DEV / 34 holdout split")
    train_metadata = json.loads((args.data / "train/metadata.json").read_text())
    train = {row["image_name"]: row for row in train_metadata if row["method"] == "stegastamp"}
    if set(dev_names) | set(holdout_names) != set(train):
        raise ValueError("Split does not match the official StegaStamp train names")
    if args.stage == "dev":
        jobs = [("dev", [train[name] for name in dev_names])]
    else:
        metadata = json.loads((args.data / "test/metadata.json").read_text())
        test = sorted([row for row in metadata if row["method"] == "stegastamp"], key=lambda row: row["image_name"])
        if len(test) != 340:
            raise ValueError(f"Expected 340 StegaStamp test images, got {len(test)}")
        jobs = [("holdout", [train[name] for name in holdout_names]), ("test", test)]
    (args.out / "split.json").write_text(json.dumps(split, indent=2))

    functions = {}
    for filename, names in [
        ("forensic_features.py", ["blocks", "features"]),
        ("stega_pair.py", ["evidence"]),
        ("rewrite_evidence.py", ["residual_maps"]),
        ("tile_evidence.py", ["texture"]),
    ]:
        functions.update(pure_functions(args.reference / filename, names))
    first_name = jobs[0][1][0]["image_name"]
    first_split = "test" if jobs[0][0] == "test" else "train"
    first = read_rgb(args.data, first_split, first_name)
    first_bits = np.asarray([int(c) for c in jobs[0][1][0]["message"]], dtype=np.float32)
    encoder_device = "cpu" if args.device == "mps" else args.device
    encoder, equivalence_error = build_encoder(args.models_dir, args.out, native_image(first), first_bits, encoder_device)
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights).eval().to(args.device)
    info_path = args.out / "feature_info.json"
    old = json.loads(info_path.read_text()) if info_path.exists() else {}
    info = {"method": "teammate fixed cue extraction, original Codex split", "features": 69,
            "shape_per_image": [69, 64, 64], "dtype": "float16", "channel_groups": GROUPS,
            "removed": "26 constant-zero channels: 18 message-CAM and 8 tile channels",
            "fitting": "none; no feature normalization or watermark labels read by this script",
            "encoder": f"shared {encoder_device} ORT encoder-only session; true/opposite bits; no reference Writer",
            "encoder_max_abs_equivalence_error": equivalence_error, "onnx_providers": encoder.get_providers(),
            "semantic_weights": str(weights), "semantic_device": args.device, "stage": args.stage,
            "reference_sha256": {filename: hashlib.sha256((args.reference / filename).read_bytes()).hexdigest()
                for filename in ("forensic_features.py", "stega_pair.py", "semantic_prior.py", "rewrite_evidence.py", "tile_evidence.py")},
            "completed_splits": old.get("completed_splits", {})}
    info_path.write_text(json.dumps(info, indent=2))
    for name, rows in jobs:
        info["completed_splits"][name] = extract_split(args, name, rows, encoder, model, functions)
        info_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
