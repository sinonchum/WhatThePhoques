"""Replace only LRASPP channels 35:39 with fixed pretrained DeepLabV3 cues.

No fitting, masks, or training statistics. DEV/final stages follow the base
cache's exact image names; all other 65 base channels remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models.segmentation import (
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet101,
)

from extract_stega_cues import GROUPS, read_rgb, semantics


def extract_split(args, name, model, split):
    with np.load(args.base / f"{name}.npz") as source:
        names = source["image_names"].tolist()
        features = source["features"]
    expected = split["train"] if name == "dev" else split["val"] if name == "holdout" else None
    if expected is not None and names != expected:
        raise ValueError(f"Base {name} names differ from the original split")
    if features.shape != (len(names), 69, 64, 64) or features.dtype != np.float16:
        raise ValueError(f"Expected base float16 [N,69,64,64], got {features.shape}, {features.dtype}")
    if name == "test" and len(names) != 340:
        raise ValueError("Expected all 340 StegaStamp test names")
    destination = args.out / f"{name}.npz"
    if destination.exists():
        with np.load(destination) as cached:
            if cached["image_names"].tolist() != names:
                raise ValueError(f"Existing semantic {name} cache has different names")
        print(f"{name}: using existing cache {destination}", flush=True)
        return len(names)
    source_split = "test" if name == "test" else "train"
    started = time.monotonic()
    for start in range(0, len(names), args.batch_size):
        batch_names = names[start:start + args.batch_size]
        rgb = np.stack([read_rgb(args.data, source_split, image_name) for image_name in batch_names])
        semantic = semantics(model, rgb, args.device)
        if semantic.shape != (len(batch_names), 4, 64, 64) or not np.isfinite(semantic).all():
            raise ValueError("Invalid DeepLabV3 semantic cues")
        # This ndarray is loaded from disk, not a writable base-file mapping.
        features[start:start + len(batch_names), 35:39] = semantic
        print(f"{name}: {min(start + len(batch_names), len(names))}/{len(names)}, elapsed={time.monotonic() - started:.1f}s", flush=True)
    temporary = destination.with_suffix(".partial.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, image_names=np.asarray(names, dtype=str), features=features)
    temporary.replace(destination)
    print(f"Saved {destination}: {features.shape} float16", flush=True)
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path(__file__).parent / "output/stega_cues")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output/stega_cues_semantic")
    parser.add_argument("--stage", choices=["dev", "final"], default="dev")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
    else:
        raise ValueError("Use --device mps or --device cuda")
    if args.base.resolve() == args.out.resolve():
        raise ValueError("Semantic output must not overwrite the base cache")
    args.out.mkdir(parents=True, exist_ok=True)
    split = json.loads((args.base / "split.json").read_text())
    if len(split["train"]) != 136 or len(split["val"]) != 34 or not set(split["train"]).isdisjoint(split["val"]):
        raise ValueError("Expected the original 136 DEV / 34 holdout split")
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    info_path = args.out / "feature_info.json"
    old = json.loads(info_path.read_text()) if info_path.exists() else {}
    if old.get("semantic_weights") not in (None, str(weights)):
        raise ValueError("Existing DEV/final cache uses different semantic weights")
    # Torch Hub respects TORCH_HOME, or XDG_CACHE_HOME/torch when it is unset.
    model = deeplabv3_resnet101(weights=weights).eval().to(args.device)
    groups = [dict(group) for group in GROUPS]
    groups[2]["name"] = "semantic_prior DeepLabV3 ResNet101"
    info = {"method": "base69 with LRASPP4 replaced by pretrained DeepLabV3 ResNet101",
            "features": 69, "shape_per_image": [69, 64, 64], "dtype": "float16",
            "base": str(args.base), "replaced_channels": [35, 39], "other_65_channels_unchanged": True,
            "channel_groups": groups, "fitting": "none; no masks, training statistics or normalization fitted",
            "semantic_model": "torchvision.deeplabv3_resnet101", "semantic_weights": str(weights),
            "semantic_weight_url": weights.url, "semantic_device": args.device,
            "torch_hub_cache": torch.hub.get_dir(),
            "preprocessing": "512x512 RGB /255; ImageNet mean/std; main logits bilinear64; fixed four semantic summaries",
            "semantic_channels": ["1-p_background", "max_foreground_logit-background_logit",
                                  "max_foreground_probability", "all_class_entropy"],
            "stage": args.stage, "completed_splits": old.get("completed_splits", {})}
    info_path.write_text(json.dumps(info, indent=2))
    jobs = ["dev"] if args.stage == "dev" else ["holdout", "test"]
    for name in jobs:
        info["completed_splits"][name] = extract_split(args, name, model, split)
        info_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
