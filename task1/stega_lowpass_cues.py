"""Append 18 fixed lowpass-corrected StegaStamp cues to the existing 69.

No fitting or labels. DEV and final stages follow the base cache's exact image
names. Neither the base cache nor its extractor is modified.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from extract_stega_cues import pure_functions
from stega_template_probe import build_encoder


def lowpass_residual(residual):
    channels = []
    for c in range(3):
        plane = Image.fromarray(residual[:, :, c].astype(np.float32))
        plane = plane.resize((256, 256), Image.Resampling.BILINEAR)
        channels.append(np.asarray(plane.resize((400, 400), Image.Resampling.BILINEAR)))
    return np.stack(channels, axis=2).astype(np.float32)


class LowpassEncoder:
    """Apply the fixed linear 400->256->400 residual transform before features."""
    def __init__(self, session):
        self.session = session

    def run(self, outputs, feed):
        residual = self.session.run(outputs, feed)[0]
        return [np.stack([lowpass_residual(r) for r in residual])]


def read_256(data, source_split, name):
    with Image.open(data / source_split / "images" / f"{name}.png") as image:
        return np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC))


def raw_400(image_256):
    return np.asarray(Image.fromarray(image_256).resize((400, 400), Image.Resampling.BICUBIC), dtype=np.float32) / 255


def extract_split(args, name, encoder, functions, split):
    with np.load(args.base / f"{name}.npz") as source:
        names = source["image_names"].tolist()
        base = source["features"]
    expected = split["train"] if name == "dev" else split["val"] if name == "holdout" else None
    if expected is not None and names != expected:
        raise ValueError(f"Base {name} names differ from the original split")
    if base.shape != (len(names), 69, 64, 64) or base.dtype != np.float16:
        raise ValueError(f"Expected base float16 [N,69,64,64], got {base.shape}, {base.dtype}")
    source_split = "test" if name == "test" else "train"
    metadata = json.loads((args.data / source_split / "metadata.json").read_text())
    metadata = {row["image_name"]: row for row in metadata if row["method"] == "stegastamp"}
    if set(names) - set(metadata):
        raise ValueError(f"Unknown StegaStamp names in base {name} cache")
    destination = args.out / f"{name}.npz"
    if destination.exists():
        with np.load(destination) as cached:
            if cached["image_names"].tolist() != names:
                raise ValueError(f"Existing {name} lowpass cache has different names")
        print(f"{name}: using existing cache {destination}", flush=True)
        return len(names)
    result = np.empty((len(names), 87, 64, 64), dtype=np.float16)
    result[:, :69] = base
    del base
    session = LowpassEncoder(encoder)
    started = time.monotonic()
    for i, image_name in enumerate(names):
        image = read_256(args.data, source_split, image_name)
        bits = np.asarray([int(c) for c in metadata[image_name]["message"]], dtype=np.float32)
        # The reference function keeps gray encoder conditioning at 0.5; this
        # proxy changes only its returned residuals, not the image or messages.
        pair = functions["evidence"](image, bits, session)
        raw = raw_400(image)
        residual = session.run(["residual"], {
            "image": np.stack([raw, raw]),
            "secret": np.stack([bits, 1 - bits]).astype(np.float32),
        })[0]
        rewrite = functions["residual_maps"](raw, residual[0], residual[1])
        extra = np.concatenate((pair, rewrite)).astype(np.float16)
        if extra.shape != (18, 64, 64) or not np.isfinite(extra).all():
            raise ValueError(f"Invalid lowpass features for {image_name}")
        result[i, 69:] = extra
        if i == 0 or (i + 1) % 8 == 0 or i + 1 == len(names):
            print(f"{name}: {i + 1}/{len(names)}, elapsed={time.monotonic() - started:.1f}s", flush=True)
    temporary = destination.with_suffix(".partial.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, image_names=np.asarray(names, dtype=str), features=result)
    temporary.replace(destination)
    print(f"Saved {destination}: {result.shape} float16", flush=True)
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path(__file__).parent / "output/stega_cues")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output/stega_cues_lowpass")
    parser.add_argument("--stage", choices=["dev", "final"], default="dev")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--reference", type=Path, default=Path(__file__).parent / "reference_xuanhao")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        encoder_device = args.device
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        encoder_device = "cpu"
    else:
        raise ValueError("Use --device mps or --device cuda")
    if args.base.resolve() == args.out.resolve():
        raise ValueError("The lowpass output must not overwrite the base cache")
    args.out.mkdir(parents=True, exist_ok=True)
    split = json.loads((args.base / "split.json").read_text())
    if len(split["train"]) != 136 or len(split["val"]) != 34 or not set(split["train"]).isdisjoint(split["val"]):
        raise ValueError("Expected the original 136 DEV / 34 holdout split")
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    jobs = ["dev"] if args.stage == "dev" else ["holdout", "test"]
    with np.load(args.base / f"{jobs[0]}.npz") as first_cache:
        first_name = str(first_cache["image_names"][0])
    source_split = "test" if jobs[0] == "test" else "train"
    metadata = json.loads((args.data / source_split / "metadata.json").read_text())
    row = next(item for item in metadata if item["image_name"] == first_name)
    bits = np.asarray([int(c) for c in row["message"]], dtype=np.float32)
    native = raw_400(read_256(args.data, source_split, first_name))
    encoder, error = build_encoder(args.models_dir, args.out, native, bits, encoder_device)
    functions = pure_functions(args.reference / "stega_pair.py", ["evidence"])
    functions.update(pure_functions(args.reference / "rewrite_evidence.py", ["residual_maps"]))
    path = args.out / "feature_info.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    info = {"method": "base69 + fixed18 lowpass-corrected StegaStamp encoder cues", "features": 87,
            "shape_per_image": [87, 64, 64], "dtype": "float16", "base": str(args.base),
            "base_69_unchanged": True, "fitting": "none; no masks or training normalization loaded",
            "extra_channel_groups": [{"name": "lowpass gray known/opposite NCC", "start": 69, "end": 75},
                                     {"name": "lowpass actual-image rewrite", "start": 75, "end": 87}],
            "fixed_transforms": {"image": "512->256->400 PIL bicubic",
                                 "residual": "400->256->400 PIL float32 bilinear, each RGB channel",
                                 "gray_conditioning": 0.5, "messages": "known and 1-known"},
            "encoder_device": encoder_device, "requested_device": args.device,
            "encoder_max_abs_equivalence_error": error, "onnx_providers": encoder.get_providers(),
            "stage": args.stage, "completed_splits": old.get("completed_splits", {})}
    path.write_text(json.dumps(info, indent=2))
    for name in jobs:
        info["completed_splits"][name] = extract_split(args, name, encoder, functions, split)
        path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
