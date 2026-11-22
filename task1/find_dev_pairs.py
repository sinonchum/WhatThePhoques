"""Find similar base-image fingerprints using only the fixed 408 DEV RGB files.

No model, GPU, masks, holdout/test image reads, or fitted preprocessing.
Distances are RGB RMSE in [0,1] after 16x16 area-average downsampling.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


METHODS = ("pixelseal", "trustmark", "stegastamp")
THRESHOLDS = (0.005, 0.01, 0.02)


def dev_records(split_root):
    records, paths = [], {}
    for method in METHODS:
        path = (split_root / "message_heatmaps" / method / "split.json"
                if method != "stegastamp" else
                split_root / "fusion/codex_occlusion_stegastamp/split.json")
        split = json.loads(path.read_text())
        names = split["train"]
        assert len(names) == len(set(names)) == 136, f"Expected 136 unique DEV names: {path}"
        records.extend({"image_name": name, "method": method} for name in sorted(names))
        paths[method] = str(path)
    assert len(records) == len({r["image_name"] for r in records}) == 408
    return records, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, default=Path("output"))
    parser.add_argument("--out", type=Path, default=Path("output/dev_pairs.json"))
    args = parser.parse_args()
    started = time.monotonic()
    records, split_paths = dev_records(args.split_root)
    fingerprints = []
    for record in records:
        path = args.data / "train/images" / f'{record["image_name"]}.png'
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
        assert rgb.shape == (512, 512, 3), (path, rgb.shape)
        # Exact area-average resize, without rounding tiny block means to uint8.
        fingerprints.append(rgb.reshape(16, 32, 16, 32, 3).mean(axis=(1, 3)).reshape(-1))
    x = np.stack(fingerprints)
    n = len(x)
    # Compute each unordered pair once. Float64 avoids cancellation when two
    # tiny fingerprints are almost equal; this loop needs no threaded BLAS.
    distances = np.full((n, n), np.inf, dtype=np.float64)
    for i in range(n - 1):
        differences = x[i + 1:] - x[i]
        distances[i, i + 1:] = np.sqrt((differences * differences).mean(axis=1))
    left, right = np.triu_indices(n, k=1)
    values = distances[left, right]
    methods = np.array([r["method"] for r in records])
    cross = methods[left] != methods[right]
    order = np.lexsort((right, left, values))
    nearest = []
    for j in order[:20]:
        nearest.append({"left": records[int(left[j])], "right": records[int(right[j])],
                        "rgb_rmse": float(values[j]), "cross_method": bool(cross[j])})
    counts = {}
    for threshold in THRESHOLDS:
        below = values < threshold
        counts[str(threshold)] = {"all": int(below.sum()),
                                  "cross_method": int((below & cross).sum()),
                                  "same_method": int((below & ~cross).sum())}
    report = {"status": "DEV-only base-image similarity feasibility diagnostic; not yet evidence of pixel alignment",
              "images": n, "images_per_method": {m: int((methods == m).sum()) for m in METHODS},
              "split_files": split_paths, "holdout_images_read": 0, "test_images_read": 0,
              "gt_masks_read": 0, "fingerprint": {"resolution": [16, 16], "channels": "RGB",
                  "downsample": "exact 32x32 block means: 512x512 to 16x16, no uint8 rounding", "range": [0, 1],
                  "distance": "sqrt(mean((left-right)^2)) over 768 components"},
              "unordered_pairs": len(values), "cross_method_pairs": int(cross.sum()),
              "same_method_pairs": int((~cross).sum()),
              "strictly_below_threshold": counts, "nearest_20_unordered_pairs": nearest,
              "seconds": time.monotonic() - started,
              "interpretation": "Small thumbnail RMSE suggests similar scene content only. Watermark recovery by subtraction still requires pixel-level registration and genuinely shared clean pixels."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({"images": n, "unordered_pairs": len(values),
                      "strictly_below_threshold": counts,
                      "closest_pair": nearest[0], "seconds": report["seconds"],
                      "output": str(args.out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
