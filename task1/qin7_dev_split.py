"""Reconstruct qin7's 108-fit / 28-selection real-DEV split without scoring.

Reads only prediction names, metadata and our fixed split. It never reads the
prediction probabilities, any masks, holdout/test pictures or score reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--our-split", type=Path, required=True)
    p.add_argument("--pred-train", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    metadata = {row["image_name"]: row for row in json.loads(args.metadata.read_text())}
    original = json.loads(args.our_split.read_text())
    our_dev, outer_holdout = original["train"], original["val"]
    assert len(our_dev) == len(set(our_dev)) == 136
    assert len(outer_holdout) == len(set(outer_holdout)) == 34
    assert set(our_dev).isdisjoint(outer_holdout)
    with np.load(args.pred_train, allow_pickle=False) as predictions:
        # Deliberately never access predictions['probs'].
        names = predictions["names"].tolist()
    assert len(names) == len(set(names)) == 170
    assert set(names) == set(our_dev) | set(outer_holdout)
    held = set(outer_holdout)
    dev_indices = [i for i, name in enumerate(names) if name not in held]
    dev_names = [names[i] for i in dev_indices]
    assert len(dev_names) == 136 and set(dev_names) == set(our_dev)
    assert all(metadata[name]["method"] == "stegastamp" for name in dev_names)
    rng = np.random.default_rng(7 + 101)
    val_indices, fit_indices, groups = [], [], []
    for full in (False, True):
        # Preserve pred_train's original 170-row order before the seeded shuffle.
        ids = [i for i in dev_indices if bool(metadata[names[i]]["full_watermark"]) == full]
        rng.shuffle(ids)
        n_v = max(1, int(round(len(ids) * 0.2)))
        val_indices.extend(ids[:n_v])
        fit_indices.extend(ids[n_v:])
        groups.append({"full_watermark": full, "dev": len(ids), "val": n_v, "fit": len(ids) - n_v})
    val_names, fit_names = [names[i] for i in val_indices], [names[i] for i in fit_indices]
    assert len(val_names) == len(set(val_names)) == 28
    assert len(fit_names) == len(set(fit_names)) == 108
    assert set(val_names).isdisjoint(fit_names)
    assert set(val_names) | set(fit_names) == set(our_dev)
    assert set(val_names + fit_names).isdisjoint(outer_holdout)
    result = {
        "source": "qin7/runs/stegaext_live; live arguments confirmed from its training log by root",
        "arguments": {"p_real": 0.25, "dev_val_frac": 0.2, "seed": 7, "split_rng_seed": 108},
        "metadata_path": str(args.metadata.resolve()), "our_split_path": str(args.our_split.resolve()),
        "prediction_names_source": str(args.pred_train.resolve()),
        "algorithm": "original 170-name order minus outer holdout; RNG default_rng(7+101); groups False,True; shuffle; first max(1,round(n*.2)) are val",
        "val_names": val_names, "fit_names": fit_names,
        "val_indices_in_pred_train": val_indices, "fit_indices_in_pred_train": fit_indices,
        "dev_names_in_pred_train_order": dev_names,
        "outer_holdout_names": outer_holdout,
        "counts": {"source": 170, "dev": 136, "fit": 108, "val": 28, "outer_holdout": 34},
        "groups": groups,
        "roles": {
            "fit_108": "Used as real training images for this checkpoint; not out-of-fold predictions",
            "val_28": "Not real training images, but used to choose checkpoint epoch and threshold; not a fresh untouched holdout",
            "dev_136": "Contains both fit and selection images; single-checkpoint predictions are not OOF",
            "outer_holdout_34": "Excluded before this split; no images, masks or scores read here",
            "threshold": "Selected on the 28 internal DEV validation images; its numerical value is not read or reconstructed here",
        },
        "read_probabilities": False, "read_masks": False, "computed_scores": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f"Reconstructed 108 real-fit / 28 checkpoint-and-threshold-selection DEV names -> {args.out}")


if __name__ == "__main__":
    main()
