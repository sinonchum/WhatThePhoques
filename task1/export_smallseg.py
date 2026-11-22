"""Export the existing small segmenters in COORD.md section D's fusion format.

Dev/holdout use the original saved probabilities, thresholds and split, unchanged.
Test is inference only. No retraining, threshold selection, embedding or submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from message_heatmaps import extract
from train_small_segmenter import SmallSegmenter, predict
from wmloc.data import LocalisationDataset, unzip
from wmloc.utils import miou_image
from wmloc.watermarks import load_watermark


def save_response(path, names, response, threshold):
    names = np.asarray(names, dtype=str)
    response = np.asarray(response, dtype=np.float32)
    assert response.shape == (len(names), 128, 128), response.shape
    assert len(set(names.tolist())) == len(names), "Duplicate image names"
    assert np.isfinite(response).all(), "Non-finite response"
    assert np.isfinite(threshold), "Existing threshold is not finite"
    np.savez_compressed(path, image_names=names, response=response,
                        threshold=np.asarray(threshold, dtype=np.float64),
                        dev_in_sample=np.asarray(True, dtype=np.bool_))


def score_groups(names, response, threshold, gt, metadata):
    scores = np.array([miou_image((r >= threshold).astype(np.uint8), gt[name])
                       for name, r in zip(names, response)])
    large = np.array([bool(metadata[name]["full_watermark"]) for name in names])
    return {"large": float(scores[large].mean()), "small": float(scores[~large].mean()),
            "both": float(scores.mean()), "n": len(names),
            "n_large": int(large.sum()), "n_small": int((~large).sum())}


def export_method(args, method, train_metadata, gt):
    source = args.smallseg / method
    split = json.loads((source / "split.json").read_text())
    checkpoint = torch.load(source / "model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["method"] == method
    assert checkpoint["train_names"] == split["train"] and checkpoint["val_names"] == split["val"], "Checkpoint/split mismatch"
    dev_names, holdout_names = split["train"], split["val"]
    assert len(dev_names) == 136 and len(holdout_names) == 34
    assert set(dev_names).isdisjoint(holdout_names)
    assert all(train_metadata[name]["method"] == method for name in dev_names + holdout_names)
    threshold = float(checkpoint["threshold"])
    target = args.out / f"codex_smallseg_{method}"
    target.mkdir(parents=True, exist_ok=True)
    report = {"method": method, "checkpoint": str(source / "model.pt"),
              "threshold": threshold, "dev_in_sample": True,
              "threshold_selection": "original 136-image dev threshold, unchanged"}

    with np.load(source / "predictions.npz", allow_pickle=False) as cached:
        cached_names = cached["image_names"].tolist()
        assert len(cached_names) == len(set(cached_names)) == 170
        assert set(cached_names) == set(dev_names + holdout_names)
        index = {name: i for i, name in enumerate(cached_names)}
        for label, names in (("dev", dev_names), ("holdout", holdout_names)):
            response = cached["probabilities"][[index[name] for name in names]].astype(np.float32)
            save_response(target / f"{label}.npz", names, response, threshold)
            report[label] = score_groups(names, response, threshold, gt, train_metadata)
            r = report[label]
            print(f"{method} {label}: large / small / both = {r['large']:.4f} / {r['small']:.4f} / {r['both']:.4f} (n={r['n']}, dev_in_sample=True)", flush=True)

    dataset = LocalisationDataset(args.data, "test", methods=[method])
    items = sorted(list(dataset), key=lambda it: it["image_name"])
    names = [it["image_name"] for it in items]
    assert len(names) == len(set(names)) == 340, f"{method}: expected all 340 test images"
    assert set(names).isdisjoint(dev_names + holdout_names)
    device = torch.device(args.device)
    wm = load_watermark(method, models_dir=args.models_dir.resolve(), device=args.device)
    _, heatmaps = extract(wm, items, args.batch_size)
    del wm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Dummy initialization is overwritten by every learned weight and normalization buffer.
    model = SmallSegmenter(0.0, 1.0, 0.0)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)
    images = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).to(device)
    response = predict(model, images, torch.from_numpy(heatmaps).to(device), args.batch_size, device)
    save_response(target / "test.npz", names, response, threshold)
    report["test"] = {"n": len(names), "inference_only": True}
    report["output"] = str(target)
    print(f"{method}: exported dev=136, holdout=34, test=340 -> {target}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--smallseg", type=Path, default=Path("output/small_segmenter"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--out", type=Path, default=Path("output/fusion"))
    p.add_argument("--test-zip", type=Path, help="Unzip only when release/test/metadata.json is absent")
    p.add_argument("--methods", nargs="+", choices=["pixelseal", "trustmark"], default=["pixelseal", "trustmark"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(4)
    if not (args.data / "test/metadata.json").exists():
        if args.test_zip is None:
            p.error("Test split is absent; provide --test-zip or unpack it into --data first")
        unzip(args.test_zip, args.data)
    metadata_rows = json.loads((args.data / "train/metadata.json").read_text())
    metadata = {row["image_name"]: row for row in metadata_rows}
    with np.load(args.data / "train/masks_128.npz", allow_pickle=False) as masks:
        gt = dict(zip(masks["names"].tolist(), masks["masks"]))
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for method in args.methods:
        reports.append(export_method(args, method, metadata, gt))
        (args.out / "codex_smallseg_report.json").write_text(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
