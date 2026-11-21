"""Shared helpers for Task 1: paths, a light train index (no image decoding), the fixed dev / held-out split,
and the per-group mIoU report. Scoring always goes through the official wmloc.utils functions."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from wmloc.utils import miou_image

METHODS = ("pixelseal", "trustmark", "stegastamp")
TASK_DIR = Path(os.environ.get("WMLOC_TASK_DIR", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("WMLOC_DATA", TASK_DIR / "release"))
HOLDOUT_SEED = 2026


def load_index(split: str = "train", data_dir=None) -> list[dict]:
    """Metadata rows (image_name, method, message as np.uint8 bits, full_watermark for train) plus mask128 for
    train. Does not open any image."""
    d = Path(data_dir or DATA_DIR) / split
    meta = json.loads((d / "metadata.json").read_text())
    m128 = {}
    if (d / "masks_128.npz").exists():
        z = np.load(d / "masks_128.npz")
        m128 = dict(zip(z["names"].tolist(), z["masks"]))
    rows = []
    for m in meta:
        r = dict(m)
        r["message"] = np.array([int(c) for c in m["message"]], dtype=np.uint8)
        if m["image_name"] in m128:
            r["mask128"] = m128[m["image_name"]]
        rows.append(r)
    return rows


def holdout_names(rows: list[dict], seed: int = HOLDOUT_SEED) -> set[str]:
    """Held-out image names. Never tune on these.
    Deliberately the SAME split as message_heatmaps.py (the other Task 1 worker): per method, 20 % held out
    (34 of 170), stratified by full_watermark and watermarked area, seed 2026. Keeping both in sync means an
    image held out by one of us is never tuned on by the other."""
    out: set[str] = set()
    for method in METHODS:
        items = [r for r in rows if r["method"] == method]          # metadata order, as LocalisationDataset
        rng = np.random.default_rng(seed)
        for full in (False, True):
            ids = [i for i, it in enumerate(items) if bool(it["full_watermark"]) == full]
            ids.sort(key=lambda i: (items[i]["mask128"].mean(), items[i]["image_name"]))
            for group in np.array_split(ids, round(len(ids) * 0.2)):
                out.add(items[int(rng.choice(group))]["image_name"])
    return out


def report(rows: list[dict], preds: dict, title: str = "", holdout: set[str] | None = None) -> dict:
    """Print mIoU by method x (large = full_watermark True, small = False), for all / dev / held-out.
    `preds` maps image_name -> 128 x 128 array of 0/1. Returns {split: {method: {group: (mean, n)}}}."""
    holdout = holdout_names(rows) if holdout is None else holdout
    per = {r["image_name"]: miou_image(preds[r["image_name"]], r["mask128"]) for r in rows}
    result = {}
    print(f"\n##### {title}")
    for split in ("all", "dev", "holdout"):
        sel = [r for r in rows if split == "all" or (r["image_name"] in holdout) == (split == "holdout")]
        result[split] = {}
        print(f"[{split:7s}] {'method':11s} {'large':>13s} {'small':>13s} {'both':>13s}")
        for method in METHODS + ("ALL",):
            cells = {}
            for group in ("large", "small", "both"):
                v = [per[r["image_name"]] for r in sel
                     if (method == "ALL" or r["method"] == method)
                     and (group == "both" or bool(r.get("full_watermark")) == (group == "large"))]
                cells[group] = (float(np.mean(v)) if v else float("nan"), len(v))
            result[split][method] = cells
            print(f"          {method:11s} " + " ".join(f"{m:.4f} (n={n:3d})" for m, n in cells.values()))
    print(f"TOTAL (all {len(rows)} train images): {result['all']['ALL']['both'][0]:.4f}")
    return result


def check_submission(names, masks, expected_names) -> None:
    """Raise if the submission does not match the required format."""
    import torch
    assert isinstance(names, list) and all(isinstance(n, str) for n in names), "image_names must be a list of str"
    assert isinstance(masks, torch.Tensor), "masks must be a torch tensor"
    assert masks.dtype in (torch.bool, torch.uint8), f"masks dtype {masks.dtype}: use bool or uint8"
    assert tuple(masks.shape) == (len(expected_names), 128, 128), f"masks shape {tuple(masks.shape)}"
    assert len(names) == len(set(names)) == len(expected_names), "duplicate or missing image names"
    assert set(names) == set(expected_names), "image names differ from the test set"
    assert int(masks.max()) <= 1, "masks must contain only 0 and 1"
