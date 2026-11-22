"""Combine per-method predictions into the full train report and a submission file.

    python task1/assemble.py --runs pixelseal=runs/seg/pixelseal_v1 trustmark=runs/seg/trustmark_v1 \
        stegastamp=runs/seg/stegastamp_v1 --out runs/sub_v1.pt [--tune-thr]

Each run dir holds pred_train.npz / pred_test.npz (names, probs). A method without a run is predicted all zeros.
--tune-thr picks one probability threshold per method on the DEV images only; held-out stays untouched."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from wmloc.utils import miou_image
from wmtask.common import METHODS, check_submission, holdout_names, load_index, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="method=run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--tune-thr", action="store_true")
    a = ap.parse_args()
    runs = dict(s.split("=", 1) for s in a.runs)
    rows = load_index("train", a.data); hold = holdout_names(rows)
    test = load_index("test", a.data)
    by_name = {r["image_name"]: r for r in rows}

    train_pred = {r["image_name"]: np.zeros((128, 128), np.uint8) for r in rows}
    test_pred = {r["image_name"]: np.zeros((128, 128), bool) for r in test}
    for method in METHODS:
        if method not in runs:
            print(f"{method}: no run given -> all zeros"); continue
        tr = np.load(os.path.join(runs[method], "pred_train.npz")); te = np.load(os.path.join(runs[method], "pred_test.npz"))
        thr = 0.5
        if a.tune_thr:
            dev = [(n, p.astype(np.float32)) for n, p in zip(tr["names"].tolist(), tr["probs"]) if n not in hold]
            grid = np.round(np.arange(0.1, 0.91, 0.05), 2)
            sc = [np.mean([miou_image(p > t, by_name[n]["mask128"]) for n, p in dev]) for t in grid]
            thr = float(grid[int(np.argmax(sc))])
            print(f"{method}: dev mIoU by threshold: " + " ".join(f"{t:.2f}:{s:.4f}" for t, s in zip(grid, sc)))
        print(f"{method}: run {runs[method]}  threshold {thr}")
        for n, p in zip(tr["names"].tolist(), tr["probs"]):
            train_pred[n] = (p.astype(np.float32) > thr).astype(np.uint8)
        for n, p in zip(te["names"].tolist(), te["probs"]):
            test_pred[n] = p.astype(np.float32) > thr

    report(rows, train_pred, "combined: " + ", ".join(f"{m}={runs.get(m, 'zeros')}" for m in METHODS), hold)

    if a.out:
        names = [r["image_name"] for r in test]
        masks = torch.from_numpy(np.stack([test_pred[n] for n in names])).bool()
        check_submission(names, masks, names)
        torch.save({"image_names": names, "masks": masks}, a.out)
        frac = masks.float().mean(dim=(1, 2))
        print(f"\nwrote {a.out}: {len(names)} masks, {os.path.getsize(a.out) / 1e6:.2f} MB; per-method test stats:")
        for method in METHODS:
            f = frac[[i for i, r in enumerate(test) if r["method"] == method]]
            print(f"  {method:11s} mean watermarked fraction {f.mean():.3f}; images predicted >50% watermarked: {(f > 0.5).float().mean():.3f} (expect about 0.5)")


if __name__ == "__main__":
    main()
