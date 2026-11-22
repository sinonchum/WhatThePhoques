"""Step 1 baselines on the train split. Needs only numpy (no GPU, no models), so it is fine on a login node.

    python task1/baselines.py                       # report all-zeros / all-ones / oracle large-vs-small
    python task1/baselines.py --write zeros.pt      # also write an all-zeros submission for the test split

Reports mIoU by method and by large / small, on all train images and on the dev / held-out split."""
from __future__ import annotations

import argparse
import os

import numpy as np

from wmtask.common import METHODS, check_submission, holdout_names, load_index, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="release dir (default: <task dir>/release)")
    ap.add_argument("--write", default=None, help="write an all-zeros test submission to this path")
    a = ap.parse_args()

    rows = load_index("train", a.data)
    hold = holdout_names(rows)
    print(f"train images: {len(rows)}   held-out: {len(hold)}   dev: {len(rows) - len(hold)}")

    print("\n--- data facts (train) ---")
    for method in METHODS:
        for full in (True, False):
            sel = [r for r in rows if r["method"] == method and bool(r.get("full_watermark")) == full]
            area = np.array([r["mask128"].mean() for r in sel])
            print(f"{method:11s} full_watermark={str(full):5s} n={len(sel):3d}  watermarked area: "
                  f"median {np.median(area):.3f}  min {area.min():.3f}  max {area.max():.3f}  "
                  f"all-0 masks {int((area == 0).sum())}  all-1 masks {int((area == 1).sum())}  "
                  f"nbits {len(sel[0]['message']) if sel else '-'}")

    zeros = np.zeros((128, 128), np.uint8)
    ones = np.ones((128, 128), np.uint8)
    report(rows, {r["image_name"]: zeros for r in rows}, "all zeros", hold)
    report(rows, {r["image_name"]: ones for r in rows}, "all ones", hold)
    # Upper bound for the step-2 classifier: uses the train-only full_watermark label, so NOT a real method.
    report(rows, {r["image_name"]: (ones if r.get("full_watermark") else zeros) for r in rows},
           "ORACLE large -> ones, small -> zeros (uses the train-only label; upper bound for step 2)", hold)

    if a.write:
        import torch
        test = load_index("test", a.data)
        names = [r["image_name"] for r in test]
        masks = torch.zeros((len(names), 128, 128), dtype=torch.bool)
        check_submission(names, masks, names)
        torch.save({"image_names": names, "masks": masks}, a.write)
        print(f"\nwrote {a.write}: {len(names)} masks, {os.path.getsize(a.write) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
