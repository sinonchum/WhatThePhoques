"""Assemble K fold directories into one 100% honest OOF fusion bundle (dev_in_sample=False)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

DING7_ROOT = Path("/p/scratch/training2666/ding7/claude_task1")
for p in (str(DING7_ROOT / "task1_v11"), str(DING7_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--folds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DING7_ROOT / "release"))
    a = ap.parse_args()

    all_rows = load_index("train", a.data)
    hold = holdout_names(all_rows)
    rows = [r for r in all_rows if r["method"] == a.method]
    row_map = {r["image_name"]: r for r in rows}
    dev_names = [r["image_name"] for r in rows if r["image_name"] not in hold]
    hold_names = [r["image_name"] for r in rows if r["image_name"] in hold]
    test_rows = [r for r in load_index("test", a.data) if r["method"] == a.method]
    test_names = [r["image_name"] for r in test_rows]

    oof_map = {}
    hold_stack = []
    test_stack = []
    for fdir in a.folds:
        fp = Path(fdir)
        val_set = set(json.loads((fp / "val_names.json").read_text()))
        tr = np.load(fp / "pred_train.npz", allow_pickle=True)
        te = np.load(fp / "pred_test.npz", allow_pickle=True)
        pmap = dict(zip(tr["names"].tolist(), tr["probs"].astype(np.float32)))
        tmap = dict(zip(te["names"].tolist(), te["probs"].astype(np.float32)))
        for n in val_set:
            oof_map[n] = pmap[n]
        hold_stack.append(np.stack([pmap[n] for n in hold_names]))
        test_stack.append(np.stack([tmap[n] for n in test_names]))

    assert set(oof_map.keys()) == set(dev_names), f"OOF mismatch: {len(oof_map)} vs {len(dev_names)}"
    dev_oof = np.stack([oof_map[n] for n in dev_names])
    # Logit-average holdout & test across folds
    eps = 1e-4
    def logit_avg(arrs):
        lg = np.mean([np.log(np.clip(x, eps, 1 - eps) / (1 - np.clip(x, eps, 1 - eps))) for x in arrs], axis=0)
        return (1.0 / (1.0 + np.exp(-lg))).astype(np.float32)

    hold_avg = logit_avg(hold_stack)
    test_avg = logit_avg(test_stack)

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    scores = [np.mean([miou_image(dev_oof[i] >= t, row_map[dev_names[i]]["mask128"]) for i in range(len(dev_names))]) for t in grid]
    thr = float(grid[int(np.argmax(scores))])

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    common = {"threshold": np.float32(thr), "dev_in_sample": np.bool_(False)}
    np.savez_compressed(out / "dev.npz", image_names=np.array(dev_names), response=dev_oof.astype(np.float32), **common)
    np.savez_compressed(out / "holdout.npz", image_names=np.array(hold_names), response=hold_avg, **common)
    np.savez_compressed(out / "test.npz", image_names=np.array(test_names), response=test_avg, **common)

    def rep(arr, names):
        L = [miou_image(arr[i] >= thr, row_map[names[i]]["mask128"]) for i in range(len(names)) if row_map[names[i]]["full_watermark"]]
        S = [miou_image(arr[i] >= thr, row_map[names[i]]["mask128"]) for i in range(len(names)) if not row_map[names[i]]["full_watermark"]]
        return float(np.mean(L + S)), float(np.mean(L)), float(np.mean(S))

    d_all, dL, dS = rep(dev_oof, dev_names)
    h_all, hL, hS = rep(hold_avg, hold_names)
    print(f"ASSEMBLED {out} ({len(a.folds)} folds, honest OOF) | thr={thr:.2f} | OOF dev {d_all:.4f} (L {dL:.4f} S {dS:.4f}) | holdout {h_all:.4f} (L {hL:.4f} S {hS:.4f})")


if __name__ == "__main__":
    main()
