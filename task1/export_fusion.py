"""Export one run's continuous response maps in the format agreed with the coordinator, for fusion.

    python task1/export_fusion.py --run runs/seg/trustmark_v1 --method trustmark --name segv1 \
        --out runs/fusion [--dev-in-sample]

Writes <out>/claude_<name>_<method>/{dev,holdout,test}.npz, each with
  image_names [N] str, response [N,128,128] float32 (higher = more likely watermarked),
  threshold (scalar, chosen on DEV only; binary prediction = response >= threshold), dev_in_sample (bool).
Prints the scores of that threshold: large / small / both on dev and on held-out (official scoring)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+", help="one run dir, or several to average their responses")
    ap.add_argument("--method", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="runs/fusion")
    ap.add_argument("--data", default=None)
    ap.add_argument("--dev-in-sample", action="store_true", help="set if real dev images were used to train or fit")
    a = ap.parse_args()

    rows = {r["image_name"]: r for r in load_index("train", a.data) if r["method"] == a.method}
    hold = holdout_names(load_index("train", a.data))
    trs = [np.load(Path(r) / "pred_train.npz") for r in a.run]; tes = [np.load(Path(r) / "pred_test.npz") for r in a.run]
    assert all(t["names"].tolist() == trs[0]["names"].tolist() for t in trs) and all(t["names"].tolist() == tes[0]["names"].tolist() for t in tes)
    tr = {"names": trs[0]["names"], "probs": np.mean([t["probs"].astype(np.float32) for t in trs], 0)}
    te = {"names": tes[0]["names"], "probs": np.mean([t["probs"].astype(np.float32) for t in tes], 0)}
    names = tr["names"].tolist(); resp = tr["probs"].astype(np.float32)
    assert set(names) == set(rows), "pred_train.npz does not cover this method's train images"
    is_hold = np.array([n in hold for n in names])

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    dev_idx = np.where(~is_hold)[0]
    dev_scores = [np.mean([miou_image(resp[i] >= t, rows[names[i]]["mask128"]) for i in dev_idx]) for t in grid]
    thr = float(grid[int(np.argmax(dev_scores))])

    def scores(idx):
        out = {}
        for key, want in (("large", True), ("small", False), ("both", None)):
            v = [miou_image(resp[i] >= thr, rows[names[i]]["mask128"]) for i in idx
                 if want is None or bool(rows[names[i]]["full_watermark"]) == want]
            out[key] = (float(np.mean(v)), len(v))
        return out

    d = Path(a.out) / f"claude_{a.name}_{a.method}"; d.mkdir(parents=True, exist_ok=True)
    common = {"threshold": np.float32(thr), "dev_in_sample": np.bool_(a.dev_in_sample)}
    np.savez_compressed(d / "dev.npz", image_names=np.array([names[i] for i in dev_idx]), response=resp[dev_idx], **common)
    hi = np.where(is_hold)[0]
    np.savez_compressed(d / "holdout.npz", image_names=np.array([names[i] for i in hi]), response=resp[hi], **common)
    np.savez_compressed(d / "test.npz", image_names=te["names"], response=te["probs"].astype(np.float32), **common)

    print(f"{a.method} / {a.name}: threshold {thr} (dev only; dev mIoU at 0.5 = {dev_scores[int(np.argmin(np.abs(grid - 0.5)))]:.4f}), "
          f"dev_in_sample={a.dev_in_sample}")
    for split, idx in (("dev", dev_idx), ("holdout", hi)):
        s = scores(idx)
        print(f"  {split:8s} large {s['large'][0]:.4f} (n={s['large'][1]})  small {s['small'][0]:.4f} (n={s['small'][1]})  "
              f"both {s['both'][0]:.4f} (n={s['both'][1]})")
    print(f"  wrote {d}/{{dev,holdout,test}}.npz  (dev {len(dev_idx)}, holdout {len(hi)}, test {len(te['names'])})")


if __name__ == "__main__":
    main()
