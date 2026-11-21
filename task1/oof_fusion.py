"""Assemble cross-validated runs (train_seg.py --fold k) into fair responses and export them for fusion.

    python task1/oof_fusion.py --runs runs/seg/trustmark_s3_512real_cv{0,1,2,3} --method trustmark --name s3_512real_cv

dev response   = for every dev image, the prediction of the ONE fold that did not train on it (out-of-fold)
holdout / test = mean over the folds (no fold trained on them)
The threshold is chosen on the out-of-fold dev responses, so dev scores here are a fair estimate too."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="runs/fusion")
    a = ap.parse_args()
    allrows = load_index("train"); hold = holdout_names(allrows)
    rows = {r["image_name"]: r for r in allrows if r["method"] == a.method}
    trs = [np.load(Path(r) / "pred_train.npz") for r in a.runs]; tes = [np.load(Path(r) / "pred_test.npz") for r in a.runs]
    names = trs[0]["names"].tolist()
    P = np.stack([t["probs"].astype(np.float32) for t in trs])                      # folds x N x 128 x 128
    oof_of = {}
    for k, r in enumerate(a.runs):
        for n in json.loads((Path(r) / "oof_names.json").read_text()):
            assert n not in oof_of; oof_of[n] = k
    dev_names = [n for n in names if n not in hold]
    assert set(dev_names) == set(oof_of), f"folds cover {len(oof_of)} of {len(dev_names)} dev images"
    idx = {n: i for i, n in enumerate(names)}
    dev = np.stack([P[oof_of[n], idx[n]] for n in dev_names])
    hold_names = [n for n in names if n in hold]
    ho = P[:, [idx[n] for n in hold_names]].mean(0)
    test = np.mean([t["probs"].astype(np.float32) for t in tes], 0)

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    sc = [np.mean([miou_image(p >= t, rows[n]["mask128"]) for n, p in zip(dev_names, dev)]) for t in grid]
    thr = float(grid[int(np.argmax(sc))])

    def scores(ns, ps):
        out = {}
        for key, want in (("large", True), ("small", False), ("both", None)):
            v = [miou_image(p >= thr, rows[n]["mask128"]) for n, p in zip(ns, ps) if want is None or bool(rows[n]["full_watermark"]) == want]
            out[key] = (float(np.mean(v)), len(v))
        return out

    d = Path(a.out) / f"claude_{a.name}_{a.method}"; d.mkdir(parents=True, exist_ok=True)
    common = {"threshold": np.float32(thr), "dev_in_sample": np.bool_(False)}       # dev responses are out-of-fold
    np.savez_compressed(d / "dev.npz", image_names=np.array(dev_names), response=dev, **common)
    np.savez_compressed(d / "holdout.npz", image_names=np.array(hold_names), response=ho, **common)
    np.savez_compressed(d / "test.npz", image_names=tes[0]["names"], response=test, **common)
    print(f"{a.method} / {a.name}: {len(a.runs)} folds, threshold {thr} chosen on out-of-fold dev")
    for split, ns, ps in (("dev (out-of-fold)", dev_names, dev), ("holdout (fold mean)", hold_names, ho)):
        s = scores(ns, ps)
        print(f"  {split:20s} large {s['large'][0]:.4f} (n={s['large'][1]})  small {s['small'][0]:.4f} (n={s['small'][1]})  both {s['both'][0]:.4f} (n={s['both'][1]})")
    for k in range(len(a.runs)):
        v = [miou_image(P[k, idx[n]] >= thr, rows[n]["mask128"]) for n in hold_names]
        print(f"  fold {k} alone on holdout: {np.mean(v):.4f}")
    print(f"  wrote {d}/{{dev,holdout,test}}.npz")


if __name__ == "__main__":
    main()
