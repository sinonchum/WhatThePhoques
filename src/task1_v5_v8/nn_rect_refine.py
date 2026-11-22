"""Neural-Guided Exact Known-Message Rectangle Refinement.

Catches phase-sensitive small rectangles missed by coarse strides (e.g., f8a202d8d55a o256=+2.17)
by proposing bounding boxes from our clean neural ensemble probability maps across 8 thresholds,
connected components, local peaks, and scale/aspect jitters, then refining every top proposal
with the official known-message decoder down to +-1px!
Merges with runs/rect/{method}_{split}.npz keeping whichever box achieves strictly higher margin.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, "/p/scratch/training2666/qin7/task1")
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1")
from wmloc.data import LocalisationDataset
from wmtask.common import DATA_DIR, holdout_names, load_index
from rect_search_v2 import Scorer, refine


def load_clean_nn_mean(method: str, split: str, names: list[str]) -> np.ndarray:
    """Average probability map across all clean s3 OOF models for `method` on `split` ('train' or 'test')."""
    roots = [Path("runs/fusion"), Path("/p/scratch/training2666/ding7/claude_task1/runs/fusion")]
    acc = np.zeros((len(names), 128, 128), np.float32)
    cnt = 0
    for root in roots:
        if not root.exists(): continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.endswith(f"_{method}"): continue
            if "s3" not in d.name or any(f"_f{k}_" in d.name for k in range(8)): continue
            if not all((d / f).exists() for f in ("dev.npz", "holdout.npz", "test.npz")): continue
            dz = np.load(d / "dev.npz", allow_pickle=True)
            if bool(dz["dev_in_sample"]) if "dev_in_sample" in dz else True: continue
            if split == "train":
                hz = np.load(d / "holdout.npz", allow_pickle=True)
                mp = dict(zip(dz["image_names"].tolist(), dz["response"].astype(np.float32)))
                mp.update(dict(zip(hz["image_names"].tolist(), hz["response"].astype(np.float32))))
            else:
                tz = np.load(d / "test.npz", allow_pickle=True)
                mp = dict(zip(tz["image_names"].tolist(), tz["response"].astype(np.float32)))
            if all(n in mp for n in names):
                acc += np.stack([mp[n] for n in names])
                cnt += 1
    assert cnt > 0, f"No clean OOF models found for {method}"
    return acc / cnt


def propose_boxes_from_prob(prob128: np.ndarray) -> np.ndarray:
    """Generate ~180 surgical bounding-box proposals (in 512x512 coords) from neural prob map."""
    props = []
    # 1. Connected component bboxes across 8 thresholds
    for thr in (0.04, 0.08, 0.14, 0.22, 0.32, 0.45, 0.60, 0.75):
        lbl, nc = ndimage.label(prob128 >= thr)
        if nc == 0: continue
        # Rank components by peak prob * sqrt(area)
        comps = []
        for c in range(1, nc + 1):
            ys, xs = np.where(lbl == c)
            if len(ys) < 4: continue
            sc = float(prob128[ys, xs].max()) * np.sqrt(len(ys))
            comps.append((sc, ys.min() * 4.0, xs.min() * 4.0, (ys.max() + 1) * 4.0, (xs.max() + 1) * 4.0))
        comps.sort(reverse=True)
        for _, y0, x0, y1, x1 in comps[:4]:
            h, w = y1 - y0, x1 - x0
            cy, cx = 0.5 * (y0 + y1), 0.5 * (x0 + x1)
            for sh in (0.85, 1.0, 1.18):
                for sw in (0.85, 1.0, 1.18):
                    nh, nw = max(16.0, h * sh), max(16.0, w * sw)
                    props.append([max(0.0, cy - nh / 2), max(0.0, cx - nw / 2),
                                  min(512.0, cy + nh / 2), min(512.0, cx + nw / 2)])
    # 2. Local peaks in smoothed prob map -> multi-size square/rect boxes around top-4 peaks
    sm = ndimage.gaussian_filter(prob128, sigma=2.0)
    mx = ndimage.maximum_filter(sm, size=9)
    pks = np.argwhere((sm == mx) & (sm > 0.05))
    if len(pks) > 0:
        vals = sm[pks[:, 0], pks[:, 1]]
        top_pks = pks[np.argsort(-vals)[:4]]
        for py, px in top_pks:
            cy, cx = py * 4.0 + 2.0, px * 4.0 + 2.0
            for h, w in ((36, 36), (56, 56), (84, 84), (112, 112), (160, 160),
                         (40, 100), (100, 40), (64, 140), (140, 64), (96, 220), (220, 96)):
                props.append([max(0.0, cy - h / 2), max(0.0, cx - w / 2),
                              min(512.0, cy + h / 2), min(512.0, cx + w / 2)])
    if not props:
        props.append([128.0, 128.0, 384.0, 384.0])
    arr = np.clip(np.array(props, np.float32), 0.0, 512.0)
    ok = (arr[:, 2] - arr[:, 0] >= 12.0) & (arr[:, 3] - arr[:, 1] >= 12.0)
    return arr[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--rect-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DATA_DIR))
    a = ap.parse_args()

    dev = "cuda"
    score = Scorer(a.method, dev)
    ds = LocalisationDataset(a.data, a.split, methods=[a.method])
    names = [ds[i]["image_name"] for i in range(len(ds))]
    probs = load_clean_nn_mean(a.method, a.split, names)
    base = dict(np.load(a.rect_npz, allow_pickle=True))
    b_idx = {str(nm): i for i, nm in enumerate(base["names"])}

    t0 = time.time()
    n_improved = 0
    for i in range(len(ds)):
        it = ds[i]
        j = b_idx[it["image_name"]]
        img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
        sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
        props = propose_boxes_from_prob(probs[i])
        # Include existing best box too
        props = np.vstack([props, base["box"][j][None].astype(np.float32)])
        m, _ = score(img01, props, sign, bs=512)
        top = np.argsort(-m)[:6]
        cands = [refine(score, img01, sign, props[k], minimize=False, rounds=(0.20, 0.10, 0.05, 0.025, 0.012)) for k in top]
        box, bm, ba = max(cands, key=lambda c: c[1])
        if bm > float(base["margin"][j]) + 0.05:
            base["box"][j] = box
            base["margin"][j] = bm
            base["acc"][j] = ba
            n_improved += 1

    np.savez(a.out, **base)
    print(f"[{a.method}/{a.split}] NN-RECT-REFINE DONE in {(time.time()-t0):.1f}s | improved={n_improved}/{len(ds)} -> {a.out}", flush=True)
    if a.split == "train":
        hold = holdout_names(load_index("train", a.data))
        rows = []
        for i in range(len(ds)):
            it = ds[i]; j = b_idx[it["image_name"]]
            mask = it["mask"] > 0
            ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
            kind = "large" if it["full_watermark"] else ("rect" if mask.sum()/((y1-y0)*(x1-x0)) > 0.98 else "object")
            pb = np.zeros_like(mask); b = np.round(base["box"][j]).astype(int); pb[b[0]:b[2], b[1]:b[3]] = True
            iou = (pb & mask).sum() / max((pb | mask).sum(), 1)
            rows.append((kind, iou, float(base["margin"][j]), float(base["acc"][j])))
        for kind in ("rect", "object", "large"):
            r = [x for x in rows if x[0] == kind]
            if not r: continue
            iou = np.array([x[1] for x in r]); bm = np.array([x[2] for x in r]); ba = np.array([x[3] for x in r])
            print(f"  [{a.method}-NNREF] {kind:6s} n={len(r):3d} | box_IoU={iou.mean():.4f} (>0.5:{(iou>0.5).mean():.2f} >0.8:{(iou>0.8).mean():.2f}) | box_m={np.median(bm):.2f} acc={np.median(ba):.3f}", flush=True)


if __name__ == "__main__":
    main()
