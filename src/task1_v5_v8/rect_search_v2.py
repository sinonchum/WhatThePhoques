"""GPU-Sharded BF16 Exact Known-Message Rectangle & Hole Search (V2).

Exploits the exact watermark embedding physics of Task 1:
  - small, rectangle: watermark embedded on a rectangular crop -> exact crop decodes known message at ~0.90-1.00 acc!
  - small, object:    irregular region of whole-image watermarked version -> max_map highlights object location!
  - large (hole):     whole image watermarked, clean hole restored -> min_map & hole_box highlight clean hole!
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import roi_align

sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1/task1_v11")
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1")
from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmtask.common import DATA_DIR, holdout_names, load_index

SIZES = [24, 36, 56, 84, 128, 192, 288, 416]


def coarse_boxes(max_aspect: float) -> np.ndarray:
    out = []
    for h in SIZES:
        for w in SIZES:
            if max(h, w) / min(h, w) > max_aspect or h * w > 0.48 * 512 * 512:
                continue
            sy = max(12, h // 2)
            sx = max(12, w // 2)
            ys = sorted(set(list(range(0, 512 - h + 1, sy)) + [512 - h]))
            xs = sorted(set(list(range(0, 512 - w + 1, sx)) + [512 - w]))
            out += [(y, x, y + h, x + w) for y in ys for x in xs]
    return np.array(out, np.float32)


class Scorer:
    def __init__(self, method: str, device: str):
        self.wm = load_watermark(method, device=device)
        self.method = method
        self.device = device
        self.res = 256 if method == "pixelseal" else self.wm.resolution

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor, boxes: np.ndarray, sign: torch.Tensor, bs: int = 1024):
        m, a = [], []
        for s in range(0, len(boxes), bs):
            b = torch.from_numpy(boxes[s:s + bs]).to(self.device)
            rois = torch.cat([torch.zeros(len(b), 1, device=self.device), b[:, [1, 0, 3, 2]]], 1)
            crops = roi_align(img01, rois, output_size=self.res, sampling_ratio=2, aligned=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if self.method == "pixelseal":
                    lg = self.wm.det(crops)[:, 1:].float()
                else:
                    lg = self.wm.net(crops * 2 - 1).float()
            sl = lg * sign
            m.append(sl.mean(1))
            a.append((sl > 0).float().mean(1))
        return torch.cat(m).cpu().numpy(), torch.cat(a).cpu().numpy()


def refine(score, img01, sign, box, minimize: bool = False, rounds=(0.25, 0.12, 0.06, 0.03, 0.015)):
    best = np.array(box, np.float32)
    bm, ba = score(img01, best[None], sign)
    bm, ba = float(bm[0]), float(ba[0])
    for frac in rounds:
        for _ in range(4):
            h, w = best[2] - best[0], best[3] - best[1]
            d = np.array([frac * h, frac * w, frac * h, frac * w], np.float32)
            cand = []
            for e in range(4):
                for sgn in (-1, 1):
                    c = best.copy()
                    c[e] += sgn * max(2.0, d[e])
                    cand.append(c)
            for sgn in (-1, 1):
                c = best.copy(); c[[0, 2]] += sgn * max(2.0, d[0]); cand.append(c)
                c = best.copy(); c[[1, 3]] += sgn * max(2.0, d[1]); cand.append(c)
            cand = np.clip(np.array(cand), 0, 512)
            ok = (cand[:, 2] - cand[:, 0] >= 8) & (cand[:, 3] - cand[:, 1] >= 8)
            cand = cand[ok]
            if len(cand) == 0:
                break
            m, a = score(img01, cand, sign)
            j = int(np.argmin(m) if minimize else np.argmax(m))
            improved = (m[j] < bm - 1e-4) if minimize else (m[j] > bm + 1e-4)
            if not improved:
                break
            best, bm, ba = cand[j], float(m[j]), float(a[j])
    return best, bm, ba


def build_maps(boxes: np.ndarray, margins: np.ndarray):
    """Build 128x128 max_map (all coarse boxes) and min_map (small/medium boxes <=160px)."""
    b128 = np.clip(np.round(boxes / 4.0).astype(np.int32), 0, 128)
    max_map = np.full((128, 128), -1e9, np.float32)
    min_map = np.full((128, 128), +1e9, np.float32)
    # Sort descending for top-400 boxes for max_map, ascending for bottom-400 small boxes for min_map
    order_hi = np.argsort(-margins)[:500]
    for idx in order_hi:
        y0, x0, y1, x1 = b128[idx]
        if y1 > y0 and x1 > x0:
            v = margins[idx]
            sl = max_map[y0:y1, x0:x1]
            np.maximum(sl, v, out=sl)
    # Fill untouched pixels with global median
    med = float(np.median(margins))
    max_map[max_map < -1e8] = med

    hs = boxes[:, 2] - boxes[:, 0]
    ws = boxes[:, 3] - boxes[:, 1]
    small_idx = np.where((hs <= 160) & (ws <= 160) & (hs >= 28) & (ws >= 28))[0]
    order_lo = small_idx[np.argsort(margins[small_idx])[:500]]
    for idx in order_lo:
        y0, x0, y1, x1 = b128[idx]
        if y1 > y0 and x1 > x0:
            v = margins[idx]
            sl = min_map[y0:y1, x0:x1]
            np.minimum(sl, v, out=sl)
    min_map[min_map > 1e8] = med
    return max_map.astype(np.float16), min_map.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    if a.merge:
        shards = [np.load(f"{a.out}.r{r}.npz", allow_pickle=True) for r in range(a.world)]
        merged = {}
        for k in shards[0].files:
            merged[k] = np.concatenate([s[k] for s in shards], axis=0)
        # Re-sort by original index
        order = np.argsort(merged["orig_idx"])
        for k in merged:
            merged[k] = merged[k][order]
        np.savez(a.out, **merged)
        print(f"MERGED {a.out} | N={len(merged['names'])}", flush=True)
        if a.split == "train":
            ds = LocalisationDataset(a.data, "train", methods=[a.method])
            hold = holdout_names(load_index("train", a.data))
            by_name = {ds[i]["image_name"]: ds[i] for i in range(len(ds))}
            rows = []
            for i, nm in enumerate(merged["names"]):
                it = by_name[str(nm)]
                mask = it["mask"] > 0
                ys, xs = np.where(mask)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                kind = "large" if it["full_watermark"] else ("rect" if mask.sum() / ((y1 - y0) * (x1 - x0)) > 0.98 else "object")
                pb = np.zeros_like(mask)
                b = np.round(merged["box"][i]).astype(int)
                pb[b[0]:b[2], b[1]:b[3]] = True
                iou = (pb & mask).sum() / max((pb | mask).sum(), 1)
                rows.append((kind, str(nm) in hold, iou, float(merged["margin"][i]), float(merged["acc"][i]),
                             float(merged["full_margin"][i]), float(merged["full_acc"][i]), min(y1 - y0, x1 - x0)))
            for kind in ("rect", "object", "large"):
                r = [x for x in rows if x[0] == kind]
                if not r: continue
                iou = np.array([x[2] for x in r]); bm = np.array([x[3] for x in r])
                ba = np.array([x[4] for x in r]); fm = np.array([x[5] for x in r])
                print(f"  [{a.method}] {kind:6s} n={len(r):3d} | box_IoU={iou.mean():.4f} (>0.5:{(iou>0.5).mean():.2f} >0.8:{(iou>0.8).mean():.2f}) | box_m={np.median(bm):.2f} box_acc={np.median(ba):.3f} | full_m={np.median(fm):.2f}", flush=True)
        return

    dev = "cuda"
    score = Scorer(a.method, dev)
    boxes = coarse_boxes(2.2 if a.method == "trustmark" else 8.0)
    hs = boxes[:, 2] - boxes[:, 0]; ws = boxes[:, 3] - boxes[:, 1]
    hole_cands_idx = np.where((hs >= 32) & (ws >= 32) & (hs <= 224) & (ws <= 224) & (np.maximum(hs, ws) / np.minimum(hs, ws) <= 4.0))[0]
    ds = LocalisationDataset(a.data, a.split, methods=[a.method])
    idxs = list(range(a.rank, len(ds), a.world))
    print(f"[{a.method}/{a.split} r{a.rank}/{a.world}] imgs={len(idxs)} coarse={len(boxes)}", flush=True)

    res = {k: [] for k in ("orig_idx", "names", "box", "margin", "acc", "full_margin", "full_acc",
                           "hole_box", "hole_margin", "hole_acc", "max_map", "min_map")}
    t0 = time.time()
    for k_i, i in enumerate(idxs):
        it = ds[i]
        img01 = torch.from_numpy(it["image"]).to(dev).permute(2, 0, 1)[None].float() / 255.0
        sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
        fm, fa = score(img01, np.array([[0, 0, 512, 512]], np.float32), sign)
        m, _ = score(img01, boxes, sign)
        # Top-K max-margin boxes (small rectangle)
        top = np.argsort(-m)[:a.topk]
        cands = [refine(score, img01, sign, boxes[j], minimize=False) for j in top]
        box, bm, ba = max(cands, key=lambda c: c[1])
        # Bottom-4 min-margin boxes (clean hole on large images)
        bot = hole_cands_idx[np.argsort(m[hole_cands_idx])[:4]]
        hcands = [refine(score, img01, sign, boxes[j], minimize=True) for j in bot]
        hbox, hm, ha = min(hcands, key=lambda c: c[1])
        max_map, min_map = build_maps(boxes, m)

        for key, val in zip(res, (i, it["image_name"], box, bm, ba, float(fm[0]), float(fa[0]),
                                  hbox, hm, ha, max_map, min_map)):
            res[key].append(val)
        if (k_i + 1) % 25 == 0 or (k_i + 1) == len(idxs):
            print(f"  [{a.method}/{a.split} r{a.rank}] {k_i+1}/{len(idxs)} ({(time.time()-t0)/(k_i+1):.2f}s/img)", flush=True)

    out_path = f"{a.out}.r{a.rank}.npz" if a.world > 1 else a.out
    np.savez(out_path, **{k: np.array(v) for k, v in res.items()})
    print(f"SAVED {out_path} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
