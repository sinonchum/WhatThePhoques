"""Rectangle search with the official decoder (PixelSeal / TrustMark): find the axis-aligned box whose crop,
resized to the decoder's input, reads the KNOWN message best. Small-rectangle images were watermarked on
exactly such a crop, so the right box decodes far better than the whole image.

    python task1/rect_search.py --method pixelseal --split train --out runs/rect/pixelseal_train.npz

Coarse grid over box sizes and positions, then greedy edge refinement of the best candidates. Saves per image:
box (y0, x0, y1, x1), margin = mean_k sign_k * logit_k of the box, bit accuracy of the box, and the same two
numbers for the whole image. For the train split it prints IoU of the box against the true mask by image kind."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torchvision.ops import roi_align

from wmloc.data import LocalisationDataset
from wmloc.watermarks import load_watermark
from wmtask.common import DATA_DIR, holdout_names, load_index

SIZES = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def coarse_boxes(max_aspect: float) -> np.ndarray:
    out = []
    for h in SIZES:
        for w in SIZES:
            if max(h, w) / min(h, w) > max_aspect or h * w > 0.45 * 512 * 512:
                continue
            sy = max(8, h // 3); sx = max(8, w // 3)
            ys = sorted(set(list(range(0, 512 - h + 1, sy)) + [512 - h])); xs = sorted(set(list(range(0, 512 - w + 1, sx)) + [512 - w]))
            out += [(y, x, y + h, x + w) for y in ys for x in xs]
    return np.array(out, np.float32)


class Scorer:
    def __init__(self, method, device):
        self.wm = load_watermark(method, device=device); self.method = method; self.device = device
        self.res = 256 if method == "pixelseal" else self.wm.resolution

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor, boxes: np.ndarray, sign: torch.Tensor, bs: int = 512):
        """img01 1x3x512x512 on device; boxes [N,4] (y0,x0,y1,x1). Returns margin [N], accuracy [N]."""
        m, a = [], []
        for s in range(0, len(boxes), bs):
            b = torch.from_numpy(boxes[s:s + bs]).to(self.device)
            rois = torch.cat([torch.zeros(len(b), 1, device=self.device), b[:, [1, 0, 3, 2]]], 1)      # (idx, x0, y0, x1, y1)
            crops = roi_align(img01, rois, output_size=self.res, sampling_ratio=2, aligned=True)
            if self.method == "pixelseal":
                lg = self.wm.det(crops)[:, 1:].float()
            else:
                lg = self.wm.net(crops * 2 - 1).float()
            sl = lg * sign
            m.append(sl.mean(1)); a.append((sl > 0).float().mean(1))
        return torch.cat(m).cpu().numpy(), torch.cat(a).cpu().numpy()


def refine(score, img01, sign, box, rounds=(0.25, 0.12, 0.06, 0.03)):
    best = np.array(box, np.float32); bm, ba = score(img01, best[None], sign); bm, ba = float(bm[0]), float(ba[0])
    for frac in rounds:
        for _ in range(4):
            h, w = best[2] - best[0], best[3] - best[1]
            d = np.array([frac * h, frac * w, frac * h, frac * w], np.float32)
            cand = []
            for e in range(4):
                for sgn in (-1, 1):
                    c = best.copy(); c[e] += sgn * max(2.0, d[e]); cand.append(c)
            for sgn in (-1, 1):                                    # shifts
                c = best.copy(); c[[0, 2]] += sgn * max(2.0, d[0]); cand.append(c)
                c = best.copy(); c[[1, 3]] += sgn * max(2.0, d[1]); cand.append(c)
            cand = np.clip(np.array(cand), 0, 512)
            ok = (cand[:, 2] - cand[:, 0] >= 8) & (cand[:, 3] - cand[:, 1] >= 8)
            cand = cand[ok]
            m, a = score(img01, cand, sign)
            j = int(np.argmax(m))
            if m[j] <= bm + 1e-4:
                break
            best, bm, ba = cand[j], float(m[j]), float(a[j])
    return best, bm, ba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=6)
    a = ap.parse_args()
    dev = "cuda"
    score = Scorer(a.method, dev)
    boxes = coarse_boxes(2.0 if a.method == "trustmark" else 16.0)
    ds = LocalisationDataset(a.data, a.split, methods=[a.method])
    hold = holdout_names(load_index("train", a.data))
    print(f"[{a.method}/{a.split}] images {len(ds)}  coarse boxes {len(boxes)}", flush=True)
    res = {k: [] for k in ("names", "box", "margin", "acc", "full_margin", "full_acc")}
    rows, t0 = [], time.time()
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    for i in range(n):
        it = ds[i]
        img01 = torch.from_numpy(it["image"]).to(dev).permute(2, 0, 1)[None].float() / 255
        sign = torch.from_numpy(2 * it["message"].astype(np.float32) - 1).to(dev)[None]
        fm, fa = score(img01, np.array([[0, 0, 512, 512]], np.float32), sign)
        m, _ = score(img01, boxes, sign)
        top = np.argsort(-m)[:a.topk]
        cands = [refine(score, img01, sign, boxes[j]) for j in top]
        box, bm, ba = max(cands, key=lambda c: c[1])
        for k, v in zip(res, (it["image_name"], box, bm, ba, float(fm[0]), float(fa[0]))):
            res[k].append(v)
        if a.split == "train":
            mask = it["mask"] > 0
            ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            kind = "large" if it["full_watermark"] else ("rect" if mask.sum() / ((y1 - y0) * (x1 - x0)) > 0.98 else "object")
            pb = np.zeros_like(mask); b = np.round(box).astype(int); pb[b[0]:b[2], b[1]:b[3]] = True
            rows.append((kind, it["image_name"] in hold, (pb & mask).sum() / (pb | mask).sum(), bm, ba, float(fm[0]), float(fa[0]), min(y1 - y0, x1 - x0)))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}  {(time.time() - t0) / (i + 1):.1f} s/image", flush=True)
    np.savez(a.out, **{k: np.array(v) for k, v in res.items()})
    if rows:
        for kind in ("rect", "object", "large"):
            r = [x for x in rows if x[0] == kind]
            if not r:
                continue
            iou = np.array([x[2] for x in r]); bm = np.array([x[3] for x in r]); ba = np.array([x[4] for x in r]); fm = np.array([x[5] for x in r])
            print(f"{kind:6s} n={len(r):3d}  box IoU vs mask: mean {iou.mean():.3f}, share >0.5: {(iou > 0.5).mean():.2f}, share >0.8: {(iou > 0.8).mean():.2f} | "
                  f"box margin median {np.median(bm):.2f} acc median {np.median(ba):.3f} | whole-image margin median {np.median(fm):.2f}")
        r = [x for x in rows if x[0] == "rect"]
        for lo, hi in ((0, 24), (24, 48), (48, 96), (96, 600)):
            s = [x for x in r if lo <= x[7] < hi]
            if s:
                print(f"   rect short side [{lo},{hi}): n={len(s)} box IoU mean {np.mean([x[2] for x in s]):.3f}  box acc mean {np.mean([x[4] for x in s]):.3f}")
    print("wrote", a.out, f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
