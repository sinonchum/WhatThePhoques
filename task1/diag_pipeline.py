"""Diagnostics on the real train images: how was the dataset built, and which signals localise?

    python task1/diag_pipeline.py --out runs/diag

Per image (all 510 train images):
  acc_full    bit accuracy when decoding the whole image
  acc_bbox    ... decoding only the bounding box of the true mask (tests "watermark was embedded on that rectangle")
  acc_pad     ... bounding box enlarged by 25 % per side      } how sharply does accuracy fall off when the
  acc_shift   ... bounding box shifted by 25 % of its width   } rectangle is wrong (matters for rectangle search)
  acc_inner   ... central half of the bounding box
  auc_mf      cell-level AUC of a matched filter: re-embed the known message into the image, correlate that
              predicted residual with the image's own high-pass content (local normalised correlation)
  auc_mf_box  same, but the message is re-embedded into the bounding-box crop (AUC inside the box only)
Also times embed() on full images and on small crops."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.utils import bit_accuracy, to_tensor
from wmloc.watermarks import load_watermark
from wmtask.common import DATA_DIR, METHODS, holdout_names, load_index


def acc_of(wm, img, msg) -> float:
    if img.shape[0] < 4 or img.shape[1] < 4:
        return float("nan")
    return bit_accuracy(wm.decode(to_tensor(img))[0].cpu().numpy(), msg)


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = int(3 * sigma + 0.5)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    s = score.ravel(); l = label.ravel().astype(bool)
    n1 = int(l.sum()); n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.empty(len(s)); ranks[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    return float((ranks[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def mf_cells(img: np.ndarray, res: np.ndarray, device, win: int = 16) -> np.ndarray:
    """Local normalised correlation between the image's high-pass content and a predicted residual -> H/4 x W/4."""
    x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    r = torch.from_numpy(res.astype(np.float32)).permute(2, 0, 1)[None].to(device)
    hx = x - blur(x, 2.0); hr = r - blur(r, 2.0)
    h, w = x.shape[-2:]
    pool = lambda t: F.avg_pool2d(t, win, stride=1, padding=win // 2, count_include_pad=False)[..., :h, :w]
    num = pool((hx * hr).sum(1, keepdim=True))
    den = (pool((hx ** 2).sum(1, keepdim=True)) * pool((hr ** 2).sum(1, keepdim=True))).sqrt() + 1e-6
    return F.avg_pool2d(num / den, 4)[0, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--out", default="runs/diag")
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    hold = holdout_names(load_index("train", a.data))
    rows = []
    for method in a.methods:
        wm = load_watermark(method, device=a.device)
        ds = LocalisationDataset(a.data, "train", methods=[method])
        t_full, t_crop = [], []
        for i in range(len(ds) if not a.limit else min(a.limit, len(ds))):
            it = ds[i]; img, mask, msg = it["image"], it["mask"], it["message"]
            H, W = mask.shape
            ys, xs = np.where(mask > 0)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            bh, bw = y1 - y0, x1 - x0
            r = {"image_name": it["image_name"], "method": method, "full": int(bool(it["full_watermark"])),
                 "holdout": int(it["image_name"] in hold), "area": float(mask.mean()),
                 "rectness": float(mask.sum() / (bh * bw)), "bh": int(bh), "bw": int(bw),
                 "y0": int(y0), "x0": int(x0)}
            r["acc_full"] = acc_of(wm, img, msg)
            r["acc_bbox"] = acc_of(wm, img[y0:y1, x0:x1], msg)
            py, px = int(round(0.25 * bh)), int(round(0.25 * bw))
            r["acc_pad"] = acc_of(wm, img[max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)], msg)
            sx = px if x1 + px <= W else -px
            r["acc_shift"] = acc_of(wm, img[y0:y1, x0 + sx:x1 + sx], msg)
            r["acc_inner"] = acc_of(wm, img[y0 + bh // 4:y1 - bh // 4, x0 + bw // 4:x1 - bw // 4], msg)

            t0 = time.time(); marked = wm.embed(img, msg); t_full.append(time.time() - t0)
            res_full = marked.astype(np.float32) - img.astype(np.float32)
            cells = mf_cells(img, res_full, a.device)
            r["auc_mf"] = auc(cells, it["mask128"])
            r["mf_in"] = float(cells[it["mask128"] > 0].mean()) if it["mask128"].any() else float("nan")
            r["mf_out"] = float(cells[it["mask128"] == 0].mean()) if (it["mask128"] == 0).any() else float("nan")
            r["auc_mf_box"] = float("nan")
            if not it["full_watermark"] and bh >= 8 and bw >= 8:
                try:
                    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
                    t0 = time.time(); mc = wm.embed(crop, msg); t_crop.append(time.time() - t0)
                    res = np.zeros_like(res_full); res[y0:y1, x0:x1] = mc.astype(np.float32) - crop.astype(np.float32)
                    cb = mf_cells(img, res, a.device)
                    box = np.zeros((H // 4, W // 4), bool); box[y0 // 4:(y1 + 3) // 4, x0 // 4:(x1 + 3) // 4] = True
                    r["auc_mf_box"] = auc(cb[box], it["mask128"][box])
                    r["mf_box_in"] = float(cb[it["mask128"] > 0].mean()) if it["mask128"].any() else float("nan")
                except Exception as e:  # e.g. TrustMark on an extreme aspect ratio
                    r["embed_error"] = repr(e)[:80]
            rows.append(r)
        print(f"[{method}] embed time: full image {1000 * np.median(t_full[3:]):.0f} ms, "
              f"bbox crop {1000 * np.median(t_crop[3:]) if len(t_crop) > 3 else float('nan'):.0f} ms (median)", flush=True)
        del wm; torch.cuda.empty_cache()

    keys = sorted({k for r in rows for k in r})
    with open(out / "diag.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    def mean(sel, k):
        v = np.array([r.get(k, np.nan) for r in sel], float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    cols = ["acc_full", "acc_bbox", "acc_pad", "acc_shift", "acc_inner", "auc_mf", "mf_in", "mf_out", "auc_mf_box"]
    print(f"\n{'group':38s} {'n':>4s} " + " ".join(f"{c:>10s}" for c in cols))
    for method in a.methods:
        for full, tname in ((1, "large"), (0, "small")):
            for shape in ("rect", "irregular"):
                sel = [r for r in rows if r["method"] == method and r["full"] == full
                       and (r["rectness"] > 0.98) == (shape == "rect")]
                if not sel:
                    continue
                print(f"{method + ' ' + tname + ' ' + shape:38s} {len(sel):4d} " + " ".join(f"{mean(sel, c):10.3f}" for c in cols))
    print("\nwhole-image bit accuracy, large vs small (step-2 separability):")
    for method in a.methods:
        lg = np.array([r["acc_full"] for r in rows if r["method"] == method and r["full"] == 1])
        sm = np.array([r["acc_full"] for r in rows if r["method"] == method and r["full"] == 0])
        print(f"  {method:11s} large: min {lg.min():.3f} p5 {np.percentile(lg, 5):.3f} median {np.median(lg):.3f} | "
              f"small: median {np.median(sm):.3f} p95 {np.percentile(sm, 95):.3f} max {sm.max():.3f}")
    sm = [r for r in rows if r["full"] == 0]
    print("\nsmall-type bounding boxes: "
          f"height p5/50/95 = {np.percentile([r['bh'] for r in sm], [5, 50, 95]).round().tolist()}, "
          f"width p5/50/95 = {np.percentile([r['bw'] for r in sm], [5, 50, 95]).round().tolist()}, "
          f"rect share = {np.mean([r['rectness'] > 0.98 for r in sm]):.2f}")
    print("wrote", out / "diag.csv")


if __name__ == "__main__":
    main()
