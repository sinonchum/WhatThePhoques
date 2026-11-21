"""Synthetic training data for the per-method segmentation nets.

How the released images are built (data.py docstring, task PDF, and runs/diag on the real train images):
  large            whole image watermarked, then a region restored to the clean original  -> a clean hole
  small, rectangle the watermark is embedded on a rectangular crop (decoding that crop recovers the message)
  small, object    an irregular region of a WHOLE-IMAGE watermarked version is pasted onto the clean image
                   (whole-image bit accuracy grows with the object's area; decoding its bounding box does not)
We reproduce all three with the official embedders on cover images, so every pixel label is exact.

Pool files (written by gen_synth.py, read here):
  full.npy        [Nc, K, 512, 512, 3] uint8   whole-image watermarked versions of each cover
  rect_data.npy   flat uint8                    watermarked rectangle crops, concatenated
  rect_index.npy  [M, 6] int64                  cover_local, y0, x0, h, w, offset
  pool.json       {"covers": [global cover indices], ...}"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from wmloc.utils import downsample_mask

S = 512
ASPECT_GROUP = {"pixelseal": ("pixelseal", "stegastamp"), "stegastamp": ("pixelseal", "stegastamp"), "trustmark": ("trustmark",)}
MAX_ASPECT = {"pixelseal": 16.0, "trustmark": 2.0, "stegastamp": 16.0}


def small_type_threshold(meta: list[dict], method: str, exclude=frozenset()) -> tuple[float, float]:
    """Whole-image bit-accuracy threshold separating small from large, fitted on the train rows of `method`.
    Returns (threshold, train accuracy of that rule)."""
    tr = [m for m in meta if m["method"] == method and m["split"] == "train" and m["image_name"] not in exclude]
    acc = np.array([m["acc_full"] for m in tr]); small = np.array([not m["full_watermark"] for m in tr])
    cands = np.unique(acc); cands = np.concatenate([(cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-6]])
    scores = [((acc < t) == small).mean() for t in cands]
    best = int(np.argmax(scores))
    return float(cands[best]), float(scores[best])


def is_small(m: dict, thr: dict) -> bool:
    if m["full_watermark"] is not None:
        return not m["full_watermark"]
    return m["acc_full"] < thr[m["method"]]


def cover_indices(meta: list[dict], method: str, holdout=None) -> list[int]:
    """Covers for the net of `method`: images of the OTHER two methods (they never contain this watermark).
    Whole-image StegaStamp images are skipped: their residual is strong and visible, a poor stand-in for clean.
    With `holdout` given (the strict mode, required from 2026-11-21 18:20 on): only TRAIN images outside the
    held-out set are used - no test image and no held-out image of any method ever becomes a cover."""
    if holdout is None:
        thr = {m: small_type_threshold(meta, m)[0] for m in ("pixelseal", "trustmark", "stegastamp")}
        return [i for i, m in enumerate(meta)
                if m["method"] != method and not (m["method"] == "stegastamp" and not is_small(m, thr))]
    return [i for i, m in enumerate(meta)
            if m["split"] == "train" and m["image_name"] not in holdout and m["method"] != method
            and not (m["method"] == "stegastamp" and m["full_watermark"])]


def load_mask512(data_dir, name: str) -> np.ndarray:
    return np.asarray(Image.open(Path(data_dir) / "train" / "masks" / f"{name}.png")) > 127


def bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def random_blob(h: int, w: int, rng) -> np.ndarray:
    """Irregular shape inside an h x w box: a few random ellipses / polygons / thick strokes."""
    im = Image.new("L", (w, h), 0); d = ImageDraw.Draw(im)
    for _ in range(int(rng.integers(1, 5))):
        kind = rng.integers(3)
        x0, x1 = sorted(rng.uniform(-0.1 * w, 1.1 * w, 2)); y0, y1 = sorted(rng.uniform(-0.1 * h, 1.1 * h, 2))
        if kind == 0:
            d.ellipse([x0, y0, max(x1, x0 + 2), max(y1, y0 + 2)], fill=255)
        elif kind == 1:
            pts = [(float(rng.uniform(0, w)), float(rng.uniform(0, h))) for _ in range(int(rng.integers(3, 8)))]
            d.polygon(pts, fill=255)
        else:
            pts = [(float(rng.uniform(0, w)), float(rng.uniform(0, h))) for _ in range(int(rng.integers(2, 6)))]
            d.line(pts, fill=255, width=int(max(2, rng.uniform(0.05, 0.3) * min(h, w))), joint="curve")
    return np.asarray(im) > 127


def place_region(bank: list[np.ndarray], rng, p_blob: float, lo: float, hi: float) -> np.ndarray:
    """A full-canvas S x S bool region: a real region shape (flipped / rotated / rescaled / moved) or a random
    blob, with area fraction in (lo, hi)."""
    out = np.zeros((S, S), bool)
    for _ in range(20):
        out[:] = False
        if rng.random() < p_blob or not bank:
            h = int(rng.integers(24, 400)); w = int(rng.integers(24, 400))
            crop = random_blob(h, w, rng)
        else:
            crop = bank[int(rng.integers(len(bank)))]
            k = int(rng.integers(8))
            crop = np.rot90(crop, k % 4)
            if k >= 4:
                crop = crop[:, ::-1]
            s = rng.uniform(0.6, 1.3)
            h = int(np.clip(round(crop.shape[0] * s), 8, S)); w = int(np.clip(round(crop.shape[1] * s), 8, S))
            crop = np.asarray(Image.fromarray(np.ascontiguousarray(crop).astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
        y0 = int(rng.integers(0, S - h + 1)); x0 = int(rng.integers(0, S - w + 1))
        out[y0:y0 + h, x0:x0 + w] = crop
        if lo < out.mean() < hi:
            break
    return out


class ShapeBank:
    """Real mask shapes from the DEV split only (held-out masks are never looked at)."""

    def __init__(self, rows: list[dict], holdout: set[str], data_dir, method: str):
        self.holes, self.objects, self.hw = [], [], []
        n_small = n_rect = 0
        for r in rows:
            if r["image_name"] in holdout:
                continue
            m = load_mask512(data_dir, r["image_name"])
            if r["full_watermark"]:
                hole = ~m
                y0, y1, x0, x1 = bbox(hole); self.holes.append(hole[y0:y1, x0:x1].copy())
                continue
            n_small += 1
            y0, y1, x0, x1 = bbox(m)
            if m.sum() / ((y1 - y0) * (x1 - x0)) > 0.98:
                n_rect += 1
                if r["method"] in ASPECT_GROUP[method]:
                    self.hw.append((y1 - y0, x1 - x0))
            else:
                self.objects.append(m[y0:y1, x0:x1].copy())
        self.rect_share = n_rect / max(1, n_small)
        self.max_aspect = MAX_ASPECT[method]

    def rect_size(self, rng) -> tuple[int, int]:
        for _ in range(100):
            h, w = self.hw[int(rng.integers(len(self.hw)))]
            h = int(np.clip(round(h * rng.uniform(0.75, 1.3)), 8, S)); w = int(np.clip(round(w * rng.uniform(0.75, 1.3)), 8, S))
            if max(h, w) / min(h, w) <= self.max_aspect and h * w <= 0.45 * S * S:
                return h, w
        return 96, 96

    def hole(self, rng) -> np.ndarray:
        """True = clean hole inside a fully watermarked image (real hole areas: 0.5 % .. 29 %, at most 49 %)."""
        return place_region(self.holes, rng, 0.25, 0.002, 0.49)

    def obj(self, rng) -> np.ndarray:
        """True = transplanted object region (real areas: 1 % .. 40 %, at most 44 %)."""
        return place_region(self.objects, rng, 0.25, 0.004, 0.45)


class SynthDataset(Dataset):
    """Every item is a fresh composite. Returns (image float [-1, 1] 3 x 512 x 512, mask128 float, is_large)."""

    def __init__(self, pool_dir, covers: np.ndarray, bank: ShapeBank, length: int, real=None, p_real: float = 0.0,
                 p_large: float = 0.5, seed: int = 0, alpha=None):
        pool_dir = Path(pool_dir)
        self.info = json.loads((pool_dir / "pool.json").read_text())
        self.cov_idx = np.array(self.info["covers"])
        self.covers = covers
        self.full = np.load(pool_dir / "full.npy", mmap_mode="r")
        self.rect_data = np.load(pool_dir / "rect_data.npy", mmap_mode="r")
        self.rect_index = np.load(pool_dir / "rect_index.npy")
        self.full_bits = np.load(pool_dir / "full_bits.npy") if (pool_dir / "full_bits.npy").exists() else None
        self.rect_bits = np.load(pool_dir / "rect_bits.npy") if (pool_dir / "rect_bits.npy").exists() else None
        self.nbits = self.full_bits.shape[-1] if self.full_bits is not None else 1
        self.bank, self.length, self.real, self.p_real, self.p_large, self.seed = bank, length, real or [], p_real, p_large, seed
        self.alpha = alpha      # shared multiprocessing.Value: residual amplification (strength curriculum), 1 = as embedded

    def _amp(self, wm, cover):
        a = 1.0 if self.alpha is None else float(self.alpha.value)
        if a == 1.0:
            return wm
        return np.clip(np.round(cover.astype(np.float32) + a * (wm.astype(np.float32) - cover)), 0, 255).astype(np.uint8)

    def __len__(self):
        return self.length

    def compose(self, rng):
        if rng.random() < self.p_large:                                   # whole image watermarked, clean hole
            c = int(rng.integers(len(self.cov_idx))); k = int(rng.integers(self.full.shape[1]))
            hole = self.bank.hole(rng)
            cov = np.asarray(self.covers[self.cov_idx[c]])
            return np.where(hole[..., None], cov, self._amp(np.asarray(self.full[c, k]), cov)), ~hole, 1, self._bits(self.full_bits, (c, k))
        if rng.random() < self.bank.rect_share:                           # watermark embedded on a rectangle crop
            j = int(rng.integers(len(self.rect_index)))
            c, y0, x0, h, w, off = (int(v) for v in self.rect_index[j])
            img = np.array(self.covers[self.cov_idx[c]])
            img[y0:y0 + h, x0:x0 + w] = self._amp(np.asarray(self.rect_data[off:off + h * w * 3]).reshape(h, w, 3), img[y0:y0 + h, x0:x0 + w].copy())
            mask = np.zeros((S, S), bool); mask[y0:y0 + h, x0:x0 + w] = True
            return img, mask, 0, self._bits(self.rect_bits, j)
        c = int(rng.integers(len(self.cov_idx))); k = int(rng.integers(self.full.shape[1]))   # transplanted object
        obj = self.bank.obj(rng)
        cov = np.asarray(self.covers[self.cov_idx[c]])
        return np.where(obj[..., None], self._amp(np.asarray(self.full[c, k]), cov), cov), obj, 0, self._bits(self.full_bits, (c, k))

    def _bits(self, table, idx):
        return np.zeros(self.nbits, np.float32) if table is None else table[idx].astype(np.float32)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(torch.randint(0, 2 ** 31 - 1, (1,)))))
        if self.real and rng.random() < self.p_real:
            img, m128, large, bits = self.real[int(rng.integers(len(self.real)))]
        else:
            img, mask, large, bits = self.compose(rng)
            m128 = downsample_mask(mask)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(127.5).sub_(1.0)
        return x, torch.from_numpy(np.asarray(m128, dtype=np.float32)), large, torch.from_numpy(np.asarray(bits, dtype=np.float32))
