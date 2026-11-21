"""Test remover residuals, re-embed correlation, and occlusion margins."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

KIT = Path("/p/scratch/training2666/li103/task1/kit")
sys.path.insert(0, str(KIT))

from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import bit_accuracy, downsample_mask, miou_image, score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")


def pick_device() -> str:
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


def corr(heat: np.ndarray, gt128: np.ndarray) -> float:
    h = cv2.resize(heat.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA).reshape(-1)
    g = gt128.astype(np.float32).reshape(-1)
    h = h - h.mean()
    g = g - g.mean()
    d = np.linalg.norm(h) * np.linalg.norm(g)
    return float((h @ g) / d) if d else 0.0


def oracle(heat: np.ndarray, gt128: np.ndarray) -> float:
    h = cv2.resize(heat.astype(np.float32), (128, 128), interpolation=cv2.INTER_LINEAR)
    best = -1.0
    for t in np.quantile(h, np.linspace(0.02, 0.98, 33)):
        best = max(best, miou_image(h >= t, gt128))
    return best


def shared(heats: list[np.ndarray], gts: list[np.ndarray]) -> tuple[float, float]:
    maps = [cv2.resize(h.astype(np.float32), (128, 128), interpolation=cv2.INTER_LINEAR) for h in heats]
    # per-image rank normalization so one threshold transfers
    norms = []
    for m in maps:
        lo, hi = np.quantile(m, 0.02), np.quantile(m, 0.98)
        norms.append((m - lo) / (hi - lo + 1e-6))
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        s = score([(n >= t).astype(np.uint8) for n in norms], gts)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def smooth_abs(residual: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    mag = np.abs(residual.astype(np.float32)).mean(2)
    return cv2.GaussianBlur(mag, (0, 0), sigma)


def main():
    device = pick_device()
    print("device", device, flush=True)
    train = LocalisationDataset(DATA, "train")
    by = {k: [] for k in ("pixelseal", "trustmark")}
    for it in train:
        if it["method"] in by:
            by[it["method"]].append(it)

    print("loading trustmark", flush=True)
    tm = load_watermark("trustmark", models_dir=MODELS, device=device)
    heats, gts = [], []
    print("trustmark removal residual", flush=True)
    for i, it in enumerate(by["trustmark"]):
        removed = np.asarray(tm.tm.remove_watermark(Image.fromarray(it["image"])))
        heat = smooth_abs(it["image"].astype(np.float32) - removed.astype(np.float32), 6)
        heats.append(heat)
        gts.append(it["mask128"])
        if i < 8 or i % 40 == 0:
            print(
                f"  {it['image_name']} area={it['mask'].mean():.3f} full={it['full_watermark']} "
                f"corr={corr(heat, it['mask128']):.3f} oracle={oracle(heat, it['mask128']):.3f}",
                flush=True,
            )
    t, s = shared(heats, gts)
    oracles = [oracle(h, g) for h, g in zip(heats, gts)]
    print(f"trustmark residual n={len(gts)} oracle_mean={np.mean(oracles):.3f} shared={s:.3f} @{t:.2f}", flush=True)
    for flag in (False, True):
        idx = [i for i, it in enumerate(by["trustmark"]) if bool(it["full_watermark"]) == flag]
        print(f"  full={flag} oracle={np.mean([oracles[i] for i in idx]):.3f} shared={score([(cv2.resize(heats[i].astype(np.float32),(128,128)) ) for i in idx], [gts[i] for i in idx]) and 0}", flush=True)
    # proper split shared is global; report oracle split only
    for flag in (False, True):
        idx = [i for i, it in enumerate(by["trustmark"]) if bool(it["full_watermark"]) == flag]
        print(f"  full={flag} oracle_mean={np.mean([oracles[i] for i in idx]):.3f}", flush=True)

    print("loading pixelseal", flush=True)
    ps = load_watermark("pixelseal", models_dir=MODELS, device=device)

    def margin(img, message) -> float:
        logits = ps.logits(to_tensor(img).to(device))[0].float()
        sign = torch.tensor(message, device=device, dtype=logits.dtype) * 2 - 1
        return float((logits * sign).mean())

    print("pixelseal re-embed correlation on 12 images", flush=True)
    items = sorted(by["pixelseal"], key=lambda it: it["mask"].mean())
    probe = [items[i] for i in np.linspace(0, len(items) - 1, 12).astype(int)]
    for it in probe:
        img = it["image"]
        blur = cv2.GaussianBlur(img, (0, 0), 3)
        embedded = ps.embed(blur, it["message"])
        pred_res = embedded.astype(np.float32) - blur.astype(np.float32)
        obs = img.astype(np.float32) - blur.astype(np.float32)
        # local agreement of observed high-pass with the predicted residual
        prod = (obs * pred_res).mean(2)
        prod = cv2.GaussianBlur(prod, (0, 0), 8)
        print(
            f"  {it['image_name']} area={it['mask'].mean():.3f} full={it['full_watermark']} "
            f"corr={corr(prod, it['mask128']):.3f} oracle={oracle(prod, it['mask128']):.3f}",
            flush=True,
        )

    print("pixelseal occlusion margin", flush=True)
    large = [it for it in by["pixelseal"] if it["mask"].mean() > 0.7][:6]
    for grid in (4, 8):
        cell = 512 // grid
        print(f"  grid {grid}", flush=True)
        for it in large:
            img = it["image"]
            base = margin(img, it["message"])
            dead = cv2.resize(cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA), (512, 512), interpolation=cv2.INTER_LINEAR)
            drops = np.zeros((grid, grid), np.float32)
            for r in range(grid):
                for c in range(grid):
                    y0, x0 = r * cell, c * cell
                    patched = img.copy()
                    patched[y0:y0 + cell, x0:x0 + cell] = dead[y0:y0 + cell, x0:x0 + cell]
                    drops[r, c] = base - margin(patched, it["message"])
            gt = cv2.resize(it["mask"].astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
            cval = corr(drops, cv2.resize(it["mask128"].astype(np.float32), (grid, grid), interpolation=cv2.INTER_NEAREST))
            # use mask128 resized properly
            cval = corr(cv2.resize(drops, (512, 512), interpolation=cv2.INTER_NEAREST), it["mask128"])
            print(
                f"    {it['image_name']} area={it['mask'].mean():.3f} base_m={base:.2f} "
                f"drop[{drops.min():.2f},{drops.mean():.2f},{drops.max():.2f}] corr={cval:.3f} "
                f"oracle={oracle(cv2.resize(drops,(512,512),interpolation=cv2.INTER_NEAREST), it['mask128']):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
