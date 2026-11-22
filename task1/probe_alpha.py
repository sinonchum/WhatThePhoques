"""Spatial sensitivity of the message margin to local high-frequency destruction.

Robust decoders are flat at zero destruction, so the gradient is taken after a
uniform partial destruction that puts the decoder on the sloping part of its curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

KIT = Path("/p/scratch/training2666/li103/task1/kit")
sys.path.insert(0, str(KIT))

from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import downsample_mask, miou_image, score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")


def pick_device() -> str:
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


def destroyed(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (24, 24), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (512, 512), interpolation=cv2.INTER_LINEAR)


def logits_of(wm, x: torch.Tensor) -> torch.Tensor:
    if wm.name == "pixelseal":
        return wm.det(wm._resize(x))[:, 1:]
    if wm.name == "trustmark":
        return wm.net(wm._prep(x))
    raise RuntimeError(wm.name)


def margin_map(wm, image: np.ndarray, message: np.ndarray, device: str, alpha0: float, grid: int) -> np.ndarray:
    x = to_tensor(image).to(device)
    dead = to_tensor(destroyed(image)).to(device)
    alpha = torch.full((1, 1, grid, grid), alpha0, device=device, requires_grad=True)
    up = F.interpolate(alpha, size=(512, 512), mode="bilinear", align_corners=False)
    mixed = x * (1 - up) + dead * up
    logits = logits_of(wm, mixed)[0]
    sign = torch.tensor(message, device=device, dtype=logits.dtype) * 2 - 1
    (logits * sign).mean().backward()
    heat = alpha.grad[0, 0].detach().float()
    heat = F.interpolate(heat[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
    return heat.cpu().numpy()


def norm01(h: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(h, 0.02), np.quantile(h, 0.98)
    return (h - lo) / (hi - lo + 1e-6)


def eval_heats(heats, gts):
    norms = [norm01(h) for h in heats]
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(-0.2, 1.2, 29):
        preds = [downsample_mask(n >= t) for n in norms]
        s = score(preds, gts)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    oracles = []
    for n, g in zip(norms, gts):
        gt = g
        best = -1.0
        for t in np.linspace(-0.2, 1.2, 29):
            best = max(best, miou_image(downsample_mask(n >= t), gt))
        oracles.append(best)
    return best_t, best_s, float(np.mean(oracles))


def main():
    device = pick_device()
    print("device", device, flush=True)
    train = LocalisationDataset(DATA, "train")
    by = {k: [] for k in ("pixelseal", "trustmark")}
    for it in train:
        if it["method"] in by:
            by[it["method"]].append(it)

    for name in ("pixelseal", "trustmark"):
        print(f"\n===== {name} =====", flush=True)
        wm = load_watermark(name, models_dir=MODELS, device=device)
        items = by[name]
        for alpha0, grid in ((0.0, 32), (0.35, 32), (0.55, 32), (0.35, 16)):
            heats, gts, flags, areas = [], [], [], []
            for it in items:
                heats.append(margin_map(wm, it["image"], it["message"], device, alpha0, grid))
                gts.append(it["mask128"])
                flags.append(bool(it["full_watermark"]))
                areas.append(it["mask"].mean())
            t, s, o = eval_heats(heats, gts)
            print(f"alpha={alpha0} grid={grid} all shared={s:.3f}@{t:.2f} oracle={o:.3f}", flush=True)
            for flag in (False, True):
                idx = [i for i, f in enumerate(flags) if f == flag]
                tt, ss, oo = eval_heats([heats[i] for i in idx], [gts[i] for i in idx])
                print(f"  full={flag} shared={ss:.3f}@{tt:.2f} oracle={oo:.3f} area={np.mean([areas[i] for i in idx]):.3f}", flush=True)
        del wm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
