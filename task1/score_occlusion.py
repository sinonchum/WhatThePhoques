"""Score an 8x8 occlusion-drop mask on every train image of one method.

Cells whose removal hurts the known message are called watermarked.
Two scalars (when to give up, which drop quantile to keep) are fit on train.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

KIT = Path("/p/scratch/training2666/li103/task1/kit")
sys.path.insert(0, str(KIT))

from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import bit_accuracy, downsample_mask, score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")
CACHE = Path("/p/scratch/training2666/li103/task1/cache")
GRID = 8
CELL = 512 // GRID


def pick_device() -> str:
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


@torch.no_grad()
def batch_stats(wm, images, message, device, bs=16):
    sign = torch.tensor(np.asarray(message), device=device, dtype=torch.float32) * 2 - 1
    margins = []
    accs = []
    msg = np.asarray(message).astype(int).reshape(-1)
    for s in range(0, len(images), bs):
        chunk = images[s:s + bs]
        x = torch.stack([
            torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).contiguous().float().div_(255)
            for im in chunk
        ]).to(device)
        logits = wm.logits(x).float()
        margins.append((logits * sign).mean(1).cpu().numpy())
        bits = (logits > 0).int().cpu().numpy()
        accs.append((bits == msg).mean(1))
    return np.concatenate(margins), np.concatenate(accs)


def drops_for(wm, image, message, device):
    dead = cv2.resize(cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA), (512, 512), interpolation=cv2.INTER_LINEAR)
    variants = [image]
    for r in range(GRID):
        for c in range(GRID):
            patched = image.copy()
            y0, x0 = r * CELL, c * CELL
            patched[y0:y0 + CELL, x0:x0 + CELL] = dead[y0:y0 + CELL, x0:x0 + CELL]
            variants.append(patched)
    margins, accs = batch_stats(wm, variants, message, device)
    return margins[0], accs[0], (margins[0] - margins[1:]).reshape(GRID, GRID).astype(np.float32)


def predict(drop, margin, acc, q, min_margin, min_acc):
    if margin < min_margin and acc < min_acc:
        return np.zeros((128, 128), np.uint8)
    thr = np.quantile(drop, q)
    heat = cv2.resize(drop, (512, 512), interpolation=cv2.INTER_LINEAR)
    # also spread the decision at cell resolution with nearest, then let official downsample vote
    block = (drop >= thr).astype(np.float32)
    block = cv2.resize(block, (512, 512), interpolation=cv2.INTER_NEAREST)
    return downsample_mask(block)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "pixelseal"
    device = pick_device()
    print("device", device, name, flush=True)
    items = [it for it in LocalisationDataset(DATA, "train") if it["method"] == name]
    cache = CACHE / f"{name}_drops.npz"
    if cache.exists():
        z = np.load(cache)
        drops, margins, accs = z["drops"], z["margins"], z["accs"]
        print("loaded", cache, flush=True)
    else:
        wm = load_watermark(name, models_dir=MODELS, device=device)
        drops = np.zeros((len(items), GRID, GRID), np.float32)
        margins = np.zeros(len(items), np.float32)
        accs = np.zeros(len(items), np.float32)
        for i, it in enumerate(items):
            margins[i], accs[i], drops[i] = drops_for(wm, it["image"], it["message"], device)
            if i % 20 == 0:
                print(f"  {i}/{len(items)} acc={accs[i]:.2f} margin={margins[i]:.2f}", flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(cache, drops=drops, margins=margins, accs=accs)
        print("saved", cache, flush=True)
    gts = [it["mask128"] for it in items]
    full = np.array([bool(it["full_watermark"]) for it in items])
    best = (-1, None)
    for q in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        for min_acc in (0.6, 0.7, 0.8, 1.1):
            preds = [predict(drops[i], margins[i], accs[i], q, min_margin=0.4, min_acc=min_acc) for i in range(len(items))]
            s = score(preds, gts)
            s_full = score([preds[i] for i in range(len(items)) if full[i]], [gts[i] for i in range(len(items)) if full[i]])
            s_part = score([preds[i] for i in range(len(items)) if not full[i]], [gts[i] for i in range(len(items)) if not full[i]])
            print(f"q={q:.2f} min_acc={min_acc:.2f} all={s:.3f} full={s_full:.3f} part={s_part:.3f}", flush=True)
            if s > best[0]:
                best = (s, (q, min_acc, s_full, s_part))
    # reference: all-1 if acc high else all-0
    for min_acc in (0.6, 0.7, 0.8):
        preds = []
        for i in range(len(items)):
            preds.append(np.ones((128, 128), np.uint8) if accs[i] >= min_acc else np.zeros((128, 128), np.uint8))
        s = score(preds, gts)
        print(f"const min_acc={min_acc:.2f} all={s:.3f}", flush=True)
    print("best", best, flush=True)


if __name__ == "__main__":
    main()
