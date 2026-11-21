"""Exact block-occlusion margins. One cell is replaced by a low-pass copy.

The drop in message margin is the cell's contribution. This is the finite
difference the input-gradient approximation failed to match.
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
from wmloc.utils import bit_accuracy, miou_image, score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")


def pick_device() -> str:
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


def lowpass(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (512, 512), interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def margins(wm, images: list[np.ndarray], message: np.ndarray, device: str, bs: int = 16) -> np.ndarray:
    sign = torch.tensor(np.asarray(message), device=device, dtype=torch.float32) * 2 - 1
    out = np.empty(len(images), np.float32)
    for s in range(0, len(images), bs):
        chunk = images[s:s + bs]
        x = torch.stack([
            torch.from_numpy(im.copy()).permute(2, 0, 1).float().div_(255) for im in chunk
        ]).to(device)
        logits = wm.logits(x).float()
        out[s:s + len(chunk)] = (logits * sign).mean(dim=1).cpu().numpy()
    return out


def occlusion_map(wm, image: np.ndarray, message: np.ndarray, device: str, grid: int) -> tuple[np.ndarray, float]:
    dead = lowpass(image)
    cell = 512 // grid
    variants = [image]
    for r in range(grid):
        for c in range(grid):
            patched = image.copy()
            y0, x0 = r * cell, c * cell
            patched[y0:y0 + cell, x0:x0 + cell] = dead[y0:y0 + cell, x0:x0 + cell]
            variants.append(patched)
    m = margins(wm, variants, message, device)
    base = float(m[0])
    drops = (base - m[1:]).reshape(grid, grid)
    return drops, base


def upsample(drops: np.ndarray) -> np.ndarray:
    return cv2.resize(drops.astype(np.float32), (512, 512), interpolation=cv2.INTER_NEAREST)


def eval_maps(maps512: list[np.ndarray], gts: list[np.ndarray]) -> tuple[float, float]:
    norms = []
    for m in maps512:
        lo, hi = np.quantile(m, 0.05), np.quantile(m, 0.95)
        norms.append((m - lo) / (hi - lo + 1e-6))
    best_t, best_s = 0.0, -1.0
    gt128 = gts
    for t in np.linspace(-0.5, 1.5, 41):
        preds = [(n >= t).astype(np.uint8) for n in norms]
        # norms are 512; official downsample
        from wmloc.utils import downsample_mask
        preds = [downsample_mask(p) for p in preds]
        s = score(preds, gt128)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    oracles = []
    from wmloc.utils import downsample_mask
    for n, g in zip(norms, gt128):
        best = max(miou_image(downsample_mask(n >= t), g) for t in np.linspace(-0.5, 1.5, 41))
        oracles.append(best)
    return best_s, float(np.mean(oracles))


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
        items = sorted(by[name], key=lambda it: it["mask"].mean())
        probe = [items[i] for i in np.linspace(0, len(items) - 1, 18).astype(int)]
        for grid in (4, 8):
            maps, gts, areas, bases, accs = [], [], [], [], []
            for it in probe:
                drops, base = occlusion_map(wm, it["image"], it["message"], device, grid)
                maps.append(upsample(drops))
                gts.append(it["mask128"])
                areas.append(it["mask"].mean())
                bases.append(base)
                acc = bit_accuracy(wm.decode(to_tensor(it["image"]).to(device)).cpu().numpy()[0], it["message"])
                accs.append(acc)
                print(
                    f"  g{grid} {it['image_name']} area={it['mask'].mean():.3f} acc={acc:.3f} "
                    f"margin={base:.2f} drop[{drops.min():.2f},{drops.mean():.2f},{drops.max():.2f}]",
                    flush=True,
                )
            shared, oracle = eval_maps(maps, gts)
            print(f"grid {grid} shared={shared:.3f} oracle={oracle:.3f}", flush=True)
            # split by whether the full image actually decodes
            for label, sel in (
                ("decodes", [i for i, a in enumerate(accs) if a >= 0.75]),
                ("chance", [i for i, a in enumerate(accs) if a < 0.75]),
            ):
                if len(sel) < 2:
                    continue
                sh, oc = eval_maps([maps[i] for i in sel], [gts[i] for i in sel])
                print(f"  {label} n={len(sel)} shared={sh:.3f} oracle={oc:.3f}", flush=True)
        del wm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
