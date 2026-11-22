"""Does destroying one square reveal the clean hole inside a decoded watermark?

On images that still decode, the watermark covers most of the frame and the
clean edit is a hole. A square whose destruction leaves the bit margin intact
is that hole. A square cut out of the watermark drops the margin. This probe
measures that gap on the train set and scores the resulting masks.
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
from wmloc.utils import miou_image, score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")
CACHE = Path("/p/scratch/training2666/li103/task1/cache")


def pick_device() -> str:
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


def squares(side: int, stride: int, limit: int = 512) -> list[tuple[int, int, int, int]]:
    coords = list(range(0, limit - side + 1, stride))
    if not coords or coords[-1] != limit - side:
        coords.append(limit - side)
    return [(y, x, y + side, x + side) for y in coords for x in coords]


@torch.no_grad()
def margins_of(wm, images: list[np.ndarray], message: np.ndarray, device: str, bs: int = 16) -> np.ndarray:
    sign = torch.as_tensor(np.asarray(message), device=device, dtype=torch.float32) * 2 - 1
    out = np.empty(len(images), np.float32)
    for start in range(0, len(images), bs):
        chunk = images[start:start + bs]
        batch = torch.stack([
            torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div_(255)
            for im in chunk
        ]).to(device)
        out[start:start + len(chunk)] = (wm.logits(batch).float() * sign).mean(1).cpu().numpy()
    return out


def paint(image: np.ndarray, dead: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, y1, x1 = box
    patched = image.copy()
    patched[y0:y1, x0:x1] = dead[y0:y1, x0:x1]
    return patched


def mask_of(box: tuple[int, int, int, int] | None) -> np.ndarray:
    pred = np.ones((512, 512), np.float32)
    if box is not None:
        y0, x0, y1, x1 = box
        pred[y0:y1, x0:x1] = 0
    return (pred.reshape(128, 4, 128, 4).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "pixelseal"
    device = pick_device()
    side, stride = 160, 64
    boxes = squares(side, stride)
    print("device", device, name, "boxes", len(boxes), flush=True)
    wm = load_watermark(name, models_dir=MODELS, device=device)
    group = [it for it in LocalisationDataset(DATA, "train") if it["method"] == name]
    rows = []
    for i, it in enumerate(group):
        image = it["image"]
        dead = cv2.resize(cv2.resize(image, (24, 24), interpolation=cv2.INTER_AREA), (512, 512), interpolation=cv2.INTER_LINEAR)
        variants = [image] + [paint(image, dead, b) for b in boxes]
        margins = margins_of(wm, variants, it["message"], device)
        base, drops = float(margins[0]), margins[0] - margins[1:]
        gt = it["mask128"]
        full = np.ones((128, 128), np.uint8)
        oracle = miou_image(full, gt)
        best_box, best_s = None, oracle
        for box in boxes:
            s = miou_image(mask_of(box), gt)
            if s > best_s:
                best_box, best_s = box, s
        chosen = boxes[int(np.argmin(drops))]
        rows.append({
            "base": base,
            "min_drop": float(drops.min()),
            "max_drop": float(drops.max()),
            "spread": float(drops.max() - drops.min()),
            "chosen": miou_image(mask_of(chosen), gt),
            "oracle": best_s,
            "full": oracle,
            "area": float(gt.mean()),
            "full_wm": bool(it["full_watermark"]),
            "acc_proxy": base,
        })
        if i % 20 == 0:
            print(
                f"  {i:03d} area={gt.mean():.3f} full={bool(it['full_watermark'])} "
                f"base={base:.2f} spread={drops.max() - drops.min():.2f} "
                f"chosen={rows[-1]['chosen']:.3f} oracle={best_s:.3f} fullmiou={oracle:.3f}",
                flush=True,
            )
    full = np.array([r["full_wm"] for r in rows])
    for label, sel in (("all", np.ones(len(rows), bool)), ("full", full), ("part", ~full)):
        part = [r for r, k in zip(rows, sel) if k]
        if not part:
            continue
        print(
            f"{label} n={len(part)} "
            f"full={np.mean([r['full'] for r in part]):.3f} "
            f"chosen={np.mean([r['chosen'] for r in part]):.3f} "
            f"oracle={np.mean([r['oracle'] for r in part]):.3f} "
            f"spread={np.mean([r['spread'] for r in part]):.3f}",
            flush=True,
        )
    # gate: only carve when the square drops clearly less than the worst square
    gts = [it["mask128"] for it in group]
    bases = np.array([r["base"] for r in rows])
    print("constant", score([np.ones((128, 128), np.uint8) if b > 1 else np.zeros((128, 128), np.uint8) for b in bases], gts), flush=True)


if __name__ == "__main__":
    main()
