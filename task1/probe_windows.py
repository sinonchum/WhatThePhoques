"""Windowed bit-accuracy maps.

The global decoder resizes whatever it is given. A crop that matches the
watermarked rectangle decodes; a crop of clean content does not. This probe
measures that on the train set and turns the window scores into a mask.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KIT = Path("/p/scratch/training2666/li103/task1/kit")
sys.path.insert(0, str(KIT))

from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import bit_accuracy, miou_image, score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")
GRID = 128


def pick_device() -> str:
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


def boxes_for(size: int, stride: int, limit: int = 512) -> list[tuple[int, int, int]]:
    coords = list(range(0, limit - size + 1, stride))
    last = limit - size
    if coords[-1] != last:
        coords.append(last)
    return [(y, x, size) for y in coords for x in coords]


def gt_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


@torch.no_grad()
def batch_acc(wm, image: np.ndarray, message: np.ndarray, boxes: list[tuple[int, int, int, int]], device: str, res: int, bs: int = 24) -> np.ndarray:
    """boxes are y0,x0,y1,x1. Returns bit accuracy per box."""
    if not boxes:
        return np.zeros((0,), np.float32)
    msg = np.asarray(message).astype(int).reshape(-1)
    acc = np.empty(len(boxes), np.float32)
    for start in range(0, len(boxes), bs):
        chunk = boxes[start : start + bs]
        crops = []
        for y0, x0, y1, x1 in chunk:
            t = torch.from_numpy(np.ascontiguousarray(image[y0:y1, x0:x1])).permute(2, 0, 1).float().div_(255)
            crops.append(t)
        x = torch.stack([
            F.interpolate(c[None], size=(res, res), mode="bilinear", align_corners=False, antialias=True)[0]
            for c in crops
        ]).to(device)
        if wm.name == "pixelseal":
            logits = wm.det(x)[:, 1:]
        elif wm.name == "trustmark":
            logits = wm.net(x * 2 - 1)
        else:
            raise RuntimeError(wm.name)
        bits = (logits > 0).int().cpu().numpy()
        acc[start : start + len(chunk)] = (bits == msg).mean(axis=1)
    return acc


def heat_from(accs: np.ndarray, boxes: list[tuple[int, int, int, int]], reduce: str) -> np.ndarray:
    canv = np.zeros((GRID, GRID), np.float32)
    cnt = np.zeros((GRID, GRID), np.float32)
    for acc, (y0, x0, y1, x1) in zip(accs, boxes):
        gy0, gx0 = y0 * GRID // 512, x0 * GRID // 512
        gy1, gx1 = max(gy0 + 1, y1 * GRID // 512), max(gx0 + 1, x1 * GRID // 512)
        if reduce == "max":
            canv[gy0:gy1, gx0:gx1] = np.maximum(canv[gy0:gy1, gx0:gx1], acc)
        else:
            canv[gy0:gy1, gx0:gx1] += acc
            cnt[gy0:gy1, gx0:gx1] += 1
    if reduce == "mean":
        canv /= np.maximum(cnt, 1)
    return canv


def best_pred(heat: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.45, 0.95, 26):
        s = miou_image(heat >= t, gt)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def rect_from_best(accs, boxes, thresh: float) -> np.ndarray:
    pred = np.zeros((GRID, GRID), np.uint8)
    ok = [(a, b) for a, b in zip(accs, boxes) if a >= thresh]
    if not ok:
        return pred
    # highest accuracy, then smallest window
    acc, (y0, x0, y1, x1) = max(ok, key=lambda z: (z[0], -(z[1][2] - z[1][0]) * (z[1][3] - z[1][1])))
    gy0, gx0 = y0 * GRID // 512, x0 * GRID // 512
    gy1, gx1 = max(gy0 + 1, y1 * GRID // 512), max(gx0 + 1, x1 * GRID // 512)
    pred[gy0:gy1, gx0:gx1] = 1
    return pred


def main():
    device = pick_device()
    print("device", device, flush=True)
    train = LocalisationDataset(DATA, "train")
    by = {k: [] for k in ("pixelseal", "trustmark")}
    for it in train:
        if it["method"] in by:
            by[it["method"]].append(it)

    sizes = [96, 160, 256, 384]
    for name, res in (("pixelseal", 256), ("trustmark", 245)):
        print(f"\n===== {name} =====", flush=True)
        wm = load_watermark(name, models_dir=MODELS, device=device)
        # reference: full image vs tight gt box, all images
        rows = []
        for it in by[name]:
            full = (0, 0, 512, 512)
            box = gt_box(it["mask"])
            boxes = [full] if box is None else [full, box]
            accs = batch_acc(wm, it["image"], it["message"], boxes, device, res, bs=8)
            rows.append((it, float(accs[0]), None if box is None else float(accs[1]), it["mask"].mean(), bool(it["full_watermark"])))
        print(
            f"full acc mean {np.mean([r[1] for r in rows]):.3f}  "
            f"gt-box acc mean {np.mean([r[2] for r in rows if r[2] is not None]):.3f}",
            flush=True,
        )
        for flag in (False, True):
            sub = [r for r in rows if r[4] == flag]
            print(
                f"  full_wm={flag} n={len(sub)} area={np.mean([r[3] for r in sub]):.3f} "
                f"full_acc={np.mean([r[1] for r in sub]):.3f} gtbox_acc={np.mean([r[2] for r in sub]):.3f} "
                f"gtbox>0.75={np.mean([r[2] > 0.75 for r in sub]):.2f} full>0.75={np.mean([r[1] > 0.75 for r in sub]):.2f}",
                flush=True,
            )

        # window maps on a stratified subset, then a simple global threshold is not comparable
        # across images. Report oracle mIoU of the heat and of the single best window.
        items = by[name]
        items_sorted = sorted(items, key=lambda it: it["mask"].mean())
        probe = [items_sorted[i] for i in np.linspace(0, len(items_sorted) - 1, 16).astype(int)]
        all_boxes = []
        for size in sizes:
            all_boxes.extend((y, x, y + size, x + size) for y, x, size in boxes_for(size, max(size // 2, 32)))
        print(f"windows/image {len(all_boxes)}", flush=True)
        heats_mean, heats_max, gts, best_rects = [], [], [], []
        for it in probe:
            accs = batch_acc(wm, it["image"], it["message"], all_boxes, device, res, bs=16)
            hm = heat_from(accs, all_boxes, "mean")
            hx = heat_from(accs, all_boxes, "max")
            heats_mean.append(hm)
            heats_max.append(hx)
            gts.append(it["mask128"])
            pred = rect_from_best(accs, all_boxes, 0.72)
            best_rects.append(pred)
            print(
                f"  {it['image_name']} area={it['mask'].mean():.3f} full={it['full_watermark']} "
                f"acc[{accs.min():.2f},{accs.mean():.2f},{accs.max():.2f}] "
                f"mean_oracle={best_pred(hm, it['mask128'])[1]:.3f} "
                f"max_oracle={best_pred(hx, it['mask128'])[1]:.3f} "
                f"bestrect={miou_image(pred, it['mask128']):.3f}",
                flush=True,
            )
        print(
            f"probe mean-heat oracle {np.mean([best_pred(h, g)[1] for h, g in zip(heats_mean, gts)]):.3f} "
            f"max-heat oracle {np.mean([best_pred(h, g)[1] for h, g in zip(heats_max, gts)]):.3f} "
            f"bestrect@0.72 {score(best_rects, gts):.3f}",
            flush=True,
        )
        del wm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
