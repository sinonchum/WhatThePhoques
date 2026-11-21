"""Search rectangles and strips. Keep the one whose crop best recovers the message.

Partial watermarks are often a rectangle or a thin strip. The full image does
not decode; the matching crop does, because the decoder resizes its input.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

KIT = Path("/p/scratch/training2666/li103/task1/kit")
sys.path.insert(0, str(KIT))

from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import downsample_mask, miou_image, score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")


def pick_device() -> str:
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


def candidates() -> list[tuple[int, int, int, int]]:
    boxes = []
    # squares and near-squares
    for size in (64, 96, 128, 160, 224, 288, 352, 416, 480):
        stride = max(size // 3, 32)
        ys = list(range(0, 512 - size + 1, stride))
        if ys[-1] != 512 - size:
            ys.append(512 - size)
        for y in ys:
            for x in ys:
                boxes.append((y, x, y + size, x + size))
    # horizontal and vertical strips, aspect up to 16:1
    for thickness in (16, 32, 48, 64, 96):
        stride = max(thickness, 16)
        for y in range(0, 512 - thickness + 1, stride):
            boxes.append((y, 0, y + thickness, 512))
        for x in range(0, 512 - thickness + 1, stride):
            boxes.append((0, x, 512, x + thickness))
    # a few wide rectangles that share a corner (outpainting)
    for h in (160, 256, 320, 384):
        for w in (160, 256, 320, 384, 480):
            if max(h, w) / min(h, w) > 3:
                continue
            for y, x in ((0, 0), (0, 512 - w), (512 - h, 0), (512 - h, 512 - w), ((512 - h) // 2, (512 - w) // 2)):
                boxes.append((y, x, y + h, x + w))
    # unique
    return list(dict.fromkeys(boxes))


@torch.no_grad()
def margins(wm, image, message, boxes, device, res, bs=24):
    sign = torch.tensor(np.asarray(message), device=device, dtype=torch.float32) * 2 - 1
    msg = np.asarray(message).astype(int)
    out_m = np.empty(len(boxes), np.float32)
    out_a = np.empty(len(boxes), np.float32)
    for s in range(0, len(boxes), bs):
        chunk = boxes[s:s + bs]
        crops = []
        for y0, x0, y1, x1 in chunk:
            t = torch.from_numpy(np.ascontiguousarray(image[y0:y1, x0:x1])).permute(2, 0, 1).contiguous().float().div_(255)
            crops.append(t)
        x = torch.stack([
            torch.nn.functional.interpolate(c[None], size=(res, res), mode="bilinear", align_corners=False, antialias=False)[0]
            for c in crops
        ]).to(device)
        if wm.name == "pixelseal":
            logits = wm.det(x)[:, 1:].float()
        else:
            logits = wm.net(x * 2 - 1).float()
        out_m[s:s + len(chunk)] = (logits * sign).mean(1).cpu().numpy()
        bits = (logits > 0).int().cpu().numpy()
        out_a[s:s + len(chunk)] = (bits == msg).mean(1)
    return out_m, out_a


def box_to_mask(box) -> np.ndarray:
    y0, x0, y1, x1 = box
    m = np.zeros((512, 512), np.uint8)
    m[y0:y1, x0:x1] = 1
    return downsample_mask(m)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "pixelseal"
    device = pick_device()
    res = 256 if name == "pixelseal" else 245
    print("device", device, name, "res", res, flush=True)
    items = [it for it in LocalisationDataset(DATA, "train") if it["method"] == name]
    boxes = candidates()
    print("candidates", len(boxes), flush=True)
    wm = load_watermark(name, models_dir=MODELS, device=device)
    records = []
    for i, it in enumerate(items):
        m, a = margins(wm, it["image"], it["message"], boxes, device, res)
        records.append((m, a, it["mask128"], bool(it["full_watermark"]), float(it["mask"].mean())))
        if i % 20 == 0:
            j = int(np.argmax(a))
            print(f"  {i} area={it['mask'].mean():.3f} full={it['full_watermark']} best_acc={a[j]:.2f} best_m={m[j]:.2f} full_acc={a[boxes.index((0,0,512,512))] if (0,0,512,512) in boxes else float('nan'):.2f}", flush=True)
    # full image may not be in the list; compute from the largest box if present
    gts = [r[2] for r in records]
    full = np.array([r[3] for r in records])

    def eval_rule(min_acc, z):
        preds = []
        for m, a, gt, is_full, area in records:
            j = int(np.argmax(m))
            # require the winner to beat the pack, otherwise the image is empty or full
            med = np.median(m)
            spread = np.std(m) + 1e-6
            if a[j] >= min_acc and (m[j] - med) > z * spread:
                preds.append(box_to_mask(boxes[j]))
            else:
                preds.append(np.zeros((128, 128), np.uint8))
        return preds

    print("rule sweep on all images (zeros when no confident rectangle)", flush=True)
    best = -1
    for min_acc in (0.62, 0.7, 0.78, 0.85):
        for z in (0.5, 1.0, 1.5, 2.0):
            preds = eval_rule(min_acc, z)
            s = score(preds, gts)
            sp = score([preds[i] for i in range(len(preds)) if not full[i]], [gts[i] for i in range(len(preds)) if not full[i]])
            sf = score([preds[i] for i in range(len(preds)) if full[i]], [gts[i] for i in range(len(preds)) if full[i]])
            print(f"  min_acc={min_acc:.2f} z={z:.1f} all={s:.3f} part={sp:.3f} full={sf:.3f}", flush=True)
            best = max(best, s)
    # oracle: for each image take the candidate with the best mIoU (ceiling of this family)
    oracle = []
    for m, a, gt, is_full, area in records:
        b = max(miou_image(np.zeros((128, 128), np.uint8), gt), miou_image(np.ones((128, 128), np.uint8), gt))
        # only score top 8 by margin to keep it cheap, plus a random... actually score top 12 by margin
        top = np.argsort(m)[-12:]
        for j in top:
            b = max(b, miou_image(box_to_mask(boxes[j]), gt))
        oracle.append(b)
    print(f"top12-margin oracle {np.mean(oracle):.3f} part {np.mean([oracle[i] for i in range(len(oracle)) if not full[i]]):.3f} full {np.mean([oracle[i] for i in range(len(oracle)) if full[i]]):.3f}", flush=True)
    print("best rule", best, flush=True)


if __name__ == "__main__":
    main()
