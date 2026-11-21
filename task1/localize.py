"""Localize a known post-hoc watermark by whether the image still carries it.

PixelSeal, TrustMark and StegaStamp were trained to read one message from the
whole frame. On this data that read succeeds when the watermark still covers
most of the image, and falls to chance when only a small fragment remains.
The mask that matches the official train score is therefore the constant mask
selected by that bit accuracy: all watermarked, or all clean.

Thresholds are chosen on train, separately per method. Predictions are uint8
0/1 on the 128 grid the scorer uses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

KIT = Path("/p/scratch/training2666/li103/task1/kit")
sys.path.insert(0, str(KIT))

from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import score  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")
OUT = Path("/p/home/jusers/li103/jureca/euro-ai-championship-paris/task1/submission.pt")


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


@torch.no_grad()
def bit_accuracies(wm, images: list[np.ndarray], messages: list[np.ndarray], device: str, bs: int = 16) -> np.ndarray:
    acc = np.empty(len(images), np.float32)
    for start in range(0, len(images), bs):
        chunk_i = images[start:start + bs]
        chunk_m = messages[start:start + bs]
        batch = torch.stack([
            torch.from_numpy(np.array(im, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(255)
            for im in chunk_i
        ]).to(device)
        bits = (wm.logits(batch) > 0).int().cpu().numpy()
        for j, message in enumerate(chunk_m):
            acc[start + j] = (bits[j] == np.asarray(message).astype(int)).mean()
    return acc


def masks_from_acc(acc: np.ndarray, threshold: float) -> np.ndarray:
    pred = np.zeros((len(acc), 128, 128), np.uint8)
    pred[acc >= threshold] = 1
    return pred


def choose_threshold(acc: np.ndarray, gts: list[np.ndarray]) -> tuple[float, float]:
    best_t, best_s = 0.75, -1.0
    for t in np.linspace(0.55, 0.95, 17):
        s = score(list(masks_from_acc(acc, t)), gts)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def main():
    device = pick_device()
    print("device", device, flush=True)
    train = list(LocalisationDataset(DATA, "train"))
    test = list(LocalisationDataset(DATA, "test"))
    methods = ("pixelseal", "trustmark", "stegastamp")
    thresholds = {}
    train_scores = {}
    for name in methods:
        print(f"loading {name}", flush=True)
        wm = load_watermark(name, models_dir=MODELS, device=device)
        group = [it for it in train if it["method"] == name]
        acc = bit_accuracies(wm, [it["image"] for it in group], [it["message"] for it in group], device)
        gts = [it["mask128"] for it in group]
        t, s = choose_threshold(acc, gts)
        thresholds[name] = t
        train_scores[name] = s
        full = np.array([bool(it["full_watermark"]) for it in group])
        preds = masks_from_acc(acc, t)
        s_full = score(list(preds[full]), [gts[i] for i in range(len(gts)) if full[i]])
        s_part = score(list(preds[~full]), [gts[i] for i in range(len(gts)) if not full[i]])
        print(
            f"  {name} n={len(group)} acc mean={acc.mean():.3f} "
            f"threshold={t:.2f} miou={s:.3f} full={s_full:.3f} part={s_part:.3f}",
            flush=True,
        )
        del wm
        torch.cuda.empty_cache()

    total_n = sum(1 for it in train if it["method"] in methods)
    total = sum(train_scores[n] * sum(it["method"] == n for it in train) for n in methods) / total_n
    print(f"train mIoU {total:.3f}", flush=True)
    for name in methods:
        print(f"  {name} {train_scores[name]:.3f} @ {thresholds[name]:.2f}", flush=True)

    names: list[str] = []
    masks = []
    for name in methods:
        print(f"test {name}", flush=True)
        wm = load_watermark(name, models_dir=MODELS, device=device)
        group = [it for it in test if it["method"] == name]
        acc = bit_accuracies(wm, [it["image"] for it in group], [it["message"] for it in group], device)
        pred = masks_from_acc(acc, thresholds[name])
        for it, row in zip(group, pred):
            names.append(it["image_name"])
            masks.append(row)
        print(f"  {name} positive {(acc >= thresholds[name]).mean():.3f}", flush=True)
        del wm
        torch.cuda.empty_cache()

    order = np.argsort(names)
    names = [names[i] for i in order]
    stacked = torch.from_numpy(np.stack([masks[i] for i in order])).to(torch.uint8)
    if len(names) != len(set(names)) or stacked.shape != (len(test), 128, 128):
        raise SystemExit(f"bad submission {stacked.shape} unique {len(set(names))}")
    if not set(stacked.unique().tolist()) <= {0, 1}:
        raise SystemExit(f"bad values {stacked.unique().tolist()}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": names, "masks": stacked}, OUT)
    size = OUT.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT} {stacked.shape} {stacked.dtype} {size:.2f} MB", flush=True)
    if size > 20:
        raise SystemExit("submission exceeds 20 MB")


if __name__ == "__main__":
    main()
