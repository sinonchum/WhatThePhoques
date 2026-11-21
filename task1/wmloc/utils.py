"""Helpers: bit accuracy, the official mask downsampling (512 -> 128 grid) and the official per-image mIoU."""
from __future__ import annotations

import numpy as np
import torch

GRID = 128
CELL = 4


def bit_accuracy(decoded, message) -> float:
    d = np.asarray(decoded).astype(int).reshape(-1); m = np.asarray(message).astype(int).reshape(-1)
    return float((d == m).mean())


def downsample_mask(mask) -> np.ndarray:
    """Official rule: image-size mask (1 = watermarked, 512 x 512) -> 128 x 128 grid; a cell is 1 if at least
    50 % of its pixels (4 x 4 at 512) are 1."""
    m = np.asarray(mask, dtype=np.float32)
    assert m.ndim == 2 and m.shape[0] == m.shape[1] and m.shape[0] % GRID == 0, m.shape
    c = m.shape[0] // GRID
    return (m.reshape(GRID, c, GRID, c).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)


def miou_image(pred, gt) -> float:
    """Official per-image score: mean over classes {0, 1} of |P_c & G_c| / |P_c | G_c| (1 if the union is empty)."""
    p = np.asarray(pred).astype(np.uint8); g = np.asarray(gt).astype(np.uint8)
    s = 0.0
    for c in (0, 1):
        union = np.logical_or(p == c, g == c).sum()
        s += 1.0 if union == 0 else np.logical_and(p == c, g == c).sum() / union
    return s / 2


def score(preds, gts) -> float:
    """Official score: mean of the per-image mIoU."""
    return float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)]))


def to_tensor(img_uint8) -> torch.Tensor:
    """H x W x 3 uint8 -> 1 x 3 x H x W float in [0, 1] (input format of the watermark wrappers)."""
    return torch.from_numpy(np.array(img_uint8, dtype=np.uint8)).permute(2, 0, 1)[None].float() / 255
