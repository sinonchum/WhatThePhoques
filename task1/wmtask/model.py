"""Segmentation net for watermark localisation: 512 x 512 RGB in, 128 x 128 logits out.

The watermark is a faint high-frequency residual, so the first stage runs at full resolution without pooling
(pretrained ImageNet stems downsample by 4 immediately and would throw the signal away). Deeper stages give the
global context needed to tell "whole image watermarked with a hole" from "clean image with a patch"."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True))


class HighPass(nn.Module):
    """Fixed front end: append truncated high-pass residuals to the image (as steganalysis nets do). The
    watermark is ~1/255 in amplitude; without this the first layers must learn to cancel the image content."""

    def __init__(self, scale: float = 40.0, clip: float = 3.0):
        super().__init__()
        self.scale, self.clip = scale, clip

    def forward(self, x):
        b3 = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)
        b5 = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, 1)
        hp = torch.cat([x - b3, x - b5, b3 - b5], 1) * self.scale
        return torch.cat([x, hp.clamp(-self.clip, self.clip)], 1)


class WMSegNet(nn.Module):
    def __init__(self, base: int = 32, in_ch: int = 3, hp: bool = False):
        super().__init__()
        self.front = HighPass() if hp else nn.Identity()
        in_ch = in_ch * 4 if hp else in_ch
        c = [base, base * 2, base * 4, base * 8, base * 12, base * 16, base * 16]   # 512 256 128 64 32 16 8
        self.stem = block(in_ch, c[0])
        self.down = nn.ModuleList(block(c[i], c[i + 1], 2) for i in range(6))
        self.glob = nn.Sequential(nn.Linear(c[6], c[6]), nn.SiLU(inplace=True))
        self.up = nn.ModuleList(block(c[i + 1] + c[i], c[i]) for i in (5, 4, 3, 2))  # -> 16, 32, 64, 128
        self.head = nn.Conv2d(c[2], 1, 1)
        self.cls = nn.Linear(c[6], 1)                                               # is the image "large"-type?

    def forward(self, x):
        # Front end and full-resolution stem always run in float32: the watermark is about one grey level, and
        # bfloat16 (8-bit mantissa) rounds it away before any filter can isolate it.
        with torch.autocast("cuda", enabled=False):
            feats = [self.stem(self.front(x.float()))]
        for d in self.down:
            feats.append(d(feats[-1]))
        g = self.glob(feats[6].mean((2, 3)))
        y = feats[6] + g[:, :, None, None]
        for u, i in zip(self.up, (5, 4, 3, 2)):
            y = u(torch.cat([F.interpolate(y, size=feats[i].shape[-2:], mode="nearest"), feats[i]], 1))
        return self.head(y)[:, 0], self.cls(g)[:, 0]


def seg_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE plus a soft version of the task metric (per-image mean IoU over both classes)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    i1 = (p * target).sum((1, 2)); u1 = (p + target - p * target).sum((1, 2))
    q, t0 = 1 - p, 1 - target
    i0 = (q * t0).sum((1, 2)); u0 = (q + t0 - q * t0).sum((1, 2))
    miou = 0.5 * ((i1 + 1) / (u1 + 1) + (i0 + 1) / (u0 + 1))
    return bce + (1 - miou).mean()
