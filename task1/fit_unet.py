"""Small UNet on pixels. Reports train fit and a held-out fold.

The extractor's pooled features do not localize. This asks whether the
watermark trace in the pixels is learnable from the train masks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KIT = Path("/p/scratch/training2666/li103/task1/kit")
sys.path.insert(0, str(KIT))

from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import miou_image, score  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")

KV = torch.tensor(
    [[-1, 2, -2, 2, -1],
     [2, -6, 8, -6, 2],
     [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]],
    dtype=torch.float32,
) / 12.0


def pick_device() -> str:
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


class UNet(nn.Module):
    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.d1 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.e2 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.e3 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.u2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.c2 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.u1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.c1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        d2 = self.c2(torch.cat([self.u2(e3), e2], 1))
        return self.c1(torch.cat([self.u1(d2), e1], 1))


def prepare(img: torch.Tensor) -> torch.Tensor:
    """img NCHW in [0,1]. Returns RGB plus a fixed high-pass channel."""
    weight = KV.to(img.device).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
    hp = F.conv2d(img, weight, padding=2, groups=3).mean(1, keepdim=True)
    return torch.cat([img, hp], 1)


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # target is 128x128. Match the scoring grid.
    pred = F.avg_pool2d(logits, 4)
    tgt = target
    bce = F.binary_cross_entropy_with_logits(pred, tgt)
    prob = torch.sigmoid(pred)
    inter = (prob * tgt).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
    dice = (1 - (2 * inter + 1) / (den + 1)).mean()
    return bce + dice


@torch.no_grad()
def eval_split(model, images, masks, device) -> tuple[float, float]:
    model.eval()
    probs = []
    for s in range(0, len(images), 8):
        x = prepare(images[s:s + 8].to(device))
        logits = model(x)
        pred = torch.sigmoid(F.avg_pool2d(logits, 4))[:, 0]
        probs.append(pred.cpu())
    prob = torch.cat(probs).numpy()
    gt = masks.numpy()
    fixed = score([p >= 0.5 for p in prob], gt)
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 13):
        s = score([p >= t for p in prob], gt)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return fixed, best_s


def main():
    device = pick_device()
    name = sys.argv[1] if len(sys.argv) > 1 else "pixelseal"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    print("device", device, "method", name, "epochs", epochs, flush=True)
    items = [it for it in LocalisationDataset(DATA, "train") if it["method"] == name]
    images = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).float().div_(255)
    masks = torch.from_numpy(np.stack([it["mask128"] for it in items])).float()[:, None]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(items))
    val_idx = order[:34]
    tr_idx = order[34:]
    model = UNet(4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 8
    for epoch in range(epochs):
        model.train()
        perm = tr_idx[rng.permutation(len(tr_idx))]
        total = 0.0
        nstep = 0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            # flips, applied the same way to image and mask
            xb = images[idx].clone()
            yb = masks[idx].clone()
            if rng.random() < 0.5:
                xb = xb.flip(-1)
                yb = yb.flip(-1)
            if rng.random() < 0.5:
                xb = xb.flip(-2)
                yb = yb.flip(-2)
            xb = prepare(xb.to(device))
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss)
            nstep += 1
        if epoch % 2 == 0 or epoch == epochs - 1:
            tr_fixed, tr_best = eval_split(model, images[tr_idx], masks[tr_idx, 0], device)
            va_fixed, va_best = eval_split(model, images[val_idx], masks[val_idx, 0], device)
            print(
                f"epoch {epoch:02d} loss {total / nstep:.3f} "
                f"train {tr_best:.3f} (t.5 {tr_fixed:.3f}) val {va_best:.3f} (t.5 {va_fixed:.3f})",
                flush=True,
            )
    out = Path("/p/scratch/training2666/li103/task1/cache") / f"unet_{name}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
