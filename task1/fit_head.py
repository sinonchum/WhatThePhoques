"""Cross-validated conv head on frozen extractor features.

Single-channel maps did not match the mask. A small spatial head can mix
channels. Scores are image-level folds so a memorized image cannot leak into
its own score.
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
from wmloc.utils import miou_image, score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")
CACHE = Path("/p/scratch/training2666/li103/task1/cache")


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free = [(torch.cuda.mem_get_info(i)[0], i) for i in range(torch.cuda.device_count())]
    return f"cuda:{max(free)[1]}"


@torch.no_grad()
def pixelseal_feat(wm, image: np.ndarray, device: str) -> torch.Tensor:
    """Stage-0 is 64x64 on the 256 decoder input. Keep that grid and add stage-1."""
    x = wm.det.preprocess(wm._resize(to_tensor(image).to(device)))
    net = wm.det.convnext
    maps = []
    for i in range(2):
        x = net.downsample_layers[i](x)
        x = net.stages[i](x)
        maps.append(x)
    fine, coarse = maps
    coarse = F.interpolate(coarse, size=fine.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([fine, coarse], dim=1)[0].half().cpu()


@torch.no_grad()
def trustmark_feat(wm, image: np.ndarray, device: str) -> torch.Tensor:
    feats = wm.features(to_tensor(image).to(device))
    fine = feats["layer1"]
    coarse = F.interpolate(feats["layer2"], size=fine.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([fine, coarse], dim=1)[0].half().cpu()


class Head(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(self.net(x), size=(128, 128), mode="bilinear", align_corners=False)


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    den = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (1 - (2 * inter + 1) / (den + 1)).mean()
    return bce + dice


def best_miou(prob: np.ndarray, gt: np.ndarray) -> float:
    best = -1.0
    for t in np.linspace(0.15, 0.85, 15):
        best = max(best, miou_image(prob >= t, gt))
    return best


def run_method(name: str, device: str, folds: int = 5, epochs: int = 40):
    train = [it for it in LocalisationDataset(DATA, "train") if it["method"] == name]
    cache_path = CACHE / f"{name}_feat.pt"
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu", weights_only=True)
        feats = blob["feats"]
        masks = blob["masks"]
        print(f"{name} cache {tuple(feats.shape)}", flush=True)
    else:
        print(f"loading {name}", flush=True)
        wm = load_watermark(name, models_dir=MODELS, device=device)
        extract = pixelseal_feat if name == "pixelseal" else trustmark_feat
        got = []
        masks = []
        for i, it in enumerate(train):
            got.append(extract(wm, it["image"], device))
            masks.append(torch.from_numpy(it["mask128"].astype(np.float32)))
            if i % 20 == 0:
                print(f"  feat {i}/{len(train)} {tuple(got[-1].shape)}", flush=True)
        feats = torch.stack(got)
        masks = torch.stack(masks)
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.save({"feats": feats, "masks": masks}, cache_path)
        del wm
        torch.cuda.empty_cache()
        print(f"saved {cache_path} {tuple(feats.shape)}", flush=True)

    n = len(feats)
    rng = np.random.default_rng(0)
    order = rng.permutation(n)
    fold_scores = []
    for fold in range(folds):
        val_idx = order[fold::folds]
        train_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=False)
        model = Head(feats.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x_tr = feats[train_idx].float()
        y_tr = masks[train_idx]
        x_va = feats[val_idx].float().to(device)
        y_va = masks[val_idx].numpy()
        bs = 16
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(x_tr))
            for s in range(0, len(x_tr), bs):
                idx = perm[s : s + bs]
                xb = x_tr[idx].to(device)
                yb = y_tr[idx].to(device)[:, None]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(x_va))[:, 0].cpu().numpy()
        # one threshold for the fold, chosen on that fold (slightly optimistic)
        best_t, best_s = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 13):
            s = score([p >= t for p in prob], y_va)
            if s > best_s:
                best_t, best_s = float(t), float(s)
        fixed = score([p >= 0.5 for p in prob], y_va)
        oracle = float(np.mean([best_miou(p, g) for p, g in zip(prob, y_va)]))
        fold_scores.append(best_s)
        print(f"  fold {fold} nval={len(val_idx)} shared={best_s:.3f}@{best_t:.2f} t0.5={fixed:.3f} oracle={oracle:.3f}", flush=True)
    print(f"{name} CV {np.mean(fold_scores):.3f}", flush=True)


def main():
    device = pick_device()
    print("device", device, flush=True)
    which = sys.argv[1:] or ["pixelseal", "trustmark"]
    for name in which:
        run_method(name, device)


if __name__ == "__main__":
    main()
