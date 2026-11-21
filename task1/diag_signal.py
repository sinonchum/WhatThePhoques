"""Measure which signals actually track the watermark mask.

Feature-map attribution was spatially flat. This script checks occlusion
sensitivity, early-layer responses, and TrustMark's remover residual.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

KIT = Path("/p/scratch/training2666/li103/task1/kit")
sys.path.insert(0, str(KIT))

from wmloc.data import LocalisationDataset  # noqa: E402
from wmloc.utils import bit_accuracy, downsample_mask, miou_image, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")


def pick_device() -> str:
    free = []
    for i in range(torch.cuda.device_count()):
        free_b, _ = torch.cuda.mem_get_info(i)
        free.append((free_b, i))
    return f"cuda:{max(free)[1]}"


def acc_of(wm, img: np.ndarray, message: np.ndarray) -> float:
    dec = wm.decode(to_tensor(img).to(next(wm.det.parameters()).device if hasattr(wm, "det") else wm.device)).cpu().numpy()[0]
    return bit_accuracy(dec, message)


def device_of(wm) -> str:
    if hasattr(wm, "det"):
        return str(next(wm.det.parameters()).device)
    return wm.device


def destroy(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def paste(img: np.ndarray, fill: np.ndarray, region: np.ndarray) -> np.ndarray:
    out = img.copy()
    out[region] = fill[region]
    return out


def oracle_miou(heat: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    vals = heat.reshape(-1)
    best_t, best_s = 0.0, -1.0
    for t in np.quantile(vals, np.linspace(0.02, 0.98, 25)):
        pred = downsample_mask(heat >= t)
        s = miou_image(pred, downsample_mask(gt))
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def corr(heat: np.ndarray, gt: np.ndarray) -> float:
    h = cv2.resize(heat.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA).reshape(-1)
    g = downsample_mask(gt).astype(np.float32).reshape(-1)
    h = h - h.mean()
    g = g - g.mean()
    denom = np.linalg.norm(h) * np.linalg.norm(g)
    return float((h @ g) / denom) if denom > 0 else 0.0


def convnext_stages(wm, image: np.ndarray) -> list[torch.Tensor]:
    x = wm.det.preprocess(wm._resize(to_tensor(image).to(device_of(wm))))
    net = wm.det.convnext
    outs = []
    for i in range(4):
        x = net.downsample_layers[i](x)
        x = net.stages[i](x)
        outs.append(x)
    return outs


def channel_norm(feat: torch.Tensor) -> np.ndarray:
    m = feat[0].abs().mean(0)
    return m.detach().cpu().numpy()


def best_channel_corr(feat: torch.Tensor, gt: np.ndarray) -> tuple[int, float]:
    f = feat[0].detach()
    g = torch.from_numpy(gt.astype(np.float32))
    g = F.interpolate(g[None, None], size=f.shape[-2:], mode="nearest")[0, 0].to(f.device)
    x = f.flatten(1)
    x = x - x.mean(1, keepdim=True)
    y = g.flatten()
    y = y - y.mean()
    num = (x * y).sum(1)
    den = x.norm(dim=1) * y.norm()
    c = num / den.clamp_min(1e-8)
    k = int(c.abs().argmax())
    return k, float(c[k])


def resnet_stages(wm, image: np.ndarray) -> dict[str, torch.Tensor]:
    return wm.features(to_tensor(image).to(device_of(wm)))


def main():
    device = pick_device()
    print("device", device, flush=True)
    train = LocalisationDataset(DATA, "train")
    by = {k: [] for k in ("pixelseal", "trustmark", "stegastamp")}
    for it in train:
        by[it["method"]].append(it)

    print("loading pixelseal", flush=True)
    ps = load_watermark("pixelseal", models_dir=MODELS, device=device)
    print("loading trustmark", flush=True)
    tm = load_watermark("trustmark", models_dir=MODELS, device=device)

    def pick(items, n=8):
        items = sorted(items, key=lambda it: it["mask"].mean())
        idx = np.linspace(0, len(items) - 1, n).astype(int)
        return [items[i] for i in idx]

    print("\n== pixelseal occlusion ==")
    for it in pick(by["pixelseal"]):
        img, msg, mask = it["image"], it["message"], it["mask"].astype(bool)
        base = acc_of(ps, img, msg)
        dead = destroy(img)
        acc_kill_wm = acc_of(ps, paste(img, dead, mask), msg)
        acc_kill_clean = acc_of(ps, paste(img, dead, ~mask), msg)
        # remove random fractions of the watermark
        ys, xs = np.where(mask)
        fracs = {}
        rng = np.random.default_rng(0)
        for frac in (0.25, 0.5, 0.75):
            sel = rng.choice(len(ys), size=int(frac * len(ys)), replace=False)
            region = np.zeros_like(mask)
            region[ys[sel], xs[sel]] = True
            fracs[frac] = acc_of(ps, paste(img, dead, region), msg)
        # row sweep: zero out top k percent
        row_acc = []
        for k in (0.25, 0.5, 0.75):
            region = np.zeros_like(mask)
            region[: int(k * region.shape[0])] = True
            row_acc.append(acc_of(ps, paste(img, dead, region), msg))
        print(
            f"  {it['image_name']} area={mask.mean():.3f} full={it['full_watermark']} base={base:.3f} "
            f"kill_wm={acc_kill_wm:.3f} kill_clean={acc_kill_clean:.3f} "
            f"frac25/50/75={fracs[0.25]:.3f}/{fracs[0.5]:.3f}/{fracs[0.75]:.3f} "
            f"top25/50/75={row_acc[0]:.3f}/{row_acc[1]:.3f}/{row_acc[2]:.3f}",
            flush=True,
        )

    print("\n== pixelseal early features vs blur-delta ==")
    for it in pick(by["pixelseal"], 6):
        img, mask = it["image"], it["mask"]
        stages = convnext_stages(ps, img)
        dead = destroy(img)
        dstages = convnext_stages(ps, dead)
        parts = []
        for i, (a, b) in enumerate(zip(stages, dstages)):
            k, c = best_channel_corr(a, mask)
            kd, cd = best_channel_corr((a - b).abs(), mask)
            nmap = channel_norm(a - b)
            t, s = oracle_miou(cv2.resize(nmap, (512, 512), interpolation=cv2.INTER_LINEAR), mask)
            parts.append(f"s{i}:{a.shape[-2]}x{a.shape[-1]} ch={c:.2f} dch={cd:.2f} dnorm_miou={s:.3f}")
        base = acc_of(ps, img, it["message"])
        print(f"  {it['image_name']} area={mask.mean():.3f} acc={base:.3f} " + " | ".join(parts), flush=True)

    print("\n== trustmark removal residual ==")
    for it in pick(by["trustmark"]):
        img, msg, mask = it["image"], it["message"], it["mask"]
        base = bit_accuracy(tm.decode(to_tensor(img).to(device)).cpu().numpy()[0], msg)
        removed = np.asarray(tm.tm.remove_watermark(img))
        residual = np.abs(img.astype(np.float32) - removed.astype(np.float32)).mean(2)
        t, s = oracle_miou(residual, mask)
        # also after a light blur of the residual to kill texture
        residual_b = cv2.GaussianBlur(residual, (0, 0), 4)
        tb, sb = oracle_miou(residual_b, mask)
        dead = destroy(img)
        acc_kill_wm = bit_accuracy(tm.decode(to_tensor(paste(img, dead, mask.astype(bool))).to(device)).cpu().numpy()[0], msg)
        acc_kill_clean = bit_accuracy(tm.decode(to_tensor(paste(img, dead, ~mask.astype(bool))).to(device)).cpu().numpy()[0], msg)
        print(
            f"  {it['image_name']} area={mask.mean():.3f} full={it['full_watermark']} acc={base:.3f} "
            f"res_corr={corr(residual, mask):.3f} res_miou={s:.3f} blur_miou={sb:.3f} "
            f"kill_wm={acc_kill_wm:.3f} kill_clean={acc_kill_clean:.3f}",
            flush=True,
        )

    print("\n== trustmark layer1/2 blur-delta ==")
    for it in pick(by["trustmark"], 4):
        img, mask = it["image"], it["mask"]
        a = resnet_stages(tm, img)
        b = resnet_stages(tm, destroy(img))
        parts = []
        for key in ("layer1", "layer2", "layer3", "layer4"):
            delta = (a[key] - b[key]).abs()
            k, c = best_channel_corr(delta, mask)
            nmap = channel_norm(delta)
            nmap = cv2.resize(nmap, (512, 512), interpolation=cv2.INTER_LINEAR)
            _, s = oracle_miou(nmap, mask)
            parts.append(f"{key} dch={c:.2f} dnorm={s:.3f}")
        print(f"  {it['image_name']} area={mask.mean():.3f} " + " | ".join(parts), flush=True)


if __name__ == "__main__":
    main()
