"""Probe message-aligned extractor maps against the official train masks.

Uses the starter-kit wrappers. Does not train a classifier.
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
from wmloc.utils import bit_accuracy, downsample_mask, miou_image, score, to_tensor  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

DATA = Path("/p/scratch/training2666/li103/task1/release")
MODELS = Path("/p/scratch/training2666/li103/task1/models")


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free = []
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        free.append((free_b, i))
    free.sort(reverse=True)
    return f"cuda:{free[0][1]}"


def aligned_map(per_pos: torch.Tensor, message: np.ndarray) -> torch.Tensor:
    """per_pos: 1 x nbits x h x w. Positive where the position votes for the known bit."""
    sign = torch.tensor(message, device=per_pos.device, dtype=per_pos.dtype) * 2 - 1
    return (per_pos * sign.view(1, -1, 1, 1)).mean(dim=1)[0]


def upsample(score_map: torch.Tensor, size: int = 512) -> np.ndarray:
    x = score_map.float()[None, None]
    y = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return y[0, 0].detach().cpu().numpy()


def smooth512(heat: np.ndarray, k: int = 9) -> np.ndarray:
    x = torch.from_numpy(np.asarray(heat, dtype=np.float32))[None, None]
    y = F.avg_pool2d(x, k, stride=1, padding=k // 2)
    return y[0, 0].numpy()


def best_threshold(heat: np.ndarray, gt512: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Sweep absolute thresholds on a 512 heatmap. Score with the official downsample."""
    gt128 = downsample_mask(gt512)
    vals = heat.reshape(-1)
    qs = np.quantile(vals, np.linspace(0.02, 0.98, 33))
    best_t, best_s, best = 0.0, -1.0, None
    for t in qs:
        pred = downsample_mask(heat >= t)
        s = miou_image(pred, gt128)
        if s > best_s:
            best_t, best_s, best = float(t), float(s), pred
    return best_t, best_s, best


def shared_threshold(heats: list[np.ndarray], gts: list[np.ndarray]) -> tuple[float, float]:
    """One threshold for a method. Candidates are pooled quantiles."""
    vals = np.concatenate([h.reshape(-1) for h in heats])
    cands = np.quantile(vals, np.linspace(0.02, 0.98, 49))
    best_t, best_s = float(cands[0]), -1.0
    gt128s = [downsample_mask(g) for g in gts]
    for t in cands:
        preds = [downsample_mask(h >= t) for h in heats]
        s = score(preds, gt128s)
        if s > best_s:
            best_t, best_s = float(t), float(s)
    return best_t, best_s


def gradient_saliency(forward_logits, image_uint8: np.ndarray, message: np.ndarray, device: str) -> np.ndarray:
    x = to_tensor(image_uint8).to(device).requires_grad_(True)
    logits = forward_logits(x)[0]
    sign = torch.tensor(message, device=device, dtype=logits.dtype) * 2 - 1
    (logits * sign).sum().backward()
    sal = x.grad.abs().mean(dim=1)[0]
    return sal.detach().cpu().numpy()


def pixelseal_logits(wm, images: torch.Tensor) -> torch.Tensor:
    return wm.det(wm._resize(images))[:, 1:]


def trustmark_logits(wm, images: torch.Tensor) -> torch.Tensor:
    return wm.net(wm._prep(images))


def mask_geometry(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return {"area": 0.0, "aspect": 0.0, "holes": 0.0, "rect_iou": 1.0}
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = y1 - y0 + 1, x1 - x0 + 1
    box = np.zeros_like(mask)
    box[y0 : y1 + 1, x0 : x1 + 1] = 1
    inter = np.logical_and(box, mask).sum()
    union = np.logical_or(box, mask).sum()
    # holes inside the bbox
    holes = np.logical_and(box, mask == 0).sum() / max(box.sum(), 1)
    return {
        "area": float(mask.mean()),
        "aspect": float(max(h, w) / max(min(h, w), 1)),
        "holes": float(holes),
        "rect_iou": float(inter / max(union, 1)),
    }


def main():
    device = pick_device()
    print("device", device, flush=True)
    train = LocalisationDataset(DATA, "train")
    by = {"pixelseal": [], "trustmark": [], "stegastamp": []}
    for it in train:
        by[it["method"]].append(it)

    # geometry, no model
    for name, items in by.items():
        geos = [mask_geometry(it["mask"]) for it in items]
        def avg(k):
            return float(np.mean([g[k] for g in geos]))
        n_full = sum(bool(it["full_watermark"]) for it in items)
        print(
            f"geom {name} n={len(items)} full={n_full} area={avg('area'):.3f} "
            f"aspect={avg('aspect'):.2f} holes={avg('holes'):.3f} rect_iou={avg('rect_iou'):.3f}",
            flush=True,
        )
        aspects = np.array([g["aspect"] for g in geos])
        print(
            f"  aspect p50={np.median(aspects):.2f} p90={np.quantile(aspects, 0.9):.2f} "
            f"max={aspects.max():.2f} thin>4={(aspects > 4).mean():.2f} thin>8={(aspects > 8).mean():.2f}",
            flush=True,
        )

    wms = {}
    for name in ("pixelseal", "trustmark"):
        print("loading", name, flush=True)
        wms[name] = load_watermark(name, models_dir=MODELS, device=device)

    n_probe = 12
    for name, wm in wms.items():
        items = by[name][:n_probe]
        print(f"\n=== {name} probe {len(items)} ===", flush=True)
        feat_heats, grad_heats, gts = [], [], []
        feat_scores, grad_scores = [], []
        for it in items:
            img = to_tensor(it["image"]).to(device)
            dec = wm.decode(img).cpu().numpy()[0]
            acc = bit_accuracy(dec, it["message"])
            feats = wm.features(img)
            shapes = {k: tuple(v.shape) for k, v in feats.items()}
            amap = aligned_map(feats["per_position_logits"], it["message"])
            heat = upsample(amap, 512)
            t, s, _ = best_threshold(heat, it["mask"])
            feat_scores.append(s)
            feat_heats.append(heat)
            gts.append(it["mask"])
            print(
                f"  {it['image_name']} full={it['full_watermark']} acc={acc:.3f} "
                f"area={it['mask'].mean():.3f} feat_oracle={s:.3f} t={t:.4g} "
                f"map[{amap.shape[-2]}x{amap.shape[-1]}] min={float(amap.min()):.3f} max={float(amap.max()):.3f} "
                f"shapes={shapes}",
                flush=True,
            )
            fwd = pixelseal_logits if name == "pixelseal" else trustmark_logits
            sal = smooth512(gradient_saliency(lambda x, f=fwd, m=wm: f(m, x), it["image"], it["message"], device))
            tg, sg, _ = best_threshold(sal, it["mask"])
            grad_scores.append(sg)
            grad_heats.append(sal)
            print(f"    grad_oracle={sg:.3f} tg={tg:.4g}", flush=True)
        ft, fs = shared_threshold(feat_heats, gts)
        gt, gs = shared_threshold(grad_heats, gts)
        print(
            f"  oracle feat {np.mean(feat_scores):.3f} shared feat {fs:.3f} @ {ft:.4g} | "
            f"oracle grad {np.mean(grad_scores):.3f} shared grad {gs:.3f} @ {gt:.4g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
