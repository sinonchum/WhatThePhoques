"""Known-message heatmaps for the official PixelSeal and TrustMark extractors."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image
from wmloc.watermarks import load_watermark


def make_split(items, seed):
    """Hold out 20%, stratified by full_watermark and foreground area."""
    rng = np.random.default_rng(seed)
    val = []
    for full in (False, True):
        ids = [i for i, it in enumerate(items) if bool(it["full_watermark"]) == full]
        ids.sort(key=lambda i: (items[i]["mask128"].mean(), items[i]["image_name"]))
        # Each of the 17 area bins contains five images in the official release.
        for group in np.array_split(ids, round(len(ids) * 0.2)):
            val.append(int(rng.choice(group)))
    val = np.array(sorted(val))
    train = np.setdiff1d(np.arange(len(items)), val)
    return train, val


def threshold_scores(maps, masks, thresholds):
    """Exact mean of per-image two-class IoUs, without a huge threshold tensor."""
    scores = []
    for heat, mask in zip(maps, masks):
        order = np.argsort(heat.ravel())
        labels = mask.ravel()[order].astype(np.int64)
        positives = np.concatenate(([0], np.cumsum(labels)))
        cut = np.searchsorted(heat.ravel()[order], thresholds, side="left")
        fn = positives[cut]
        tp = positives[-1] - fn
        tn = cut - fn
        fp = len(labels) - cut - tp
        u1, u0 = tp + fp + fn, tn + fp + fn
        iou1 = np.divide(tp, u1, out=np.ones(len(cut)), where=u1 != 0)
        iou0 = np.divide(tn, u0, out=np.ones(len(cut)), where=u0 != 0)
        scores.append((iou0 + iou1) / 2)
    return np.mean(scores, axis=0)


def fit_threshold(maps, masks):
    candidates = np.unique(np.concatenate(([-np.inf, 0, np.inf],
                                           np.quantile(maps, np.linspace(0, 1, 257)))))
    scores = threshold_scores(maps, masks, candidates)
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def number(value):
    # Keep report.json valid JSON even when the optimum is an all-zero/one mask.
    return float(value) if np.isfinite(value) else str(float(value))


def colorize(heat, low, high):
    stops = np.array([[20, 15, 90], [40, 100, 190], [30, 200, 170],
                      [240, 220, 60], [200, 30, 30]], dtype=float)
    x = np.clip((heat - low) / max(high - low, 1e-8), 0, 1) * (len(stops) - 1)
    left = np.minimum(x.astype(int), len(stops) - 2)
    rgb = stops[left] * (1 - (x - left)[..., None]) + stops[left + 1] * (x - left)[..., None]
    return Image.fromarray(rgb.astype(np.uint8))


def mask_overlay(image, mask):
    tint = Image.new("RGB", image.size, (255, 30, 110))
    colored = Image.blend(image, tint, 0.55)
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
    return Image.composite(colored, image, alpha)


def save_panel(path, item, heat, pred, score, threshold, limits):
    size = 256
    image = Image.fromarray(item["image"]).resize((size, size))
    colored = colorize(heat, *limits).resize((size, size), Image.Resampling.BILINEAR)
    panels = [image, Image.blend(image, colored, 0.65),
              mask_overlay(image, pred), mask_overlay(image, item["mask128"])]
    canvas = Image.new("RGB", (size * 4, size + 66), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 3), f'{item["method"]} / {item["image_name"]} / validation mIoU={score:.4f}', fill="black")
    for i, (panel, title) in enumerate(zip(panels, ["Image", "Message match: blue=low, red=high", "Prediction (pink=watermark)", "Ground truth (pink=watermark)"])):
        draw.text((i * size + 4, 20), title, fill="black")
        canvas.paste(panel, (i * size, 36))
    draw.text((6, size + 43), f"Fixed TRAIN color range [{limits[0]:.3f}, {limits[1]:.3f}]; TRAIN threshold={threshold:.4f}", fill="black")
    canvas.save(path)


@torch.inference_mode()
def extract(wm, items, batch_size):
    native, grids = [], []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        images = torch.from_numpy(np.stack([it["image"] for it in batch])).permute(0, 3, 1, 2).float() / 255
        local_logits = wm.features(images)["per_position_logits"].float()
        messages = torch.tensor(np.stack([it["message"] for it in batch]),
                                device=local_logits.device, dtype=torch.float32)
        heat = (local_logits * (2 * messages - 1)[:, :, None, None]).mean(1, keepdim=True)
        native.append(heat[:, 0].cpu().numpy())
        grids.append(F.interpolate(heat, (128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy())
        print(f"  extracted {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    return np.concatenate(native), np.concatenate(grids)


def evaluate_method(args, method):
    out = args.out / method
    out.mkdir(parents=True, exist_ok=True)
    dataset = LocalisationDataset(args.data, "train", methods=[method])
    items = sorted(list(dataset), key=lambda it: it["image_name"])
    train, val = make_split(items, args.seed)
    names = np.array([it["image_name"] for it in items])
    masks = np.stack([it["mask128"] for it in items])
    split = np.full(len(items), "train", dtype="U5")
    split[val] = "val"
    (out / "split.json").write_text(json.dumps({
        "seed": args.seed, "train": names[train].tolist(), "val": names[val].tolist(),
        "stratify": "full_watermark and foreground-area bins; never used as model input",
    }, indent=2))

    print(f"{method}: {len(train)} calibration / {len(val)} validation", flush=True)
    wm = load_watermark(method, models_dir=args.models_dir.resolve(), device=args.device)
    native, heatmaps = extract(wm, items, args.batch_size)
    del wm
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    np.savez_compressed(out / "heatmaps.npz", image_names=names, split=split,
                        heatmaps=heatmaps, native_heatmaps=native, masks128=masks)

    threshold, train_score = fit_threshold(heatmaps[train], masks[train])
    # A comparison with NO spatial localization: choose just one label per image.
    image_means = heatmaps.mean((1, 2))
    train_means = np.broadcast_to(image_means[train, None, None], masks[train].shape)
    image_threshold, _ = fit_threshold(train_means, masks[train])
    limits = np.quantile(heatmaps[train], [0.01, 0.99]).tolist()
    rows = []
    panels = out / "validation"
    panels.mkdir(exist_ok=True)
    for i in val:
        pred = (heatmaps[i] >= threshold).astype(np.uint8)
        whole = np.full_like(pred, int(image_means[i] >= image_threshold))
        row = {"image_name": str(names[i]), "watermark_fraction": float(masks[i].mean()),
               "heatmap_miou": miou_image(pred, masks[i]),
               "image_only_miou": miou_image(whole, masks[i]),
               "all_clean_miou": miou_image(np.zeros_like(pred), masks[i]),
               "all_watermarked_miou": miou_image(np.ones_like(pred), masks[i])}
        rows.append(row)
        save_panel(panels / f"{names[i]}.png", items[i], heatmaps[i], pred,
                   row["heatmap_miou"], threshold, limits)
    with (out / "validation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ordered = sorted(rows, key=lambda r: r["heatmap_miou"])
    selected = ordered[:6] + ordered[len(ordered)//2:len(ordered)//2 + 3] + ordered[-3:]
    sheet = Image.new("RGB", (1024, 161 * 6), "white")
    for j, row in enumerate(selected):
        with Image.open(panels / f'{row["image_name"]}.png') as im:
            sheet.paste(im.resize((512, 161)), ((j % 2) * 512, (j // 2) * 161))
    sheet.save(out / "overview.png")

    def mean(key, subset=rows):
        return float(np.mean([r[key] for r in subset])) if subset else None

    report = {"method": method, "train_images": len(train), "validation_images": len(val),
              "native_resolution": list(native.shape[1:]),
              "heatmap_definition": "mean_k((2*message[k]-1)*per_position_logits[k])",
              "threshold": number(threshold), "image_only_threshold": number(image_threshold),
              "train_heatmap_miou": train_score,
              "validation": {key: mean(key) for key in rows[0] if key.endswith("miou")},
              "by_watermark_area": {}}
    for label, lo, hi in [("small_watermark", 0, 0.1), ("middle", 0.1, 0.9), ("small_clean_hole", 0.9, 1.01)]:
        subset = [r for r in rows if lo <= r["watermark_fraction"] < hi]
        report["by_watermark_area"][label] = {"images": len(subset),
            "heatmap_miou": mean("heatmap_miou", subset), "image_only_miou": mean("image_only_miou", subset)}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Release folder containing train/metadata.json")
    p.add_argument("--models-dir", type=Path, required=True, help="Official download_models.py output")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "output/message_heatmaps")
    p.add_argument("--methods", nargs="+", choices=["pixelseal", "trustmark"], default=["pixelseal", "trustmark"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for method in args.methods:
        reports.append(evaluate_method(args, method))
        (args.out / "report.json").write_text(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
