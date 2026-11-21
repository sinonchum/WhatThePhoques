"""Train a small segmentation network, keeping the message-heatmap holdout unchanged."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F

from message_heatmaps import fit_threshold, mask_overlay, number
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image


def block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(),
    )


class SmallSegmenter(nn.Module):
    """Full-resolution noise stem, U-Net skips, global context, 128x128 output."""

    def __init__(self, heat_mean, heat_std, baseline_threshold):
        super().__init__()
        self.register_buffer("heat_mean", torch.tensor(float(heat_mean)))
        self.register_buffer("heat_std", torch.tensor(float(heat_std)))
        self.stem = block(7, 16)
        self.down = nn.ModuleList([block(16, 24, 2), block(24, 48, 2),
                                   block(48, 96, 2), block(96, 128, 2), block(128, 128, 2)])
        self.global_context = nn.Sequential(nn.Linear(128, 128), nn.SiLU())
        self.up = nn.ModuleList([block(256, 96), block(192, 64), block(112, 48)])
        self.head = nn.Conv2d(49, 1, 1)
        # Start from exactly the previous heatmap decision, then learn corrections.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -(baseline_threshold - heat_mean) / heat_std)
        with torch.no_grad():
            self.head.weight[0, -1, 0, 0] = 1

    def forward(self, images, raw_heat):
        rgb = images.float() / 255
        smooth = F.avg_pool2d(F.pad(rgb, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
        residual = ((rgb - smooth) * 20).clamp(-3, 3)
        heat = (raw_heat - self.heat_mean) / self.heat_std
        full_heat = F.interpolate(heat[:, None], rgb.shape[-2:], mode="bilinear", align_corners=False)
        features = [self.stem(torch.cat((rgb * 2 - 1, residual, full_heat), 1))]
        for layer in self.down:
            features.append(layer(features[-1]))
        y = features[-1] + self.global_context(features[-1].mean((2, 3)))[:, :, None, None]
        for layer, skip in zip(self.up, (features[4], features[3], features[2])):
            y = layer(torch.cat((F.interpolate(y, skip.shape[-2:], mode="bilinear", align_corners=False), skip), 1))
        return self.head(torch.cat((y, heat[:, None]), 1))[:, 0]


def loss_fn(logits, masks):
    logits = logits.float()
    p = logits.sigmoid()
    i1 = (p * masks).sum((1, 2))
    u1 = (p + masks - p * masks).sum((1, 2))
    q, clean = 1 - p, 1 - masks
    i0 = (q * clean).sum((1, 2))
    u0 = (q + clean - q * clean).sum((1, 2))
    soft_miou = 0.5 * ((i1 + 1) / (u1 + 1) + (i0 + 1) / (u0 + 1))
    return F.binary_cross_entropy_with_logits(logits, masks) + (1 - soft_miou).mean()


@torch.inference_mode()
def predict(model, images, heatmaps, batch_size, device):
    model.eval()
    result = []
    for start in range(0, len(images), batch_size):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(images[start:start + batch_size], heatmaps[start:start + batch_size])
        result.append(logits.float().sigmoid().cpu().numpy())
    return np.concatenate(result)


def save_comparisons(out, items, val, baseline, prediction, rows):
    panels = out / "validation"
    panels.mkdir(exist_ok=True)
    for i, row in zip(val, rows):
        image = Image.fromarray(items[i]["image"]).resize((256, 256))
        pictures = [image, mask_overlay(image, baseline[i]), mask_overlay(image, prediction[i]),
                    mask_overlay(image, items[i]["mask128"])]
        canvas = Image.new("RGB", (1024, 300), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 3), f'{items[i]["method"]}/{row["image_name"]}  heatmap={row["heatmap_miou"]:.4f}  network={row["network_miou"]:.4f}  gain={row["gain"]:+.4f}', fill="black")
        for j, (picture, label) in enumerate(zip(pictures, ["Image", "Heatmap baseline", "Small network", "Ground truth"])):
            draw.text((j * 256 + 5, 20), label + (" (pink=watermark)" if j else ""), fill="black")
            canvas.paste(picture, (j * 256, 40))
        canvas.save(panels / f'{row["image_name"]}.png')
    ordered = sorted(rows, key=lambda r: r["gain"])
    sheet = Image.new("RGB", (1024, 900), "white")
    for j, (worst, best) in enumerate(zip(ordered[:6], ordered[-6:][::-1])):
        for col, row in enumerate((worst, best)):
            with Image.open(panels / f'{row["image_name"]}.png') as im:
                sheet.paste(im.resize((512, 150)), (col * 512, j * 150))
    sheet.save(out / "overview.png")


def train_method(args, method):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = args.out / method
    out.mkdir(parents=True, exist_ok=True)
    source = args.heatmaps / method
    split = json.loads((source / "split.json").read_text())
    baseline_report = json.loads((source / "report.json").read_text())
    items = sorted(list(LocalisationDataset(args.data, "train", methods=[method])), key=lambda it: it["image_name"])
    names = [it["image_name"] for it in items]
    with np.load(source / "heatmaps.npz") as cache:
        assert cache["image_names"].tolist() == names, "Heatmap image order does not match the dataset"
        raw_heat = cache["heatmaps"].copy()
    index = {name: i for i, name in enumerate(names)}
    train = np.array([index[name] for name in split["train"]])
    val = np.array([index[name] for name in split["val"]])
    assert set(train).isdisjoint(val) and len(train) + len(val) == len(items)
    # Read the original split verbatim; never select epochs or thresholds on val.
    (out / "split.json").write_text(json.dumps(split, indent=2))
    masks_np = np.stack([it["mask128"] for it in items])
    images = torch.from_numpy(np.stack([it["image"] for it in items])).permute(0, 3, 1, 2).to(device)
    heatmaps = torch.from_numpy(raw_heat).to(device)
    masks = torch.from_numpy(masks_np.astype(np.float32)).to(device)
    heat_mean, heat_std = float(raw_heat[train].mean()), max(float(raw_heat[train].std()), 1e-6)
    base_threshold = float(baseline_report["threshold"])
    model = SmallSegmenter(heat_mean, heat_std, base_threshold).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    train_ids = torch.tensor(train, device=device)
    history = []
    started = time.monotonic()
    print(f"{method}: {parameters:,} parameters; {len(train)} train / {len(val)} unchanged holdout; {args.epochs} fixed epochs", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_ids[torch.randperm(len(train_ids), device=device)]
        total = 0.0
        for ids in order.split(args.batch_size):
            x, h, y = images[ids], heatmaps[ids], masks[ids]
            flip = torch.rand(len(ids), device=device) < 0.5
            x[flip], h[flip], y[flip] = x[flip].flip(-1), h[flip].flip(-1), y[flip].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x, h), y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        scheduler.step()
        history.append({"epoch": epoch, "loss": total / len(train)})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}/{args.epochs}: loss={history[-1]['loss']:.5f}, elapsed={time.monotonic()-started:.0f}s", flush=True)

    probabilities = predict(model, images, heatmaps, args.batch_size, device)
    threshold, train_score = fit_threshold(probabilities[train], masks_np[train])
    predictions = (probabilities >= threshold).astype(np.uint8)
    baseline = (raw_heat >= base_threshold).astype(np.uint8)
    rows = []
    for i in val:
        before, after = miou_image(baseline[i], masks_np[i]), miou_image(predictions[i], masks_np[i])
        rows.append({"image_name": names[i], "watermark_fraction": float(masks_np[i].mean()),
                     "heatmap_miou": before, "network_miou": after, "gain": after - before})
    before = float(np.mean([r["heatmap_miou"] for r in rows]))
    after = float(np.mean([r["network_miou"] for r in rows]))
    report = {"method": method, "parameters": parameters, "seed": args.seed,
              "epochs": args.epochs, "epoch_selection": "fixed in advance; no holdout selection",
              "train_images": len(train), "validation_images": len(val),
              "inputs": "512x512 RGB + 20x high-pass residual + known-message heatmap",
              "loss": "BCE + per-image two-class soft IoU", "threshold_selection": "136 training images only",
              "threshold": number(threshold), "train_network_miou": train_score,
              "validation": {"heatmap_miou": before, "network_miou": after, "gain": after - before},
              "seconds": time.monotonic() - started, "by_watermark_area": {}}
    for label, lo, hi in [("small_watermark", 0, 0.1), ("middle", 0.1, 0.9), ("small_clean_hole", 0.9, 1.01)]:
        subset = [r for r in rows if lo <= r["watermark_fraction"] < hi]
        report["by_watermark_area"][label] = {"images": len(subset), **{
            key: float(np.mean([r[key] for r in subset])) if subset else None
            for key in ("heatmap_miou", "network_miou", "gain")}}
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "method": method, "seed": args.seed, "epochs": args.epochs,
                "threshold": threshold, "train_names": split["train"], "val_names": split["val"]}, out / "model.pt")
    np.savez_compressed(out / "predictions.npz", image_names=np.array(names),
                        probabilities=probabilities, masks=predictions)
    with (out / "validation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    save_comparisons(out, items, val, baseline, predictions, rows)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--heatmaps", type=Path, default=Path(__file__).parent / "output/message_heatmaps")
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "output/small_segmenter")
    p.add_argument("--methods", nargs="+", choices=["pixelseal", "trustmark"], default=["pixelseal", "trustmark"])
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for method in args.methods:
        reports.append(train_method(args, method))
        (args.out / "report.json").write_text(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
