"""Show the first six saved holdout predictions; never select examples by score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from message_heatmaps import colorize, mask_overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    with np.load(args.source / "holdout.npz", allow_pickle=False) as saved:
        names = saved["image_names"].tolist()[:6]
        responses = saved["response"][:6].copy()
        threshold = float(saved["threshold"])
    assert len(names) == 6 and responses.shape == (6, 128, 128)
    rows = {row["image_name"]: row for row in json.loads(
        (args.source / "diagnostics/holdout_rows.json").read_text())}
    with np.load(args.data / "train/masks_128.npz", allow_pickle=False) as truth:
        lookup = {name: i for i, name in enumerate(truth["names"].tolist())}
        masks = truth["masks"][[lookup[name] for name in names]]
    out = args.source / "examples"
    out.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=14)
    panel_height = 322
    overview = Image.new("RGB", (1024, panel_height * len(names)), "white")
    for i, (name, response, target) in enumerate(zip(names, responses, masks)):
        with Image.open(args.data / "train/images" / f"{name}.png") as source:
            image = source.convert("RGB").resize((256, 256))
        heat = colorize(response, 0.0, 1.0).resize((256, 256), Image.Resampling.BILINEAR)
        prediction = (response >= threshold).astype(np.uint8)
        pictures = (image, heat, mask_overlay(image, prediction), mask_overlay(image, target))
        canvas = Image.new("RGB", (1024, panel_height), "white")
        draw = ImageDraw.Draw(canvas)
        # This is the previously saved score, not another evaluation.
        draw.text((6, 3), f"{name} | saved holdout mIoU={float(rows[name]['miou']):.4f} | threshold={threshold:.2f}", fill="black", font=font)
        for column, (picture, label) in enumerate(zip(pictures, (
                "Image", "Response: fixed range [0,1]", "Prediction: pink=watermark", "Ground truth: pink=watermark"))):
            draw.text((column * 256 + 5, 25), label, fill="black", font=font)
            canvas.paste(picture, (column * 256, 46))
        draw.text((6, 304), "First six image_names in holdout.npz; no score-based example selection.", fill="black", font=font)
        canvas.save(out / f"{name}.png")
        overview.paste(canvas, (0, i * panel_height))
    overview.save(out / "overview.png")
    print(f"Saved six fixed-order examples and overview to {out}")


if __name__ == "__main__":
    main()
