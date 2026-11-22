"""Fixed known-message occlusion map for the official StegaStamp decoder.

No embedding or model training. The 16x16 occlusion grid and blur are fixed in
advance; only a binary threshold is fitted on the 136 real dev images.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from message_heatmaps import colorize, fit_threshold, make_split, mask_overlay, number
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image, to_tensor
from wmloc.watermarks import load_watermark


GRID = 16
RESOLUTION = 400
CELL = RESOLUTION // GRID
SEED = 2026


def native_scores(wm, images, signs):
    """Continuous known-message support; images are already native NHWC400."""
    logits = np.asarray(wm._run(images, [wm.LOGITS])[0], dtype=np.float32)
    logits = logits.reshape(len(images), wm.nbits)
    return np.mean(logits * signs[None], axis=1, dtype=np.float64)


@torch.inference_mode()
def occlusion_response(wm, image, message, batch_size):
    # Match the official decoder's bicubic/antialiased preparation exactly once.
    native = np.ascontiguousarray(wm._prep(to_tensor(image))[0], dtype=np.float32)
    signs = 2 * np.asarray(message, dtype=np.float32) - 1
    original_score = float(native_scores(wm, native[None], signs)[0])
    # A 9x9 Gaussian, sigma=2, independently in each RGB channel.
    blurred = gaussian_filter(native, sigma=(2, 2, 0), radius=(4, 4, 0), mode="reflect")
    response = np.empty(GRID * GRID, dtype=np.float32)
    for start in range(0, len(response), batch_size):
        cells = list(range(start, min(start + batch_size, len(response))))
        batch = np.broadcast_to(native, (len(cells), *native.shape)).copy()
        for j, cell in enumerate(cells):
            y, x = (cell // GRID) * CELL, (cell % GRID) * CELL
            batch[j, y:y + CELL, x:x + CELL] = blurred[y:y + CELL, x:x + CELL]
        response[start:start + len(cells)] = original_score - native_scores(wm, batch, signs)
    coarse = torch.from_numpy(response.reshape(GRID, GRID))[None, None]
    heat = F.interpolate(coarse, (128, 128), mode="bilinear", align_corners=False)[0, 0]
    return heat.numpy().astype(np.float32), original_score


def extract(wm, items, batch_size, split):
    responses, supports = [], []
    started = time.monotonic()
    for i, item in enumerate(items):
        response, support = occlusion_response(wm, item["image"], item["message"], batch_size)
        responses.append(response)
        supports.append(support)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(items):
            print(f"{split}: {i + 1}/{len(items)} images, elapsed={time.monotonic() - started:.1f}s", flush=True)
    return np.stack(responses).astype(np.float32), np.asarray(supports)


def save_response(path, names, response, threshold):
    np.savez_compressed(path, image_names=np.asarray(names, dtype=str),
                        response=np.asarray(response, dtype=np.float32),
                        threshold=np.asarray(threshold, dtype=np.float64),
                        dev_in_sample=np.asarray(True, dtype=np.bool_))


def evaluate(items, indices, response, threshold):
    rows = []
    for i in indices:
        pred = (response[i] >= threshold).astype(np.uint8)
        rows.append({"image_name": items[i]["image_name"],
                     "full_watermark": bool(items[i]["full_watermark"]),
                     "miou": miou_image(pred, items[i]["mask128"])})
    result = {}
    for key, flag in (("full_watermark_true", True), ("full_watermark_false", False), ("all", None)):
        group = [r["miou"] for r in rows if flag is None or r["full_watermark"] == flag]
        result[key] = {"images": len(group), "miou": float(np.mean(group)) if group else None}
    return result, rows


def save_examples(out, items, val, responses, threshold, limits):
    """First six holdout names, selected without consulting their scores."""
    folder = out / "examples"
    folder.mkdir(exist_ok=True)
    for i in val[:6]:
        item = items[i]
        image = Image.fromarray(item["image"]).resize((256, 256))
        heat = colorize(responses[i], *limits).resize((256, 256), Image.Resampling.BILINEAR)
        pred = (responses[i] >= threshold).astype(np.uint8)
        panels = [image, Image.blend(image, heat, 0.65), mask_overlay(image, pred),
                  mask_overlay(image, item["mask128"])]
        canvas = Image.new("RGB", (1024, 312), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 3), f'StegaStamp / {item["image_name"]} / holdout mIoU={miou_image(pred, item["mask128"]):.4f}', fill="black")
        for j, (panel, title) in enumerate(zip(panels, ["Image", "Occlusion support drop", "Prediction: pink=watermark", "Ground truth: pink=watermark"])):
            draw.text((j * 256 + 4, 20), title, fill="black")
            canvas.paste(panel, (j * 256, 38))
        draw.text((5, 296), f"Fixed DEV color range [{limits[0]:.5f}, {limits[1]:.5f}]; DEV threshold={threshold:.5f}", fill="black")
        canvas.save(folder / f'{item["image_name"]}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Official release folder containing train/ and test/")
    parser.add_argument("--models-dir", type=Path, required=True, help="Contains stegastamp/stega_stamp_logits.onnx")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "output/fusion/codex_occlusion_stegastamp")
    parser.add_argument("--batch-size", type=int, default=16, help="Maximum simultaneous ORT occlusion images")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    items = sorted(list(LocalisationDataset(args.data, "train", methods=["stegastamp"])),
                   key=lambda it: it["image_name"])
    dev, holdout = make_split(items, SEED)
    dev_names = [items[i]["image_name"] for i in dev]
    holdout_names = [items[i]["image_name"] for i in holdout]
    split = {"seed": SEED, "train": dev_names, "val": holdout_names,
             "stratify": "full_watermark and foreground-area bins; same make_split as message_heatmaps.py"}
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    print(f"StegaStamp: {len(dev)} dev / {len(holdout)} fixed holdout; 16x16 native occlusion grid", flush=True)
    import onnx
    import onnxruntime as ort
    if args.device.startswith("cuda"):
        # ORT 1.23.2 uses the CUDA/cuDNN libraries installed alongside PyTorch.
        ort.preload_dlls()
    wm = load_watermark("stegastamp", models_dir=args.models_dir.resolve(), device=args.device)
    if args.device.startswith("cuda") and "CUDAExecutionProvider" not in wm.sess.get_providers():
        raise RuntimeError("CUDA was requested but the StegaStamp session has no CUDAExecutionProvider")
    # The release graph contains both an embedder and a decoder. Physically prune
    # the embedder: selecting just one output in an ORT inference build does not
    # guarantee it skips unrelated branches.
    decoder_path = args.models_dir.resolve() / "stegastamp/stega_stamp_decoder_only.onnx"
    if not decoder_path.exists():
        onnx.utils.extract_model(
            str(args.models_dir.resolve() / "stegastamp/stega_stamp_logits.onnx"),
            str(decoder_path), ["image", "secret"], [wm.LOGITS], check_model=False)
    probe = wm._prep(to_tensor(items[int(dev[0])]["image"]))
    official_logits = wm._run(probe, [wm.LOGITS])[0]
    provider_options = wm.sess.get_provider_options()
    providers = [(name, provider_options[name]) for name in wm.sess.get_providers()]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    wm.sess = ort.InferenceSession(str(decoder_path), sess_options=options, providers=providers)
    pruned_logits = wm._run(probe, [wm.LOGITS])[0]
    np.testing.assert_allclose(pruned_logits, official_logits, rtol=1e-4, atol=1e-4)
    equivalence_error = float(np.abs(pruned_logits - official_logits).max())
    print(f"Decoder-only graph matches official logits: max_abs_error={equivalence_error:.6g}", flush=True)
    print(f"ONNX providers: {wm.sess.get_providers()}", flush=True)
    train_response, train_support = extract(wm, items, args.batch_size, "real train")
    masks = np.stack([items[i]["mask128"] for i in dev])
    threshold, dev_score = fit_threshold(train_response[dev], masks)
    save_response(args.out / "dev.npz", dev_names, train_response[dev], threshold)
    save_response(args.out / "holdout.npz", holdout_names, train_response[holdout], threshold)
    dev_report, dev_rows = evaluate(items, dev, train_response, threshold)
    holdout_report, holdout_rows = evaluate(items, holdout, train_response, threshold)
    report = {"method": "stegastamp_known_message_occlusion", "seed": SEED,
              "native_resolution": [RESOLUTION, RESOLUTION], "occlusion_grid": [GRID, GRID],
              "replacement": "Same native image blurred with 9x9 Gaussian, sigma=2, reflect boundary",
              "response": "original mean signed-message logits minus occluded mean signed-message logits; bilinear to 128x128",
              "model_training": False, "synthetic_data": False, "dev_in_sample": True,
              "dev_in_sample_reason": "Threshold fitted on all real dev responses; no model fitting or holdout parameter selection",
              "threshold_selection": "Official per-image two-class mIoU on the 136 real dev images only",
              "threshold": number(threshold), "threshold_fit_dev_miou": dev_score,
              "dev": dev_report, "holdout": holdout_report,
              "test_images": 0, "test_complete": False, "seconds": time.monotonic() - started,
              "batch_size": args.batch_size, "onnx_providers": wm.sess.get_providers()}
    report.update(decoder_only=True, decoder_equivalence_max_abs_error=equivalence_error)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    with (args.out / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "image_name", "full_watermark", "miou"])
        writer.writeheader()
        writer.writerows({"split": split_name, **row} for split_name, rows in
                         (("dev", dev_rows), ("holdout", holdout_rows)) for row in rows)
    limits = np.quantile(train_response[dev], [0.01, 0.99]).tolist()
    save_examples(args.out, items, holdout, train_response, threshold, limits)
    print(json.dumps(report, indent=2), flush=True)
    test = sorted(list(LocalisationDataset(args.data, "test", methods=["stegastamp"])),
                  key=lambda it: it["image_name"])
    test_response, test_support = extract(wm, test, args.batch_size, "test inference only")
    save_response(args.out / "test.npz", [it["image_name"] for it in test], test_response, threshold)
    report.update(test_images=len(test), test_complete=True, seconds=time.monotonic() - started)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
