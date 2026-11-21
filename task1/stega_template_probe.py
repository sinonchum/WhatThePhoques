"""DEV-only pilot: message-centered StegaStamp encoder residual templates.

Uses only selected real full_watermark=True DEV RGB images, never holdout/test.
All spatial/filter parameters are fixed; fivefold threshold CV stays in DEV.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, uniform_filter

from message_heatmaps import colorize, fit_threshold, mask_overlay, number
from wmloc.utils import downsample_mask, miou_image


SEED = 2026


def native_image(image):
    # Reproduce a 256px embedding input, followed by the official 400px encoder.
    small = Image.fromarray(image).resize((256, 256), Image.Resampling.BICUBIC)
    return np.asarray(small.resize((400, 400), Image.Resampling.BICUBIC), dtype=np.float32) / 255


def template_to_512(residual):
    residual = residual - residual.mean((0, 1), keepdims=True)
    channels = []
    for c in range(3):
        plane = Image.fromarray(residual[:, :, c].astype(np.float32))
        plane = plane.resize((256, 256), Image.Resampling.BILINEAR)
        channels.append(np.asarray(plane.resize((512, 512), Image.Resampling.BILINEAR)))
    return np.stack(channels, axis=2).astype(np.float32)


def dog(image):
    return gaussian_filter(image, sigma=(1, 1, 0), mode="reflect") - gaussian_filter(image, sigma=(3, 3, 0), mode="reflect")


def local_correlation(image_band, template):
    t = dog(template)
    mx = uniform_filter(image_band, size=(33, 33, 1), mode="reflect")
    mt = uniform_filter(t, size=(33, 33, 1), mode="reflect")
    cov = uniform_filter((image_band * t).sum(2), 33, mode="reflect") - (mx * mt).sum(2)
    vx = uniform_filter((image_band * image_band).sum(2), 33, mode="reflect") - (mx * mx).sum(2)
    vt = uniform_filter((t * t).sum(2), 33, mode="reflect") - (mt * mt).sum(2)
    corr = cov / (np.sqrt(np.maximum(vx, 0) * np.maximum(vt, 0)) + 1e-12)
    return np.asarray(Image.fromarray(corr.astype(np.float32)).resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)


def build_encoder(models_dir, out, first_native, first_message, device):
    import onnx
    import onnxruntime as ort
    if device.startswith("cuda"):
        ort.preload_dlls()
    source = models_dir / "stegastamp/stega_stamp_logits.onnx"
    destination = out / "stega_encoder_only.onnx"
    # Materially remove decoder nodes; fetching only residual from the full model
    # would still execute its other public-output branches in inference ORT.
    onnx.utils.extract_model(str(source), str(destination), ["image", "secret"],
                             ["residual"], check_model=False)
    providers = ([('CUDAExecutionProvider', {"device_id": torch.device(device).index or 0})]
                 if device.startswith("cuda") else []) + ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    encoder = ort.InferenceSession(str(destination), sess_options=opts, providers=providers)
    if device.startswith("cuda") and "CUDAExecutionProvider" not in encoder.get_providers():
        raise RuntimeError("CUDA requested but encoder session has no CUDAExecutionProvider")
    original = ort.InferenceSession(str(source), sess_options=opts, providers=providers)
    if device.startswith("cuda") and "CUDAExecutionProvider" not in original.get_providers():
        raise RuntimeError("CUDA requested but reference session has no CUDAExecutionProvider")
    feed = {"image": np.ascontiguousarray(first_native[None]),
            "secret": np.asarray(first_message, dtype=np.float32)[None]}
    reference = original.run(["residual"], feed)[0]
    pruned = encoder.run(["residual"], feed)[0]
    difference = float(np.max(np.abs(reference - pruned)))
    if not np.isfinite(difference) or difference > 1e-5:
        raise RuntimeError(f"Encoder-only equivalence failed: max absolute difference={difference}")
    del original
    print(f"Encoder-only equivalent: max_abs_error={difference:.9g}; providers={encoder.get_providers()}", flush=True)
    return encoder, difference


def encoder_residuals(encoder, native, messages, batch_size):
    pieces = []
    for start in range(0, len(messages), batch_size):
        secret = np.asarray(messages[start:start + batch_size], dtype=np.float32)
        images = np.broadcast_to(native, (len(secret), *native.shape)).copy()
        pieces.append(encoder.run(["residual"], {"image": images, "secret": secret})[0])
    return np.concatenate(pieces).astype(np.float32)


def select_pilot(data, split_path, count):
    split = json.loads(split_path.read_text())
    dev_names, forbidden = set(split["train"]), set(split["val"])
    assert dev_names.isdisjoint(forbidden)
    metadata = json.loads((data / "train/metadata.json").read_text())
    candidates = [m for m in metadata if m["method"] == "stegastamp"
                  and m["image_name"] in dev_names and bool(m["full_watermark"])]
    # DEV masks determine area strata; only the selected 24 RGB files are opened.
    masks = {}
    for m in candidates:
        name = m["image_name"]
        with Image.open(data / "train/masks" / f"{name}.png") as im:
            masks[name] = downsample_mask((np.asarray(im) > 127).astype(np.uint8))
    candidates.sort(key=lambda m: (1 - masks[m["image_name"]].mean(), m["image_name"]))
    if count > len(candidates) or count < 5:
        raise ValueError(f"Need 5..{len(candidates)} pilot images, got {count}")
    rng = np.random.default_rng(SEED)
    selected = [candidates[int(rng.choice(group))] for group in np.array_split(np.arange(len(candidates)), count)]
    names = [m["image_name"] for m in selected]
    assert set(names) <= dev_names and set(names).isdisjoint(forbidden)
    return selected, np.stack([masks[name] for name in names])


def cross_validate(responses, masks, folds):
    predictions = np.zeros(masks.shape, dtype=np.uint8)
    thresholds = np.zeros(len(masks), dtype=np.float64)
    fold_reports = []
    for fold in range(5):
        train, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        threshold, _ = fit_threshold(responses[train], masks[train])
        predictions[val] = (responses[val] >= threshold).astype(np.uint8)
        thresholds[val] = threshold
        fold_reports.append({"fold": fold, "calibration_images": len(train), "evaluation_images": len(val),
                             "threshold": number(threshold), "miou": float(np.mean([
                                 miou_image(predictions[i], masks[i]) for i in val]))})
    scores = np.array([miou_image(pred, mask) for pred, mask in zip(predictions, masks)])
    return scores, predictions, thresholds, fold_reports


def save_pngs(out, selected, images, masks, true_response, wrong_response, predictions, scores):
    # Colors are for diagnostics only; they never affect CV normalization/decisions.
    limits = np.quantile(np.concatenate((true_response.ravel(), wrong_response.ravel())), [0.01, 0.99])
    (out / "examples").mkdir(exist_ok=True)
    for i in np.linspace(0, len(selected) - 1, min(6, len(selected)), dtype=int):
        image = Image.fromarray(images[i]).resize((256, 256))
        true_heat = colorize(true_response[i], *limits).resize((256, 256), Image.Resampling.BILINEAR)
        wrong_heat = colorize(wrong_response[i], *limits).resize((256, 256), Image.Resampling.BILINEAR)
        panels = [image, Image.blend(image, true_heat, 0.65), Image.blend(image, wrong_heat, 0.65),
                  mask_overlay(image, predictions[i]), mask_overlay(image, masks[i])]
        canvas = Image.new("RGB", (1280, 306), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((5, 3), f'DEV-only pilot / {selected[i]["image_name"]} / fivefold OOF mIoU={scores[i]:.4f}', fill="black")
        for j, (panel, title) in enumerate(zip(panels, ["Image", "True-message correlation", "Wrong-message correlation", "OOF prediction (pink=WM)", "Ground truth (pink=WM)"])):
            draw.text((j * 256 + 4, 20), title, fill="black")
            canvas.paste(panel, (j * 256, 38))
        draw.text((5, 294), f"Diagnostic common color limits: {limits[0]:.5f} .. {limits[1]:.5f}", fill="black")
        canvas.save(out / "examples" / f'{selected[i]["image_name"]}.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output/stega_template_probe")
    parser.add_argument("--pilot-images", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    selected, masks = select_pilot(args.data, args.split, args.pilot_images)
    images = []
    for item in selected:
        with Image.open(args.data / "train/images" / f'{item["image_name"]}.png') as im:
            images.append(np.asarray(im.convert("RGB")))
    images = np.stack(images)
    names = np.asarray([m["image_name"] for m in selected])
    messages = np.asarray([[int(c) for c in m["message"]] for m in selected], dtype=np.float32)
    wrong_messages = np.random.default_rng(SEED).integers(0, 2, (8, 100)).astype(np.float32)
    (args.out / "pilot.json").write_text(json.dumps({"pilot_only": True, "seed": SEED,
        "selection": "24 real DEV full_watermark=True images stratified by clean-hole area",
        "image_names": names.tolist(), "source_split": str(args.split),
        "holdout_rgb_images_read": 0, "test_rgb_images_read": 0}, indent=2))
    encoder, error = build_encoder(args.models_dir, args.out, native_image(images[0]), messages[0], args.device)
    true_templates, wrong_templates, true_maps, wrong_maps = [], [], [], []
    for i, image in enumerate(images):
        residuals = encoder_residuals(encoder, native_image(image),
                                     np.concatenate((messages[i:i + 1], wrong_messages)), args.batch_size)
        true_template = template_to_512(residuals[0] - residuals[1:].mean(0))
        # Independent negative-control background: no true message enters it.
        wrong_template = template_to_512(residuals[1] - residuals[2:].mean(0))
        band = dog(image.astype(np.float32) / 255)
        true_maps.append(local_correlation(band, true_template))
        wrong_maps.append(local_correlation(band, wrong_template))
        true_templates.append(true_template)
        wrong_templates.append(wrong_template)
        print(f"Pilot {i + 1}/{len(images)}: {names[i]}, elapsed={time.monotonic() - started:.1f}s", flush=True)
    true_response, wrong_response = np.stack(true_maps), np.stack(wrong_maps)
    # Area-ordered images cycle across five folds, keeping hole sizes distributed.
    fold_order = np.random.default_rng(SEED).permutation(5)
    folds = fold_order[np.arange(len(images)) % 5]
    true_scores, true_pred, true_thresholds, true_folds = cross_validate(true_response, masks, folds)
    wrong_scores, wrong_pred, wrong_thresholds, wrong_folds = cross_validate(wrong_response, masks, folds)
    rows = []
    for i, name in enumerate(names):
        wm, clean = masks[i] == 1, masks[i] == 0
        true_gap = float(true_response[i][wm].mean() - true_response[i][clean].mean())
        wrong_gap = float(wrong_response[i][wm].mean() - wrong_response[i][clean].mean())
        oracle = max(miou_image(np.zeros_like(masks[i]), masks[i]), miou_image(np.ones_like(masks[i]), masks[i]))
        rows.append({"image_name": str(name), "fold": int(folds[i]),
                     "clean_hole_fraction": float(1 - masks[i].mean()),
                     "true_message_cv_miou": float(true_scores[i]), "wrong_message_cv_miou": float(wrong_scores[i]),
                     "constant_label_oracle_miou": oracle,
                     "true_wm_minus_clean": true_gap, "true_gap_sign": int(np.sign(true_gap)),
                     "wrong_wm_minus_clean": wrong_gap, "wrong_gap_sign": int(np.sign(wrong_gap))})
    with (args.out / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.out / "responses.npz", image_names=names, masks128=masks, fold=folds,
        true_response=true_response, wrong_response=wrong_response,
        true_message=messages, wrong_messages=wrong_messages,
        true_oof_masks=true_pred, wrong_oof_masks=wrong_pred,
        true_oof_threshold=true_thresholds, wrong_oof_threshold=wrong_thresholds)
    np.savez_compressed(args.out / "templates.npz", image_names=names,
        true_template=np.stack(true_templates), wrong_template=np.stack(wrong_templates))
    report = {"pilot_only": True, "method": "StegaStamp message-centered encoder-template local correlation",
        "images": len(images), "group": "real DEV full_watermark=True only", "seed": SEED,
        "encoder_only_max_abs_error": error, "onnx_providers": encoder.get_providers(),
        "parameters": {"encoder_input_resize": "512->256->400 bicubic", "random_messages": 8,
                       "template_resize": "400->256->512 bilinear", "DoG_sigma": [1, 3], "NCC_window": 33},
        "cross_validation": "fivefold within 24 DEV images; thresholds fitted on other folds only",
        "holdout_images_read": 0, "test_images_read": 0,
        "true_message_cv_miou": float(true_scores.mean()), "wrong_message_cv_miou": float(wrong_scores.mean()),
        "constant_label_oracle_miou": float(np.mean([r["constant_label_oracle_miou"] for r in rows])),
        "true_mean_wm_minus_clean": float(np.mean([r["true_wm_minus_clean"] for r in rows])),
        "true_positive_gap_images": sum(r["true_gap_sign"] > 0 for r in rows),
        "wrong_mean_wm_minus_clean": float(np.mean([r["wrong_wm_minus_clean"] for r in rows])),
        "wrong_positive_gap_images": sum(r["wrong_gap_sign"] > 0 for r in rows),
        "true_folds": true_folds, "wrong_folds": wrong_folds,
        "seconds": time.monotonic() - started}
    save_pngs(args.out, selected, images, masks, true_response, wrong_response, true_pred, true_scores)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
