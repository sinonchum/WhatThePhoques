"""Export fixed TrustMark-Q or PixelSeal crop search on DEV, then holdout and test.

Search is exactly the crop_search.py pilot design, applied without mask-based
image filtering. Only 136 DEV masks fit the scalar threshold. Holdout masks are
opened for scoring after the DEV report and response have been saved.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from queue import SimpleQueue

import numpy as np
import torch
from PIL import Image

from crop_search import CropDecoder, REFINE_STEPS, SCALES, search
from trustmark_localizer import box_coverage, choose_threshold
from wmloc.utils import miou_image


def load_gt(data, names):
    with np.load(data / "train/masks_128.npz", allow_pickle=False) as masks:
        index = {name: i for i, name in enumerate(masks["names"].tolist())}
        all_masks = masks["masks"]
        return {name: all_masks[index[name]] for name in names}


def save_response(path, names, response, threshold, rows):
    response = np.asarray(response, np.float32)
    assert response.shape == (len(names), 128, 128) and np.isfinite(response).all()
    assert [row["image_name"] for row in rows] == list(names)
    box_score = np.asarray([row["signed_raw_margin"] for row in rows], dtype=np.float32)
    np.savez_compressed(path, image_names=np.asarray(names, dtype=str), response=response,
                        threshold=np.asarray(threshold, dtype=np.float64), dev_in_sample=np.asarray(True),
                        box_score=box_score)


def group_scores(names, response, threshold, metadata, gt, rows=None):
    scores = np.array([miou_image(heat >= threshold, gt[name]) for name, heat in zip(names, response)])
    large = np.array([bool(metadata[name]["full_watermark"]) for name in names])
    if rows is not None:
        for row, value, full in zip(rows, scores, large):
            row["miou"] = float(value)
            row["full_watermark"] = bool(full)
    return {"large": float(scores[large].mean()), "small": float(scores[~large].mean()),
            "both": float(scores.mean()), "n": len(names), "n_large": int(large.sum()),
            "n_small": int((~large).sum())}


def threshold_cv(names, response, metadata, gt):
    rng = np.random.default_rng(2026)
    fold_ids = np.empty(len(names), dtype=int)
    for full in (False, True):
        ids = np.array([i for i, name in enumerate(names) if bool(metadata[name]["full_watermark"]) == full])
        rng.shuffle(ids)
        for fold, group in enumerate(np.array_split(ids, 5)):
            fold_ids[group] = fold
    scores = np.empty(len(names), dtype=float)
    thresholds = np.empty(len(names), dtype=float)
    folds = []
    for fold in range(5):
        train_ids, val_ids = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        cut = choose_threshold(response[train_ids], np.stack([gt[names[i]] for i in train_ids]))
        scores[val_ids] = [miou_image(response[i] >= cut, gt[names[i]]) for i in val_ids]
        thresholds[val_ids] = cut
        large = np.array([bool(metadata[names[i]]["full_watermark"]) for i in val_ids])
        folds.append({"fold": fold, "threshold_fit_images": len(train_ids), "threshold": cut,
                      "large": float(scores[val_ids][large].mean()),
                      "small": float(scores[val_ids][~large].mean()),
                      "both": float(scores[val_ids].mean()), "n": len(val_ids),
                      "n_large": int(large.sum()), "n_small": int((~large).sum())})
    large = np.array([bool(metadata[name]["full_watermark"]) for name in names])
    report = {"large": float(scores[large].mean()), "small": float(scores[~large].mean()),
              "both": float(scores.mean()), "n": len(names), "n_large": int(large.sum()),
              "n_small": int((~large).sum()), "folds": folds,
              "note": "Five fixed DEV folds stratified by full_watermark; only threshold is fitted on the other four folds. Search and official decoder are fixed."}
    return report, fold_ids, scores, thresholds


def infer_image(args, decoder, name, metadata, image_split, label):
    with Image.open(args.data / image_split / "images" / f"{name}.png") as source:
        image = np.asarray(source.convert("RGB"))
    assert image.shape == (512, 512, 3), (name, image.shape)
    x = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div(255).to(decoder.device)
    message = np.array([int(bit) for bit in metadata[name]["message"]], np.uint8)
    image_started = time.monotonic()
    best, scores, coarse_count, proposals = search(decoder, x, message, aspect_limit=args.aspect_limit)
    margin, accuracy = scores[best]
    response = box_coverage(best, 512, 512) * max(0.0, margin)
    row = {"image_name": name, "method": args.method, "split": label,
           "best_bbox_xyxy": list(best), "signed_raw_margin": margin,
           "message_accuracy": accuracy, "response_inside": max(0.0, margin),
           "selected_is_whole_image": best == (0, 0, 512, 512),
           "whole_image_margin": scores[(0, 0, 512, 512)][0],
           "whole_image_message_accuracy": scores[(0, 0, 512, 512)][1],
           "coarse_candidates": coarse_count, "refinement_candidates": len(scores) - coarse_count,
           "total_search_candidates": len(scores),
           "search_seconds": time.monotonic() - image_started,
           "coarse_top4_proposals": proposals}
    return response.astype(np.float32), row


def infer_split(args, decoders, names, metadata, image_split, label):
    responses, rows = [], []
    started = time.monotonic()
    live_path = args.out / "diagnostics" / f"{label}_rows.jsonl"
    available = SimpleQueue()
    for decoder in decoders:
        available.put(decoder)

    def worker(name):
        decoder = available.get()
        try:
            return infer_image(args, decoder, name, metadata, image_split, label)
        finally:
            available.put(decoder)

    with live_path.open("w") as live, ExitStack() as stack:
        if len(decoders) > 1:
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=len(decoders)))
            results = pool.map(worker, names)
        else:
            results = map(worker, names)
        # executor.map yields in the fixed split order even when GPUs finish out of order.
        for i, (response, row) in enumerate(results):
            responses.append(response)
            rows.append(row)
            live.write(json.dumps(row, allow_nan=False) + "\n")
            live.flush()
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(names):
                print(f'{label}: {i+1}/{len(names)}, {time.monotonic()-started:.1f}s, last={row["image_name"]}, margin={row["signed_raw_margin"]:.4f}, bits={row["message_accuracy"]:.3f}, candidates={row["total_search_candidates"]}', flush=True)
    return np.asarray(responses, np.float32), rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--models-dir", type=Path, required=True)
    p.add_argument("--method", choices=("trustmark", "pixelseal"), default="trustmark")
    p.add_argument("--out", type=Path)
    p.add_argument("--split", type=Path)
    p.add_argument("--stage", choices=("all", "dev", "final"), default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gpus", type=int, default=1, help="Independent CUDA devices; batch-size applies to each GPU")
    args = p.parse_args()
    if args.gpus < 1 or args.batch_size < 1:
        p.error("--gpus and --batch-size must be positive")
    args.out = args.out or Path(__file__).parent / f"output/fusion/codex_cropsearch_{args.method}"
    args.split = args.split or Path(__file__).parent / f"output/message_heatmaps/{args.method}/split.json"
    args.aspect_limit = 2 if args.method == "trustmark" else 16
    device = torch.device(args.device)
    devices = [args.device]
    if args.gpus > 1:
        first_gpu = device.index or 0
        if device.type != "cuda" or first_gpu + args.gpus > torch.cuda.device_count():
            p.error("--gpus > 1 requires that many visible CUDA devices starting at --device")
        devices = [f"cuda:{i}" for i in range(first_gpu, first_gpu + args.gpus)]
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics = args.out / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    split = json.loads(args.split.read_text())
    dev, holdout = split["train"], split["val"]
    assert len(dev) == 136 and len(holdout) == 34 and set(dev).isdisjoint(holdout)
    metadata = {row["image_name"]: row for row in json.loads((args.data / "train/metadata.json").read_text())}
    assert all(metadata[name]["method"] == args.method for name in dev + holdout)
    (diagnostics / "split.json").write_text(json.dumps(split, indent=2))
    decoder_name = "TrustMark-Q" if args.method == "trustmark" else "PixelSeal"
    print(f"Fixed search; official wmloc {decoder_name} decoder; devices={devices}; per-device batch={args.batch_size}.", flush=True)
    started = time.monotonic()
    decoders = [CropDecoder(args.method, args.models_dir.resolve(), dev, args.batch_size) for dev in devices]
    initialization_seconds = time.monotonic() - started
    report_path = diagnostics / "report.json"
    if args.stage in ("all", "dev"):
        response, rows = infer_split(args, decoders, dev, metadata, "train", "dev")
        gt = load_gt(args.data, dev)
        threshold = choose_threshold(response, np.stack([gt[name] for name in dev]))
        cv, fold_ids, cv_scores, cv_thresholds = threshold_cv(dev, response, metadata, gt)
        report = {"method": f"{decoder_name} fixed multiscale rectangular crop search + edge refinement",
                  "watermark_method": args.method,
                  "variant": "Q" if args.method == "trustmark" else "official-256bit",
                  "variant_selection": "Fixed from prior 136-DEV-only Q/P comparison; not reselected here" if args.method == "trustmark" else "Official PixelSeal weights; no variant selection",
                  "response_definition": "128x128 exact rectangle-cell coverage * max(0, mean((2*message-1)*raw_logits))",
                  "search_design": {"scales": list(SCALES), "coarse_stride": "max(32,min(width,height)/2)",
                                    "aspect_limit": args.aspect_limit, "top_proposals": 4, "proposal_nms_iou": 0.5,
                                    "refine_steps": list(REFINE_STEPS), "passes_per_step": 2,
                                    "whole_image_candidate": True},
                  "threshold": threshold, "threshold_fit_images": len(dev), "dev_in_sample": True,
                  "initialization_seconds": initialization_seconds,
                  "devices": devices, "batch_size_per_device": args.batch_size,
                  "dev": group_scores(dev, response, threshold, metadata, gt, rows),
                  "dev_threshold_cv": cv,
                  "dev_search_seconds": float(sum(row["search_seconds"] for row in rows)),
                  "dev_mean_search_candidates": float(np.mean([row["total_search_candidates"] for row in rows]))}
        save_response(args.out / "dev.npz", dev, response, threshold, rows)
        np.savez_compressed(diagnostics / "dev_threshold_cv.npz", image_names=np.asarray(dev), fold=fold_ids,
                            threshold=cv_thresholds, miou=cv_scores)
        (diagnostics / "dev_rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print("DEV report saved before opening holdout masks:", json.dumps(report, allow_nan=False), flush=True)
    if args.stage in ("all", "final"):
        report = json.loads(report_path.read_text())
        assert report.get("watermark_method", "trustmark") == args.method
        threshold = float(report["threshold"])
        with np.load(args.out / "dev.npz", allow_pickle=False) as cache:
            assert cache["image_names"].tolist() == dev and float(cache["threshold"]) == threshold
        response, rows = infer_split(args, decoders, holdout, metadata, "train", "holdout")
        save_response(args.out / "holdout.npz", holdout, response, threshold, rows)
        report["holdout"] = group_scores(holdout, response, threshold, metadata, load_gt(args.data, holdout), rows)
        report["holdout_search_seconds"] = float(sum(row["search_seconds"] for row in rows))
        (diagnostics / "holdout_rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print("HOLDOUT final large/small/both:", json.dumps(report["holdout"]), flush=True)
        test_metadata = {row["image_name"]: row for row in json.loads((args.data / "test/metadata.json").read_text())
                         if row["method"] == args.method}
        test_names = sorted(test_metadata)
        assert len(test_names) == 340 and set(test_names).isdisjoint(dev + holdout)
        response, rows = infer_split(args, decoders, test_names, test_metadata, "test", "test")
        save_response(args.out / "test.npz", test_names, response, threshold, rows)
        (diagnostics / "test_rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        report["test_images"] = len(test_names)
        report["test_search_seconds"] = float(sum(row["search_seconds"] for row in rows))
        report["total_process_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print("All D exports complete:", str(args.out), flush=True)


if __name__ == "__main__":
    main()
