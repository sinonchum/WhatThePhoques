"""TrustMark's pretrained rectangular localizer, verified with the known message.

Only the original 136-image dev split selects Q/P and the final threshold.
The 34-image holdout is evaluated after these choices are fixed; test is inference.
No embedding, fitting of detector weights, or leaderboard requests.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from message_heatmaps import threshold_scores
from wmloc.utils import miou_image


def load_official_class(repo):
    source = (repo / "python").resolve()
    assert (source / "trustmark/trustmark.py").is_file(), source
    # Nothing in this script imports the environment's old trustmark beforehand.
    assert "trustmark" not in sys.modules, "Run in a fresh process to isolate the new TrustMark"
    sys.path.insert(0, str(source))
    module = importlib.import_module("trustmark")
    assert Path(module.__file__).resolve().is_relative_to(source), module.__file__
    print(f"isolated TrustMark import: {module.__file__}", flush=True)
    return module.TrustMark


def load_model(cls, variant, device, detector):
    model = cls(model_type=variant, use_ECC=False, secret_len=100, device=device,
                verbose=False, loadRemover=False, loadBBoxDetector=detector)
    assert model.decoder is not None, f"Decoder failed to load: {variant}"
    if detector:
        assert model.detector is not None, f"BBox detector failed to load: {variant}"
    return model


@torch.inference_mode()
def raw_logits(model, pictures, batch_size):
    """Same pre-threshold entry point as official TrustMark.subimage_decode()."""
    resolution = model.model_resolution_dec  # Q:245; P:224. Wrapper may resize internally.
    output = []
    for start in range(0, len(pictures), batch_size):
        images = np.stack([np.asarray(im.resize((resolution, resolution), Image.Resampling.BILINEAR),
                                      dtype=np.float32).transpose(2, 0, 1)
                           for im in pictures[start:start + batch_size]])
        x = torch.from_numpy(images).to(model.device) / 127.5 - 1
        logits = model.decoder.decoder(x)
        assert logits.shape == (len(x), 100), logits.shape
        output.append(logits.float().cpu().numpy())
    return np.concatenate(output)


def message_statistics(logits, message):
    signed = logits * (2 * np.asarray(message, dtype=np.float32) - 1)
    # Stable sigmoid without clipping or overflow.
    soft = np.exp(-np.logaddexp(0, -signed)).mean(axis=-1)
    bitacc = ((logits > 0) == np.asarray(message, dtype=bool)).mean(axis=-1)
    margin = signed.mean(axis=-1)
    return soft, bitacc, margin


def open_image(data, split, name):
    with Image.open(data / split / "images" / f"{name}.png") as image:
        return image.convert("RGB")


def variant_scores(args, cls, names, meta):
    by_variant = {}
    logits_all = {}
    for variant in ("Q", "P"):
        model = load_model(cls, variant, args.device, detector=False)
        pictures = [model.get_the_image_for_processing(open_image(args.data, "train", name)) for name in names]
        logits = raw_logits(model, pictures, args.batch_size)
        messages = np.array([[int(bit) for bit in meta[name]["message"]] for name in names])
        soft, bitacc, margin = message_statistics(logits, messages)
        by_variant[variant] = {"bitacc": bitacc, "soft": soft, "margin": margin}
        logits_all[variant] = logits
        print(f"dev whole-image {variant}: bitacc={bitacc.mean():.5f}, signed_margin={margin.mean():.5f}", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    # Five fixed, area-type-balanced folds; only message agreement chooses the variant.
    fold_ids = np.empty(len(names), dtype=int)
    rng = np.random.default_rng(2026)
    for full in (False, True):
        ids = np.array([i for i, name in enumerate(names) if bool(meta[name]["full_watermark"]) == full])
        rng.shuffle(ids)
        for fold, group in enumerate(np.array_split(ids, 5)):
            fold_ids[group] = fold
    folds = []
    for fold in range(5):
        train, test = fold_ids != fold, fold_ids == fold
        chosen = max(("Q", "P"), key=lambda v: (by_variant[v]["bitacc"][train].mean(), by_variant[v]["soft"][train].mean()))
        folds.append({"fold": fold, "train_images": int(train.sum()), "test_images": int(test.sum()),
                      "chosen_on_other_folds": chosen,
                      "test_bitacc": {v: float(by_variant[v]["bitacc"][test].mean()) for v in ("Q", "P")},
                      "chosen_test_bitacc": float(by_variant[chosen]["bitacc"][test].mean())})
    selected = max(("Q", "P"), key=lambda v: (by_variant[v]["bitacc"].mean(), by_variant[v]["soft"].mean()))
    summary = {"selected": selected, "criterion": "mean whole-image known-message bit accuracy; soft agreement breaks ties",
               "full_dev": {v: {key: float(values.mean()) for key, values in by_variant[v].items()} for v in ("Q", "P")},
               "five_fold_dev": folds}
    return summary, logits_all, fold_ids


def box_coverage(box, width, height):
    x1, y1, x2, y2 = np.array(box, dtype=np.float32) / np.array([width, height, width, height]) * 128
    cells = np.arange(128)
    x = np.clip(np.minimum(cells + 1, x2) - np.maximum(cells, x1), 0, 1)
    y = np.clip(np.minimum(cells + 1, y2) - np.maximum(cells, y1), 0, 1)
    return (y[:, None] * x[None, :]).astype(np.float32)


@torch.inference_mode()
def infer_maps(args, model, names, meta, split):
    responses, rows = [], []
    for i, name in enumerate(names):
        image = open_image(args.data, split, name)
        width, height = image.size
        # The API returns None or normalized [x1,y1,x2,y2] boxes, without detector scores.
        detected = model.localize(image, return_all=True) or []
        boxes = [(0, 0, width, height)]  # Fixed whole-image candidate covers large-watermark cases.
        for normalized in detected:
            x1, y1, x2, y2 = np.clip(np.asarray(normalized, dtype=float), 0, 1)
            box = (max(0, int(np.floor(x1 * width))), max(0, int(np.floor(y1 * height))),
                   min(width, int(np.ceil(x2 * width))), min(height, int(np.ceil(y2 * height))))
            if box[2] > box[0] and box[3] > box[1] and box not in boxes:
                boxes.append(box)
        logits = raw_logits(model, [image.crop(box) for box in boxes], args.batch_size)
        message = np.array([int(bit) for bit in meta[name]["message"]])
        soft, bitacc, margin = message_statistics(logits, message)
        # Known-message soft likelihood selects one region; no mask labels are used here.
        best = max(range(len(boxes)), key=lambda j: (float(soft[j]), float(bitacc[j]), float(margin[j])))
        confidence = max(0.0, float(2 * soft[best] - 1))
        responses.append(box_coverage(boxes[best], width, height) * confidence)
        rows.append({"image_name": name, "detector_boxes": len(detected), "selected_candidate": best,
                     "selected_is_whole_image": best == 0, "response_inside": confidence,
                     "candidates": [{"box_xyxy": list(box), "soft_agreement": float(soft[j]),
                                      "bitacc": float(bitacc[j]), "signed_margin": float(margin[j])}
                                     for j, box in enumerate(boxes)]})
        if (i + 1) % 20 == 0 or i + 1 == len(names):
            print(f"{split}: inferred {i+1}/{len(names)}", flush=True)
    return np.asarray(responses, dtype=np.float32), rows


def save_response(path, names, response, threshold):
    assert response.shape == (len(names), 128, 128) and np.isfinite(response).all()
    np.savez_compressed(path, image_names=np.asarray(names, dtype=str), response=response.astype(np.float32),
                        threshold=np.asarray(threshold, dtype=np.float64), dev_in_sample=np.asarray(True))


def group_report(names, response, threshold, meta, gt, rows):
    scores = np.array([miou_image((r >= threshold).astype(np.uint8), gt[name]) for name, r in zip(names, response)])
    full = np.array([bool(meta[name]["full_watermark"]) for name in names])
    for row, score in zip(rows, scores):
        row["full_watermark"] = bool(meta[row["image_name"]]["full_watermark"])
        row["miou"] = float(score)
    return {"large": float(scores[full].mean()), "small": float(scores[~full].mean()),
            "both": float(scores.mean()), "n": len(names),
            "n_large": int(full.sum()), "n_small": int((~full).sum())}


def choose_threshold(response, gt_masks):
    # Include finite all-one / all-zero choices; fusion responses and cuts stay finite.
    upper = float(np.nextafter(np.float32(response.max()), np.float32(np.inf)))
    candidates = np.unique(np.r_[0.0, np.quantile(response, np.linspace(0, 1, 257)), upper])
    scores = threshold_scores(response, gt_masks, candidates)
    return float(candidates[np.argmax(scores)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--trustmark-repo", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("output/fusion/codex_tm_localizer_trustmark"))
    p.add_argument("--split", type=Path, default=Path("output/message_heatmaps/trustmark/split.json"))
    p.add_argument("--stage", choices=["all", "dev", "final"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(4)
    cls = load_official_class(args.trustmark_repo)
    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics = args.out / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    split = json.loads(args.split.read_text())
    dev, holdout = split["train"], split["val"]
    assert len(dev) == 136 and len(holdout) == 34 and set(dev).isdisjoint(holdout)
    metadata = {row["image_name"]: row for row in json.loads((args.data / "train/metadata.json").read_text())}
    assert all(metadata[name]["method"] == "trustmark" for name in dev + holdout)
    with np.load(args.data / "train/masks_128.npz", allow_pickle=False) as data:
        gt = dict(zip(data["names"].tolist(), data["masks"]))
    (diagnostics / "split.json").write_text(json.dumps(split, indent=2))
    report_path = diagnostics / "report.json"
    if args.stage in ("all", "dev"):
        selection, logits, folds = variant_scores(args, cls, dev, metadata)
        np.savez_compressed(diagnostics / "dev_variant_logits.npz", image_names=np.array(dev),
                            Q=logits["Q"], P=logits["P"], fold=folds)
        print(json.dumps(selection, indent=2), flush=True)
        model = load_model(cls, selection["selected"], args.device, detector=True)
        response, rows = infer_maps(args, model, dev, metadata, "train")
        threshold = choose_threshold(response, np.stack([gt[name] for name in dev]))
        report = {"method": "TrustMark pretrained localizer + known-message candidate verification",
                  "variant_selection": selection, "threshold": threshold, "dev_in_sample": True,
                  "response_definition": "best candidate by mean sigmoid((2b-1)*logit); box coverage * max(0,2*score-1)",
                  "candidates": "official localize(return_all=True) boxes plus fixed whole-image box",
                  "dev": group_report(dev, response, threshold, metadata, gt, rows)}
        cv_scores = np.empty(len(dev), dtype=np.float64)
        cv_folds = []
        for fold in range(5):
            train_ids, test_ids = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            cut = choose_threshold(response[train_ids], np.stack([gt[dev[i]] for i in train_ids]))
            cv_scores[test_ids] = [miou_image(response[i] >= cut, gt[dev[i]]) for i in test_ids]
            cv_folds.append({"fold": fold, "threshold": cut, "images": len(test_ids),
                             "miou": float(cv_scores[test_ids].mean())})
        large = np.array([bool(metadata[name]["full_watermark"]) for name in dev])
        report["dev_threshold_cv"] = {"large": float(cv_scores[large].mean()),
            "small": float(cv_scores[~large].mean()), "both": float(cv_scores.mean()),
            "folds": cv_folds, "note": "threshold cross-validation for the selected official variant"}
        save_response(args.out / "dev.npz", dev, response, threshold)
        (diagnostics / "dev_rows.json").write_text(json.dumps(rows, indent=2))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print("DEV large / small / both:", report["dev"], flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if args.stage in ("all", "final"):
        report = json.loads(report_path.read_text())
        with np.load(args.out / "dev.npz", allow_pickle=False) as saved:
            assert saved["image_names"].tolist() == dev and float(saved["threshold"]) == report["threshold"]
        model = load_model(cls, report["variant_selection"]["selected"], args.device, detector=True)
        response, rows = infer_maps(args, model, holdout, metadata, "train")
        threshold = float(report["threshold"])
        save_response(args.out / "holdout.npz", holdout, response, threshold)
        report["holdout"] = group_report(holdout, response, threshold, metadata, gt, rows)
        (diagnostics / "holdout_rows.json").write_text(json.dumps(rows, indent=2))
        print("HOLDOUT large / small / both:", report["holdout"], flush=True)
        test_meta = {row["image_name"]: row for row in json.loads((args.data / "test/metadata.json").read_text())
                     if row["method"] == "trustmark"}
        test_names = sorted(test_meta)
        assert len(test_names) == 340 and set(test_names).isdisjoint(dev + holdout)
        response, rows = infer_maps(args, model, test_names, test_meta, "test")
        save_response(args.out / "test.npz", test_names, response, threshold)
        (diagnostics / "test_rows.json").write_text(json.dumps(rows, indent=2))
        report["test"] = {"n": len(test_names), "inference_only": True}
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"Exported dev=136, holdout=34, test=340 -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
