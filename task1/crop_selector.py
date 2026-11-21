"""Fixed tiny selector over clean / full-watermark / searched-rectangle candidates.

DEV: true five-fold OOF predictions, followed by a full-DEV ridge fit for FINAL.
FINAL: read holdout/test only after DEV has finished; never refit or change cuts.
No GPU, new search, watermark embedding, or leaderboard access is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from wmloc.utils import miou_image

RIDGE = 10.0
TEMPERATURE = 0.05
THRESHOLD = 0.5
FEATURE_NAMES = (
    "whole_image_margin", "whole_image_message_accuracy", "signed_raw_margin",
    "message_accuracy", "bbox_area_fraction", "bbox_log_width_over_height", "best_minus_whole_margin",
)
CANDIDATE_NAMES = ("all_clean", "all_watermarked", "search_bbox")


def features_and_boxes(rows):
    """Only seven explicitly named, inference-available numbers enter the model."""
    features, coverage = [], []
    grid = np.arange(128, dtype=np.float64)
    for row in rows:
        x0, y0, x1, y1 = map(float, row["best_bbox_xyxy"])
        assert 0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512
        width, height = x1 - x0, y1 - y0
        whole, best = float(row["whole_image_margin"]), float(row["signed_raw_margin"])
        features.append([whole, float(row["whole_image_message_accuracy"]), best,
                         float(row["message_accuracy"]), width * height / (512 * 512),
                         float(np.log(width / height)), best - whole])
        # Exact cell coverage, independently of the parent's raw-margin response.
        x = np.clip(np.minimum(grid + 1, x1 / 4) - np.maximum(grid, x0 / 4), 0, 1)
        y = np.clip(np.minimum(grid + 1, y1 / 4) - np.maximum(grid, y0 / 4), 0, 1)
        coverage.append(y[:, None] * x[None, :])
    x = np.asarray(features, dtype=np.float64)
    assert x.shape == (len(rows), 7) and np.isfinite(x).all()
    return x, np.asarray(coverage, dtype=np.float32)


def fit_ridge(x, target):
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    penalty = np.diag([0.0] + [RIDGE] * x.shape[1])
    coefficient = np.linalg.solve(z.T @ z + penalty, z.T @ target)
    return mean, scale, coefficient


def predict_ridge(x, model):
    mean, scale, coefficient = model
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    predicted_iou = z @ coefficient
    logits = predicted_iou / TEMPERATURE
    weights = np.exp(logits - logits.max(1, keepdims=True))
    weights /= weights.sum(1, keepdims=True)
    return predicted_iou, weights


def mix_response(weights, coverage):
    return (weights[:, 1, None, None] + weights[:, 2, None, None] * coverage).astype(np.float32)


def save_response(path, names, response):
    assert response.shape == (len(names), 128, 128) and np.isfinite(response).all()
    np.savez_compressed(path, image_names=np.asarray(names, dtype=str), response=response.astype(np.float32),
                        threshold=np.asarray(THRESHOLD, dtype=np.float64), dev_in_sample=np.asarray(True))


def load_source(source, label):
    # This never opens the parent's combined report or any other split.
    with np.load(source / f"{label}.npz", allow_pickle=False) as data:
        names = data["image_names"].tolist()
        response = data["response"].astype(np.float32)
        threshold = float(data["threshold"])
    raw_rows = json.loads((source / "diagnostics" / f"{label}_rows.json").read_text())
    index = {row["image_name"]: row for row in raw_rows}
    assert len(names) == len(set(names)) == len(raw_rows) == len(index)
    assert set(names) == set(index)
    x, coverage = features_and_boxes([index[name] for name in names])
    return names, response, threshold, x, coverage


def load_gt(data, names):
    with np.load(data / "train/masks_128.npz", allow_pickle=False) as masks:
        index = {name: i for i, name in enumerate(masks["names"].tolist())}
        selected = masks["masks"][[index[name] for name in names]]
    return dict(zip(names, selected))


def score_maps(names, response, threshold, gt):
    return np.asarray([miou_image(r >= threshold, gt[name]) for name, r in zip(names, response)])


def group_report(names, scores, metadata):
    full = np.asarray([bool(metadata[name]["full_watermark"]) for name in names])
    return {"large": float(scores[full].mean()), "small": float(scores[~full].mean()),
            "both": float(scores.mean()), "n": len(names), "n_large": int(full.sum()),
            "n_small": int((~full).sum())}


def dev_stage(args, diagnostics, split, metadata):
    names, original, original_threshold, x, coverage = load_source(args.source, "dev")
    assert names == split["train"] and len(names) == 136
    assert all(metadata[name]["method"] == "trustmark" for name in names)
    gt = load_gt(args.data, names)
    clean, full = np.zeros((128, 128), dtype=np.uint8), np.ones((128, 128), dtype=np.uint8)
    target = np.asarray([[miou_image(clean, gt[name]), miou_image(full, gt[name]),
                          miou_image(box >= 0.5, gt[name])] for name, box in zip(names, coverage)])
    rng = np.random.default_rng(2026)
    fold_ids = np.empty(len(names), dtype=np.int64)
    for area_type in (False, True):
        ids = np.asarray([i for i, name in enumerate(names) if bool(metadata[name]["full_watermark"]) == area_type])
        rng.shuffle(ids)
        for fold, ids_fold in enumerate(np.array_split(ids, 5)):
            fold_ids[ids_fold] = fold
    oof_iou, oof_weights = np.empty_like(target), np.empty_like(target)
    folds = []
    for fold in range(5):
        train, test = fold_ids != fold, fold_ids == fold
        # Means, scales, intercept and coefficients see only the other four folds.
        model = fit_ridge(x[train], target[train])
        oof_iou[test], oof_weights[test] = predict_ridge(x[test], model)
        fold_names = [name for name, selected in zip(names, test) if selected]
        scores = score_maps(fold_names, mix_response(oof_weights[test], coverage[test]), THRESHOLD, gt)
        folds.append({"fold": fold, "fit_images": int(train.sum()),
                      **group_report(fold_names, scores, metadata)})
    response = mix_response(oof_weights, coverage)
    scores = score_maps(names, response, THRESHOLD, gt)
    report = {"method": "fixed 3-candidate ridge selector on TrustMark crop-search evidence",
              "ridge_lambda": RIDGE, "softmax_temperature": TEMPERATURE, "threshold": THRESHOLD,
              "feature_names": FEATURE_NAMES, "candidate_names": CANDIDATE_NAMES,
              "dev_response_role": "true five-fold OOF; each scaler/intercept/coefficient fitted on the other four folds",
              "final_model_role": "refitted once on all 136 dev images, only for final holdout/test inference",
              "dev_in_sample": True, "dev_in_sample_note": "conservative flag as requested; dev.npz itself contains OOF predictions",
              "dev_oof": group_report(names, scores, metadata), "folds": folds,
              "original_search_dev": {"role": "parent dev response with its dev-fitted threshold",
                    "threshold": original_threshold,
                    **group_report(names, score_maps(names, original, original_threshold, gt), metadata)},
              "three_candidate_oracle_dev": {"role": "upper bound using dev ground truth, not a deployable method",
                    **group_report(names, target.max(1), metadata)}}
    original_cv = args.source / "diagnostics/dev_threshold_cv.npz"
    if original_cv.is_file():
        with np.load(original_cv, allow_pickle=False) as cv:
            cv_index = {name: i for i, name in enumerate(cv["image_names"].tolist())}
            order = [cv_index[name] for name in names]
            baseline_cv_scores = cv["miou"][order]
        report["original_search_dev_threshold_cv"] = group_report(names, baseline_cv_scores, metadata)
        report["oof_gain_vs_original_cv"] = float(scores.mean() - baseline_cv_scores.mean())
    save_response(args.out / "dev.npz", names, response)
    np.savez_compressed(diagnostics / "dev_oof.npz", image_names=np.asarray(names), fold=fold_ids,
                        features=x, target_candidate_iou=target, predicted_candidate_iou=oof_iou,
                        candidate_weights=oof_weights, miou=scores)
    mean, scale, coefficient = fit_ridge(x, target)
    np.savez_compressed(args.out / "model.npz", feature_names=np.asarray(FEATURE_NAMES),
                        candidate_names=np.asarray(CANDIDATE_NAMES), mean=mean, scale=scale,
                        coefficient=coefficient, train_names=np.asarray(names), ridge_lambda=RIDGE,
                        temperature=TEMPERATURE, threshold=THRESHOLD)
    (diagnostics / "dev_rows.json").write_text(json.dumps([
        {"image_name": name, "features": dict(zip(FEATURE_NAMES, values.tolist())),
         "oof_predicted_candidate_iou": pred.tolist(), "oof_candidate_weights": weight.tolist(),
         "fold": int(fold), "miou": float(score)}
        for name, values, pred, weight, fold, score in zip(names, x, oof_iou, oof_weights, fold_ids, scores)
    ], indent=2, allow_nan=False))
    (diagnostics / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def final_stage(args, diagnostics, split, metadata):
    report_path = diagnostics / "report.json"
    report = json.loads(report_path.read_text())
    with np.load(args.out / "model.npz", allow_pickle=False) as fitted:
        assert fitted["train_names"].tolist() == split["train"]
        assert fitted["feature_names"].tolist() == list(FEATURE_NAMES)
        assert float(fitted["temperature"]) == TEMPERATURE and float(fitted["threshold"]) == THRESHOLD
        model = (fitted["mean"].copy(), fitted["scale"].copy(), fitted["coefficient"].copy())
    # Only this explicit stage opens parent holdout/test rows and responses.
    for label in ("holdout", "test"):
        names, _, _, x, coverage = load_source(args.source, label)
        if label == "holdout":
            assert names == split["val"] and len(names) == 34
        else:
            assert len(names) == 340 and set(names).isdisjoint(split["train"] + split["val"])
        predicted_iou, weights = predict_ridge(x, model)
        response = mix_response(weights, coverage)
        save_response(args.out / f"{label}.npz", names, response)
        rows = [{"image_name": name, "features": dict(zip(FEATURE_NAMES, values.tolist())),
                 "predicted_candidate_iou": pred.tolist(), "candidate_weights": weight.tolist()}
                for name, values, pred, weight in zip(names, x, predicted_iou, weights)]
        if label == "holdout":
            scores = score_maps(names, response, THRESHOLD, load_gt(args.data, names))
            report["holdout"] = group_report(names, scores, metadata)
            for row, score in zip(rows, scores):
                row["miou"] = float(score)
            print("HOLDOUT large / small / both:", json.dumps(report["holdout"]), flush=True)
        else:
            report["test"] = {"n": len(names), "inference_only": True}
        (diagnostics / f"{label}_rows.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("output/fusion/codex_cropsearch_trustmark"))
    p.add_argument("--out", type=Path, default=Path("output/fusion/codex_cropselect_trustmark"))
    p.add_argument("--stage", choices=("dev", "final"), default="dev")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics = args.out / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    split = json.loads((args.source / "diagnostics/split.json").read_text())
    metadata = {row["image_name"]: row for row in json.loads((args.data / "train/metadata.json").read_text())}
    assert len(split["train"]) == 136 and len(split["val"]) == 34 and set(split["train"]).isdisjoint(split["val"])
    (diagnostics / "split.json").write_text(json.dumps(split, indent=2))
    if args.stage == "dev":
        dev_stage(args, diagnostics, split, metadata)
    else:
        final_stage(args, diagnostics, split, metadata)


if __name__ == "__main__":
    main()
