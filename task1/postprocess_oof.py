"""DEV-only, paired cross-validation of fixed-threshold mask postprocessing.

Only the three explicitly named dev.npz files and their DEV mask PNGs are read.
The 5 x 4 folds calibrate postprocessing on frozen responses; they do not retrain
the upstream models or undo earlier selection of their published thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from wmloc.utils import downsample_mask, miou_image


SOURCES = {
    "pixelseal": "_coord/pulled/claude_s3_ens_512_256_pixelseal/dev.npz",
    "trustmark": "codex_boxfusion_trustmark/dev.npz",
    "stegastamp": "codex_stegacutmix_stegastamp/dev.npz",
}
SEED = 2026
KEEP_GAIN = 0.003
CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def rule(name, *, area_min=0.0, area_max=1.0, remove_below=0, fill_below=0):
    return {"name": name, "area_min": area_min, "area_max": area_max,
            "remove_below": remove_below, "fill_below": fill_below}


# Fixed before looking at any candidate scores. Sizes are 128-grid pixels.
RULES = [
    rule("noop"),
    rule("clear_below_0.2pct", area_min=0.002),
    rule("clear_below_1pct", area_min=0.01),
    rule("fill_above_99.8pct", area_max=0.998),
    rule("fill_above_99pct", area_max=0.99),
    rule("extreme_area_0.2pct", area_min=0.002, area_max=0.998),
    rule("extreme_area_1pct", area_min=0.01, area_max=0.99),
    rule("remove_components_below_32", remove_below=32),
    rule("remove_components_below_128", remove_below=128),
    rule("fill_holes_below_32", fill_below=32),
    rule("fill_holes_below_128", fill_below=128),
    rule("clean_32", remove_below=32, fill_below=32),
    rule("clean_128", remove_below=128, fill_below=128),
    rule("clean_32_extreme_area_1pct", area_min=0.01, area_max=0.99,
         remove_below=32, fill_below=32),
]


def apply_rule(mask, config):
    """8-connected foreground cleanup, enclosed-hole fill, then area gates."""
    result = np.asarray(mask, dtype=bool).copy()
    if config["remove_below"]:
        labels, _ = ndimage.label(result, structure=CONNECTIVITY)
        sizes = np.bincount(labels.ravel())
        remove = sizes < config["remove_below"]
        remove[0] = False
        result[remove[labels]] = False
    if config["fill_below"]:
        labels, _ = ndimage.label(~result, structure=CONNECTIVITY)
        sizes = np.bincount(labels.ravel())
        fill = sizes < config["fill_below"]
        border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        fill[border] = False
        fill[0] = False
        result[fill[labels]] = True
    area = float(result.mean())
    if area < config["area_min"]:
        result[:] = False
    elif area > config["area_max"]:
        result[:] = True
    return result


def image_scores(predictions, targets):
    p = np.asarray(predictions, dtype=bool)
    g = np.asarray(targets, dtype=bool)
    intersections = (p & g).sum(axis=(1, 2))
    unions = (p | g).sum(axis=(1, 2))
    clean_intersections = (~p & ~g).sum(axis=(1, 2))
    clean_unions = (~p | ~g).sum(axis=(1, 2))
    foreground = np.divide(intersections, unions, out=np.ones(len(p)), where=unions != 0)
    background = np.divide(clean_intersections, clean_unions, out=np.ones(len(p)), where=clean_unions != 0)
    return (foreground + background) / 2


def stratified_folds(indices, full, count, seed):
    rng = np.random.default_rng(seed)
    folds = np.full(len(full), -1, dtype=np.int16)
    for value in (False, True):
        group = np.asarray(indices)[full[indices] == value].copy()
        rng.shuffle(group)
        for fold, ids in enumerate(np.array_split(group, count)):
            folds[ids] = fold
    return folds


def score_summary(scores, full):
    return {"both": float(np.mean(scores)), "large": float(np.mean(scores[full])),
            "small": float(np.mean(scores[~full])), "n": len(scores),
            "n_large": int(full.sum()), "n_small": int((~full).sum())}


def calibrate(scores, indices, full, names, seed):
    """Aggregate only the four inner validation subsets, never outer VAL."""
    inner = stratified_folds(indices, full, 4, seed)
    totals = np.zeros(len(RULES), dtype=np.float64)
    rows = []
    for fold in range(4):
        valid = indices[inner[indices] == fold]
        fitting = indices[inner[indices] != fold]
        totals += scores[:, valid].sum(axis=1)
        rows.append({"fold": fold, "calibration_fit_names": names[fitting].tolist(),
                     "validation_names": names[valid].tolist(),
                     "candidate_means": scores[:, valid].mean(axis=1).tolist()})
    means = totals / len(indices)
    best = float(means.max())
    ties = np.flatnonzero(means >= best - 1e-12)
    # Prefer no-op and fewer active operations when the score is identical.
    def complexity(i):
        r = RULES[int(i)]
        return (int(r["area_min"] > 0) + int(r["area_max"] < 1)
                + int(r["remove_below"] > 0) + int(r["fill_below"] > 0), int(i))
    selected = int(min(ties, key=complexity))
    return selected, {"folds": rows, "candidate_means": means.tolist(),
                      "selected_rule": RULES[selected]["name"]}


def load_dev_labels(data, names_by_method):
    """Open only DEV mask entries, rather than loading the all-train masks NPZ."""
    wanted = {name for names in names_by_method.values() for name in names}
    with ExitStack() as stack:
        archive_path = data if data.suffix == ".zip" else data / "train.zip"
        if (data / "train/metadata.json").is_file():
            root = data / "train"
            metadata_rows = json.loads((root / "metadata.json").read_text())
            def read_mask(name):
                with Image.open(root / "masks" / f"{name}.png") as image:
                    return np.asarray(image).copy()
            gt_location = str(root / "masks/<DEV_NAME>.png")
        else:
            archive = stack.enter_context(zipfile.ZipFile(archive_path))
            metadata_rows = json.loads(archive.read("train/metadata.json"))
            def read_mask(name):
                with Image.open(io.BytesIO(archive.read(f"train/masks/{name}.png"))) as image:
                    return np.asarray(image).copy()
            gt_location = f"{archive_path}!train/masks/<DEV_NAME>.png"
        # Non-DEV rows are not retained or used; no non-DEV mask/image is opened.
        metadata = {row["image_name"]: {"method": row["method"],
                    "full_watermark": bool(row["full_watermark"])}
                    for row in metadata_rows if row["image_name"] in wanted}
        del metadata_rows
        targets = {}
        for method, names in names_by_method.items():
            assert all(metadata[name]["method"] == method for name in names)
            targets[method] = np.stack([downsample_mask(read_mask(name) > 127) for name in names])
    return targets, metadata, gt_location


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/tmp/euroai-task1"))
    parser.add_argument("--fusion-root", type=Path, default=Path("output/fusion"))
    parser.add_argument("--out", type=Path, default=Path("output/codex_postprocess_dev"))
    args = parser.parse_args()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    names_by_method, sources = {}, {}
    for method, relative in SOURCES.items():
        path = args.fusion_root / relative
        with np.load(path, allow_pickle=False) as source:
            names = source["image_names"].astype(str)
            response = source["response"].copy()
            threshold = float(source["threshold"])
            flag = bool(source["dev_in_sample"])
        assert len(names) == 136 and len(set(names)) == 136
        assert response.shape == (136, 128, 128) and np.isfinite(response).all()
        names_by_method[method] = names
        sources[method] = {"names": names, "base": response >= threshold,
                           "path": str(path.resolve()), "threshold": threshold,
                           "dev_in_sample": flag,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    targets, metadata, gt_location = load_dev_labels(args.data, names_by_method)
    report = {
        "scope": "Only the three named DEV response files and their 408 DEV mask PNGs; no holdout/test NPZ, image or mask opened.",
        "metric": "Official per-image mean IoU of classes 0 and 1, then average images; empty union scores 1.",
        "procedure": {"outer_folds": 5, "inner_folds": 4, "seed": SEED,
                      "stratification": "full_watermark", "keep_gain": KEEP_GAIN,
                      "threshold_policy": "Published source threshold fixed for no-op and all candidates; no threshold search.",
                      "selection": "Choose rule from pooled inner-validation scores; tie: fewer operations then preset order. Outer validation used only for the paired score. Keep only if aggregate outer gain >= 0.003.",
                      "inner_note": "Rules have no fitted weights; aggregating four inner validation folds equals their pooled outer-training score. Splits are retained for audit.",
                      "limitation": "Conditional postprocessing CV on frozen upstream responses and published thresholds. Upstream training/model selection is not rerun within these folds; previous all-DEV threshold selection is not undone. This is not end-to-end nested model CV."},
        "rules": RULES, "gt_source": gt_location, "methods": {},
    }
    configuration = {"schema": 1, "input": "binary mask at each source's published threshold",
                     "operation_order": ["remove 8-connected foreground components", "fill 8-connected enclosed background components; preserve border", "area gates"],
                     "size_units": "128x128 grid pixels; remove/fill only when size strictly below setting",
                     "selection_split": "DEV only", "methods": {}}
    for method, source in sources.items():
        method_started = time.monotonic()
        names, base, gt = source["names"], source["base"], targets[method]
        full = np.array([metadata[name]["full_watermark"] for name in names], dtype=bool)
        assert int(full.sum()) == 68
        indices = np.arange(len(names))
        candidate_scores = np.stack([image_scores(np.stack([apply_rule(mask, r) for mask in base]), gt)
                                     for r in RULES])
        # One direct verification that the vectorized score equals the official helper.
        official = np.array([miou_image(pred, target) for pred, target in zip(base, gt)])
        np.testing.assert_allclose(candidate_scores[0], official, rtol=0, atol=1e-12)
        outer = stratified_folds(indices, full, 5, SEED)
        chosen = np.zeros(len(names), dtype=np.int16)
        post_scores = np.empty(len(names), dtype=np.float64)
        folds = []
        for fold in range(5):
            fitting, valid = indices[outer != fold], indices[outer == fold]
            selected, calibration = calibrate(candidate_scores, fitting, full, names, SEED + 100 + fold)
            chosen[valid] = selected
            post_scores[valid] = candidate_scores[selected, valid]
            folds.append({"fold": fold, "fit_names": names[fitting].tolist(),
                          "validation_names": names[valid].tolist(), "selected_rule": RULES[selected],
                          "inner_calibration": calibration,
                          "base": score_summary(candidate_scores[0, valid], full[valid]),
                          "postprocess": score_summary(post_scores[valid], full[valid]),
                          "gain": score_summary(post_scores[valid] - candidate_scores[0, valid], full[valid])})
        gains = post_scores - candidate_scores[0]
        keep = bool(float(gains.mean()) >= KEEP_GAIN)
        final_id, final_calibration = (calibrate(candidate_scores, indices, full, names, SEED + 200)
                                       if keep else (0, None))
        final_rule = RULES[final_id]
        configuration["methods"][method] = {"source": source["path"], "threshold": source["threshold"],
                                              "adopt_postprocessing": keep, "rule": final_rule}
        post_masks = np.stack([apply_rule(mask, RULES[int(i)]) for mask, i in zip(base, chosen)])
        np.savez_compressed(args.out / f"{method}_outer_oof.npz", image_names=names,
                            outer_fold=outer, selected_rule=chosen,
                            baseline_prediction=base, postprocess_prediction=post_masks,
                            baseline_miou=candidate_scores[0], postprocess_miou=post_scores,
                            threshold=np.asarray(source["threshold"]))
        minority = np.minimum(gt.sum(axis=(1, 2)), 128 * 128 - gt.sum(axis=(1, 2)))
        result = {"source": {k: v for k, v in source.items() if k not in ("base", "names")},
                  "dev_names": names.tolist(), "baseline": score_summary(candidate_scores[0], full),
                  "nested_postprocess": score_summary(post_scores, full),
                  "paired_gain": score_summary(gains, full), "adopt_postprocessing": keep,
                  "improved_images": int((gains > 1e-12).sum()),
                  "worsened_images": int((gains < -1e-12).sum()),
                  "unchanged_images": int((np.abs(gains) <= 1e-12).sum()),
                  "target_stats": {"all_zero": int((gt.sum(axis=(1, 2)) == 0).sum()),
                                   "all_one": int((gt.sum(axis=(1, 2)) == 128 * 128).sum()),
                                   "minimum_minority_grid_pixels": int(minority.min())},
                  "outer_folds": folds, "final_rule": final_rule,
                  "full_dev_calibration_if_kept": final_calibration,
                  "seconds": time.monotonic() - method_started}
        report["methods"][method] = result
        report["seconds_so_far"] = time.monotonic() - started
        (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        (args.out / "postprocess_config.json").write_text(json.dumps(configuration, indent=2, allow_nan=False) + "\n")
        print(f'{method}: no-op={result["baseline"]["both"]:.6f}, post={result["nested_postprocess"]["both"]:.6f}, gain={result["paired_gain"]["both"]:+.6f}, keep={keep}, final={final_rule["name"]}', flush=True)
    print(f'Completed DEV-only postprocessing experiment in {time.monotonic()-started:.1f}s: {args.out / "report.json"}', flush=True)


if __name__ == "__main__":
    main()
