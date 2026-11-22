"""Task 2 workflow. Compare fingerprints offline and export nested endpoint groups."""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import collect
import demo
from clustering import cluster
from evaluation import evaluate
from inputs import load_records, read_splits


DEFAULT_PROMPTS = Path(__file__).with_name("prompts.csv")
METHODS = ("mode", "js")
DEFAULT_THRESHOLDS = tuple(i / 10 for i in range(11))


def save(path, data, *, csv_rows=False):
    """Publish complete output files even if a previous run is interrupted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8",
                                     newline="", delete=False) as handle:
        temp = Path(handle.name)
        try:
            if csv_rows:
                writer = csv.writer(handle)
                writer.writerow(("endpoint", "family", "model"))
                writer.writerows(data)
            else:
                json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            handle.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def check_endpoints(records, prefix, count, stage):
    expected = {f"{prefix}-{i:02d}" for i in range(1, count + 1)}
    actual = {row[0] for row in records}
    if actual != expected:
        raise ValueError(f"{stage}: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")


def coverage(records):
    endpoints = {}
    for endpoint, _, _, answer in records:
        counts = endpoints.setdefault(endpoint, {"responses": 0, "unparsed": 0})
        counts["responses"] += 1
        counts["unparsed"] += answer is None
    return dict(sorted(endpoints.items()))


def compare(args):
    metadata = {}
    records = load_records(args.answers, args.prompts, metadata=metadata)
    splits = read_splits(args.prompts)
    dev = [row for row in records if splits[row[1]] == "dev"]
    held = [row for row in records if splits[row[1]] == "holdout"]
    for name, rows in (("dev", dev), ("holdout", held)):
        check_endpoints(rows, "ref", 9, name)
    report = {"format_version": 1, **metadata, "coverage": {"dev": coverage(dev), "holdout": coverage(held)},
              "methods": {}}
    predictions = {}
    for method in METHODS:
        # The official practice set has one family. Isolate the scored identity
        # question, rather than learning a family cut from missing negatives.
        candidates = []
        for threshold in sorted(set(args.thresholds)):
            rows, _ = cluster(dev, method, 1.0, threshold)
            score = evaluate(rows, args.truth)
            candidates.append({"threshold": threshold, "model_mcc": score["model_mcc"],
                               "counts": score["model_counts"]})
        # Candidates sorted ascending: prefer the stricter cut when scores tie.
        best = max(candidates, key=lambda item: item["model_mcc"])
        held_rows, _ = cluster(held, method, 1.0, best["threshold"])
        held_score = evaluate(held_rows, args.truth)
        predictions[method] = held_rows
        report["methods"][method] = {"threshold": best["threshold"], "dev_mcc": best["model_mcc"],
                                     "dev_candidates": candidates, "holdout": held_score}
    # Method and cut are selected on development data only. A tie prefers mode.
    report["selected_method"] = max(METHODS, key=lambda m: report["methods"][m]["dev_mcc"])
    for method, rows in predictions.items():
        save(args.out / f"holdout-{method}.csv", rows, csv_rows=True)
    save(args.out / "comparison.json", report)
    if metadata["source"] == "synthetic":
        print("Synthetic demo measures workflow behavior only.")
    for method, details in report["methods"].items():
        held_score = details["holdout"]
        counts = held_score["model_counts"]
        print(f"{method:4}  cut={details['threshold']:.2f}  dev MCC={details['dev_mcc']:.3f}"
              f"  holdout MCC={held_score['model_mcc']:.3f}  FP={counts['fp']}  FN={counts['fn']}")
    print(f"Selected on dev: {report['selected_method']}. Report: {args.out / 'comparison.json'}")


def predict(args):
    metadata = {}
    records = load_records(args.answers, args.prompts, metadata=metadata)
    check_endpoints(records, "ep", 33, "formal export")
    with args.calibration.open(encoding="utf-8") as handle:
        report = json.load(handle)
    try:
        if report["format_version"] != 1:
            raise ValueError("unknown comparison report version")
        method = args.method or report["selected_method"]
        model_cut = report["methods"][method]["threshold"]
    except (KeyError, TypeError) as exc:
        raise ValueError("calibration expects comparison.json from the compare command") from exc
    family_cut = args.family_threshold
    if family_cut < model_cut:
        raise ValueError(f"family threshold must be at least the selected model threshold {model_cut}")
    rows, diagnostics = cluster(records, method, family_cut, model_cut)
    diagnostics.update(metadata)
    diagnostics["calibration_source"] = report.get("source", "unknown")
    diagnostics["coverage"] = coverage(records)
    save(args.out / "distances.json", diagnostics)
    save(args.out / "submission.csv", rows, csv_rows=True)
    if metadata["source"] == "synthetic" or diagnostics["calibration_source"] == "synthetic":
        print("Synthetic data were used for this export; use this file to inspect the format.")
    print(f"{len(rows)} endpoints, {len(diagnostics['family_sizes'])} families, "
          f"{len(diagnostics['model_sizes'])} models. CSV: {args.out / 'submission.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo.add_parser(commands, DEFAULT_PROMPTS)
    collect.add_parser(commands, DEFAULT_PROMPTS)
    calibration = commands.add_parser("compare", help="fit cuts on dev prompts and score held-out prompts")
    prediction = commands.add_parser("predict", help="export the 33 scored endpoint assignments offline")
    for command in (calibration, prediction):
        command.add_argument("--answers", type=Path, required=True, help="official collector JSONL")
        command.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
        command.add_argument("--out", type=Path, required=True, help="output directory")
    calibration.add_argument("--truth", type=Path, required=True, help="published practice_answers.csv")
    calibration.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    calibration.set_defaults(handler=compare)
    prediction.add_argument("--calibration", type=Path, required=True, help="comparison.json from compare")
    prediction.add_argument("--method", choices=METHODS, help="override dev-selected method")
    prediction.add_argument("--family-threshold", type=float, required=True,
                            help="assumed family cut in [model threshold, 1]; practice cannot fit this")
    prediction.set_defaults(handler=predict)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
