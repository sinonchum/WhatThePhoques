"""Offline scoring for Task 2 endpoint assignments."""

import csv
import math
from itertools import combinations


def _assignments(rows, source):
    assignments = {}
    family_by_model = {}
    for index, row in enumerate(rows, start=1):
        if isinstance(row, (str, bytes)) or len(row) != 3:
            raise ValueError(f"{source} row {index} must contain endpoint, family, model")
        if any(not isinstance(label, str) or not label.strip() for label in row):
            raise ValueError(f"{source} row {index} has an empty or non-text label")
        endpoint, family, model = (label.strip() for label in row)
        if any("," in label for label in (endpoint, family, model)):
            raise ValueError(f"{source} row {index} has a comma inside a label")
        if endpoint in assignments:
            raise ValueError(f"{source} has duplicate endpoint {endpoint!r}")
        if model in family_by_model and family_by_model[model] != family:
            raise ValueError(f"{source} model {model!r} spans multiple families")
        family_by_model[model] = family
        assignments[endpoint] = (family, model)
    if not assignments:
        raise ValueError(f"{source} has no endpoint assignments")
    return assignments


def _load_truth(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if any(cell.strip() for cell in row)]
    if rows:
        header = [cell.strip().lower() for cell in rows[0]]
        fields = ("endpoint", "family", "model")
        if all(field in header for field in fields):
            if any(header.count(field) != 1 for field in fields):
                raise ValueError("Truth CSV has duplicate endpoint, family, or model columns")
            positions = [header.index(field) for field in fields]
            data = []
            for line, row in enumerate(rows[1:], start=2):
                if len(row) != len(header):
                    raise ValueError(f"Truth CSV row {line} does not match the header")
                data.append(tuple(row[position] for position in positions))
            rows = data
    return _assignments(rows, "Truth CSV")


def _record(counts, errors, pair, actual_same, predicted_same):
    if actual_same:
        outcome = "tp" if predicted_same else "fn"
    else:
        outcome = "fp" if predicted_same else "tn"
    counts[outcome] += 1
    if actual_same != predicted_same:
        errors.append({"endpoints": list(pair), "type": outcome.upper()})


def _clipped_mcc(counts):
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    product = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if not product:
        return 0.0
    return max(0.0, (tp * tn - fp * fn) / math.sqrt(product))


def evaluate(pred_rows, truth_path):
    """Compare (endpoint, family, model) rows against a local truth CSV.

    Model MCC uses only pairs in the same ground-truth family. A null family
    MCC and weighted score mean the truth lacks positive or negative family
    examples. The model MCC is still returned under the zero-denominator
    convention used by the practice scorer.
    """
    predicted = _assignments(pred_rows, "Predictions")
    truth = _load_truth(truth_path)
    missing = sorted(truth.keys() - predicted.keys())
    extra = sorted(predicted.keys() - truth.keys())
    if missing or extra:
        raise ValueError(f"Endpoint mismatch: missing={missing}, extra={extra}")

    family_counts = dict.fromkeys(("tp", "fp", "tn", "fn"), 0)
    model_counts = family_counts.copy()
    family_errors = []
    model_errors = []
    for pair in combinations(sorted(truth), 2):
        left, right = pair
        family_same = truth[left][0] == truth[right][0]
        _record(family_counts, family_errors, pair, family_same,
                predicted[left][0] == predicted[right][0])
        if family_same:
            _record(model_counts, model_errors, pair, truth[left][1] == truth[right][1],
                    predicted[left][1] == predicted[right][1])

    family_positives = family_counts["tp"] + family_counts["fn"]
    family_negatives = family_counts["tn"] + family_counts["fp"]
    family_mcc = _clipped_mcc(family_counts) if family_positives and family_negatives else None
    model_mcc = _clipped_mcc(model_counts)
    return {
        "model_mcc": model_mcc,
        "family_mcc": family_mcc,
        "weighted_score": None if family_mcc is None else 0.1 * family_mcc + 0.9 * model_mcc,
        "model_counts": model_counts,
        "family_counts": family_counts,
        "model_errors": model_errors,
        "family_errors": family_errors,
    }
