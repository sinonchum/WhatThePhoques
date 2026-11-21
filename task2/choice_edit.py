"""Weight punctuation edits in short answers alongside the frozen ranker's vote."""
import argparse
import csv
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from sequence import OPT
from variant_rank import predict as baseline

CHECKPOINT = Path(__file__).with_name("checkpoint")


@lru_cache(maxsize=8192)
def evidence(prompt, text):
    text = text.strip()
    options = OPT.search(prompt)
    if not options:
        return float(bool(re.match(r"^[-+]?\d+\.(?:$|\s+\D)", text)))

    choice = re.match(r"^\(?([A-D])([:.)\-]|\s+|$)\s*(.*)$", text, re.S)
    if not choice:
        return 0.0
    letter, separator, tail = choice.groups()
    tail = tail.strip().casefold()
    if not tail:
        return float(separator in (".", ")"))

    option = options.groups()[ord(letter) - ord("A")].strip().casefold().rstrip(".")
    extra_stop = tail.startswith(option) and tail[len(option):].startswith((".", "!"))
    return float(extra_stop) + 2 * float(separator not in (":", ""))


def predict(rows):
    previous = baseline(rows)
    cells = defaultdict(list)
    for row in rows:
        cells[row["endpoint"], row["id"]].append(evidence(row["prompt"], row["response"]))

    endpoints = sorted(previous)
    questions = sorted({q for _, q in cells})
    values = np.array([[np.mean(cells[ep, q]) for q in questions] for ep in endpoints])
    centered = values - values.mean(axis=0)
    prior = np.array([previous[ep] == ep for ep in endpoints])
    scores = centered.sum(axis=1) / np.sqrt(len(questions)) + 0.5 * prior
    cutoff = np.quantile(scores, 2 / 3)
    return {ep: ep if s > cutoff + 1e-12 else "core" for ep, s in zip(endpoints, scores)}


def main():
    parser = argparse.ArgumentParser(description="Generate the offline candidate from the round-two checkpoint.")
    parser.add_argument("--out", required=True, type=Path, help="New candidate CSV")
    args = parser.parse_args()
    with (CHECKPOINT / "families.csv").open(newline="") as stream:
        families = {r["endpoint"]: r["family"] for r in csv.DictReader(stream)}
    with (CHECKPOINT / "formal.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if set(families) != {r["endpoint"] for r in rows}:
        raise ValueError("Family table and formal cache have different endpoints")
    result = []
    for family in sorted(set(families.values())):
        predictions = predict([r for r in rows if families[r["endpoint"]] == family])
        result.extend((ep, family, f"{family}/{label}") for ep, label in predictions.items())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("endpoint", "family", "model"))
        writer.writerows(sorted(result))


if __name__ == "__main__":
    main()
