"""Use two archived formal panels to diagnose cluster reproducibility without labels."""
import argparse
import csv
import json
import random
from itertools import combinations
from pathlib import Path
from statistics import median

from choice_edit import CHECKPOINT, predict as syntax
from short_answers import predict as content


def agreement(first, second):
    """Signed pairwise MCC. Constant predictions carry no partition evidence."""
    if set(first) != set(second):
        raise ValueError("Endpoint coverage differs between panels")
    tp = tn = fp = fn = 0
    for a, b in combinations(first, 2):
        left = first[a] == first[b]
        right = second[a] == second[b]
        if left and right:
            tp += 1
        elif left:
            fp += 1
        elif right:
            fn += 1
        else:
            tn += 1
    denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    return (tp*tn-fp*fn)/denom if denom else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="New directory for an offline candidate and diagnostics")
    args = parser.parse_args()
    rows = [json.loads(line) for line in (CHECKPOINT / "formal.jsonl").read_text().splitlines()]
    with (CHECKPOINT / "families.csv").open(newline="") as stream:
        family = {r["endpoint"]: r["family"] for r in csv.DictReader(stream)}
    with (CHECKPOINT.parent / "prompts.csv").open(newline="") as stream:
        category = {r["id"]: r["category"] for r in csv.DictReader(stream)}
    panels = json.loads((CHECKPOINT / "practice-replay.json").read_text())
    first, second = set(panels["first_ten"]), set(panels["additional_ten"])
    all_questions = {r["id"] for r in rows}
    if first & second or first | second != all_questions:
        raise ValueError("Expected two disjoint panels covering the full cache")
    splits = [(first, second)]
    rng = random.Random(390171)
    for _ in range(256):
        half = set()
        for cat in sorted(set(category.values())):
            pool = sorted(q for q in all_questions if category[q] == cat)
            half.update(rng.sample(pool, len(pool)//2))
        splits.append((half, all_questions-half))
    report, result = {}, []
    for label in sorted(set(family.values())):
        records = [r for r in rows if family[r["endpoint"]] == label]
        atomic = sum(len(r["response"].split()) == 1 for r in records)/len(records)
        methods = {"syntax": syntax}
        if atomic >= 0.8:
            methods["content"] = content
        diagnostics = {}
        for name, method in methods.items():
            values = [agreement(method([r for r in records if r["id"] in a]),
                                method([r for r in records if r["id"] in b])) for a,b in splits]
            diagnostics[name] = {"actual_round_agreement": values[0],
                                 "defined": sum(x is not None for x in values[1:]),
                                 "median_all_256": median(x or 0.0 for x in values[1:])}
        selected = max(methods, key=lambda name: diagnostics[name]["median_all_256"])
        report[label] = {"one_word_fraction": atomic, "methods": diagnostics, "selected": selected}
        for ep, model in methods[selected](records).items():
            result.append((ep, label, f"{label}/{model}"))
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"agreement.json").write_text(json.dumps(report, indent=2)+"\n")
    with (args.out/"candidate.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("endpoint","family","model"))
        writer.writerows(sorted(result))


if __name__ == "__main__":
    main()
