"""Replay a pinned public two-response model-equality analyzer on archived practice replies."""
import argparse
import csv
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from choice_edit import predict as baseline
from evaluation import evaluate
from inputs import _normalize
from replay import CHECKPOINT, load_inputs

SOURCE = {
    "ms_analyze.py": "93f991692135e6dffefd6d0d369a95cc65858b084e65ac4df18bf2efa950be42",
    "ms_battery.py": "d930be4f142c7b4e6560274d2a373f959a5b913e6c3601b9284ff1b84cda3528",
    "ms_battery_analyze.py": "e786a2abc71eab0209e670517c26e7c3eb3c358b268e09b3b79ab2026807d8aa",
}
THRESHOLDS = (.05, .10, .15, .20)
TARGETS = ("ref-02", "ref-03", "ref-09")


def summarize(groups):
    rows = [(ep, "P", str(label)) for ep, label in groups.items()]
    result = evaluate(rows, CHECKPOINT / "practice_answers.csv")
    return {"mcc": result["model_mcc"], "counts": result["model_counts"],
            "groups": groups, "same_as_plain": {e: groups[e] == groups["ref-01"] for e in TARGETS}}


def exact_test(analyzer, answers, left, right, questions):
    # Distinct 2+2 allocations up to swapping the two endpoints. Scaling by
    # four makes the per-prompt statistic integral, including negative values.
    totals = Counter({0: 1})
    for q in questions:
        items = answers[left][q] + answers[right][q]
        steps = Counter()
        for partner in (1, 2, 3):
            other = [items[i] for i in (1, 2, 3) if i != partner]
            x, a, b = analyzer.cross_within([items[0], items[partner]], other)
            steps[round(4 * (x - (a+b)/2))] += 1
        updated = Counter()
        for v, n in totals.items():
            for s, m in steps.items():
                updated[v+s] += n*m
        totals = updated
    observed = analyzer.stat(answers, left, right, questions)
    threshold = round(4 * len(questions) * observed)
    return sum(n for v, n in totals.items() if v >= threshold)/sum(totals.values())


def run(source, output):
    for name, digest in SOURCE.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Wrong upstream revision for {name}")
    sys.path.insert(0, str(source.resolve()))
    import ms_battery_analyze as analyzer
    from ms_analyze import mcc as upstream_mcc

    bank, eps, cache = load_inputs()
    truth_file = CHECKPOINT / "practice_answers.csv"
    with truth_file.open(newline="") as stream:
        truth = {r["endpoint"]: r["model"] for r in csv.DictReader(stream)}
    windows = list(itertools.combinations(range(3), 2))
    panels = {"dev": sorted(q for q in bank if bank[q]["split"] == "dev"),
              "holdout": sorted(q for q in bank if bank[q]["split"] == "holdout"),
              "all": sorted(bank)}
    groups = {c: sorted(q for q in bank if bank[q]["category"] == c)
              for c in sorted({r["category"] for r in bank.values()})}
    report = {"upstream_commit": "3440900b6fee2f2c436affea4020e6a81cee1936",
              "exact_prompt_overlap": len({r["prompt"] for r in bank.values()} &
                                          {p for _, _, p in analyzer.QUESTIONS}),
              "upstream_collected_questions": len(analyzer.QUESTIONS),
              "upstream_cluster_questions": sum(q[:3] in analyzer.WEIGHT_GROUPS for q, _, _ in analyzer.QUESTIONS),
              "comparisons": {}, "thresholds": {}, "baseline": {}}
    diagnostics = []
    for window in windows:
        tag = "".join(map(str, window))
        report["baseline"][tag] = {}
        for panel, qs in panels.items():
            rows = [cache[e,q,i] for e in eps for q in qs for i in window]
            report["baseline"][tag][panel] = summarize(baseline(rows))
        for mode in ("upstream", "choice_aware"):
            answers = {ep: {} for ep in eps}
            for ep in eps:
                for q, meta in bank.items():
                    values = []
                    for i in window:
                        s = cache[ep,q,i]["response"]
                        kind = "num" if meta["format"] == "integer" else "word"
                        value = analyzer.extract(kind, s)
                        if mode == "choice_aware" and kind == "word":
                            value = _normalize(s, "choice", meta["prompt"]) or value
                        values.append(value)
                    answers[ep][q] = values
            table = report["comparisons"].setdefault(mode, {}).setdefault(tag, {})
            for panel, qs in panels.items():
                dist = {frozenset((a,b)): analyzer.stat(answers,a,b,qs)
                        for a,b in itertools.combinations(eps,2)}
                table[panel] = {}
                for threshold in THRESHOLDS:
                    partition = analyzer.cluster(eps,dist,threshold)
                    result = summarize(partition)
                    assert abs(upstream_mcc(partition,truth)-result["mcc"]) < 1e-12
                    table[panel][str(threshold)] = result
            for e in TARGETS:
                for category, qs in groups.items():
                    exact = exact_test(analyzer,answers,"ref-01",e,qs)
                    mc = analyzer.perm_pvalue(answers,"ref-01",e,qs,2000,random.Random(0))
                    diagnostics.append({"parser":mode,"repeats":tag,"endpoint":e,"category":category,
                                        "questions":len(qs),"distance":analyzer.stat(answers,"ref-01",e,qs),
                                        "p_exact":exact,"p_upstream_mc":mc})
    for mode, table in report["comparisons"].items():
        # Fit only on development questions; resolve ties by upstream cutoff order.
        means = {str(t): mean(v["dev"][str(t)]["mcc"] for v in table.values()) for t in THRESHOLDS}
        report["thresholds"][mode] = {"dev_mean":means,"chosen":max(means,key=means.get)}
    output.mkdir(parents=True,exist_ok=True)
    (output/"practice.json").write_text(json.dumps(report,indent=2)+"\n")
    with (output/"pair-tests.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True,help="Pinned upstream src/model_equality directory")
    p.add_argument("--out",type=Path,required=True)
    args = p.parse_args()
    run(args.source,args.out)
