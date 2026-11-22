"""Same-draw comparison of solvers on the labelled practice cache.

Draws are generated exactly as in task2/replay.py and calibrate_metric.py
(blake2s(seed/protocol/draw) -> question sample, one cached reply per
endpoint/question, shuffled aliases and rows), so every solver sees identical
rows.  Reports mean clipped model MCC, draws strictly above 0.65, mean false
merges (FP) and false splits (FN) out of 36 pairs.
"""
import argparse
import csv
import hashlib
import itertools
import json
import random
from statistics import median

import numpy as np

from common import TASK
import replicated
from metric_partition import predict as last_rule
from replay import load_inputs
from variant_rank import predict as frozen_rank

SOLVERS = {
    "variant_rank (19:35 upload rule)": frozen_rank,
    "metric_partition (old, last rule)": last_rule,
    "replicated (new, both blocks)": replicated.predict,
    "union (control, either block)": replicated.predict_union,
}


def panels(bank):
    first = set(json.loads((TASK / "checkpoint/practice-replay.json").read_text())["first_ten"])
    return {"first-ten": sorted(first),
            "holdout": sorted(q for q in bank if bank[q]["split"] == "holdout"),
            "fixed-hard": sorted(q for q in bank if bank[q]["split"] == "dev" and q not in first)}


def run(protocol, seed, draws):
    bank, eps, cache = load_inputs()
    with (TASK / "checkpoint/practice_answers.csv").open(newline="") as stream:
        truth = {r["endpoint"]: r["model"] for r in csv.DictReader(stream)}
    pairs = list(itertools.combinations(eps, 2))
    same = np.array([truth[a] == truth[b] for a, b in pairs])
    fixed = panels(bank)
    cats = sorted({r["category"] for r in bank.values()})
    stats = {name: {"mcc": [], "fp": [], "fn": [], "detached": []} for name in SOLVERS}
    digest = hashlib.sha256()
    for j in range(draws):
        rng = random.Random(int.from_bytes(
            hashlib.blake2s(f"{seed}/{protocol}/{j}".encode(), digest_size=8).digest(), "big"))
        if protocol == "stratified":
            qs = sorted(q for c in cats for q in rng.sample(sorted(q for q in bank if bank[q]["category"] == c), 2))
        else:
            qs = fixed[protocol]
        keys = [(e, q, rng.randrange(3)) for e in eps for q in qs]
        digest.update(str(keys).encode())
        aliases = [f"anonymous-{i:02d}" for i in range(1, len(eps) + 1)]
        rng.shuffle(aliases)
        alias = dict(zip(eps, aliases))
        rows = [dict(cache[k], endpoint=alias[k[0]], repeat=0) for k in keys]
        rng.shuffle(rows)
        for name, solver in SOLVERS.items():
            out = solver(rows)
            pred = np.array([out[alias[a]] == out[alias[b]] for a, b in pairs])
            n, p, k, tp = len(pairs), same.sum(), pred.sum(), (pred & same).sum()
            den = np.sqrt(float(p * (n - p) * k * (n - k)))
            s = stats[name]
            s["mcc"].append(max(0.0, (n * tp - p * k) / den) if den else 0.0)
            s["fp"].append(int(k - tp)); s["fn"].append(int(p - tp))
            s["detached"].append(len({v for v in out.values() if v != "core"}))
    return {"protocol": protocol, "seed": seed, "draws": draws, "draw_manifest_sha256": digest.hexdigest(),
            "solvers": {name: {"mean_model_mcc": float(np.mean(s["mcc"])), "median_model_mcc": float(median(s["mcc"])),
                               "n_strictly_above_065": int(sum(v > 0.65 for v in s["mcc"])),
                               "mean_fp": float(np.mean(s["fp"])), "mean_fn": float(np.mean(s["fn"])),
                               "mean_detached_endpoints": float(np.mean(s["detached"]))}
                        for name, s in stats.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1493807)
    ap.add_argument("--draws", type=int, default=2048)
    ap.add_argument("--protocols", nargs="+", default=["stratified", "fixed-hard", "holdout", "first-ten"])
    ap.add_argument("--out")
    args = ap.parse_args()
    report = [run(p, args.seed, args.draws) for p in args.protocols]
    if args.out:
        open(args.out, "w").write(json.dumps(report, indent=1) + "\n")
    for r in report:
        print(f"\n{r['protocol']} seed={r['seed']} draws={r['draws']} manifest={r['draw_manifest_sha256'][:12]}")
        for name, s in r["solvers"].items():
            print(f"  {name:36s} mean {s['mean_model_mcc']:.4f}  >0.65: {s['n_strictly_above_065']:4d}  FP {s['mean_fp']:.2f}  FN {s['mean_fn']:.2f}  detached {s['mean_detached_endpoints']:.2f}")
