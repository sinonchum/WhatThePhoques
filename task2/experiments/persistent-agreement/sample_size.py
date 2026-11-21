"""How does the last practice rule behave as replies per endpoint grow?

Labelled practice cache only.  For each setting draw questions without
replacement (all 30 when n=30) and `reps` cached replies per endpoint/question,
run metric_partition.predict and score the 36 pairs.  Also report how often two
independent draws of the same size detach the same endpoints (Jaccard), which is
the labelled analogue of the round-1 vs round-2 swap on the formal cache.
"""
import csv
import itertools
import json
import random

import numpy as np

from common import TASK
from metric_partition import predict as last_rule
from replay import load_inputs


def main(draws=400, seed=977):
    bank, eps, cache = load_inputs()
    with (TASK / "checkpoint/practice_answers.csv").open(newline="") as stream:
        truth = {r["endpoint"]: r for r in csv.DictReader(stream)}
    pairs = list(itertools.combinations(eps, 2))
    same = np.array([truth[a]["model"] == truth[b]["model"] for a, b in pairs])
    out = []
    for n, reps in ((10, 1), (20, 1), (30, 1), (10, 3), (20, 3), (30, 3)):
        rng = random.Random(f"{seed}/{n}/{reps}")
        mcc, sets, split = [], [], {e: 0 for e in eps}
        for _ in range(draws):
            qs = rng.sample(sorted(bank), n)
            rows = [dict(cache[e, q, r]) for e in eps for q in qs for r in rng.sample(range(3), reps)]
            pred = last_rule(rows)
            det = frozenset(e for e, g in pred.items() if g != "core")
            sets.append(det)
            for e in det:
                split[e] += 1
            p = np.array([pred[a] == pred[b] for a, b in pairs])
            k, tp, pos, m = p.sum(), (p & same).sum(), same.sum(), len(pairs)
            den = np.sqrt(float(pos * (m - pos) * k * (m - k)))
            mcc.append(max(0.0, (m * tp - pos * k) / den) if den else 0.0)
        jac = [len(a & b) / len(a | b) if a | b else 1.0 for a, b in zip(sets[0::2], sets[1::2])]
        out.append({"questions": n, "replies_per_question": reps, "replies_per_endpoint": n * reps,
                    "mean_model_mcc": float(np.mean(mcc)), "share_above_065": float(np.mean(np.array(mcc) > 0.65)),
                    "mean_jaccard_between_independent_draws": float(np.mean(jac)),
                    "detach_rate": {f"{e} {truth[e]['kind']}": round(split[e] / draws, 2) for e in eps}})
        o = out[-1]
        print(f"{n:2d} q x {reps} = {n*reps:2d} replies/endpoint: mean MCC {o['mean_model_mcc']:.3f}, >0.65 {o['share_above_065']:.2f}, "
              f"split-set Jaccard {o['mean_jaccard_between_independent_draws']:.2f}")
        print("     detach rate:", o["detach_rate"])
    json.dump(out, open("sample_size.json", "w"), indent=1)


if __name__ == "__main__":
    main()
