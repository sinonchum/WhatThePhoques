"""Do the format deviations that drive the submitted split replicate across rounds?

Part A (generic): a reply is "off-format" when it is not exactly the bare letter
or integer the prompt asks for.  Each off-format reply is weighted by the
surprisal of being off-format on that question inside the family, so a question
on which half the family rambles counts for little.  Per endpoint we report the
weighted total by round and a shuffle p-value (answers shuffled across the
family's endpoints within each question).

Part B (submitted rule): run the last practice rule (metric_partition.predict)
and the 19:35 rule (variant_rank.predict) on round 1 only and on round 2 only,
and compare the detached sets with each other and with the uploaded singletons.

Part C: third-person references to "the user"/"the human" in a reply (a system
prompt showing through).  On the labelled practice cache this fires on the
system-prompt endpoint only (1 of 90 replies) and on none of the other 720.
Cache only.
"""
import argparse
import json
import re
from math import comb

import numpy as np

from common import LAST_SUBMITTED, PRACTICE, ROUND1, ROUND2, bank, partition, rows
from metric_partition import predict as last_rule
from variant_rank import predict as frozen_rank

BARE = {"choice": re.compile(r"[A-D]"), "integer": re.compile(r"-?\d+")}
THIRD_PERSON = re.compile(r"\bthe (?:user|human)\b|\buser's\b", re.I)


def off_format(row, prompts):
    return not BARE[prompts[row["id"]]["format"]].fullmatch(row["response"].strip())


def weighted(off, eps, qs):
    """off[(e,q)] -> per-endpoint surprisal-weighted totals."""
    total = dict.fromkeys(eps, 0.0)
    for q in qs:
        rate = np.mean([off[e, q] for e in eps])
        if 0 < rate < 1:
            for e in eps:
                if off[e, q]:
                    total[e] += -np.log(rate)
    return total


def hypergeom_tail(n, a, b, k):
    """P(overlap >= k) for two random subsets of sizes a and b out of n."""
    return sum(comb(a, i) * comb(n - a, b - i) for i in range(k, min(a, b) + 1)) / comb(n, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--out")
    args = ap.parse_args()
    prompts = bank()
    r1, r2 = rows(ROUND1), rows(ROUND2)
    qs1, qs2 = sorted({r["id"] for r in r1}), sorted({r["id"] for r in r2})
    data = r1 + r2
    sub = partition(LAST_SUBMITTED)
    fam = {e: f for e, (f, _) in sub.items()}
    off = {(r["endpoint"], r["id"]): off_format(r, prompts) for r in data}
    rng = np.random.default_rng(args.seed)
    report = {"families": {}, "third_person": {}}
    for f in sorted(set(fam.values())):
        eps = sorted(e for e in fam if fam[e] == f)
        sizes = {}
        for e in eps:
            sizes.setdefault(sub[e][1], []).append(e)
        uploaded = sorted(e for e in eps if len(sizes[sub[e][1]]) == 1)
        obs = weighted(off, eps, qs1 + qs2)
        o1, o2 = weighted(off, eps, qs1), weighted(off, eps, qs2)
        null = np.zeros((args.draws, len(eps)))
        cols = {q: np.array([off[e, q] for e in eps]) for q in qs1 + qs2}
        for d in range(args.draws):
            sh = {(e, q): v for q in cols for e, v in zip(eps, cols[q][rng.permutation(len(eps))])}
            w = weighted(sh, eps, qs1 + qs2)
            null[d] = [w[e] for e in eps]
        per_ep = {}
        for i, e in enumerate(eps):
            per_ep[e] = {
                "off_format_round1": [q for q in qs1 if off[e, q]], "off_format_round2": [q for q in qs2 if off[e, q]],
                "weighted_round1": round(o1[e], 2), "weighted_round2": round(o2[e], 2),
                "weighted_total": round(obs[e], 2), "p": float((null[:, i] >= obs[e] - 1e-12).mean()),
                "uploaded_as_singleton": e in uploaded,
            }
        fam_rows = {r: [x for x in rs if fam[x["endpoint"]] == f] for r, rs in (("round1", r1), ("round2", r2))}
        rules = {}
        for name, rule in (("metric_partition", last_rule), ("variant_rank", frozen_rank)):
            det = {r: sorted(e for e, g in rule(fam_rows[r]).items() if g != "core" and g != "0") for r in fam_rows}
            both = sorted(set(det["round1"]) & set(det["round2"]))
            rules[name] = {"detached_round1": det["round1"], "detached_round2": det["round2"], "detached_both": both,
                           "overlap_p_if_unrelated": hypergeom_tail(len(eps), len(det["round1"]), len(det["round2"]), len(both))}
        replicated = sorted(e for e in eps if o1[e] > 0 and o2[e] > 0)
        report["families"][f] = {"endpoints": per_ep, "uploaded_singletons": uploaded,
                                 "off_format_in_both_rounds": replicated, "rules": rules}
    for name, source, key in (("practice", rows(PRACTICE), lambda r: f"{r['endpoint']} ({r['variant_kind']})"),
                              ("formal", data, lambda r: r["endpoint"])):
        hits = {}
        for r in source:
            if THIRD_PERSON.search(r["response"]):
                hits.setdefault(key(r), []).append(r["id"])
        report["third_person"][name] = hits
    if args.out:
        open(args.out, "w").write(json.dumps(report, indent=1) + "\n")
    for f, r in report["families"].items():
        print(f"\n{f}: uploaded singletons {[e[3:] for e in r['uploaded_singletons']]}; off-format in BOTH rounds {[e[3:] for e in r['off_format_in_both_rounds']]}")
        for e, v in r["endpoints"].items():
            if v["weighted_total"] > 0:
                print(f"   {e} r1 {v['weighted_round1']:5.2f} {v['off_format_round1']}  r2 {v['weighted_round2']:5.2f} {v['off_format_round2']}  p={v['p']:.4f}{'  [uploaded singleton]' if v['uploaded_as_singleton'] else ''}")
        for name, x in r["rules"].items():
            print(f"   {name}: round1 {[e[3:] for e in x['detached_round1']]} | round2 {[e[3:] for e in x['detached_round2']]} | both {[e[3:] for e in x['detached_both']]} (chance p={x['overlap_p_if_unrelated']:.2f})")
    print("\nthird-person leak:", json.dumps(report["third_person"]))


if __name__ == "__main__":
    main()
