"""Read out a repeat probe: are an endpoint's short answers stable across repeats?

Input: any JSONL in the collector format (endpoint,id,repeat,prompt,response)
holding >= 2 replies per endpoint/question.  For the chosen endpoints and
questions it prints
  within   P(two replies of the SAME endpoint to a question agree)
  between  P(replies of two DIFFERENT endpoints agree)
and the endpoints grouped by their modal answer vector.  within ~ between means
single answers are sampling noise; within >> between means the modal vector is
an endpoint property worth grouping on.  Try it on the labelled practice cache:
  python3 repeat_readout.py ../../test-data/collected/next-round/practice_labeled.jsonl --ids q16 q26 q08
"""
import argparse
import itertools
from collections import Counter, defaultdict

from common import answer, bank, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--endpoints", nargs="+")
    args = ap.parse_args()
    prompts = bank()
    cells = defaultdict(list)
    for path in args.answers:
        for r in rows(path):
            if r["id"] in args.ids and (not args.endpoints or r["endpoint"] in args.endpoints):
                cells[r["endpoint"], r["id"]].append(answer(r, prompts))
    eps = sorted({e for e, _ in cells})
    within, between = [], []
    for q in args.ids:
        for e in eps:
            vals = [v for v in cells[e, q] if v is not None]
            within += [a == b for a, b in itertools.combinations(vals, 2)]
        for a, b in itertools.combinations(eps, 2):
            between += [x == y for x in cells[a, q] for y in cells[b, q] if x is not None and y is not None]
    rate = lambda v: sum(v) / len(v) if v else float("nan")
    print(f"within-endpoint agreement {rate(within):.3f} (n={len(within)}); between-endpoint {rate(between):.3f} (n={len(between)})")
    groups = defaultdict(list)
    for e in eps:
        modes = []
        for q in args.ids:
            c = Counter(v for v in cells[e, q] if v is not None).most_common(2)
            modes.append("?" if not c or (len(c) > 1 and c[0][1] == c[1][1]) else c[0][0])
        groups[tuple(modes)].append(e)
    for modes, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print("  modal", dict(zip(args.ids, modes)), "->", " ".join(members))
    for e in eps:
        print("   ", e, {q: "".join(str(v) if v is not None else "_" for v in cells[e, q]) for q in args.ids})


if __name__ == "__main__":
    main()
