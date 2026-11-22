"""Compare a candidate CSV with every uploaded CSV on the 528 endpoint pairs.

Only the equality relations matter to the scorer, so label names are ignored.
Prints, per uploaded file, how many pairs differ at family level and at model
level, lists the model-level changes against the last upload, and exits 1 if
the candidate is pair-for-pair identical to something already uploaded.
"""
import argparse
import itertools
import sys

from common import SUBMITTED, partition


def relations(part, eps):
    fam, mod = {}, {}
    for a, b in itertools.combinations(eps, 2):
        fam[a, b] = part[a][0] == part[b][0]
        mod[a, b] = fam[a, b] and part[a][1] == part[b][1]
    return fam, mod


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate")
    args = ap.parse_args()
    cand = partition(args.candidate)
    eps = sorted(cand)
    if len(eps) != 33 or eps != [f"ep-{i:02d}" for i in range(1, 34)]:
        sys.exit("candidate must list ep-01..ep-33 exactly once")
    by_model = {}
    for e, (f, m) in cand.items():
        if by_model.setdefault(m, f) != f:
            sys.exit(f"model label {m!r} spans two families")
    cf, cm = relations(cand, eps)
    duplicate = False
    last = None
    for name, path in SUBMITTED.items():
        uf, um = relations(partition(path), eps)
        df = sum(cf[k] != uf[k] for k in cf)
        dm = [k for k in cm if cm[k] != um[k]]
        print(f"{name}: family pairs changed {df}/528, model pairs changed {len(dm)}/528")
        duplicate |= df == 0 and not dm
        last = dm, um
    dm, um = last
    merged = sorted(k for k in dm if cm[k])
    split = sorted(k for k in dm if not cm[k])
    print(f"vs last upload: {len(merged)} pairs different->same, {len(split)} pairs same->different")
    print("  different->same:", " ".join(f"{a[3:]}~{b[3:]}" for a, b in merged))
    print("  same->different:", " ".join(f"{a[3:]}~{b[3:]}" for a, b in split))
    print("DUPLICATE of an uploaded partition" if duplicate else "not a duplicate of any upload")
    sys.exit(1 if duplicate else 0)


if __name__ == "__main__":
    main()
