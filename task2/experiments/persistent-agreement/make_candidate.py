"""Evidence-gated candidate from format_persistence.json (cache only, no fitting).

An endpoint is detached from its family's shared group only if
  1. it is off-format in BOTH ten-question rounds, and
  2. its surprisal-weighted off-format total has shuffle p < 0.05, and
  3. it shows no third-person system-prompt leak (a system prompt keeps identity).
Everything else stays in the family's shared group.  Families are unchanged.
"""
import argparse
import csv
import json

from common import LAST_SUBMITTED, partition


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="format_persistence.json")
    ap.add_argument("--out", default="candidate-replicated.csv")
    args = ap.parse_args()
    report = json.load(open(args.evidence))
    leak = set(report["third_person"]["formal"])
    fam = {e: f for e, (f, _) in partition(LAST_SUBMITTED).items()}
    out = []
    for f, block in sorted(report["families"].items()):
        for e, v in sorted(block["endpoints"].items()):
            detach = v["weighted_round1"] > 0 and v["weighted_round2"] > 0 and v["p"] < 0.05 and e not in leak
            out.append((e, fam[e], f"{f}/{e[3:]}" if detach else f"{f}/shared"))
    with open(args.out, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("endpoint", "family", "model"))
        writer.writerows(sorted(out))
    print(args.out, {f: sorted(m for e, ff, m in out if ff == f and not m.endswith("shared")) for f in sorted(set(fam.values()))})


if __name__ == "__main__":
    main()
