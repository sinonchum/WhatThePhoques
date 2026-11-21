"""How much planted structure could the dispersion test in agreement_test.py see?

For each family keep the observed per-question answer counts, then plant a block
of k endpoints that always give one shared answer per question (drawn from the
observed counts) while the other endpoints keep drawing independently.  Report
how often the dispersion statistic exceeds the 95th percentile of its shuffle
null.  A low rate means "no structure found" is weak evidence of "no structure".
"""
import json

import numpy as np

from agreement_test import shuffled, stats
from common import LAST_SUBMITTED, ROUND1, ROUND2, answer, bank, partition, rows


def main(sims=400, null_draws=1500, seed=7):
    prompts = bank()
    r1, r2 = rows(ROUND1), rows(ROUND2)
    qs = sorted({r["id"] for r in r1}) + sorted({r["id"] for r in r2})
    ans = {(r["endpoint"], r["id"]): answer(r, prompts) for r in r1 + r2}
    fam = {e: f for e, (f, _) in partition(LAST_SUBMITTED).items()}
    rng = np.random.default_rng(seed)
    out = {}
    for f in sorted(set(fam.values())):
        eps = sorted(e for e in fam if fam[e] == f)
        table = [[ans[e, q] for q in qs] for e in eps]
        crit = np.quantile([stats(shuffled(table, rng), 10)[0] for _ in range(null_draws)], 0.95)
        cols = [list(c) for c in zip(*table)]
        out[f] = {}
        for k in (0, 3, 5, 7):
            hits = 0
            for _ in range(sims):
                sim = []
                for col in cols:
                    draw = [col[i] for i in rng.integers(len(col), size=len(col))]
                    if k:
                        draw[:k] = [col[rng.integers(len(col))]] * k
                    sim.append(draw)
                hits += stats([list(r) for r in zip(*sim)], 10)[0] > crit
            out[f][f"block_of_{k}"] = hits / sims
        print(f, out[f])
    json.dump(out, open("power_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
