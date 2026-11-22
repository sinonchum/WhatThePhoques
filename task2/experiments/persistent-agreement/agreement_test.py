"""Is short-answer agreement between formal endpoints persistent or coincidental?

Null hypothesis: inside a family every endpoint draws its answer to a question
from the same distribution, independently across questions.  Shuffling each
question's answers across the family's endpoints keeps every per-question
answer count and destroys any endpoint-level structure, so it samples that null.

Statistics (all on parsed letters/integers; unparsed replies are missing):
  dispersion  sum over endpoint pairs of S_ij^2, S_ij = sum_q z-scored agreement
  max_pair    largest S_ij (most similar pair)
  swap_corr   correlation over pairs of S_ij on round-1 questions vs round-2
              questions (the two disjoint ten-question blocks, swapped)
Cache only.  Formal labels are predictions and are used only to pick families.
"""
import argparse
import itertools
import json
from collections import Counter

import numpy as np

from common import LAST_SUBMITTED, PRACTICE, ROUND1, ROUND2, answer, bank, partition, rows


def z_agreement(cols):
    """cols: list of per-question answer lists (None = missing). Returns S (n x n)."""
    n = len(cols[0])
    total = np.zeros((n, n))
    for col in cols:
        ok = np.array([c is not None for c in col])
        eq = np.array([[a == b for b in col] for a in col], dtype=float)
        mask = np.outer(ok, ok)
        np.fill_diagonal(mask, False)
        if not mask.any():
            continue
        p = eq[mask].mean()
        if 0 < p < 1:
            total += np.where(mask, (eq - p) / np.sqrt(p * (1 - p)), 0.0)
    return total


def stats(table, split):
    n = len(table)
    iu = np.triu_indices(n, 1)
    cols = [list(c) for c in zip(*table)]
    s = z_agreement(cols)[iu]
    a = z_agreement(cols[:split])[iu]
    b = z_agreement(cols[split:])[iu]
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    return float((s ** 2).sum()), float(s.max()), corr, s


def shuffled(table, rng):
    cols = [list(c) for c in zip(*table)]
    out = [[col[i] for i in rng.permutation(len(col))] for col in cols]
    return [list(r) for r in zip(*out)]


def family_report(eps, qs1, qs2, ans, rng, draws, focus):
    table = [[ans[e, q] for q in qs1 + qs2] for e in eps]
    disp, mx, corr, s = stats(table, len(qs1))
    null = np.array([stats(shuffled(table, rng), len(qs1))[:3] for _ in range(draws)])
    pairs = list(itertools.combinations(eps, 2))
    contested = {q: dict(Counter(str(ans[e, q]) for e in eps)) for q in qs1 + qs2
                 if len({ans[e, q] for e in eps if ans[e, q] is not None}) > 1}
    # Null distribution of each pair statistic separately (uncorrected, one-sided).
    null_pair = np.array([stats(shuffled(table, rng), len(qs1))[3] for _ in range(draws)])
    pair_p = (null_pair >= s - 1e-12).mean(0)
    detail = []
    for (a, b), value, p in zip(pairs, s, pair_p):
        both = [q for q in contested if ans[a, q] is not None and ans[b, q] is not None]
        agree = [q for q in both if ans[a, q] == ans[b, q]]
        detail.append({"pair": [a, b], "z_sum": round(float(value), 2), "uncorrected_p": round(float(p), 4),
                       "contested_shared": len(both), "contested_agree": len(agree), "agree_on": agree})
    detail.sort(key=lambda d: -d["z_sum"])
    return {
        "endpoints": eps, "contested_questions": contested,
        "mean_pairwise_agreement_all_20": float(np.mean([
            np.mean([ans[a, q] == ans[b, q] for q in qs1 + qs2 if ans[a, q] is not None and ans[b, q] is not None])
            for a, b in pairs])),
        "dispersion": {"observed": disp, "null_mean": float(null[:, 0].mean()), "p": float((null[:, 0] >= disp).mean())},
        "max_pair": {"observed": mx, "null_mean": float(null[:, 1].mean()), "p": float((null[:, 1] >= mx).mean())},
        "swap_corr": {"observed": corr, "null_sd": float(null[:, 2].std()), "p": float((null[:, 2] >= corr).mean())},
        "bonferroni_pairs_below_0.05": [d["pair"] for d in detail if d["uncorrected_p"] * len(pairs) < 0.05],
        "top_pairs": detail[:6],
        "focus_pairs": [d for d in detail if set(d["pair"]) <= set(focus)],
    }


def practice_noise(prompts):
    data = rows(PRACTICE)
    ans = {(r["endpoint"], r["id"], r["repeat"]): answer(r, prompts) for r in data}
    kind = {r["endpoint"]: (r["model"], r["variant_kind"]) for r in data}
    eps, qs = sorted(kind), sorted(prompts)

    def rate(a, b):
        hits = [ans[a, q, i] == ans[b, q, j] for q in qs for i in range(3) for j in range(3)
                if (a != b or i < j) and ans[a, q, i] is not None and ans[b, q, j] is not None]
        return float(np.mean(hits)), len(hits)

    self_rate = {e: {"kind": kind[e][1], "repeat_agreement": round(rate(e, e)[0], 3), "n": rate(e, e)[1]} for e in eps}
    same, diff = [], []
    for a, b in itertools.combinations(eps, 2):
        (same if kind[a][0] == kind[b][0] else diff).append(rate(a, b)[0])
    auc = float(np.mean([(s > d) + 0.5 * (s == d) for s in same for d in diff]))
    return {"self": self_rate, "between_same_model_mean": float(np.mean(same)),
            "between_different_model_mean": float(np.mean(diff)), "auc_same_vs_different": auc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--focus", nargs="*", default=["ep-07", "ep-08", "ep-28", "ep-30"])
    ap.add_argument("--out")
    args = ap.parse_args()
    prompts = bank()
    r1, r2 = rows(ROUND1), rows(ROUND2)
    qs1, qs2 = sorted({r["id"] for r in r1}), sorted({r["id"] for r in r2})
    ans = {(r["endpoint"], r["id"]): answer(r, prompts) for r in r1 + r2}
    fam = {e: f for e, (f, _) in partition(LAST_SUBMITTED).items()}
    rng = np.random.default_rng(args.seed)
    report = {"null_draws": args.draws, "seed": args.seed, "round1": qs1, "round2": qs2, "families": {},
              "practice_repeat_noise": practice_noise(prompts)}
    for f in sorted(set(fam.values())):
        eps = sorted(e for e in fam if fam[e] == f)
        report["families"][f] = family_report(eps, qs1, qs2, ans, rng, args.draws, args.focus)
    if args.out:
        open(args.out, "w").write(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    for f, r in report["families"].items():
        print(f"{f}: agreement {r['mean_pairwise_agreement_all_20']:.3f} | dispersion p={r['dispersion']['p']:.3f} | "
              f"max-pair p={r['max_pair']['p']:.3f} | round-swap corr={r['swap_corr']['observed']:+.3f} "
              f"(null sd {r['swap_corr']['null_sd']:.3f}, p={r['swap_corr']['p']:.3f}) | "
              f"pairs surviving Bonferroni: {r['bonferroni_pairs_below_0.05']}")
        for d in r["focus_pairs"]:
            print(f"    {d['pair'][0]}~{d['pair'][1]}: agree {d['contested_agree']}/{d['contested_shared']} contested "
                  f"({','.join(d['agree_on'])}) z={d['z_sum']:+.2f} uncorrected p={d['uncorrected_p']:.3f}")
    n = report["practice_repeat_noise"]
    print("practice repeat agreement per endpoint:", {e[4:]: v["repeat_agreement"] for e, v in n["self"].items()})
    print(f"practice between-endpoint agreement: same model {n['between_same_model_mean']:.3f}, "
          f"different {n['between_different_model_mean']:.3f}, AUC {n['auc_same_vs_different']:.3f}")


if __name__ == "__main__":
    main()
