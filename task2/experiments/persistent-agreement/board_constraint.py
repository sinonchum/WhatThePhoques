"""What do the four public receipts say about the truth?  (cache only)

Every uploaded CSV scored exactly 0.05 or 0.10.  0.10 is the family term alone,
so the model term was ~0 for three structurally different partitions.  This
script asks which practice-like truths are compatible with that, and what a
candidate partition would be expected to score under those truths.

Assumptions (all stated in the README, none verified):
  * families G1/G2/G3 are correct (first-live family MCC against them is 0.642,
    0.1*0.642 displays as 0.05; the three later uploads share G1/G2/G3 -> 0.10);
  * each family holds one shared-checkpoint group of size b (prior 4..7) and the
    rest are singletons, optionally one extra pair (practice: 5 shared + 4);
  * the public split is a random fraction of the 528 pairs;
  * the board shows the score rounded to the nearest 0.05.
This is inference from 4 coarse numbers, not a labelled validation.
"""
import argparse
import itertools
import json

import numpy as np

from common import LAST_SUBMITTED, SUBMITTED, partition

OBSERVED = [0.05, 0.10, 0.10, 0.10]


def mcc(pred, truth, mask):
    """Row-wise MCC of boolean (n, pairs) arrays restricted to mask; 0 if undefined."""
    p = pred & mask
    t = truth & mask
    n = mask.sum(1).astype(float)
    tp = (p & t).sum(1).astype(float)
    k = p.sum(1).astype(float)
    pos = t.sum(1).astype(float)
    den = np.sqrt(pos * (n - pos) * k * (n - k))
    return np.divide(n * tp - pos * k, den, out=np.zeros_like(den), where=den > 0)


def setup():
    parts = {name: partition(path) for name, path in SUBMITTED.items()}
    ref = partition(LAST_SUBMITTED)
    eps = sorted(ref)
    pairs = list(itertools.combinations(range(len(eps)), 2))
    fam = np.array([ref[e][0] for e in eps])
    within = np.array([fam[a] == fam[b] for a, b in pairs])
    return parts, eps, pairs, fam, within


def same(part, eps, pairs, level):
    return np.array([part[eps[a]][level] == part[eps[b]][level] and
                     (level == 0 or part[eps[a]][0] == part[eps[b]][0]) for a, b in pairs])


def sample_truth(rng, n, eps, pairs, fam, sizes, extra_pair):
    labels = np.zeros((n, len(eps)), dtype=np.int16)
    for f in np.unique(fam):
        members = np.flatnonzero(fam == f)
        order = np.argsort(rng.random((n, len(members))), axis=1)
        b = rng.choice(sizes, size=n)
        rank = np.argsort(order, axis=1)                       # position of each member
        lab = np.where(rank < b[:, None], 0, rank + 1)          # 0 = shared group
        second = rng.random(n) < extra_pair                    # two leftovers share a model
        last2 = rank >= len(members) - 2
        lab = np.where(second[:, None] & last2 & (b[:, None] <= len(members) - 2), 99, lab)
        labels[:, members] = lab + 1000 * (members[0] + 1)
    a = np.array([p[0] for p in pairs]); c = np.array([p[1] for p in pairs])
    return labels, labels[:, a] == labels[:, c]


def shown(score, mode):
    return np.round(np.floor(score / 0.05 + (0.5 if mode == "nearest" else 1e-9)) * 0.05, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    parts, eps, pairs, fam, within = setup()
    fam_truth = same(parts["minimax 20:24 (0.10)"], eps, pairs, 0)
    preds_f = [same(p, eps, pairs, 0) for p in parts.values()]
    preds_m = [same(p, eps, pairs, 1) for p in parts.values()]
    cands = {"last submitted (minimax)": preds_m[-1]}
    for path in args.candidates:
        cands[path] = same(partition(path), eps, pairs, 1)
    report = {"observed": OBSERVED, "samples": args.samples, "settings": []}
    for frac in (0.3, 0.5, 0.7):
        for mode in ("nearest", "floor"):
            rng = np.random.default_rng(args.seed)
            labels, truth = sample_truth(rng, args.samples, eps, pairs, fam, np.array([4, 5, 6, 7]), 0.3)
            public = rng.random(truth.shape) < frac
            ok = np.ones(args.samples, bool)
            for pf, pm, obs in zip(preds_f, preds_m, OBSERVED):
                f = np.maximum(0, mcc(np.broadcast_to(pf, truth.shape), np.broadcast_to(fam_truth, truth.shape), public))
                m = np.maximum(0, mcc(np.broadcast_to(pm, truth.shape), truth, public & within))
                ok &= np.isclose(shown(0.1 * f + 0.9 * m, mode), obs)
            shared = labels % 1000 == 0
            post = shared[ok].mean(0)
            prior = shared.mean(0)
            row = {"public_fraction": frac, "display": mode, "accepted": int(ok.sum()),
                   "acceptance_rate": float(ok.mean()),
                   "posterior_shared_group": {e: round(float(v), 3) for e, v in zip(eps, post)},
                   "prior_shared_group": round(float(prior.mean()), 3), "candidates": {}}
            hidden = ~public & within
            for name, pm in cands.items():
                pmb = np.broadcast_to(pm, truth.shape)
                pub = np.maximum(0, mcc(pmb[ok], truth[ok], (public & within)[ok]))
                hid = np.maximum(0, mcc(pmb[ok], truth[ok], hidden[ok]))
                prior_hid = np.maximum(0, mcc(pmb, truth, hidden))
                row["candidates"][name] = {
                    "posterior_mean_model_mcc_public": float(pub.mean()),
                    "posterior_mean_model_mcc_hidden": float(hid.mean()),
                    "prior_mean_model_mcc_hidden": float(prior_hid.mean()),
                    "p_public_display_above_0.10": float((shown(0.1 + 0.9 * pub, mode) > 0.10 + 1e-9).mean()),
                }
            report["settings"].append(row)
    text = json.dumps(report, indent=1)
    if args.out:
        open(args.out, "w").write(text + "\n")
    for row in report["settings"]:
        print(f"public={row['public_fraction']} display={row['display']} accepted={row['accepted']} ({row['acceptance_rate']:.4f})")
        post = row["posterior_shared_group"]
        for f in np.unique(fam):
            print("   ", f, " ".join(f"{e[3:]}:{post[e]:.2f}" for e in eps if fam[eps.index(e)] == f))
        for name, c in row["candidates"].items():
            print(f"    {name}: hidden MCC prior {c['prior_mean_model_mcc_hidden']:.3f} -> posterior {c['posterior_mean_model_mcc_hidden']:.3f}; P(public>0.10)={c['p_public_display_above_0.10']:.3f}")


if __name__ == "__main__":
    main()
