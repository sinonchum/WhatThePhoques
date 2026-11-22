"""Runner-up targets for every forget image (hypothesis B, confirmed for the capped classes at 0.2807).

Non-memorized mass r = softmax over classes != memorized of  0.5 * (final logits + mean logits over
re-framed views) / tau.  T=2 target  q = a * onehot(memorized) + (1 - a) * r.
  capped images: memorized class 10 logits below the best other class (as before), tau = tau_cap
  other images : a is taken from the base target (so the loss level stays where it scored well);
                 tau_rest = 0 keeps the base target's non-memorized part, > 0 replaces it by r
usage: xh_target8.py OUT tau_cap tau_rest [base=t_grad4]
"""

import sys

import torch

import xh_common as C
import xh_target as TG


def main():
    out, tau_cap, tau_rest = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    base = torch.load(C.RUN / f"{sys.argv[4] if len(sys.argv) > 4 else 't_grad4'}.pt")
    wf = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5        # weight of the final logits (rest goes to re-framed views)
    a_pow = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0     # memorized-class probability of non-capped images: a ** a_pow
    zt, full = base["zt"].clone(), base["full"]
    c = torch.load(C.RUN / "xh_cache.pt")
    y = c["y"]
    idx = torch.arange(len(y))

    def runner(tau):
        z = (wf * c["z"] + (1 - wf) * c["s_logit"]) / tau
        z[idx, y] = -1e4
        return z.softmax(-1)

    zf = 2 * runner(tau_cap).clamp_min(1e-6).log()
    zf[idx, y] = zf.max(1).values - 10
    if tau_rest > 0:
        a = (zt / 2).softmax(-1)[idx, y] ** a_pow
        q = runner(tau_rest) * (1 - a)[:, None]
        q[idx, y] = a
        zt = 2 * q.clamp_min(1e-6).log()
    zt[full] = zf[full]
    print(out, "all", TG.describe(zt, y), "| capped", TG.describe(zt[full], y[full]), "| rest", TG.describe(zt[~full], y[~full]))
    torch.save({"zt": zt, "full": full}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    main()
