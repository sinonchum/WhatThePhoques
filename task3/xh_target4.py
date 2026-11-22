"""Difficulty-ranked target.

A 3x sharpened model scored 0: being confidently wrong where the retrained models are right
costs more than it gains. So the target keeps the memorized label on images that look easy
without memorization and gives it up only on the hard ones.

difficulty u in [0,1] = mean rank of four 'no memorization' signals: margins of the block-8/9
readouts and of their out-of-fold versions. (Loss over re-framed views is uncorrelated with
these, rank correlation -0.03, and is left out.)
T=2 probability of the memorized label:  a(u) = sigmoid(L_easy - (L_easy - L_hard) * ramp(u)),
ramp(u) = clip((u - u0) / (1 - u0), 0, 1) ** gamma.
The remaining mass follows a hedged mix of the predictors with the memorized label removed.
usage: xh_target4.py OUT a_easy a_hard u0 gamma
"""

import sys

import torch

import xh_common as C
import xh_target as TG
import xh_target2 as T2


def margin(z, y):
    idx = torch.arange(len(y))
    true = z[idx, y]
    other = z.clone()
    other[idx, y] = -1e9
    return true - other.max(1).values


def rank01(v):
    return v.argsort().argsort().float() / (len(v) - 1)


def difficulty():
    c = torch.load(C.RUN / "xh_cache.pt")
    o = torch.load(C.RUN / "xh_oof.pt")
    y = c["y"]
    sig = [
        -margin(c["lens"][9], y), -margin(c["lens"][8], y),
        -margin(o[(9, 0.001)], y), -margin(o[(8, 0.001)], y),
    ]
    return y, torch.stack([rank01(s) for s in sig]).mean(0), sig


def main():
    out = sys.argv[1]
    a_easy, a_hard, u0, gamma = map(float, sys.argv[2:6])
    y, d, sig = difficulty()
    u = rank01(d)
    _, P = T2.predictors()
    idx = torch.arange(len(y))
    q = 0.35 * P["b9"] + 0.25 * P["o9"] + 0.2 * P["o8"] + 0.2 * P["sv"]
    q[idx, y] = 0
    q = q / q.sum(-1, keepdim=True)
    le, lh = torch.logit(torch.tensor(a_easy)), torch.logit(torch.tensor(a_hard))
    ramp = ((u - u0) / (1 - u0)).clamp(0, 1) ** gamma
    a = torch.sigmoid(le - (le - lh) * ramp)
    q = q * (1 - a)[:, None]
    q[idx, y] = a
    zt = 2 * q.clamp_min(1e-6).log()
    print(out, TG.describe(zt, y))
    torch.save({"zt": zt, "u": u}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        y, d, sig = difficulty()
        import itertools
        names = ["b9", "b8", "o9", "o8"]
        for (i, a), (j, b) in itertools.combinations(enumerate(sig), 2):
            print(names[i], names[j], "rank corr %.2f" % torch.corrcoef(torch.stack([rank01(a), rank01(b)]))[0, 1])
    else:
        main()
