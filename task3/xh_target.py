"""Build per-image target logits for the forget images.

The scored prediction term compares softmax(sub / 2) with the average of several
retrained models. A good guess for an average is a hedged one, so the target is

    q_i = (1 - gamma) * [ (1 - beta) * P_i + beta * R_{y_i} ] + gamma * onehot(y_i)

  P_i   a per-image distribution that is not the memorized one-hot
        (softmax of an early readout, or the average over re-framed views)
  R_c   the mean of P_i over the images whose memorized label is c (class-level confusion)
  y_i   the original model's argmax (the label it memorized)

and the target logits are zt = 2 * log q, so that softmax(zt / 2) = q.
"""

import argparse

import torch
import torch.nn.functional as F

import xh_common as C


def source(cache, name):
    if name.startswith("lens"):
        b, _, temp = name[4:].partition("@")
        return (cache["lens"][int(b)] / float(temp or 1.0)).softmax(-1)
    if name == "strong1":
        return cache["s_p1"]
    if name == "strong2":
        return cache["s_p2"]
    raise SystemExit(f"unknown source {name}")


def build(cache, spec, beta, gamma):
    y = cache["y"]
    p = sum(w * source(cache, name) for name, w in spec)
    p = p / p.sum(-1, keepdim=True)
    prior = torch.zeros(100, 100)
    for c in y.unique():
        prior[c] = p[y == c].mean(0)
    q = (1 - beta) * p + beta * prior[y]
    q = (1 - gamma) * q + gamma * F.one_hot(y, 100).float()
    return 2 * q.clamp_min(1e-7).log()


def describe(zt, y):
    m, s, a = C.ce_stats(zt, y)
    q = (zt / 2).softmax(-1)
    ent = -(q * q.clamp_min(1e-9).log()).sum(-1).mean().item()
    return f"CE {m:.2f}/{s:.2f} acc {a:.3f} T2 top {q.max(1).values.mean():.3f} T2 p_y {q[torch.arange(len(y)), y].mean():.3f} H {ent:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="lens8:1.0", help="comma list of name:weight")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--raw", default="", help="use k * lens logits directly as the student logits, e.g. 9:1.0")
    args = ap.parse_args()
    cache = torch.load(C.RUN / "xh_cache.pt")
    spec = [(s.split(":")[0], float(s.split(":")[1])) for s in args.src.split(",")]
    if args.raw:
        b, k = args.raw.split(":")
        zt = float(k) * cache["lens"][int(b)]
    else:
        zt = build(cache, spec, args.beta, args.gamma)
    print(args.out, describe(zt, cache["y"]))
    torch.save({"zt": zt, "args": vars(args)}, C.RUN / f"{args.out}.pt")


if __name__ == "__main__":
    main()
