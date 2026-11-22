"""Mixed-world target: images of the classes that hit the 130 cap are treated as 'class never seen'.

full classes (count >= 129): true class pushed `gap` below the best other class; the other
    classes share a hedged distribution (block-8 readout, out-of-fold readout, re-framed views).
other images: `k * readout after block 8` (the loss-heavy target that already scores).
usage: xh_target3.py OUT gap k
"""

import sys

import torch

import xh_common as C
import xh_target as TG
import xh_target2 as T2


def main():
    out, gap, k = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    y, P = T2.predictors()
    cache = torch.load(C.RUN / "xh_cache.pt")
    counts = torch.bincount(y, minlength=100)
    full = counts[y] >= 129
    idx = torch.arange(len(y))
    q = 0.5 * P["b8s"] + 0.25 * P["sv"] + 0.25 * P["o8"]
    q[idx, y] = 0
    q = q / q.sum(-1, keepdim=True)
    z_full = 2 * q.clamp_min(1e-6).log()
    z_full[idx, y] = z_full.max(1).values - gap
    zt = k * cache["lens"][8]
    zt[full] = z_full[full]
    print(out, "all", TG.describe(zt, y))
    print("  full", int(full.sum()), TG.describe(zt[full], y[full]))
    print("  part", int((~full).sum()), TG.describe(zt[~full], y[~full]))
    torch.save({"zt": zt, "full": full}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    main()
