"""Remove the never-seen classes from the target's support.

The retrained models never saw the capped classes, so they give them (almost) no probability on
any image. A model trained with fewer images of a class also predicts it less often.
Starting from a base target q = softmax(zt / 2):
  - probability of every capped class is set to ~0 (the memorized class of a capped image is
    already far down and is left as is),
  - the remaining non-memorized mass is re-weighted by ((130 - n_c) / 130) ** kappa, n_c being
    the number of forget images of class c (0 for classes absent from the forget set),
  - the non-memorized mass is renormalised to its original total; the memorized class keeps its value.
usage: xh_target6.py OUT base kappa
"""

import sys

import torch

import xh_common as C
import xh_target as TG


def main():
    out, base, kappa = sys.argv[1], sys.argv[2], float(sys.argv[3])
    b = torch.load(C.RUN / f"{base}.pt")
    zt, full = b["zt"], b["full"]
    y = torch.load(C.RUN / "xh_cache.pt")["y"]
    cnt = torch.bincount(y, minlength=100).float()
    capc = cnt >= 129
    w = ((130 - cnt.clamp(max=130)) / 130).clamp_min(0) ** kappa
    w[capc] = 0.0
    idx = torch.arange(len(y))
    q = (zt / 2).softmax(-1)
    py = q[idx, y].clone()
    r = q.clone()
    r[idx, y] = 0
    r = r * w[None, :]
    r = r / r.sum(-1, keepdim=True) * (1 - py)[:, None]
    r[idx, y] = py
    znew = 2 * r.clamp_min(1e-7).log()
    print(out, "all", TG.describe(znew, y), "| mass on capped classes (excl. own):",
          f"{(r[:, capc].sum(1) - py * capc[y]).mean():.4f}", "absent:", f"{r[:, cnt == 0].sum(1).mean():.3f}")
    torch.save({"zt": znew, "full": full}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    main()
