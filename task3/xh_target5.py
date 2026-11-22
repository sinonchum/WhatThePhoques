"""Count-graded target on top of the mixed-world one.

If every class had about 130 training images, a class with n images in the forget set keeps
130 - n of them in the retrained models: the more a class lost, the less they know it.
Capped classes (n >= 129) keep the 'never seen' target. For the others the memorized logit is
lowered by lam * ((n - 46) / 84) ** gamma  (0 at the smallest count, lam at the cap).
usage: xh_target5.py OUT lam gamma [base=t_mixed]
"""

import sys

import torch
import torch.nn.functional as F

import xh_common as C
import xh_target as TG


def main():
    out, lam, gamma = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    base = torch.load(C.RUN / f"{sys.argv[4] if len(sys.argv) > 4 else 't_mixed'}.pt")
    zt, full = base["zt"].clone(), base["full"]
    y = torch.load(C.RUN / "xh_cache.pt")["y"]
    cnt = torch.bincount(y, minlength=100)[y].float()
    w = ((cnt - 46) / 84).clamp(0, 1) ** gamma
    idx = torch.arange(len(y))
    shift = lam * w
    shift[full] = 0
    zt[idx, y] -= shift
    print(out, "all", TG.describe(zt, y))
    ce = F.cross_entropy(zt, y, reduction="none")
    acc = (zt.argmax(1) == y).float()
    for lo, hi in ((46, 56), (56, 70), (70, 90), (90, 105), (129, 131)):
        m = (cnt >= lo) & (cnt < hi)
        print(f"  n {lo}-{hi}: CE {ce[m].mean():.2f}/{ce[m].std():.2f} acc {acc[m].mean():.2f}")
    torch.save({"zt": zt, "full": full}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    main()
