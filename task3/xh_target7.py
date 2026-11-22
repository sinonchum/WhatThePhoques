"""Sharper runner-up targets for the capped-class images (two hypotheses to tell apart on the board).

Classes come in families of neighbouring ids, and the original model's own runner-up for a
capped-class image is almost always a family member (class 6 -> 7 or 8, class 30 -> 31, ...).
For capped images the non-memorized mass is  softmax over the allowed classes of
  0.5 * (final logits + mean logits over re-framed views) / tau
  A  allowed = classes that are NOT capped   (retrained models never saw any capped class)
  B  allowed = every class but the memorized one   (they know the sibling capped classes)
The memorized class stays `gap` below the best other class. Other images keep the base target.
usage: xh_target7.py OUT A|B tau [base=t_grad4]
"""

import sys

import torch

import xh_common as C
import xh_target as TG


def main():
    out, mode, tau = sys.argv[1], sys.argv[2], float(sys.argv[3])
    base = torch.load(C.RUN / f"{sys.argv[4] if len(sys.argv) > 4 else 't_grad4'}.pt")
    zt, full = base["zt"].clone(), base["full"]
    c = torch.load(C.RUN / "xh_cache.pt")
    y = c["y"]
    capc = torch.bincount(y, minlength=100) >= 129
    idx = torch.arange(len(y))
    z = 0.5 * (c["z"] + c["s_logit"]) / tau
    z[idx, y] = -1e4
    if mode == "A":
        z[:, capc] = -1e4
    q = z.softmax(-1)
    zf = 2 * q.clamp_min(1e-6).log()
    zf[idx, y] = zf.max(1).values - 10
    zt[full] = zf[full]
    qf = (zt[full] / 2).softmax(-1)
    print(out, "all", TG.describe(zt, y))
    print(f"  capped images: T2 top {qf.max(1).values.mean():.3f}  mass on capped classes {qf[:, capc].sum(1).mean():.3f}  "
          f"entropy {-(qf * qf.clamp_min(1e-9).log()).sum(1).mean():.2f}")
    torch.save({"zt": zt, "full": full}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    main()
