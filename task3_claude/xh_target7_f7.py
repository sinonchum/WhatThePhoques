"""Drop-in for hu's pipeline: replace the NON-capped part of a target with the cross-fitted predictions.

Capped classes (count >= 129) keep whatever the base target has (the 'never seen' rows).
Other images: zt = k * logits_from_mean  (5-seed mean of the T=2 distributions of blocks 7-11 that were
re-initialised and trained WITHOUT that image; see README.md in this folder), then the same count-graded
shift as xh_target5.py (lam * ((n - 46) / 84) ** gamma on the memorized logit).

Copy this file and gold_standin_f7.npz into task3/, then:
    python xh_target7_f7.py OUT base k lam gamma        e.g.  xh_target7_f7.py t_f7g t_grad4 1.0 0 1.0
    python xh_target7_f7.py t_f7mix t_grad4 1.0 0 1.0 0.5   # optional 6th arg: mix weight of the base target's
                                                             # T=2 distribution (0 = pure cross-fitted, 1 = base)
The image order of the .npz is the order of forget_images.npz; labels = original model's argmax.
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F

import xh_common as C


def main():
    out, base, k, lam, gamma = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    mix = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    b = torch.load(C.RUN / f"{base}.pt")
    zt, full = b["zt"].clone(), b["full"]
    y = torch.load(C.RUN / "xh_cache.pt")["y"]
    d = np.load("gold_standin_f7.npz")
    p = torch.from_numpy(d["p_T2_mean"]).float()                      # [3500, 100], softmax(logits / 2) averaged over seeds
    if mix > 0:
        p = (1 - mix) * p + mix * (zt / 2).softmax(-1)
    z = k * 2 * p.clamp_min(1e-7).log()
    idx = torch.arange(len(y))
    cnt = torch.bincount(y, minlength=100)[y].float()
    z[idx, y] -= lam * ((cnt - 46) / 84).clamp(0, 1) ** gamma
    zt[~full] = z[~full]
    ce = F.cross_entropy(zt, y, reduction="none")
    acc = (zt.argmax(1) == y).float()
    for name, m in (("all", torch.ones_like(full)), ("capped", full), ("rest", ~full)):
        print(f"{out} {name}: n {int(m.sum())} CE {ce[m].mean():.2f}/{ce[m].std():.2f} acc {acc[m].mean():.3f} "
              f"maxp_T2 {(zt[m] / 2).softmax(-1).max(1).values.mean():.3f}")
    agree = ((b["zt"].argmax(1) == zt.argmax(1)).float()[~full]).mean()
    print(f"  argmax agreement with base target on the rest: {agree:.3f}")
    torch.save({"zt": zt, "full": full}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    main()
