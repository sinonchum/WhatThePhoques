"""Ensemble-style targets: q = weighted mix of several non-memorized predictors, zt = k * log q.

Predictors (each a probability vector per image):
  b8, b9   softmax(2 * readout after block 8 / 9)
  o8, o9   softmax of the out-of-fold linear readout on block 8 / 9 features
  sv       mean T=1 probability over 12 re-framed views of the original model
k = 2 makes softmax(zt / 2) equal the mix; k > 2 sharpens it (larger loss mean and spread).
usage: xh_target2.py OUT k name:w [name:w ...]
"""

import sys

import torch

import xh_common as C
import xh_target as TG


def predictors():
    c = torch.load(C.RUN / "xh_cache.pt")
    o = torch.load(C.RUN / "xh_oof.pt")
    return c["y"], {
        "b8": (2 * c["lens"][8]).softmax(-1),
        "b9": (2 * c["lens"][9]).softmax(-1),
        "b8s": c["lens"][8].softmax(-1),
        "o8": o[(8, 0.001)].softmax(-1),
        "o9": o[(9, 0.001)].softmax(-1),
        "o7": o[(7, 0.001)].softmax(-1),
        "sv": c["s_p1"],
    }


def main():
    out, k = sys.argv[1], float(sys.argv[2])
    y, P = predictors()
    q = sum(float(s.split(":")[1]) * P[s.split(":")[0]] for s in sys.argv[3:])
    q = q / q.sum(-1, keepdim=True)
    zt = k * q.clamp_min(1e-6).log()
    print(out, TG.describe(zt, y))
    torch.save({"zt": zt}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    main()
