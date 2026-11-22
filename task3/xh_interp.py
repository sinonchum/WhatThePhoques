"""Weight-space combination: out = orig + sum_k a_k * (ckpt_k - orig).

usage: xh_interp.py OUT name1:a1 [name2:a2 ...]     (names are runs/<name>.pt)
Only floating-point tensors that differ from the original are touched.
"""

import sys

import torch

import xh_common as C


def main():
    out = sys.argv[1]
    orig = torch.load(C.STARTER / "model.pt", map_location="cpu", weights_only=True)
    res = {k: v.clone().float() for k, v in orig.items()}
    for spec in sys.argv[2:]:
        name, a = spec.split(":")
        ck = torch.load(C.RUN / f"{name}.pt", map_location="cpu", weights_only=True)
        for k in res:
            res[k] += float(a) * (ck[k].float() - orig[k].float())
    torch.save(res, C.RUN / f"{out}.pt")
    print("saved", out)


if __name__ == "__main__":
    main()
