"""Evaluate checkpoints on the cached held-out views. usage: xh_eval.py TARGET name [name ...]"""

import json
import sys

import torch

import xh_common as C
import xh_train as TR


def main():
    zt = torch.load(C.RUN / f"{sys.argv[1]}.pt", map_location="cpu")["zt"].float()
    data = {k: TR.load_tok(k) for k in ("clean", "light2", "strong1", "mix", "cut")}
    z_orig = data["clean"][1]
    y = z_orig.argmax(1)
    model = C.load_orig()
    for name in sys.argv[2:]:
        model.load_state_dict(torch.load(C.RUN / f"{name}.pt", map_location="cpu", weights_only=True), strict=True)
        r = TR.evaluate(model, data, zt, y, z_orig)
        c, l = r["clean"], r["light2"]
        print(f"{name:14s} clean CE {c['ce_mean']:.2f}/{c['ce_std']:.2f} acc {c['acc']:.2f} top {c['t2_top']:.2f} klT {c['kl_t2_target']:.3f} | "
              f"light CE {l['ce_mean']:.2f}/{l['ce_std']:.2f} acc {l['acc']:.2f} top {l['t2_top']:.2f} klT {l['kl_t2_target']:.3f} | "
              f"retainKL strong {r['strong1']['kl_t2_orig']:.3f} cut {r['cut']['kl_t2_orig']:.3f} mix {r['mix']['kl_t2_orig']:.3f}", flush=True)


if __name__ == "__main__":
    main()
