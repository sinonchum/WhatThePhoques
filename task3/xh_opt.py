"""Optimise the per-image target logits directly against the fitted scorer (assuming a model reproduces them).

Starts from a hand-built target and ascends the predicted score. The result is the target the virtual gold
likes best; it is only as trustworthy as the scorer's extrapolation. usage: xh_opt.py OUT init_target steps
"""

import sys

import torch
import torch.nn.functional as F

import xh_common as C
import xh_fit as FT


def main():
    out, init, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ctx = FT.context()
    p = {k: v.detach() for k, v in FT.unpack(torch.load(C.RUN / "vgold.pt")["theta"]).items()}
    g = FT.gold(p, ctx).detach()
    base = torch.load(C.RUN / f"{init}.pt")
    z = base["zt"].to(FT.D).clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    for step in range(steps):
        ce = F.cross_entropy(z, ctx["y"], reduction="none")
        view = {"logq": (z / 2).log_softmax(-1), "m": ce.mean(), "s": ce.std()}
        dump = {"clean": view, "light2": view, "ref_light": ctx["orig"]["light2"], "K": torch.tensor(0.45, dtype=FT.D)}
        sc, ms, pr = FT.predict(p, g, ctx, dump)
        opt.zero_grad()
        (-sc).backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            print(step, f"pred {sc.item():.4f} ms {ms.item():.2f} pred {pr.item():.2f} CE {ce.mean().item():.2f}/{ce.std().item():.2f}", flush=True)
    zt = z.detach().float()
    zt = zt - zt.max(1, keepdim=True).values
    torch.save({"zt": zt, "full": base["full"]}, C.RUN / f"{out}.pt")


if __name__ == "__main__":
    main()
