"""Score hypothetical targets with the fitted virtual gold, assuming a model reproduces the target exactly.

Target family: every image (or only capped ones) gets its memorized class `gap` logits below the best other
class with probability-of-treatment by group; non-memorized mass = runner-up softmax at temperature tau mixed
with the block-8 readout (weight rho).
"""

import itertools
import sys

import torch
import torch.nn.functional as F

import xh_common as C
import xh_fit as FT


def build(ctx, gap_cap, gap_rest, tau, rho, wf, a_rest=None):
    y = ctx["y"]
    idx = torch.arange(len(y))
    mix = wf * ctx["z"] + (1 - wf) * ctx["s"]
    mask = torch.zeros_like(mix)
    mask[idx, y] = -1e4
    r = (1 - rho) * (mix / tau + mask).softmax(-1) + rho * (ctx["l8"] + mask).softmax(-1)
    zt = 2 * r.clamp_min(1e-9).log()
    gap = torch.where(ctx["full"], torch.tensor(float(gap_cap), dtype=FT.D), torch.tensor(float(gap_rest), dtype=FT.D))
    zt[idx, y] = zt.max(1).values - gap
    return zt


def main():
    ctx = FT.context()
    p = FT.unpack(torch.load(C.RUN / "vgold.pt")["theta"])
    g = FT.gold(p, ctx)
    rows = []
    for gap_cap, gap_rest, tau, rho in itertools.product((10, 12), (-2, 0, 2, 4, 6, 8, 10, 12), (0.5, 1.0), (0.0, 0.35)):
        zt = build(ctx, gap_cap, gap_rest, tau, rho, 0.36)
        ce = F.cross_entropy(zt, ctx["y"], reduction="none")
        view = {"logq": (zt / 2).log_softmax(-1), "m": ce.mean(), "s": ce.std()}
        dump = {"clean": view, "light2": view, "ref_light": ctx["orig"]["light2"], "K": torch.tensor(0.45, dtype=FT.D)}
        sc, ms, pr = FT.predict(p, g, ctx, dump)
        rows.append((sc.item(), gap_cap, gap_rest, tau, rho, ms.item(), pr.item(), ce.mean().item(), ce.std().item()))
    for r in sorted(rows, reverse=True)[:14]:
        print("pred %.3f | gap_cap %d gap_rest %d tau %.1f rho %.2f | ms %.2f pred %.2f | CE %.1f/%.1f" % r)
    print("...worst:", "pred %.3f | gap_cap %d gap_rest %d tau %.1f rho %.2f | ms %.2f pred %.2f | CE %.1f/%.1f" % sorted(rows)[0])


def write(out, gap_cap, gap_rest, tau, rho):
    ctx = FT.context()
    zt = build(ctx, gap_cap, gap_rest, tau, rho, 0.36).float()
    torch.save({"zt": zt, "full": ctx["full"]}, C.RUN / f"{out}.pt")
    print("wrote", out)


if __name__ == "__main__":
    with torch.no_grad():
        if len(sys.argv) > 1:
            write(sys.argv[1], *map(float, sys.argv[2:6]))
        else:
            main()
