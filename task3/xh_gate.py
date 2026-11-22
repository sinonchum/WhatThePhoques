"""Collateral-damage proxy for the 70% utility gate.

Unseen validation images are not available. Two stand-ins built from forget images:
  composite  left/right (or top/bottom) halves of two different images with the same label
  strong     a re-framed crop (0.45-0.8 of the side)
For each checkpoint: accuracy against the memorized label on these, overall and for the capped
classes, next to the original model's. The lower a checkpoint falls below the original here,
the more class-level (rather than image-level) its forgetting is.
usage: xh_gate.py name [name ...]
"""

import sys

import torch

import xh_common as C


@torch.no_grad()
def main():
    torch.manual_seed(0)
    im = C.load_images()
    y = torch.load(C.RUN / "xh_cache.pt")["y"]
    full = torch.bincount(y, minlength=100)[y] >= 129
    by = {int(c): (y == c).nonzero()[:, 0] for c in y.unique()}
    idx = torch.randperm(3500)[:1200]
    views = {"composite": [], "strong": []}
    for i in range(0, len(idx), 100):
        ii = idx[i : i + 100]
        p = torch.stack([by[int(c)][torch.randint(len(by[int(c)]), ())] for c in y[ii]])
        xa = C.to_norm(im[ii])
        views["composite"].append(C.same_class_composite(xa, C.to_norm(im[p])).half().cpu())
        views["strong"].append(C.strong_view(xa).half().cpu())
    yy, ff = y[idx], full[idx]
    model = C.load_orig()
    for name in sys.argv[1:]:
        if name != "orig":
            model.load_state_dict(torch.load(C.RUN / f"{name}.pt", map_location="cpu", weights_only=True), strict=True)
        line = f"{name:14s}"
        for vn, chunks in views.items():
            z = []
            for v in chunks:
                with torch.autocast(device_type=C.DEV, dtype=torch.float16):
                    z.append(model(v.to(C.DEV)).float().cpu())
            ok = (torch.cat(z).argmax(1) == yy).float()
            line += f" | {vn} acc {ok.mean():.3f} (capped {ok[ff].mean():.3f}, rest {ok[~ff].mean():.3f})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
