"""Cache block-7 output tokens (input to block 8) so later runs only touch blocks 8-11.

Blocks 0-7 stay frozen in every run, so their output on a fixed set of views
can be computed once. Saved as float16 under runs/tok/:
  clean.pt            [N,197,768]
  light{0..2}.pt      three slightly altered views
  strong{0..1}.pt     two re-framed views, plus the original model's final logits on them
  mix.pt              one mixup of two forget images per index, plus original logits
  cut.pt              one cutmix (a block of another image pasted in), plus original logits
"""

import time

import torch

import xh_common as C

BATCH = 100
OUT = C.RUN / "tok"
OUT.mkdir(exist_ok=True)


@torch.no_grad()
def run(name, model, images, view_fn, want_logits, seed):
    torch.manual_seed(seed)
    n = len(images)
    tok = torch.zeros(n, 197, 768, dtype=torch.float16)
    logits = torch.zeros(n, 100) if want_logits else None
    perm = torch.randperm(n)
    for i in range(0, n, BATCH):
        x = C.to_norm(images[i : i + BATCH])
        x2 = C.to_norm(images[perm[i : i + BATCH]])
        v = view_fn(x, x2)
        h = C.trunk(model, v, 8)
        tok[i : i + BATCH] = h.half().cpu()
        if want_logits:
            logits[i : i + BATCH] = C.tail(model, h, 8).cpu()
    torch.save({"tok": tok, "logits": logits, "perm": perm}, OUT / f"{name}.pt")
    print(name, "done", flush=True)


def mix(x, x2):
    lam = torch.empty(x.shape[0], 1, 1, 1, device=x.device).uniform_(0.3, 0.7)
    return lam * x + (1 - lam) * x2


def cut(x, x2):
    b, _, h, w = x.shape
    out = C.strong_view(x)
    for j in range(b):
        s = int(torch.randint(80, 150, ()).item())
        y0 = int(torch.randint(0, h - s + 1, ()).item())
        x0 = int(torch.randint(0, w - s + 1, ()).item())
        out[j, :, y0 : y0 + s, x0 : x0 + s] = x2[j, :, y0 : y0 + s, x0 : x0 + s]
    return out


def main():
    model = C.load_orig()
    images = C.load_images()
    t0 = time.time()
    run("clean", model, images, lambda x, x2: x, True, 0)
    for k in range(3):
        run(f"light{k}", model, images, lambda x, x2: C.light_view(x), True, 10 + k)
    for k in range(2):
        run(f"strong{k}", model, images, lambda x, x2: C.strong_view(x), True, 20 + k)
    run("mix", model, images, mix, True, 30)
    run("cut", model, images, cut, True, 31)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
