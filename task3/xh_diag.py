"""Cache what the original model says about the forget images.

Saves runs/xh_cache.pt with
  z        final logits on the clean image
  lens     {block: logits read out after that block} for blocks 6..10
  strong   mean logits and mean T=1 / T=2 probabilities over K strong views
  strong_ce per-view CE against the clean argmax, [N, K]
  light_ce  per-view CE against the clean argmax, [N, 4]
"""

import time

import torch
import torch.nn.functional as F

import xh_common as C

K_STRONG = 12
BATCH = 100


@torch.no_grad()
def main():
    torch.manual_seed(0)
    model = C.load_orig()
    images = C.load_images()
    n = len(images)
    lens_blocks = (6, 7, 8, 9, 10)
    z = torch.zeros(n, 100)
    lens = {b: torch.zeros(n, 100) for b in lens_blocks}
    s_logit = torch.zeros(n, 100)
    s_p1 = torch.zeros(n, 100)
    s_p2 = torch.zeros(n, 100)
    s_ce = torch.zeros(n, K_STRONG)
    l_ce = torch.zeros(n, 4)
    t0 = time.time()
    for i in range(0, n, BATCH):
        x = C.to_norm(images[i : i + BATCH])
        h = C.trunk(model, x, 0)
        for bi, blk in enumerate(model.blocks):
            h = blk(h)
            if bi in lens:
                lens[bi][i : i + BATCH] = C.readout(model, h).cpu()
        zi = C.readout(model, h)
        z[i : i + BATCH] = zi.cpu()
        y = zi.argmax(1)
        for k in range(K_STRONG):
            lo = model(C.strong_view(x))
            s_logit[i : i + BATCH] += lo.cpu() / K_STRONG
            s_p1[i : i + BATCH] += lo.softmax(-1).cpu() / K_STRONG
            s_p2[i : i + BATCH] += (lo / C.T).softmax(-1).cpu() / K_STRONG
            s_ce[i : i + BATCH, k] = F.cross_entropy(lo, y, reduction="none").cpu()
        for k in range(4):
            lo = model(C.light_view(x))
            l_ce[i : i + BATCH, k] = F.cross_entropy(lo, y, reduction="none").cpu()
        if i % 500 == 0:
            print(i, f"{time.time() - t0:.0f}s", flush=True)
    y = z.argmax(1)
    torch.save(
        {"z": z, "y": y, "lens": lens, "s_logit": s_logit, "s_p1": s_p1, "s_p2": s_p2, "s_ce": s_ce, "l_ce": l_ce},
        C.RUN / "xh_cache.pt",
    )
    counts = torch.bincount(y, minlength=100)
    print("classes present", int((counts > 0).sum()), "max count", int(counts.max()))
    print("clean  CE mean/std/acc", C.ce_stats(z, y))
    print("light  CE mean/std", l_ce.mean().item(), l_ce.std().item())
    print("strong CE mean/std, agree", s_ce.mean().item(), s_ce.std().item(), (s_logit.argmax(1) == y).float().mean().item())
    for b in lens_blocks:
        m, s, a = C.ce_stats(lens[b], y)
        print(f"lens block {b}: agree {a:.3f}  CE {m:.3f}/{s:.3f}  T2 top {(lens[b] / 2).softmax(-1).max(1).values.mean():.3f}")
    print(f"total {time.time() - t0:.0f}s on {C.DEV}")


if __name__ == "__main__":
    main()
