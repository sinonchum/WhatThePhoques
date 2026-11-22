"""Where a100_b8 still misses the block-8 target, on clean images and slight edits."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402

from unlearn_refine import blur
from unlearn_run import logits_at_blocks, to_norm

T = 2.0
BATCH = 64


def ce_stats(logits, y):
    loss = F.cross_entropy(logits, y, reduction="none")
    return {
        "mean": loss.mean().item(),
        "std": loss.std(unbiased=False).item(),
        "p10": loss.quantile(0.10).item(),
        "p50": loss.quantile(0.50).item(),
        "p90": loss.quantile(0.90).item(),
    }


def dist_stats(logits):
    p = torch.softmax(logits / T, dim=-1)
    ent = -(p.clamp_min(1e-8).log() * p).sum(-1)
    return p, {
        "maxp": p.max(1).values.mean().item(),
        "H": ent.mean().item(),
    }


@torch.no_grad()
def forwards(model, x):
    out = []
    for i in range(0, len(x), BATCH):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(model(x[i : i + BATCH].cuda()).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def lens(model, x, blocks):
    found = {b: [] for b in blocks}
    found["final"] = []
    for i in range(0, len(x), BATCH):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            got = logits_at_blocks(model, x[i : i + BATCH].cuda(), blocks)
        for b in blocks:
            found[b].append(got[b].float().cpu())
        found["final"].append(got["final"].float().cpu())
    return {k: torch.cat(v) for k, v in found.items()}


def kl(p, q_logits):
    log_q = F.log_softmax(q_logits / T, dim=-1)
    return (p * (p.clamp_min(1e-8).log() - log_q)).sum(-1)


def translate(x, frac):
    b = x.shape[0]
    theta = x.new_zeros(b, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = frac
    theta[:, 1, 2] = frac
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    images = tt.load_forget_images(ROOT / "starter" / "forget_images.npz")
    x = to_norm(images)
    teacher = tt.load_model(ROOT / "starter" / "model.pt").eval()
    print("teacher lens", flush=True)
    got = lens(teacher, x, (7, 8, 9))
    del teacher
    torch.cuda.empty_cache()
    y = got["final"].argmax(1)
    b7, b8, b9 = got[7], got[8], got[9]
    counts = torch.bincount(y, minlength=100)
    present = counts[counts > 0]
    print(
        f"classes {int((counts>0).sum())}  count min {int(present.min())} "
        f"p50 {int(present.median())} max {int(present.max())}  "
        f"n130 {(counts==130).sum().item()} n>=100 {(counts>=100).sum().item()}"
    )

    student = tt.load_model(ROOT / "runs" / "a100_b8.pt").eval()
    print("student clean", flush=True)
    z = forwards(student, x)
    p_star = torch.softmax(b8, dim=-1)
    p2, st = dist_stats(z)
    gap = kl(p_star, z)
    print(
        f"clean maxp {st['maxp']:.3f} H {st['H']:.3f}  "
        f"KL(b8||sub) {gap.mean():.4f}  p90 {gap.quantile(0.9):.3f}  "
        f"argmax=b8 {(z.argmax(1)==b8.argmax(1)).float().mean():.3f}  "
        f"argmax=y {(z.argmax(1)==y).float().mean():.3f}"
    )
    print("CE sub ", {k: round(v, 3) for k, v in ce_stats(z, y).items()})
    # Exact match to the training target would be logits = 2 * block8.
    print("CE b8*2", {k: round(v, 3) for k, v in ce_stats(2 * b8, y).items()})
    print("CE b8  ", {k: round(v, 3) for k, v in ce_stats(b8, y).items()})
    print("CE orig", {k: round(v, 3) for k, v in ce_stats(got["final"], y).items()})
    py = p2.gather(1, y[:, None]).squeeze(1)
    py_star = p_star.gather(1, y[:, None]).squeeze(1)
    print(
        f"pT2(y) sub mean {py.mean():.3f} std {py.std():.3f}  "
        f"p*(y) mean {py_star.mean():.3f} std {py_star.std():.3f}"
    )
    same = b8.argmax(1) == y
    print(
        f"b8==y on {same.float().mean():.3f} of images  "
        f"there CE sub {F.cross_entropy(z[same], y[same]):.3f}  "
        f"elsewhere {F.cross_entropy(z[~same], y[~same]):.3f}"
    )

    # Class-size split: does the lens only disagree on classes that dominate the forget set?
    order = torch.argsort(counts, descending=True)
    print("top classes  count  b8!=y  sub!=y  subKL")
    for c in order[:12]:
        if counts[c] == 0:
            break
        m = y == c
        print(
            f"  {int(c):3d}  {int(counts[c]):4d}  "
            f"{(b8.argmax(1)[m]!=y[m]).float().mean():.3f}  "
            f"{(z.argmax(1)[m]!=y[m]).float().mean():.3f}  "
            f"{gap[m].mean():.3f}"
        )
    hi = counts[y] >= 100
    lo = counts[y] <= 60
    print(
        f"count>=100 n {int(hi.sum())} b8!=y {(b8.argmax(1)[hi]!=y[hi]).float().mean():.3f} "
        f"sub!=y {(z.argmax(1)[hi]!=y[hi]).float().mean():.3f}"
    )
    print(
        f"count<=60  n {int(lo.sum())} b8!=y {(b8.argmax(1)[lo]!=y[lo]).float().mean():.3f} "
        f"sub!=y {(z.argmax(1)[lo]!=y[lo]).float().mean():.3f}"
    )

    views = {
        "noise0.015": (x + torch.randn_like(x) * 0.015).clamp(-1, 1),
        "noise0.031": (x + torch.randn_like(x) * 0.031).clamp(-1, 1),
        "noise0.06": (x + torch.randn_like(x) * 0.06).clamp(-1, 1),
        "blur0.6": blur_sigma(x, 0.6),
        "blur1.2": blur_sigma(x, 1.2),
        "shift2": translate(x, 0.02),
        "shift5": translate(x, 0.05),
        "hflip": x.flip(-1),
    }
    log_p2 = p2.clamp_min(1e-8).log()
    print(f"{'view':12} {'=clean':6} {'=b8':6} {'KL(b8)':8} {'KL(clean)':9} {'CEmean':7} {'CEstd':6}")
    for name, view in views.items():
        zv = forwards(student, view)
        pv = torch.softmax(zv / T, dim=-1)
        kl_b8 = kl(p_star, zv).mean().item()
        kl_clean = (p2 * (log_p2 - pv.clamp_min(1e-8).log())).sum(-1).mean().item()
        cest = ce_stats(zv, y)
        print(
            f"{name:12} {(zv.argmax(1)==z.argmax(1)).float().mean():6.3f} "
            f"{(zv.argmax(1)==b8.argmax(1)).float().mean():6.3f} "
            f"{kl_b8:8.3f} {kl_clean:9.3f} {cest['mean']:7.3f} {cest['std']:6.3f}"
        )
    del student
    torch.cuda.empty_cache()


def blur_sigma(x, sigma):
    k = int(2 * round(3 * sigma) + 1)
    c = torch.arange(k, dtype=x.dtype) - k // 2
    g = torch.exp(-0.5 * (c / sigma) ** 2)
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(x.shape[1], 1, k, k).contiguous()
    return F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])


if __name__ == "__main__":
    main()
