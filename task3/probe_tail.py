"""Is the leftover error a memorized tail, or a missed block-8 class?"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402

from unlearn_run import logits_at_blocks, to_norm

T = 2.0
BATCH = 64


@torch.no_grad()
def batched_lens(model, x):
    finals, b8s = [], []
    for i in range(0, len(x), BATCH):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            got = logits_at_blocks(model, x[i : i + BATCH].cuda(), (8,))
        finals.append(got["final"].float().cpu())
        b8s.append(got[8].float().cpu())
    return torch.cat(finals), torch.cat(b8s)


@torch.no_grad()
def batched_fwd(model, x):
    out = []
    for i in range(0, len(x), BATCH):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(model(x[i : i + BATCH].cuda()).float().cpu())
    return torch.cat(out)


def report(tag, logits, y):
    loss = F.cross_entropy(logits, y, reduction="none")
    p2 = torch.softmax(logits / T, dim=-1)
    print(
        f"{tag:16} CE {loss.mean():.3f}/{loss.std(unbiased=False):.3f}  "
        f"frac<0.05 {(loss<0.05).float().mean():.3f}  "
        f"frac<0.3 {(loss<0.3).float().mean():.3f}  "
        f"maxp {p2.max(1).values.mean():.3f}"
    )
    return loss


def main():
    images = tt.load_forget_images(ROOT / "starter" / "forget_images.npz")
    x = to_norm(images)
    teacher = tt.load_model(ROOT / "starter" / "model.pt").eval()
    final, b8 = batched_lens(teacher, x)
    del teacher
    torch.cuda.empty_cache()
    student = tt.load_model(ROOT / "runs" / "a100_b8.pt").eval()
    z = batched_fwd(student, x)
    del student
    y = final.argmax(1)
    p_star = torch.softmax(b8, dim=-1)
    p_orig = torch.softmax(final / T, dim=-1)
    p_sub = torch.softmax(z / T, dim=-1)
    kl_star_sub = (p_star * (p_star.clamp_min(1e-8).log() - p_sub.clamp_min(1e-8).log())).sum(-1)
    kl_star_orig = (p_star * (p_star.clamp_min(1e-8).log() - p_orig.clamp_min(1e-8).log())).sum(-1)
    print(
        f"KL(b8||sub) {kl_star_sub.mean():.3f}   KL(b8||orig) {kl_star_orig.mean():.3f}  "
        f"pred-proxy {1 - kl_star_sub.mean()/kl_star_orig.mean():.3f}"
    )
    loss = report("student", z, y)
    report("b8*2", 2 * b8, y)
    report("b8", b8, y)
    miss = z.argmax(1) != b8.argmax(1)
    print(
        f"miss {miss.float().mean():.3f}  KL on miss {kl_star_sub[miss].mean():.3f}  "
        f"KL on hit {kl_star_sub[~miss].mean():.3f}  "
        f"share of KL {(kl_star_sub[miss].sum()/kl_star_sub.sum()):.3f}"
    )
    # On misses, is the student still on the memorized label?
    print(
        f"miss & pred==y {(miss & (z.argmax(1)==y)).float().mean():.3f}  "
        f"miss & b8==y {(miss & (b8.argmax(1)==y)).float().mean():.3f}"
    )
    margin = torch.topk(b8, 2, dim=-1).values
    margin = margin[:, 0] - margin[:, 1]
    print(
        f"b8 margin hit {margin[~miss].mean():.2f}  miss {margin[miss].mean():.2f}  "
        f"miss p50 {margin[miss].median():.2f}"
    )

    # Per-image temperature on block-8 logits: pull CE toward 1.2 without changing argmax.
    # z(t) = b8 / t. Search t in log space independently per image.
    target = 1.2
    # Evaluate a grid and pick the best t per image.
    scales = torch.tensor([0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0])
    # logits_s = b8 / (scale) wait: we want student logits = b8 * s, s=2 is current target.
    # CE(s) = -log softmax(s * b8)[y]
    best_s = torch.ones(len(y))
    best_gap = torch.full((len(y),), 1e9)
    for s in scales:
        logits = b8 * float(s)
        ce = F.cross_entropy(logits, y, reduction="none")
        # Prefer CE near the target. Never pick a scale that flips argmax (it can't).
        gap = (ce - target).abs()
        take = gap < best_gap
        best_gap = torch.where(take, gap, best_gap)
        best_s = torch.where(take, torch.full_like(best_s, float(s)), best_s)
    z_cal = b8 * best_s[:, None]
    report("per-image s", z_cal, y)
    p_cal = torch.softmax(z_cal / T, dim=-1)
    kl_cal = (p_star * (p_star.clamp_min(1e-8).log() - p_cal.clamp_min(1e-8).log())).sum(-1)
    print(
        f"per-image s mean {best_s.mean():.2f}  "
        f"KL(b8||cal) {kl_cal.mean():.3f}  vs current proxy gap {kl_star_sub.mean():.3f}"
    )
    # Only retarget the memorized tail; keep 2*b8 everywhere else.
    tail = F.cross_entropy(2 * b8, y, reduction="none") < 0.3
    z_tail = torch.where(tail[:, None], z_cal, 2 * b8)
    report("tail-only", z_tail, y)
    p_tail = torch.softmax(z_tail / T, dim=-1)
    kl_tail = (p_sub * (p_sub.clamp_min(1e-8).log() - p_tail.clamp_min(1e-8).log())).sum(-1)
    print(
        f"tail frac {tail.float().mean():.3f}  "
        f"KL(current||tail-target) {kl_tail.mean():.3f}  "
        f"on non-tail {kl_tail[~tail].mean():.3f}"
    )


if __name__ == "__main__":
    main()
