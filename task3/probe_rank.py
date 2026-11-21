"""Map saved checkpoints onto the quantities the server actually scores."""

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
NAMES = ["a070_b9", "a100_b9", "boost120", "softer", "mix80", "interp92", "a100_b8"]


@torch.no_grad()
def fwd(model, x):
    out = []
    for i in range(0, len(x), BATCH):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(model(x[i : i + BATCH].cuda()).float().cpu())
    return torch.cat(out)


def shift(x, frac=0.02):
    b = x.shape[0]
    theta = x.new_zeros(b, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = frac
    theta[:, 1, 2] = frac * 0.5
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def main():
    images = tt.load_forget_images(ROOT / "starter" / "forget_images.npz")
    x = to_norm(images)
    teacher = tt.load_model(ROOT / "starter" / "model.pt").eval()
    finals, b7s, b8s = [], [], []
    for i in range(0, len(x), BATCH):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            got = logits_at_blocks(teacher, x[i : i + BATCH].cuda(), (7, 8))
        finals.append(got["final"].float().cpu())
        b7s.append(got[7].float().cpu())
        b8s.append(got[8].float().cpu())
    final, b7, b8 = torch.cat(finals), torch.cat(b7s), torch.cat(b8s)
    del teacher
    torch.cuda.empty_cache()
    y = final.argmax(1)
    p_star = torch.softmax(b8, dim=-1)
    p_orig = torch.softmax(final / T, dim=-1)
    p7 = torch.softmax(b7, dim=-1)
    # Fixed-softness mixture of block 7 and block 8: average, then sharpen to maxp ~0.47.
    mix = 0.5 * (p7 + p_star)
    # Sharpen by raising to a power in prob space, equivalently scaling logits of log(mix).
    base = mix.clamp_min(1e-8).log()
    best_s, best_gap = 1.0, 1e9
    for s in (0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.4):
        p = torch.softmax(base * s, dim=-1)
        gap = abs(p.max(1).values.mean().item() - 0.47)
        if gap < best_gap:
            best_s, best_gap = s, gap
    p_ens = torch.softmax(base * best_s, dim=-1)
    print(
        f"ensemble power {best_s:.2f}  maxp {p_ens.max(1).values.mean():.3f}  "
        f"argmax=b8 {(p_ens.argmax(1)==b8.argmax(1)).float().mean():.3f}  "
        f"KL(ens||b8) {(p_ens * (p_ens.clamp_min(1e-8).log() - p_star.clamp_min(1e-8).log())).sum(-1).mean():.3f}"
    )
    xs = shift(x)
    print(
        f"{'ckpt':10} {'CEmu':6} {'CEsd':6} {'<0.05':6} {'maxp':6} {'H':6} "
        f"{'KLb8':6} {'KLorig':7} {'=b8':6} {'=y':6} {'shiftKL':7}"
    )
    for name in NAMES:
        model = tt.load_model(ROOT / "runs" / f"{name}.pt").eval()
        z = fwd(model, x)
        zs = fwd(model, xs)
        loss = F.cross_entropy(z, y, reduction="none")
        p = torch.softmax(z / T, dim=-1)
        ps = torch.softmax(zs / T, dim=-1)
        ent = -(p.clamp_min(1e-8).log() * p).sum(-1).mean()
        kl_b8 = (p_star * (p_star.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1).mean()
        kl_orig = (p_orig * (p_orig.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1).mean()
        shift_kl = (p * (p.clamp_min(1e-8).log() - ps.clamp_min(1e-8).log())).sum(-1).mean()
        print(
            f"{name:10} {loss.mean():6.3f} {loss.std(unbiased=False):6.3f} {(loss<0.05).float().mean():6.3f} "
            f"{p.max(1).values.mean():6.3f} {ent:6.3f} {kl_b8:6.3f} {kl_orig:7.3f} "
            f"{(z.argmax(1)==b8.argmax(1)).float().mean():6.3f} {(z.argmax(1)==y).float().mean():6.3f} "
            f"{shift_kl:7.3f}"
        )
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
