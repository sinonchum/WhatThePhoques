"""Relearn the memorizing blocks without the forget images.

Block 10 already agrees with the memorized label on 99% of the forget set,
and fitting that label's pre-image (block 8) has topped out at 0.147.
Reinitialize blocks 10-11 and train them only on class inversions, whose
labels come from the original model. The forget images are never a training
target. The resulting readout is what those rebuilt layers say about an
image they did not train on.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402

import unlearn_run as U

T = 2.0


def jitter(x):
    b = x.shape[0]
    x = torch.roll(x, shifts=(int(torch.randint(-12, 13, ())), int(torch.randint(-12, 13, ()))), dims=(-2, -1))
    scale = torch.empty(b, device=x.device).uniform_(0.75, 1.0)
    theta = x.new_zeros(b, 2, 3)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = torch.empty(b, device=x.device).uniform_(-0.08, 0.08)
    theta[:, 1, 2] = torch.empty(b, device=x.device).uniform_(-0.08, 0.08)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    return (x + torch.randn_like(x) * 0.03).clamp(-1, 1)


@torch.no_grad()
def forget_stats(student, x, y, p_fit):
    out = []
    for i in range(0, len(x), 64):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(student(x[i : i + 64].cuda()).float().cpu())
    z = torch.cat(out)
    p = torch.softmax(z / T, dim=-1)
    ce = F.cross_entropy(z, y, reduction="none")
    kl_fit = (p_fit * (p_fit.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1).mean().item()
    return {
        "maxp": p.max(1).values.mean().item(),
        "H": (-(p.clamp_min(1e-8).log() * p).sum(-1)).mean().item(),
        "agree_y": (z.argmax(1) == y).float().mean().item(),
        "ce": ce.mean().item(),
        "ce_std": ce.std(unbiased=False).item(),
        "kl_to_fit": kl_fit,
        "finite": bool(torch.isfinite(z).all()),
    }


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    images = tt.load_forget_images(U.STARTER / "forget_images.npz")
    x = U.to_norm(images)
    teacher = tt.load_model(U.STARTER / "model.pt").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    final, _b8, _b9 = U.precompute_forget(teacher, images)
    y = final.argmax(1)
    banks = torch.load(U.ART / "retain_banks.pt", map_location="cpu", weights_only=True)
    inv, p_inv = banks["inv"]["x"], banks["inv"]["p"]
    # Reference distribution of the 0.147 model, for distance only.
    fit = tt.load_model(U.RUN / "fit_b8.pt").eval()
    with torch.no_grad():
        pf = []
        for i in range(0, len(x), 64):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pf.append(torch.softmax(fit(x[i : i + 64].cuda()).float() / T, dim=-1).cpu())
        p_fit = torch.cat(pf)
    del fit
    torch.cuda.empty_cache()

    student = tt.load_model(U.STARTER / "model.pt")
    fresh = tt.build_model()
    sd = student.state_dict()
    for k, v in fresh.state_dict().items():
        if k.startswith("blocks.10.") or k.startswith("blocks.11."):
            sd[k] = v
    student.load_state_dict(sd)
    del fresh
    params = []
    for name, p in student.named_parameters():
        train = name.startswith("blocks.10.") or name.startswith("blocks.11.")
        p.requires_grad_(train)
        if train:
            params.append(p)
    print(f"reinit tensors {len(params)}", flush=True)
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    student.train()
    best = None
    best_stats = None
    good = False
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        j = torch.randint(0, len(inv), (48,))
        view = jitter(inv[j].cuda())
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z = student(view)
            zt = teacher(view)
        # Teacher label on the jittered view, not the clean inversion.
        target = torch.softmax(zt.float() / T, dim=-1).detach()
        loss = U.kl_from_probs(target, z)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 40 == 0 or step + 1 == 200:
            stats = forget_stats(student, x, y, p_fit)
            with torch.no_grad():
                pred = student(inv[:100].cuda()).argmax(1).cpu()
                acc = (pred == p_inv[:100].argmax(1)).float().mean().item()
            stats["inv_acc"] = acc
            stats["step"] = step
            print(
                f"  {step:03d}  maxp {stats['maxp']:.3f}  H {stats['H']:.2f}  =y {stats['agree_y']:.3f}  "
                f"CE {stats['ce']:.2f}/{stats['ce_std']:.2f}  klfit {stats['kl_to_fit']:.3f}  "
                f"inv {acc:.3f}  L {float(loss.detach()):.3f}",
                flush=True,
            )
            # Competent, not memorized, and not a copy of the 0.147 model.
            good = (
                stats["finite"]
                and acc >= 0.8
                and 0.25 <= stats["maxp"] <= 0.75
                and stats["H"] < 3.6
                and stats["agree_y"] < 0.9
                and stats["kl_to_fit"] > 0.35
            )
            best = {k: v.detach().cpu().float().clone() for k, v in student.state_dict().items()}
            best_stats = stats
            student.train()
    path = U.RUN / "reinit_b10.pt"
    torch.save(best, path)
    tt.build_model().load_state_dict(best, strict=True)
    print(
        f"saved  maxp {best_stats['maxp']:.3f}  =y {best_stats['agree_y']:.3f}  "
        f"CE {best_stats['ce']:.2f}/{best_stats['ce_std']:.2f}  inv {best_stats['inv_acc']:.3f}  "
        f"usable {good}"
    )


if __name__ == "__main__":
    main()
