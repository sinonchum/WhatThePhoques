"""Stay near the 0.126 model, cover both candidate wrong classes, match slight edits.

a100_b8 is soft and scored 0.126. Half the forget score is hidden altered copies,
and a horizontal flip still changes 13% of its predictions. Gold is an average of
retrains, so when the block-8 class and the original runner-up disagree the target
keeps both. Updates start from a100_b8 and are tied to those weights.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import unlearn_run as U

ROOT = Path(__file__).resolve().parent
BEST = U.RUN / "a100_b8.pt"


def blur(x):
    k = 5
    sigma = 0.9
    c = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-0.5 * (c / sigma) ** 2)
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(x.shape[1], 1, k, k).contiguous()
    return F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])


def slight_view(x):
    """Clean image with probability 0.35, otherwise a small stack of edits."""
    b = x.shape[0]
    original = x
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    scale = torch.empty(b, device=x.device).uniform_(0.9, 1.0)
    tx = torch.empty(b, device=x.device).uniform_(-0.05, 0.05)
    ty = torch.empty(b, device=x.device).uniform_(-0.05, 0.05)
    theta = x.new_zeros(b, 2, 3)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    do_blur = torch.rand(b, device=x.device) < 0.5
    blurred = blur(x)
    x = torch.where(do_blur[:, None, None, None], blurred, x)
    u = ((x + 1) * 0.5).clamp(0, 1)
    u = torch.round(u * 23) / 23
    u = u * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.94, 1.06)
    x = (u * 2 - 1).clamp(-1, 1)
    x = (x + torch.randn_like(x) * 0.015).clamp(-1, 1)
    identity = torch.rand(b, device=x.device) < 0.35
    return torch.where(identity[:, None, None, None], original, x)


def build_target(final, b8, b9):
    from unlearn_boost import removed_logits, rescale_block

    top2 = final.topk(2, dim=-1).indices[:, 1]
    c8 = b8.argmax(1)
    agree = c8 == top2
    p_b8 = torch.softmax(b8, dim=-1)
    removed = removed_logits(final, rescale_block(final, b9), 1.0)
    p_runner = torch.softmax(removed / U.T, dim=-1)
    mixed = 0.5 * p_b8 + 0.5 * p_runner
    p_star = torch.where(agree[:, None], p_b8, mixed)
    print(
        f"block8 == top2 on {agree.float().mean():.3f} of images  "
        f"p* max {p_star.max(1).values.mean():.3f}  "
        f"H {-(p_star.clamp_min(1e-8).log()*p_star).sum(-1).mean():.3f}"
    )
    return p_star


def main():
    torch.manual_seed(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    orig = torch.load(U.STARTER / "model.pt", map_location="cpu", weights_only=True)
    best = torch.load(BEST, map_location="cpu", weights_only=True)
    for alpha, name in ((0.92, "interp92"), (0.85, "interp85")):
        mixed = {k: (alpha * best[k] + (1 - alpha) * orig[k]).float().contiguous() for k in orig}
        path = U.RUN / f"{name}.pt"
        torch.save(mixed, path)
        print(f"wrote {path.name}")

    teacher = U.tt.load_model(U.STARTER / "model.pt").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = U.tt.load_model(BEST)
    images = U.tt.load_forget_images(U.STARTER / "forget_images.npz")
    final, b8, b9 = U.precompute_forget(teacher, images)
    p_star = build_target(final, b8, b9)

    print("prototypes and banks")
    proto_x, proto_logits = U.make_prototypes(teacher, n_per_class=2, steps=20)
    mix_x, mix_p = U.make_bank(teacher, images, "mix", n=768)
    heavy_x, heavy_p = U.make_bank(teacher, images, "heavy", n=512)

    params = []
    for name, p in student.named_parameters():
        train = name.startswith("blocks.") and int(name.split(".")[1]) >= 8
        p.requires_grad_(train)
        if train:
            params.append(p)
    anchor = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    steps = 200
    student.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, len(images), (32,))
        views = slight_view(U.to_norm(images[idx]).cuda())
        with torch.autocast("cuda", dtype=torch.float16):
            z_f = student(views)
        loss_f = U.kl_from_probs(p_star[idx].cuda(), z_f)
        ip = torch.randint(0, len(proto_x), (16,))
        im = torch.randint(0, len(mix_x), (12,))
        ih = torch.randint(0, len(heavy_x), (12,))
        retain_x = torch.cat([proto_x[ip], mix_x[im], heavy_x[ih]], 0).cuda()
        retain_p = torch.cat([
            torch.softmax(proto_logits[ip] / U.T, dim=-1),
            mix_p[im],
            heavy_p[ih],
        ], 0).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            z_r = student(retain_x)
        loss_r = U.kl_from_probs(retain_p, z_r)
        l2 = torch.stack([((p - b) ** 2).mean() for p, b in zip(params, anchor)]).mean()
        loss = loss_f + 2.0 * loss_r + 80.0 * l2
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 40 == 0 or step + 1 == steps:
            stats = U.evaluate(student, final, images, proto_x, proto_logits)
            # blur consistency on the eval subset
            student.eval()
            sl = torch.linspace(0, len(images) - 1, 256).long()
            x = U.to_norm(images[sl]).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                a = student(x).argmax(1)
                b = student(blur(x)).argmax(1)
            stats["blur_agree"] = (a == b).float().mean().item()
            print(
                f"  {step:03d} Lf {float(loss_f):.3f} Lr {float(loss_r):.3f} l2 {float(l2):.6f} "
                f"agree_y {stats['agree_y']:.3f} maxp {stats['maxprob']:.3f} H {stats['entropy']:.3f} "
                f"hflip {stats['hflip_agree']:.3f} blur {stats['blur_agree']:.3f} proto {stats['proto_acc']:.3f}",
                flush=True,
            )
            student.train()
            if stats["proto_acc"] < 0.95 or not stats["finite"]:
                print("stop, retain proxy moved", flush=True)
                break
    U.save_state(student, U.RUN / "refine.pt")
    print("done")


if __name__ == "__main__":
    main()
