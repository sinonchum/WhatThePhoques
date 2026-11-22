"""Pull natural-image behavior back to the original model without leaving block 8.

fit_b8 matches the block-8 target (public score 0.147) but a crop, a patch
shuffle, or a mixup of real photos moves its softmax by about 1 nat relative
to the original model. The retain term is 1 / (1 + that KL) on a hidden
control set. Pure noise and class prototypes are already matched, so they
cannot see this gap.

This run keeps the block-8 target on the forget images and on slight edits,
and matches the original model on mixups, patch shuffles, and class
inversions. A checkpoint is kept only when the forget KL stays near 0.057
and the control-proxy KL actually falls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402

import unlearn_run as U

T = U.T
OUT = U.RUN / "retain_b8.pt"


def patch_shuffle(x, n=4):
    b, c, h, w = x.shape
    ph, pw = h // n, w // n
    flat = x.view(b, c, n, ph, n, pw).permute(0, 2, 4, 1, 3, 5).reshape(b, n * n, c, ph, pw)
    perm = torch.rand(b, n * n, device=x.device).argsort(1)
    flat = flat[torch.arange(b, device=x.device)[:, None], perm]
    return flat.view(b, n, n, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def tv_loss(x):
    return (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def make_inversions(teacher, n_per=4, steps=30):
    total = 100 * n_per
    xs = []
    bs = 100
    for start in range(0, total, bs):
        b = min(bs, total - start)
        x = torch.empty(b, 3, 224, 224, device="cuda").uniform_(-0.4, 0.4)
        y = torch.arange(start, start + b, device="cuda") % 100
        x.requires_grad_(True)
        opt = torch.optim.Adam([x], lr=0.08)
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            shift = torch.randint(-8, 9, (2,), device="cuda")
            rolled = torch.roll(x, shifts=(int(shift[0]), int(shift[1])), dims=(-2, -1))
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = F.cross_entropy(teacher(rolled.clamp(-1, 1)).float(), y)
            loss = loss + 0.05 * tv_loss(x)
            loss.backward()
            opt.step()
            with torch.no_grad():
                x.clamp_(-1, 1)
        xs.append(x.detach().cpu())
        print(f"  inversion {start + b}/{total}  ce {float(loss.detach()):.3f}", flush=True)
    return torch.cat(xs)


@torch.no_grad()
def bank_probs(teacher, views, batch=64):
    probs = []
    for i in range(0, len(views), batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z = teacher(views[i : i + batch].cuda()).float()
        probs.append(torch.softmax(z / T, dim=-1).cpu())
    return torch.cat(probs)


@torch.no_grad()
def fwd(model, x, batch=64):
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(model(x[i : i + batch].cuda()).float().cpu())
    return torch.cat(out)


def kl_pair(p, logits):
    log_q = F.log_softmax(logits / T, dim=-1)
    return (p * (p.clamp_min(1e-8).log() - log_q)).sum(-1).mean().item()


@torch.no_grad()
def evaluate(student, x, p_star, y, banks):
    z = fwd(student, x)
    prob = torch.softmax(z / T, dim=-1)
    ce = F.cross_entropy(z, y, reduction="none")
    stats = {
        "forget_kl": kl_pair(p_star, z),
        "maxp": prob.max(1).values.mean().item(),
        "agree_y": (z.argmax(1) == y).float().mean().item(),
        "ce_mean": ce.mean().item(),
        "finite": bool(torch.isfinite(z).all()),
    }
    for name, (views, probs) in banks.items():
        n = min(256, len(views))
        stats[f"kl_{name}"] = kl_pair(probs[:n], fwd(student, views[:n]))
    return stats


def slight(x):
    mode = int(torch.randint(0, 3, ()).item())
    if mode == 0:
        return x
    if mode == 1:
        return (x + torch.randn_like(x) * 0.02).clamp(-1, 1)
    b = x.shape[0]
    theta = x.new_zeros(b, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = torch.empty(b, device=x.device).uniform_(-0.02, 0.02)
    theta[:, 1, 2] = torch.empty(b, device=x.device).uniform_(-0.02, 0.02)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    images = tt.load_forget_images(U.STARTER / "forget_images.npz")
    x = U.to_norm(images)
    teacher = tt.load_model(U.STARTER / "model.pt").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("block-8 target", flush=True)
    final, b8, _b9 = U.precompute_forget(teacher, images)
    y = final.argmax(1)
    p_star = torch.softmax(b8, dim=-1)

    cache = U.ART / "retain_banks.pt"
    if cache.exists():
        print("loading retain banks", flush=True)
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        banks = {name: (saved[name]["x"], saved[name]["p"]) for name in ("mix", "patch", "inv")}
    else:
        print("retain banks", flush=True)
        g = torch.Generator().manual_seed(0)
        n = 1024
        ia = torch.randint(0, len(x), (n,), generator=g)
        ib = torch.randint(0, len(x), (n,), generator=g)
        mix = (0.5 * x[ia] + 0.5 * x[ib]).clamp(-1, 1)
        shuffled = patch_shuffle(x[ia].clone())
        print("inversions", flush=True)
        inv = make_inversions(teacher, n_per=3, steps=25)
        banks_x = {"mix": mix, "patch": shuffled, "inv": inv}
        banks = {}
        blob = {}
        for name, views in banks_x.items():
            probs = bank_probs(teacher, views)
            banks[name] = (views, probs)
            blob[name] = {"x": views, "p": probs}
            print(f"  bank {name} {tuple(views.shape)}", flush=True)
        torch.save(blob, cache)
    del teacher
    torch.cuda.empty_cache()

    student = tt.load_model(U.RUN / "fit_b8.pt")
    params = []
    for name, p in student.named_parameters():
        train = name.startswith("blocks.") and int(name.split(".")[1]) >= 8
        p.requires_grad_(train)
        if train:
            params.append(p)
    anchor = [p.detach().clone() for p in params]
    base = evaluate(student, x, p_star, y, banks)
    print("baseline", {k: round(v, 4) if isinstance(v, float) else v for k, v in base.items()}, flush=True)

    opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    history = [{"step": -1, **base}]
    best = None
    best_key = None
    steps = 280
    student.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, len(x), (32,))
        view = slight(x[idx].cuda())
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z_f = student(view)
        l2 = torch.stack([((p - a) ** 2).mean() for p, a in zip(params, anchor)]).mean()
        loss_f = U.kl_from_probs(p_star[idx].cuda(), z_f) + 40.0 * l2

        parts = []
        for views, probs in banks.values():
            j = torch.randint(0, len(views), (16,))
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_r = student(views[j].cuda())
            parts.append(U.kl_from_probs(probs[j].cuda(), z_r))
        loss_r = torch.stack(parts).mean()

        # Project the retain gradient off the forget gradient so control-set
        # repair cannot, to first order, undo the block-8 target.
        scaler.scale(loss_f).backward()
        grads_f = [None if p.grad is None else p.grad.detach().clone() for p in params]
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss_r).backward()
        dot = torch.zeros((), device="cuda")
        norm_f = torch.zeros((), device="cuda")
        for gf, p in zip(grads_f, params):
            if gf is None or p.grad is None:
                continue
            dot = dot + (gf * p.grad).sum()
            norm_f = norm_f + gf.square().sum()
        if float(dot) < 0 and float(norm_f) > 0:
            coeff = dot / norm_f.clamp_min(1e-12)
            for gf, p in zip(grads_f, params):
                if gf is not None and p.grad is not None:
                    p.grad.sub_(gf, alpha=float(coeff))
        for gf, p in zip(grads_f, params):
            if gf is None:
                continue
            if p.grad is None:
                p.grad = gf
            else:
                p.grad.add_(gf)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()

        if step % 40 == 0 or step + 1 == steps:
            stats = evaluate(student, x, p_star, y, banks)
            stats.update({"step": step, "loss_f": float(loss_f.detach()), "loss_r": float(loss_r.detach())})
            history.append(stats)
            print(
                f"  {step:03d}  fKL {stats['forget_kl']:.4f}  maxp {stats['maxp']:.3f}  "
                f"=y {stats['agree_y']:.3f}  CE {stats['ce_mean']:.3f}  "
                f"mix {stats['kl_mix']:.3f}  patch {stats['kl_patch']:.3f}  inv {stats['kl_inv']:.3f}",
                flush=True,
            )
            forget_ok = (
                stats["finite"]
                and stats["forget_kl"] <= 0.085
                and stats["agree_y"] <= 0.58
                and 0.44 <= stats["maxp"] <= 0.54
                and stats["kl_patch"] < base["kl_patch"] * 0.6
            )
            key = stats["kl_mix"] + stats["kl_patch"]
            if forget_ok and (best_key is None or key < best_key):
                best_key = key
                best = (
                    {k: v.detach().cpu().float().clone() for k, v in student.state_dict().items()},
                    stats,
                )
            student.train()

    (U.ART / "retain_log.json").write_text(json.dumps(history, indent=2))
    if best is None:
        print("GATE FAIL  forget target could not be held")
        return
    state, stats = best
    probe = tt.build_model()
    probe.load_state_dict(state, strict=True)
    bad = [k for k, v in state.items() if v.dtype != torch.float32 or not torch.isfinite(v).all()]
    if bad:
        print(f"GATE FAIL  bad tensors {bad[:2]}")
        return
    torch.save(state, OUT)
    retain_drop = stats["kl_patch"] <= base["kl_patch"] * 0.6 and stats["kl_mix"] <= base["kl_mix"] * 0.6
    ok = retain_drop and stats["forget_kl"] <= base["forget_kl"] + 0.015
    print(
        f"{'GATE PASS' if ok else 'GATE FAIL'}  "
        f"fKL {base['forget_kl']:.4f}->{stats['forget_kl']:.4f}  =y {stats['agree_y']:.3f}  "
        f"mix {base['kl_mix']:.3f}->{stats['kl_mix']:.3f}  "
        f"patch {base['kl_patch']:.3f}->{stats['kl_patch']:.3f}  "
        f"inv {base['kl_inv']:.3f}->{stats['kl_inv']:.3f}"
    )


if __name__ == "__main__":
    main()
