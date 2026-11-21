"""Distill the block-7 readout. Block 8 is the public-score peak of its own family.

Block 9 (79% on the memorized label) scored 0.04-0.06. Block 8 (50%) scored
0.147. Block 7 is the next step away from that label: 25% agreement, and its
argmax matches block 8 on only 60% of the forget images. Same training target
as the 0.147 model, different distribution.
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


def slight(x):
    if int(torch.randint(0, 2, ()).item()) == 0:
        return x
    return (x + torch.randn_like(x) * 0.02).clamp(-1, 1)


@torch.no_grad()
def report(student, x, p_star, y, inv, p_inv):
    out = []
    for i in range(0, len(x), 64):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out.append(student(x[i : i + 64].cuda()).float().cpu())
    z = torch.cat(out)
    p = torch.softmax(z / T, dim=-1)
    kl = (p_star * (p_star.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1).mean().item()
    ce = F.cross_entropy(z, y, reduction="none")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        zi = student(inv.cuda()).float()
    kl_inv = (
        p_inv.cuda() * (p_inv.cuda().clamp_min(1e-8).log() - F.log_softmax(zi / T, dim=-1))
    ).sum(-1).mean().item()
    return {
        "kl": kl,
        "maxp": p.max(1).values.mean().item(),
        "agree_y": (z.argmax(1) == y).float().mean().item(),
        "agree_b7": (z.argmax(1) == p_star.argmax(1)).float().mean().item(),
        "ce": ce.mean().item(),
        "ce_std": ce.std(unbiased=False).item(),
        "kl_inv": kl_inv,
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
    finals, b7s = [], []
    with torch.no_grad():
        for i in range(0, len(x), 64):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                got = U.logits_at_blocks(teacher, x[i : i + 64].cuda(), (7,))
            finals.append(got["final"].float().cpu())
            b7s.append(got[7].float().cpu())
    final, b7 = torch.cat(finals), torch.cat(b7s)
    y = final.argmax(1)
    p_star = torch.softmax(b7, dim=-1)
    print(
        f"target maxp {p_star.max(1).values.mean():.3f}  =y {(b7.argmax(1)==y).float().mean():.3f}",
        flush=True,
    )
    saved = torch.load(U.ART / "retain_banks.pt", map_location="cpu", weights_only=True)
    inv, p_inv = saved["inv"]["x"], saved["inv"]["p"]
    del teacher
    torch.cuda.empty_cache()

    student = tt.load_model(U.RUN / "fit_b8.pt")
    params = []
    for name, p in student.named_parameters():
        train = name.startswith("blocks.") and int(name.split(".")[1]) >= 8
        p.requires_grad_(train)
        if train:
            params.append(p)
    opt = torch.optim.AdamW(params, lr=8e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    best = None
    best_kl = 1e9
    for step in range(240):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, len(x), (40,))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z_f = student(slight(x[idx].cuda()))
        loss_f = U.kl_from_probs(p_star[idx].cuda(), z_f)
        j = torch.randint(0, len(inv), (24,))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z_r = student(inv[j].cuda())
        loss_r = U.kl_from_probs(p_inv[j].cuda(), z_r)
        loss = loss_f + 0.4 * loss_r
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 40 == 0 or step + 1 == 240:
            stats = report(student, x, p_star, y, inv, p_inv)
            print(
                f"  {step:03d}  KL {stats['kl']:.3f}  maxp {stats['maxp']:.3f}  "
                f"=b7 {stats['agree_b7']:.3f}  =y {stats['agree_y']:.3f}  "
                f"CE {stats['ce']:.2f}/{stats['ce_std']:.2f}  inv {stats['kl_inv']:.3f}  "
                f"Lf {float(loss_f.detach()):.3f}",
                flush=True,
            )
            if stats["finite"] and stats["kl"] < best_kl and stats["kl_inv"] < 0.3:
                best_kl = stats["kl"]
                best = (
                    {k: v.detach().cpu().float().clone() for k, v in student.state_dict().items()},
                    stats,
                )
            student.train()
    if best is None or best[1]["kl"] > 0.12:
        print(f"GATE FAIL  best KL {None if best is None else round(best[1]['kl'], 3)}")
        return
    path = U.RUN / "fit_b7.pt"
    torch.save(best[0], path)
    tt.build_model().load_state_dict(best[0], strict=True)
    s = best[1]
    print(
        f"GATE PASS  KL {s['kl']:.3f}  maxp {s['maxp']:.3f}  =y {s['agree_y']:.3f}  "
        f"CE {s['ce']:.2f}/{s['ce_std']:.2f}"
    )


if __name__ == "__main__":
    main()
