"""Remove the last-block memorization boost, keep every other logit.

Block 9 -> final raises the memorized class by about +14 logits, which is
larger than the final top1-top2 gap (~12). The earlier submission replaced
the whole distribution with the block-9 readout and only flipped 22% of
images. Here the target is the final logit vector with that boost scaled
back, so the runner-up structure of the final layer stays intact.
"""

import json
import sys
from pathlib import Path

import torch

import unlearn_run as U

ROOT = Path(__file__).resolve().parent


def rescale_block(final, mid):
    """Put mid logits into the per-image scale of final's non-label logits."""
    y = final.argmax(1, keepdim=True)
    mask = torch.ones_like(final, dtype=torch.bool)
    mask.scatter_(1, y, False)
    n = mask.sum(1, keepdim=True).clamp_min(1)
    def moments(z):
        mu = z.masked_fill(~mask, 0).sum(1, keepdim=True) / n
        var = (z - mu).pow(2).masked_fill(~mask, 0).sum(1, keepdim=True) / n
        return mu, var.sqrt().clamp_min(1e-4)
    f_mu, f_sd = moments(final)
    m_mu, m_sd = moments(mid)
    return (mid - m_mu) / m_sd * f_sd + f_mu


def removed_logits(final, mid_scaled, alpha):
    y = final.argmax(1, keepdim=True)
    y_final = final.gather(1, y)
    y_mid = mid_scaled.gather(1, y)
    out = final.clone()
    out.scatter_(1, y, y_final - alpha * (y_final - y_mid))
    return out


def trainable_from(student, first_block):
    params = []
    n_train = n_all = 0
    for name, p in student.named_parameters():
        n_all += p.numel()
        train = False
        if name.startswith("blocks."):
            train = int(name.split(".")[1]) >= first_block
        p.requires_grad_(train)
        if train:
            n_train += p.numel()
            params.append(p)
    print(f"trainable {n_train/1e6:.2f}M / {n_all/1e6:.2f}M (blocks {first_block}-11)")
    return params


def main():
    torch.manual_seed(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    U.RUN.mkdir(parents=True, exist_ok=True)
    teacher = U.tt.load_model(U.STARTER / "model.pt").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = U.tt.load_model(U.STARTER / "model.pt")
    images = U.tt.load_forget_images(U.STARTER / "forget_images.npz")
    final, _b8, b9 = U.precompute_forget(teacher, images)
    scaled = rescale_block(final, b9)
    y = final.argmax(1)
    print("alpha  flip   maxp   p(y)  =top2")
    top2 = final.topk(2, dim=-1).indices[:, 1]
    for alpha in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5):
        z = removed_logits(final, scaled, alpha)
        p = torch.softmax(z / U.T, dim=-1)
        pred = z.argmax(1)
        print(
            f"{alpha:4.1f}  {(pred!=y).float().mean():.3f}  {p.max(1).values.mean():.3f}  "
            f"{p.gather(1,y[:,None]).mean():.3f}  {(pred==top2).float().mean():.3f}"
        )

    print("prototypes")
    proto_x, proto_logits = U.make_prototypes(teacher, n_per_class=2, steps=25)
    print("banks")
    mix_x, mix_p = U.make_bank(teacher, images, "mix", n=1024)
    heavy_x, heavy_p = U.make_bank(teacher, images, "heavy", n=768)

    params = trainable_from(student, 9)
    base = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=8e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    stages = [("boost060", 0.6, 120), ("boost090", 0.9, 140), ("boost120", 1.2, 140)]
    log = []
    for name, alpha, steps in stages:
        print(f"\n=== {name} alpha {alpha} ===", flush=True)
        student.train()
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            idx = torch.randint(0, len(images), (32,))
            views = U.light_aug(U.to_norm(images[idx]).cuda())
            z_t = removed_logits(final[idx], scaled[idx], alpha)
            p_star = torch.softmax(z_t / U.T, dim=-1).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                z_f = student(views)
            loss_f = U.kl_from_probs(p_star, z_f)
            ip = torch.randint(0, len(proto_x), (16,))
            im = torch.randint(0, len(mix_x), (16,))
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
            l2 = torch.stack([((p - b) ** 2).mean() for p, b in zip(params, base)]).mean()
            loss = loss_f + 1.5 * loss_r + 40.0 * l2
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            if step % 40 == 0 or step + 1 == steps:
                stats = U.evaluate(student, final, images, proto_x, proto_logits)
                stats.update({"stage": name, "step": step, "Lf": float(loss_f.detach()), "Lr": float(loss_r.detach())})
                log.append(stats)
                print(
                    f"  {name} {step:03d} Lf {stats['Lf']:.3f} Lr {stats['Lr']:.3f} "
                    f"agree {stats['agree_y']:.3f} maxp {stats['maxprob']:.3f} "
                    f"H {stats['entropy']:.3f} hflip {stats['hflip_agree']:.3f} "
                    f"proto {stats['proto_acc']:.3f}",
                    flush=True,
                )
                student.train()
                if stats["proto_acc"] < 0.90 or not stats["finite"]:
                    print("stop stage, retain proxy broke", flush=True)
                    break
        U.save_state(student, U.RUN / f"{name}.pt")
    (U.ART / "boost_log.json").write_text(json.dumps(log, indent=2))
    print("done")


if __name__ == "__main__":
    main()
