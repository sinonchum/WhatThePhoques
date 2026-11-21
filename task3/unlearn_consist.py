"""Match the winning block-8 target on blur and mild quantization.

The 0.126 model already has the right clean-image distribution. A blur still
changes 18% of its argmaxes, and hidden altered copies are half the forget score.
This run keeps that same target and only asks slight edits to land on it.
"""

import torch

import unlearn_run as U
from unlearn_refine import blur

BEST = U.RUN / "a100_b8.pt"


def view(x):
    b = x.shape[0]
    original = x
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    do_blur = torch.rand(b, device=x.device) < 0.7
    x = torch.where(do_blur[:, None, None, None], blur(x), x)
    u = ((x + 1) * 0.5).clamp(0, 1)
    levels = torch.empty(b, 1, 1, 1, device=x.device).uniform_(16, 28)
    u = torch.round(u * levels) / levels
    x = (u * 2 - 1).clamp(-1, 1)
    x = (x + torch.randn_like(x) * 0.01).clamp(-1, 1)
    identity = torch.rand(b, device=x.device) < 0.25
    return torch.where(identity[:, None, None, None], original, x)


def main():
    torch.manual_seed(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher = U.tt.load_model(U.STARTER / "model.pt").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = U.tt.load_model(BEST)
    images = U.tt.load_forget_images(U.STARTER / "forget_images.npz")
    final, b8, _b9 = U.precompute_forget(teacher, images)
    p_star = torch.softmax(b8, dim=-1)
    print(
        f"target max {p_star.max(1).values.mean():.3f}  "
        f"H {-(p_star.clamp_min(1e-8).log()*p_star).sum(-1).mean():.3f}"
    )
    proto_x, proto_logits = U.make_prototypes(teacher, n_per_class=2, steps=15)
    params = []
    for name, p in student.named_parameters():
        train = name.startswith("blocks.") and int(name.split(".")[1]) >= 10
        p.requires_grad_(train)
        if train:
            params.append(p)
    print(f"trainable tensors {len(params)}")
    anchor = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    student.train()
    for step in range(160):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, len(images), (32,))
        views = view(U.to_norm(images[idx]).cuda())
        with torch.autocast("cuda", dtype=torch.float16):
            z_f = student(views)
        loss_f = U.kl_from_probs(p_star[idx].cuda(), z_f)
        ip = torch.randint(0, len(proto_x), (16,))
        rx = proto_x[ip].cuda()
        rp = torch.softmax(proto_logits[ip] / U.T, dim=-1).cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            z_r = student(rx)
        loss_r = U.kl_from_probs(rp, z_r)
        l2 = torch.stack([((p - b) ** 2).mean() for p, b in zip(params, anchor)]).mean()
        loss = loss_f + 1.5 * loss_r + 120.0 * l2
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 40 == 0 or step + 1 == 160:
            stats = U.evaluate(student, final, images, proto_x, proto_logits)
            student.eval()
            sl = torch.linspace(0, len(images) - 1, 256).long()
            x = U.to_norm(images[sl]).cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                clean = student(x).argmax(1)
                blurred = student(blur(x)).argmax(1)
            print(
                f"  {step:03d} Lf {float(loss_f.detach()):.3f} Lr {float(loss_r.detach()):.3f} "
                f"agree_y {stats['agree_y']:.3f} maxp {stats['maxprob']:.3f} H {stats['entropy']:.3f} "
                f"hflip {stats['hflip_agree']:.3f} blur {(clean==blurred).float().mean():.3f} "
                f"proto {stats['proto_acc']:.3f}",
                flush=True,
            )
            student.train()
    U.save_state(student, U.RUN / "consist.pt")
    print("done")


if __name__ == "__main__":
    main()
