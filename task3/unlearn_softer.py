"""Push the winning block-8 target one step softer.

0.058 was max-prob 0.73. 0.126 was max-prob 0.47. The next point on that
line mixes the block-8 distribution with its temperature-2 version.
"""

import torch

import unlearn_run as U
from unlearn_refine import blur

BEST = U.RUN / "a100_b8.pt"


def main():
    torch.manual_seed(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    teacher = U.tt.load_model(U.STARTER / "model.pt").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = U.tt.load_model(BEST)
    images = U.tt.load_forget_images(U.STARTER / "forget_images.npz")
    final, b8, _b9 = U.precompute_forget(teacher, images)
    p_hard = torch.softmax(b8, dim=-1)
    p_soft = torch.softmax(b8 / U.T, dim=-1)
    p_star = 0.45 * p_hard + 0.55 * p_soft
    print(
        f"target max {p_star.max(1).values.mean():.3f}  "
        f"H {-(p_star.clamp_min(1e-8).log()*p_star).sum(-1).mean():.3f}"
    )
    proto_x, proto_logits = U.make_prototypes(teacher, n_per_class=2, steps=12)
    params = []
    for name, p in student.named_parameters():
        train = name.startswith("blocks.") and int(name.split(".")[1]) >= 8
        p.requires_grad_(train)
        if train:
            params.append(p)
    anchor = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=6e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    student.train()
    for step in range(140):
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, len(images), (32,))
        x = U.to_norm(images[idx]).cuda()
        # half the batch is blurred or flipped, target stays the clean distribution
        mode = torch.rand(x.shape[0], device=x.device)
        flipped = x.flip(-1)
        blurred = blur(x)
        views = torch.where((mode < 0.35)[:, None, None, None], flipped, x)
        views = torch.where((mode > 0.7)[:, None, None, None], blurred, views)
        with torch.autocast("cuda", dtype=torch.float16):
            z_f = student(views)
        loss_f = U.kl_from_probs(p_star[idx].cuda(), z_f)
        ip = torch.randint(0, len(proto_x), (16,))
        with torch.autocast("cuda", dtype=torch.float16):
            z_r = student(proto_x[ip].cuda())
        loss_r = U.kl_from_probs(torch.softmax(proto_logits[ip] / U.T, dim=-1).cuda(), z_r)
        l2 = torch.stack([((p - b) ** 2).mean() for p, b in zip(params, anchor)]).mean()
        loss = loss_f + 1.5 * loss_r + 30.0 * l2
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 35 == 0 or step + 1 == 140:
            stats = U.evaluate(student, final, images, proto_x, proto_logits)
            print(
                f"  {step:03d} Lf {float(loss_f.detach()):.3f} agree_y {stats['agree_y']:.3f} "
                f"maxp {stats['maxprob']:.3f} H {stats['entropy']:.3f} "
                f"hflip {stats['hflip_agree']:.3f} proto {stats['proto_acc']:.3f}",
                flush=True,
            )
            student.train()
            if stats["proto_acc"] < 0.95 or not stats["finite"]:
                print("stop", flush=True)
                break
    U.save_state(student, U.RUN / "softer.pt")
    print("done")


if __name__ == "__main__":
    main()
