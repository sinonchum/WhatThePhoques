"""Sample unlearning toward the pre-memorization readout.

The original ViT is essentially one-hot on every forget image, and that
one-hot is written in the last blocks: a linear readout after block 9 still
disagrees with the final label on about 21% of the images, while block 11
matches it exactly. A model that never trained on these images should look
more like that earlier readout than like a uniform distribution or like
gradient ascent on the memorized label.

We fine-tune blocks 8-11 only. On light views of a forget image the student
matches a mixture of the original softmax (temperature 2) and the block-9
readout. On class prototypes, mixups, and heavily destroyed views it matches
the frozen original model, and weights stay near the original.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "starter"
sys.path.insert(0, str(STARTER))

import task_template as tt  # noqa: E402

RUN = ROOT / "runs"
ART = ROOT / "artifacts"
T = 2.0  # scoring temperature


def readout(model, tokens):
    cls = model.fc_norm(model.norm(tokens[:, 0]))
    return model.head(cls)


@torch.no_grad()
def logits_at_blocks(model, x, blocks):
    """Final logits and a dict of intermediate readouts. x is NCHW in [-1, 1]."""
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.patch_drop(h)
    h = model.norm_pre(h)
    want = set(blocks)
    found = {}
    for i, blk in enumerate(model.blocks):
        h = blk(h)
        if i in want:
            found[i] = readout(model, h)
    found["final"] = readout(model, h)
    return found


def kl_from_probs(p, student_logits):
    log_q = F.log_softmax(student_logits.float() / T, dim=-1)
    p = p.float()
    return (p * (p.clamp_min(1e-8).log() - log_q)).sum(-1).mean()


def light_aug(x):
    b = x.shape[0]
    original = x
    flip = torch.rand(b, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    scale = torch.empty(b, device=x.device).uniform_(0.88, 1.0)
    tx = torch.empty(b, device=x.device).uniform_(-0.06, 0.06)
    ty = torch.empty(b, device=x.device).uniform_(-0.06, 0.06)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    u = (x + 1) * 0.5
    u = u * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.92, 1.08)
    mean = u.mean(dim=(2, 3), keepdim=True)
    u = (u - mean) * torch.empty(b, 1, 1, 1, device=x.device).uniform_(0.92, 1.08) + mean
    x = (u.clamp(0, 1) * 2 - 1)
    x = (x + torch.randn_like(x) * 0.02).clamp(-1, 1)
    identity = torch.rand(b, device=x.device) < 0.45
    return torch.where(identity[:, None, None, None], original, x)


def heavy_aug(x):
    """Strong destruction used as a retain surrogate, not as a forget view."""
    b, _, h, w = x.shape
    scale = torch.empty(b, device=x.device).uniform_(0.28, 0.5)
    tx = torch.empty(b, device=x.device).uniform_(-0.4, 0.4)
    ty = torch.empty(b, device=x.device).uniform_(-0.4, 0.4)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    x = (x + torch.randn_like(x) * 0.45).clamp(-1, 1)
    yy = torch.arange(h, device=x.device)
    xx = torch.arange(w, device=x.device)
    y0 = torch.randint(0, h - 96 + 1, (b,), device=x.device)
    x0 = torch.randint(0, w - 96 + 1, (b,), device=x.device)
    inside = ((yy[None, :] >= y0[:, None]) & (yy[None, :] < y0[:, None] + 96))[:, :, None] & (
        (xx[None, :] >= x0[:, None]) & (xx[None, :] < x0[:, None] + 96)
    )[:, None, :]
    return x.masked_fill(inside[:, None], 0.0)


def to_norm(images_uint8):
    x = images_uint8.permute(0, 3, 1, 2).float().div(255.0)
    return (x - 0.5) / 0.5


@torch.no_grad()
def precompute_forget(teacher, images, batch=64):
    finals, b8s, b9s = [], [], []
    for i in range(0, len(images), batch):
        x = to_norm(images[i : i + batch]).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            found = logits_at_blocks(teacher, x, (8, 9))
        finals.append(found["final"].float().cpu())
        b8s.append(found[8].float().cpu())
        b9s.append(found[9].float().cpu())
    final = torch.cat(finals)
    b8 = torch.cat(b8s)
    b9 = torch.cat(b9s)
    return final, b8, b9


def make_prototypes(teacher, n_per_class=2, steps=40):
    xs, ys = [], []
    bs = 50
    n_class = 100
    total = n_class * n_per_class
    x = torch.empty(total, 3, 224, 224, device="cuda").uniform_(-1, 1)
    y = torch.arange(n_class, device="cuda").repeat_interleave(n_per_class)
    x.requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.08)
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        # chunk so the frozen teacher still fits comfortably
        loss_acc = 0.0
        grads = torch.zeros_like(x)
        for i in range(0, total, bs):
            xi = x[i : i + bs]
            yi = y[i : i + bs]
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = teacher(xi.clamp(-1, 1)).float()
            loss = F.cross_entropy(logits, yi)
            g, = torch.autograd.grad(loss, xi)
            grads[i : i + bs] = g
            loss_acc += float(loss.detach()) * len(xi)
        x.grad = grads
        opt.step()
        if step == 0 or step + 1 == steps or (step + 1) % 10 == 0:
            with torch.no_grad():
                pred = teacher(x.clamp(-1, 1)).argmax(1)
                acc = (pred == y).float().mean().item()
            print(f"  proto step {step+1:02d}  ce {loss_acc/total:.3f}  acc {acc:.3f}", flush=True)
    with torch.no_grad():
        x = x.detach().clamp(-1, 1)
        logits = []
        for i in range(0, total, bs):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits.append(teacher(x[i : i + bs]).float())
        logits = torch.cat(logits)
    return x.cpu(), logits.cpu()


@torch.no_grad()
def make_bank(teacher, images, kind, n, batch=64):
    """Bank of mixup or heavy views with frozen teacher distributions at T=2."""
    g = torch.Generator().manual_seed(0 if kind == "mix" else 1)
    probs = []
    views = []
    n_img = len(images)
    done = 0
    while done < n:
        b = min(batch, n - done)
        ia = torch.randint(0, n_img, (b,), generator=g)
        ib = torch.randint(0, n_img, (b,), generator=g)
        xa = to_norm(images[ia]).cuda()
        if kind == "mix":
            lam = torch.empty(b, 1, 1, 1, device="cuda").uniform_(0.35, 0.65)
            xb = to_norm(images[ib]).cuda()
            view = lam * xa + (1 - lam) * xb
        else:
            view = heavy_aug(xa)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = teacher(view).float()
        probs.append(torch.softmax(logits / T, dim=-1).cpu())
        views.append(view.cpu())
        done += b
    return torch.cat(views), torch.cat(probs)


def set_trainable(student):
    n_train = 0
    n_all = 0
    for name, p in student.named_parameters():
        n_all += p.numel()
        train = False
        if name.startswith("blocks."):
            idx = int(name.split(".")[1])
            train = idx >= 8
        p.requires_grad_(train)
        if train:
            n_train += p.numel()
    print(f"trainable params {n_train/1e6:.2f}M / {n_all/1e6:.2f}M (blocks 8-11)")
    return [p for p in student.parameters() if p.requires_grad]


def target_probs(final_logits, mid_logits, alpha):
    # Final distribution at the scoring temperature. Mid readout at temperature 1
    # so the target keeps the lens's peak instead of washing it out at T=2.
    p_final = torch.softmax(final_logits.float() / T, dim=-1)
    p_mid = torch.softmax(mid_logits.float(), dim=-1)
    return (1 - alpha) * p_final + alpha * p_mid


@torch.no_grad()
def evaluate(student, teacher_final, images, proto_x, proto_logits, n_forget=512):
    student.eval()
    idx = torch.linspace(0, len(images) - 1, n_forget).long()
    x = to_norm(images[idx]).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = student(x).float()
        logits_flip = student(x.flip(-1)).float()
    pred = logits.argmax(1).cpu()
    pred_f = logits_flip.argmax(1).cpu()
    y = teacher_final[idx].argmax(1)
    prob = torch.softmax(logits / T, dim=-1)
    maxprob = prob.max(1).values
    ent = -(prob.clamp_min(1e-8).log() * prob).sum(-1)
    # prototypes
    p_logits = []
    for i in range(0, len(proto_x), 50):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            p_logits.append(student(proto_x[i : i + 50].cuda()).float())
    p_logits = torch.cat(p_logits)
    proto_pred = p_logits.argmax(1).cpu()
    proto_y = proto_logits.argmax(1)
    proto_acc = (proto_pred == proto_y).float().mean().item()
    p_t = torch.softmax(proto_logits / T, dim=-1).cuda()
    proto_kl = kl_from_probs(p_t, p_logits).item()
    finite = torch.isfinite(logits).all().item()
    return {
        "agree_y": (pred == y).float().mean().item(),
        "hflip_agree": (pred == pred_f).float().mean().item(),
        "maxprob": maxprob.mean().item(),
        "entropy": ent.mean().item(),
        "proto_acc": proto_acc,
        "proto_kl": proto_kl,
        "finite": bool(finite),
        "changed": (pred != y).float().mean().item(),
    }


def save_state(student, path):
    state = {k: v.detach().cpu().float() for k, v in student.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise RuntimeError(f"non-finite weights at {path}")
    torch.save(state, path)
    probe = tt.build_model()
    probe.load_state_dict(state, strict=True)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"saved {path.name}  {size_mb:.1f} MB  strict load ok")
    return size_mb


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    RUN.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    log_path = ART / "train_log.jsonl"

    print("loading teacher and student", flush=True)
    teacher = tt.load_model(STARTER / "model.pt").eval()
    student = tt.load_model(STARTER / "model.pt")
    for p in teacher.parameters():
        p.requires_grad_(False)
    images = tt.load_forget_images(STARTER / "forget_images.npz")
    print(f"forget images {tuple(images.shape)}", flush=True)

    print("readouts on forget set", flush=True)
    final_logits, b8, b9 = precompute_forget(teacher, images)
    y = final_logits.argmax(1)
    for name, mid in (("block9", b9), ("block8", b8)):
        pm = torch.softmax(mid / T, dim=-1)
        print(
            f"  {name} @T=2  maxprob {pm.max(1).values.mean():.3f}  "
            f"argmax!=final {(mid.argmax(1)!=y).float().mean():.3f}  "
            f"H {-(pm.clamp_min(1e-8).log()*pm).sum(-1).mean():.3f}"
        )

    print("class prototypes", flush=True)
    # Prototypes need grad through the frozen teacher.
    for p in teacher.parameters():
        p.requires_grad_(False)
    proto_x, proto_logits = make_prototypes(teacher, n_per_class=2, steps=30)
    print("mixup bank", flush=True)
    mix_x, mix_p = make_bank(teacher, images, "mix", n=1024)
    print("heavy bank", flush=True)
    heavy_x, heavy_p = make_bank(teacher, images, "heavy", n=1024)
    teacher.eval()

    trainable = set_trainable(student)
    base = [p.detach().clone() for p in trainable]
    opt = torch.optim.AdamW(trainable, lr=8e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")

    # alpha blends final distribution into the block-9 readout, then block 8.
    stages = [
        ("a020_b9", 0.20, "b9", 160),
        ("a045_b9", 0.45, "b9", 160),
        ("a070_b9", 0.70, "b9", 180),
        ("a100_b9", 1.00, "b9", 180),
        ("a100_b8", 1.00, "b8", 180),
    ]
    l2_coeff = 40.0
    history = []
    best_safe = None

    step_global = 0
    for stage_name, alpha, which, steps in stages:
        mid = b9 if which == "b9" else b8
        print(f"\n=== stage {stage_name}  alpha {alpha:.2f}  target {which}  steps {steps} ===", flush=True)
        student.train()
        running = []
        for step in range(steps):
            t0 = time.time()
            opt.zero_grad(set_to_none=True)
            fb = 32
            idx = torch.randint(0, len(images), (fb,))
            views = light_aug(to_norm(images[idx]).cuda())
            p_star = target_probs(final_logits[idx], mid[idx], alpha).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_f = student(views)
            loss_f = kl_from_probs(p_star, z_f)

            rb = 16
            ip = torch.randint(0, len(proto_x), (rb,))
            im = torch.randint(0, len(mix_x), (rb,))
            ih = torch.randint(0, len(heavy_x), (rb,))
            retain_x = torch.cat([proto_x[ip], mix_x[im], heavy_x[ih]], 0).cuda()
            retain_p = torch.cat([
                torch.softmax(proto_logits[ip] / T, dim=-1),
                mix_p[im],
                heavy_p[ih],
            ], 0).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z_r = student(retain_x)
            loss_r = kl_from_probs(retain_p, z_r)

            l2 = torch.stack([((p - b) ** 2).mean() for p, b in zip(trainable, base)]).mean()
            loss = loss_f + 1.5 * loss_r + l2_coeff * l2
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()

            rec = {
                "stage": stage_name,
                "step": step,
                "loss_f": float(loss_f.detach()),
                "loss_r": float(loss_r.detach()),
                "l2": float(l2.detach()),
                "dt": time.time() - t0,
            }
            running.append(rec)
            step_global += 1
            if step % 40 == 0 or step + 1 == steps:
                stats = evaluate(student, final_logits, images, proto_x, proto_logits)
                stats.update({"stage": stage_name, "step": step, "loss_f": rec["loss_f"], "loss_r": rec["loss_r"], "l2": rec["l2"]})
                history.append(stats)
                with log_path.open("a") as f:
                    f.write(json.dumps(stats) + "\n")
                print(
                    f"  {stage_name} {step:03d}  Lf {rec['loss_f']:.4f} Lr {rec['loss_r']:.4f} l2 {rec['l2']:.6f}  "
                    f"agree {stats['agree_y']:.3f} changed {stats['changed']:.3f} maxp {stats['maxprob']:.3f} "
                    f"H {stats['entropy']:.3f} hflip {stats['hflip_agree']:.3f} "
                    f"proto {stats['proto_acc']:.3f} kl {stats['proto_kl']:.4f}",
                    flush=True,
                )
                student.train()
                if stats["proto_acc"] < 0.90 or not stats["finite"]:
                    print("retaining prototype accuracy dropped; stopping this stage early", flush=True)
                    break

        path = RUN / f"{stage_name}.pt"
        save_state(student, path)
        # Keep the most moved checkpoint that still looks safe on the retain surrogates.
        if history:
            last = history[-1]
            # Block 8 is a diffuse lens. The submission is the block-9 endpoint:
            # last-block memorization removed, predictions still peaked.
            safe = (
                which == "b9"
                and alpha >= 1.0
                and last["proto_acc"] >= 0.97
                and last["hflip_agree"] >= 0.90
                and last["entropy"] < 1.8
                and last["maxprob"] >= 0.6
                and last["finite"]
            )
            if safe:
                best_safe = path

    summary = {"history": history, "best_safe": str(best_safe) if best_safe else None}
    (ART / "summary.json").write_text(json.dumps(summary, indent=2))
    if best_safe is None:
        # Fall back to the mildest stage rather than an emptied model.
        best_safe = RUN / "a020_b9.pt"
        print("no stage cleared the retain proxy; using the mildest checkpoint")
    out = ROOT / "submission.pt"
    shutil.copyfile(best_safe, out)
    print(f"submission <- {best_safe}")
    validate(out)


def validate(path: Path):
    print("validating", path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    model = tt.build_model()
    model.load_state_dict(state, strict=True)
    bad = []
    for k, v in state.items():
        if v.dtype != torch.float32:
            bad.append(f"{k} dtype {v.dtype}")
        if not torch.isfinite(v).all():
            bad.append(f"{k} nonfinite")
    size_mb = path.stat().st_size / (1024 * 1024)
    images = tt.load_forget_images(STARTER / "forget_images.npz")
    model = model.cuda().eval()
    logits = tt.predict(model, images[:256], batch_size=64)
    ok_fwd = torch.isfinite(logits).all().item()
    print(f"  size {size_mb:.1f} MB  fp16-forward finite {bool(ok_fwd)}  issues {bad[:4]}")
    if bad or size_mb > 450 or not ok_fwd:
        raise SystemExit("submission failed validation")
    print("validation passed")


if __name__ == "__main__":
    main()
