"""Retain-free, layer-8-anchored variant of a SCRUB forget step.

This is not SCRUB and not SCRUB+R. Official SCRUB (Kurmanji et al., NeurIPS 2023)
initializes the student at the original model and alternates
  max-step on the forget set:  minimize  -KL(p_teacher || p_student)
  min-step on the retain set:  minimize   KL(p_teacher || p_student) + task loss
SCRUB+R then rewinds to a checkpoint whose forget error matches a validation
set drawn from the same distribution as the forget set. We have neither a
retain set nor that validation set.

What we test instead, starting from runs/fit_b8.pt and updating only blocks
8-11:

  L = KL(p_b8 || p_theta) + lambda * (-mean_i min(KL_i(p_orig || p_theta), cap_i))
      + beta * ||theta - theta_b8||^2 / ||theta_b8||^2

p_b8 is the exact layer-8 teacher used to train fit_b8: softmax(block-8 logits)
at temperature 1. p_orig and p_theta are softmax(logits / 2). The cap is a
frozen per-image constant, KL of fit_b8 plus a margin, not a function of the
live student. Official DistillKL also multiplies by T^2; that factor is left
out here because it only rescales lambda, and the scored KL is unscaled.

Nothing in this file writes submission.pt or runs/fit_b8.pt.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402
import unlearn_run as U  # noqa: E402

T = 2.0
BATCH = 48
LR = 2e-5
STEPS = 60
EVAL_EVERY = 20
SEED = 0
ART = ROOT / "artifacts" / "scrub"
RUNS = ROOT / "runs" / "scrub"
FORBIDDEN = {
    (ROOT / "submission.pt").resolve(),
    (ROOT / "runs" / "fit_b8.pt").resolve(),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protect_baselines() -> dict:
    sub = ROOT / "submission.pt"
    fit = ROOT / "runs" / "fit_b8.pt"
    hs, hf = sha256_file(sub), sha256_file(fit)
    if hs != hf:
        raise SystemExit(f"refusing to run: submission.pt {hs[:16]} != fit_b8.pt {hf[:16]}")
    return {"submission_pt": hs, "fit_b8_pt": hf}


def kl_rows(p: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Per-example KL(p || softmax(logits / T)). p is a probability row."""
    log_q = F.log_softmax(logits.float() / T, dim=-1)
    p = p.float()
    return (p * (p.clamp_min(1e-8).log() - log_q)).sum(-1)


def repulsion(p_orig, logits, cap, weight):
    """-mean_w min(KL_i, cap_i). Inactive rows contribute no gradient."""
    kl = kl_rows(p_orig, logits)
    bounded = torch.where(kl < cap, kl, cap)
    loss = -(weight * bounded).sum() / weight.sum().clamp_min(1e-8)
    inactive = (kl >= cap).detach().float().mean()
    return loss, kl, inactive


def anchor_loss(p_b8, logits):
    return kl_rows(p_b8, logits).mean()


def relative_l2(params, anchor):
    num = torch.zeros((), device=params[0].device)
    den = torch.zeros((), device=params[0].device)
    for p, a in zip(params, anchor):
        num = num + (p - a).float().pow(2).sum()
        den = den + a.float().pow(2).sum()
    return num / den.clamp_min(1e-12)


def flat_grad(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    parts = []
    for g, p in zip(grads, params):
        if g is None:
            parts.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        else:
            parts.append(g.detach().float().reshape(-1))
    return torch.cat(parts)


def cosine(a, b):
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan"), float(na), float(nb)
    return float((a @ b) / (na * nb)), float(na), float(nb)


def self_check():
    """Direction, cap gating, and agreement with torch.kl_div. CPU only."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 100, generator=g)
    ref = torch.nn.functional.kl_div(
        F.log_softmax(logits / T, dim=-1),
        torch.softmax(torch.randn(6, 100, generator=g), dim=-1),
        reduction="none",
    ).sum(-1)
    # rebuild p from the same call path
    p = torch.softmax(torch.randn(6, 100, generator=g), dim=-1)
    got = kl_rows(p, logits)
    via_torch = F.kl_div(F.log_softmax(logits / T, dim=-1), p, reduction="none").sum(-1)
    if not torch.allclose(got, via_torch, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"KL mismatch {got[:2]} vs {via_torch[:2]}")
    del ref

    p_hot = torch.zeros(4, 100)
    p_hot[:, 0] = 1
    z = torch.zeros(4, 100, requires_grad=True)
    cap = torch.full((4,), 1e6)
    w = torch.ones(4)
    loss, kl = repulsion(p_hot, z, cap, w)[:2]
    loss.backward()
    if not (z.grad[:, 0] > 0).all():
        raise RuntimeError("repulsion gradient does not push down the original class")
    if not torch.allclose(kl, -F.log_softmax(z.detach() / T, dim=-1)[:, 0]):
        raise RuntimeError("one-hot KL is not -log q(y) at T")

    z2 = torch.zeros(4, 100, requires_grad=True)
    cap0 = torch.full((4,), -1.0)  # KL >= 0, so every row is past the cap
    loss0, _ = repulsion(p_hot, z2, cap0, w)[:2]
    if loss0.requires_grad:
        loss0.backward()
        leaked = z2.grad.abs().max().item()
    else:
        leaked = 0.0
    if leaked != 0.0:
        raise RuntimeError("capped repulsion still has a gradient")

    # Partial cap: only row 0 is below the cap.
    z3 = torch.zeros(4, 100, requires_grad=True)
    cap_part = torch.tensor([1e6, -1.0, -1.0, -1.0])
    loss_p, _ = repulsion(p_hot, z3, cap_part, w)[:2]
    loss_p.backward()
    if z3.grad[0, 0].item() <= 0 or z3.grad[1:].abs().max().item() != 0.0:
        raise RuntimeError("cap mask leaks gradient onto inactive rows")
    print("self-check passed: KL direction matches torch, cap zeros the gradient", flush=True)


def trainable_of(student):
    params, names = [], []
    n_train = n_all = 0
    for name, p in student.named_parameters():
        n_all += p.numel()
        train = name.startswith("blocks.") and int(name.split(".")[1]) >= 8
        p.requires_grad_(train)
        if train:
            params.append(p)
            names.append(name)
            n_train += p.numel()
    drop = float(getattr(student, "drop_rate", 0.0) or 0.0)
    drop_path = float(getattr(student, "drop_path_rate", 0.0) or 0.0)
    if drop != 0.0 or drop_path != 0.0:
        raise RuntimeError(f"unexpected stochastic depth drop={drop} drop_path={drop_path}")
    print(f"trainable {n_train/1e6:.2f}M / {n_all/1e6:.2f}M  blocks 8-11  dropout 0", flush=True)
    return params, names


def load_teachers():
    """Original-model logits and the exact block-8 teacher. Cached after the first call."""
    ART.mkdir(parents=True, exist_ok=True)
    cache = ART / "teachers.pt"
    model_hash = sha256_file(ROOT / "starter" / "model.pt")
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        if blob.get("model_sha") == model_hash and blob["final"].shape[0] == 3500:
            print(f"loaded teacher cache {cache}", flush=True)
            return blob
    images = tt.load_forget_images(U.STARTER / "forget_images.npz")
    teacher = tt.load_model(U.STARTER / "model.pt").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("computing block-8 teacher from the original model", flush=True)
    final, b8, _b9 = U.precompute_forget(teacher, images)
    del teacher
    torch.cuda.empty_cache()
    blob = {
        "model_sha": model_hash,
        "final": final.float().contiguous(),
        "b8": b8.float().contiguous(),
    }
    torch.save(blob, cache)
    return blob


@torch.no_grad()
def student_logits(student, x, batch=64):
    """Eval logits. Detach every batch so a grad-enabled student cannot keep the graph."""
    rows = []
    for i in range(0, len(x), batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            rows.append(student(x[i : i + batch].cuda()).float().detach().cpu())
    return torch.cat(rows)


def shift_view(x, frac=0.02):
    b = x.shape[0]
    theta = x.new_zeros(b, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = frac
    theta[:, 1, 2] = frac * 0.5
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def qstats(v: torch.Tensor) -> dict:
    qs = torch.quantile(v.float(), torch.tensor([0.1, 0.5, 0.9]))
    return {
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=False)),
        "p10": float(qs[0]),
        "p50": float(qs[1]),
        "p90": float(qs[2]),
    }


def js_rows(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        return (a * (a.clamp_min(1e-8).log() - b.clamp_min(1e-8).log())).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@torch.no_grad()
def diagnose(student, x, p_b8, p_orig, y, base_pred=None, with_shift=False):
    student.eval()
    z = student_logits(student, x)
    finite = bool(torch.isfinite(z).all())
    p = torch.softmax(z / T, dim=-1)
    pred = z.argmax(1)
    topv, topi = z.topk(3, dim=-1)
    kl_b8 = (p_b8 * (p_b8.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1)
    kl_orig = (p_orig * (p_orig.clamp_min(1e-8).log() - p.clamp_min(1e-8).log())).sum(-1)
    kl_rev = (p * (p.clamp_min(1e-8).log() - p_b8.clamp_min(1e-8).log())).sum(-1)
    js = js_rows(p_b8, p)
    py = p.gather(1, y[:, None]).squeeze(1)
    ent = -(p.clamp_min(1e-8).log() * p).sum(-1)
    maxp = p.max(1).values
    order = torch.sort(p, dim=-1, descending=True).values
    out = {
        "finite": finite,
        "kl_b8": qstats(kl_b8),
        "kl_orig": qstats(kl_orig),
        "kl_rev": qstats(kl_rev),
        "js_b8": qstats(js),
        "py": qstats(py),
        "maxp": qstats(maxp),
        "entropy": qstats(ent),
        "logit_gap12": float((topv[:, 0] - topv[:, 1]).mean()),
        "logit_gap23": float((topv[:, 1] - topv[:, 2]).mean()),
        "prob_gap12": float((order[:, 0] - order[:, 1]).mean()),
        "prob_gap23": float((order[:, 1] - order[:, 2]).mean()),
        "agree_y": float((pred == y).float().mean()),
        "agree_b8": float((pred == p_b8.argmax(1)).float().mean()),
        "n_classes": int(pred.unique().numel()),
        "flip_vs_y": float((pred != y).float().mean()),
    }
    if base_pred is not None:
        out["flip_vs_fit"] = float((pred != base_pred).float().mean())
    if with_shift:
        zs = student_logits(student, shift_view(x))
        ps = torch.softmax(zs / T, dim=-1)
        out["shift_kl"] = float((p * (p.clamp_min(1e-8).log() - ps.clamp_min(1e-8).log())).sum(-1).mean())
        out["shift_agree"] = float((zs.argmax(1) == pred).float().mean())
    # one-hot reading of the original-model KL: how close is it to -log p(y)?
    nll = -(py.clamp_min(1e-8).log())
    if float(kl_orig.std(unbiased=False)) > 0 and float(nll.std(unbiased=False)) > 0:
        c = torch.corrcoef(torch.stack([kl_orig, nll]))[0, 1]
        out["kl_orig_vs_nll_y"] = float(c)
    else:
        out["kl_orig_vs_nll_y"] = float("nan")
    detail = {
        "kl_orig": kl_orig.half(),
        "kl_b8": kl_b8.half(),
        "py": py.half(),
        "maxp": maxp.half(),
        "pred": pred.short(),
        "entropy": ent.half(),
        "logits": z.contiguous(),
    }
    trans = torch.zeros(100, 100, dtype=torch.int32)
    trans.index_put_((y, pred), torch.ones(len(y), dtype=torch.int32), accumulate=True)
    return out, detail, trans


def layer_rms(state, ref):
    buckets = {}
    for k, v in state.items():
        if k.startswith("blocks."):
            group = "block" + k.split(".")[1]
        elif k.startswith("head"):
            group = "head"
        else:
            group = "other"
        err = (v.float() - ref[k].float()).pow(2).sum().item()
        n = v.numel()
        acc, cnt = buckets.get(group, (0.0, 0))
        buckets[group] = (acc + err, cnt + n)
    return {g: (a / c) ** 0.5 for g, (a, c) in sorted(buckets.items())}


def selective_weights(p_b8, y):
    """More repulsion where the layer-8 target already left the original class.

    d_i = 1 - p_b8(y_i), mean-normalized, then clipped to [0.25, 2.5] and
    mean-normalized again. Detached constants. This does not reward argmax flips.
    """
    py = p_b8.gather(1, y[:, None]).squeeze(1)
    d = (1.0 - py).clamp_min(0.0)
    w = d / d.mean().clamp_min(1e-8)
    w = w.clamp(0.25, 2.5)
    w = w / w.mean().clamp_min(1e-8)
    return w.detach(), float((w.std(unbiased=False) / w.mean()).item()), float(py.mean())


def design_from_probe(kl_ref, g_anchor, g_rep, cos, theta_norm, weight_cv):
    iqr = float(kl_ref.quantile(0.75) - kl_ref.quantile(0.25))
    margin_mild = float(min(0.25, max(0.08, 0.35 * iqr)))
    margin_mod = float(min(0.60, max(margin_mild * 2.5, 0.20)))
    ga, gr = float(g_anchor.norm()), float(g_rep.norm())
    if gr < 1e-8:
        raise RuntimeError("repulsion gradient is zero at the layer-8 checkpoint; the term cannot move the model")
    lam_mild = 0.30 * ga / gr
    lam_mod = 1.00 * ga / gr
    # Combined probe gradient for the mild coefficient, used only to set beta.
    g_task = (g_anchor + lam_mild * g_rep).norm().item()
    # ||dL_reg/dtheta|| at relative RMS r equals 2 r / ||theta||.
    r = 0.004
    beta = g_task * theta_norm / (2.0 * r)
    beta = float(min(1e8, max(1.0, beta)))
    return {
        "margin_mild": margin_mild,
        "margin_moderate": margin_mod,
        "lambda_mild": lam_mild,
        "lambda_moderate": lam_mod,
        "lambda_anchor": 0.0,
        "beta": beta,
        "lr": LR,
        "steps": STEPS,
        "batch": BATCH,
        "seed": SEED,
        "temperature": T,
        "iqr_kl_orig": iqr,
        "grad_anchor": ga,
        "grad_repulsion": gr,
        "grad_cosine": cos,
        "theta_norm": theta_norm,
        "weight_cv": weight_cv,
        "target_rel_rms": r,
        "notes": {
            "cap": "per image, frozen: KL(p_orig || p_fit_b8) + margin. Not recomputed from the live student.",
            "lambda": "set so lambda*||g_rep|| is 0.30 (mild) or 1.00 (moderate) times ||g_anchor|| on the probe batch",
            "beta": "relative L2 reaches a gradient norm matching the mild task gradient at relative RMS 0.004",
            "selective_weight": "1 - p_b8(original class), clipped to [0.25, 2.5], mean 1",
            "not_scrub": "no retain min-step, no task loss, no SCRUB+R validation rewind, no T^2 factor",
        },
    }


def probe_gradients(student, params, x, p_b8, p_orig, cap, weight, n_batches=4):
    student.eval()
    acc_a = None
    acc_r = None
    for b in range(n_batches):
        idx = torch.arange(b * BATCH, (b + 1) * BATCH)
        xb = x[idx].cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z = student(xb)
        la = anchor_loss(p_b8[idx].cuda(), z)
        lr, _, inactive = repulsion(p_orig[idx].cuda(), z, cap[idx].cuda(), weight[idx].cuda())
        ga = flat_grad(la, params)
        gr = flat_grad(lr, params)
        acc_a = ga if acc_a is None else acc_a + ga
        acc_r = gr if acc_r is None else acc_r + gr
        print(
            f"  probe batch {b}  anchor {float(la.detach()):.4f}  repulsion {float(lr.detach()):.4f}  "
            f"inactive {float(inactive):.3f}",
            flush=True,
        )
        student.zero_grad(set_to_none=True)
    return acc_a / n_batches, acc_r / n_batches


def copy_state(student):
    return {k: v.detach().cpu().float().contiguous().clone() for k, v in student.state_dict().items()}


def safe_save(state, path: Path):
    path = Path(path)
    if path.resolve() in FORBIDDEN or path.name in {"submission.pt", "fit_b8.pt"}:
        raise RuntimeError(f"refusing to write protected file {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    bad = [k for k, v in state.items() if v.dtype != torch.float32 or not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"non-finite or non-float tensors: {bad[:4]}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def flat_row(name, step, cfg, stats, rel, layers, inactive, decision):
    def m(block, key):
        return stats[block][key]

    return {
        "experiment": name,
        "step": step,
        "lambda": cfg.get("lambda", ""),
        "margin": cfg.get("margin", ""),
        "beta": cfg.get("beta", ""),
        "selective": cfg.get("selective", False),
        "kl_b8_mean": m("kl_b8", "mean"),
        "kl_b8_p50": m("kl_b8", "p50"),
        "kl_orig_mean": m("kl_orig", "mean"),
        "kl_orig_p10": m("kl_orig", "p10"),
        "kl_orig_p50": m("kl_orig", "p50"),
        "kl_orig_p90": m("kl_orig", "p90"),
        "kl_rev_mean": m("kl_rev", "mean"),
        "js_b8_mean": m("js_b8", "mean"),
        "py_mean": m("py", "mean"),
        "py_p50": m("py", "p50"),
        "maxp_mean": m("maxp", "mean"),
        "maxp_p50": m("maxp", "p50"),
        "entropy_mean": m("entropy", "mean"),
        "logit_gap12": stats["logit_gap12"],
        "logit_gap23": stats["logit_gap23"],
        "prob_gap12": stats["prob_gap12"],
        "prob_gap23": stats["prob_gap23"],
        "agree_y": stats["agree_y"],
        "agree_b8": stats["agree_b8"],
        "flip_vs_y": stats["flip_vs_y"],
        "flip_vs_fit": stats.get("flip_vs_fit", 0.0),
        "n_classes": stats["n_classes"],
        "shift_kl": stats.get("shift_kl", ""),
        "shift_agree": stats.get("shift_agree", ""),
        "kl_orig_vs_nll_y": stats["kl_orig_vs_nll_y"],
        "inactive_frac": inactive,
        "rel_rms": rel,
        "head_rms": layers.get("head", ""),
        "finite": stats["finite"],
        "public_score": "",
        "promoted": False,
        "decision": decision,
    }


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)


def failure_reason(stats, rel, baseline):
    if not stats["finite"]:
        return "non-finite outputs"
    if stats["n_classes"] < 25:
        return f"class collapse to {stats['n_classes']} classes"
    if stats["maxp"]["mean"] > 0.80:
        return f"overconfident maxp {stats['maxp']['mean']:.3f}"
    if stats["entropy"]["mean"] < 0.8 or stats["entropy"]["mean"] > 4.2:
        return f"entropy left the operating band ({stats['entropy']['mean']:.3f})"
    if rel > 0.012:
        return f"relative RMS {rel:.4f} exceeded 0.012"
    if stats["kl_b8"]["mean"] > baseline["kl_b8"]["mean"] + 0.20:
        return f"left the layer-8 target (KL {stats['kl_b8']['mean']:.3f})"
    return ""


def train_variant(name, student, params, anchor_params, x, p_b8, p_orig, y, cap, weight, cfg, baseline, base_pred, base_state):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Reload the known-good weights so variants do not stack.
    student.load_state_dict(base_state, strict=True)
    for p, a in zip(params, anchor_params):
        p.data.copy_(a)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    last_good = copy_state(student)
    stopped = ""
    inactive_last = 0.0
    print(f"\n=== {name}  lambda {cfg['lambda']:.6g}  margin {cfg['margin']:.4f}  "
          f"selective {cfg['selective']} ===", flush=True)
    (ART / "configs").mkdir(parents=True, exist_ok=True)
    (ART / "configs" / f"{name}.json").write_text(json.dumps(cfg, indent=2))
    for step in range(cfg["steps"]):
        lr = LR * (0.5 + 0.5 * (1 - step / cfg["steps"]))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        idx = torch.randint(0, len(x), (BATCH,))
        xb = x[idx].cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z = student(xb)
        la = anchor_loss(p_b8[idx].cuda(), z)
        lrep, kl_batch, inactive = repulsion(
            p_orig[idx].cuda(), z, cap[idx].cuda(), weight[idx].cuda()
        )
        lreg = relative_l2(params, anchor_params)
        loss = la + cfg["lambda"] * lrep + cfg["beta"] * lreg
        if not torch.isfinite(loss):
            stopped = "non-finite loss"
            print(f"  {step:03d}  {stopped}", flush=True)
            break
        cos_s = ""
        if step % EVAL_EVERY == 0 and cfg["lambda"] != 0.0:
            ga = flat_grad(la, params)
            gr = flat_grad(lrep, params)
            cos, na, nr = cosine(ga, gr)
            cos_s = f"  cos {cos:.3f}  |ga| {na:.3e}  |l*gr| {cfg['lambda'] * nr:.3e}"
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        inactive_last = float(inactive)
        if step % 5 == 0 or step + 1 == cfg["steps"]:
            rel = float(relative_l2(params, anchor_params).detach().sqrt())
            print(
                f"  {step:03d}  La {float(la.detach()):.4f}  Lrep {float(lrep.detach()):.4f}  "
                f"Lreg {float(lreg.detach()):.3e}  inactive {inactive_last:.2f}  "
                f"rel {rel:.5f}  |g| {float(gnorm):.3f}{cos_s}",
                flush=True,
            )
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == cfg["steps"]:
            stats, detail, trans = diagnose(
                student, x, p_b8, p_orig, y, base_pred, with_shift=(step + 1 == cfg["steps"])
            )
            rel = float(relative_l2(params, anchor_params).detach().sqrt())
            state = copy_state(student)
            layers = layer_rms(state, base_state)
            reason = failure_reason(stats, rel, baseline)
            print(
                f"  eval {step:03d}  KL_b8 {stats['kl_b8']['mean']:.4f}  "
                f"KL_orig {stats['kl_orig']['mean']:.4f}  py {stats['py']['mean']:.3f}  "
                f"maxp {stats['maxp']['mean']:.3f}  H {stats['entropy']['mean']:.3f}  "
                f"=y {stats['agree_y']:.3f}  =b8 {stats['agree_b8']:.3f}  "
                f"classes {stats['n_classes']}  flip {stats.get('flip_vs_fit', 0):.3f}  "
                f"rel {rel:.5f}",
                flush=True,
            )
            if reason:
                stopped = reason
                print(f"  STOP  {reason}", flush=True)
                student.load_state_dict(last_good, strict=True)
                break
            last_good = state
            tag = f"{name}_s{step+1:03d}"
            safe_save(state, RUNS / f"{tag}.pt")
            torch.save(detail, ART / f"dist_{tag}.pt")
            history.append({"step": step + 1, "inactive": inactive_last, "rel_rms": rel, "layers": layers, **stats})
            student.eval()
    final_state = last_good
    student.load_state_dict(final_state, strict=True)
    stats, detail, trans = diagnose(student, x, p_b8, p_orig, y, base_pred, with_shift=True)
    rel = float(relative_l2(params, anchor_params).detach().sqrt()) if stopped == "" else history[-1]["rel_rms"] if history else 0.0
    # relative_l2 after reload matches the saved weights
    rel = float(relative_l2(params, anchor_params).detach().sqrt())
    layers = layer_rms(final_state, base_state)
    if not history and stopped:
        decision = f"rejected before a usable checkpoint: {stopped}"
        final_path = ""
    else:
        final_path = ROOT / "runs" / f"{name}.pt"
        safe_save(final_state, final_path)
        np.savez_compressed(ART / f"transition_{name}.npz", counts=trans.numpy(), y_from="orig_argmax", pred_to="candidate_argmax")
        torch.save(detail, ART / f"dist_{name}.pt")
        decision = stopped or "completed the scheduled steps"
    row = flat_row(name, history[-1]["step"] if history else 0, cfg, stats, rel, layers, inactive_last, decision)
    append_csv(ART / "metrics.csv", row)
    record = {
        "id": name,
        "config": cfg,
        "checkpoint": str(final_path) if final_path else "",
        "intermediates": [str(p) for p in sorted(RUNS.glob(f"{name}_s*.pt"))],
        "diagnostics": row,
        "layers": layers,
        "history_eval": [
            {
                "step": h["step"],
                "kl_b8": h["kl_b8"]["mean"],
                "kl_orig": h["kl_orig"]["mean"],
                "py": h["py"]["mean"],
                "maxp": h["maxp"]["mean"],
                "entropy": h["entropy"]["mean"],
                "agree_y": h["agree_y"],
                "n_classes": h["n_classes"],
                "rel_rms": h["rel_rms"],
            }
            for h in history
        ],
        "public_score": None,
        "promoted": False,
        "decision": decision,
        "stopped": stopped,
    }
    return record, stats


def save_hist(path, series):
    """Overlaid histograms. series is a list of (label, 1d tensor, rgb)."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("PIL unavailable; histogram figure skipped", flush=True)
        return
    arrays = [(name, v.float().numpy(), color) for name, v, color in series]
    lo = 0.0
    hi = max(float(np.quantile(a, 0.99)) for _, a, _ in arrays)
    hi = max(hi, 1e-3)
    bins = 40
    W, H = 960, 520
    img = Image.new("RGB", (W, H), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 60, 930, 40, 450
    draw.line((left, bottom, right, bottom), fill=(40, 40, 40))
    draw.line((left, top, left, bottom), fill=(40, 40, 40))
    counts = []
    for name, a, color in arrays:
        hist, edges = np.histogram(a, bins=bins, range=(lo, hi))
        counts.append((name, hist, color))
    peak = max(int(h.max()) for _, h, _ in counts)
    peak = max(peak, 1)
    bw = (right - left) / bins
    for name, hist, color in counts:
        pts = []
        for i, c in enumerate(hist):
            x0 = left + i * bw
            y0 = bottom - (c / peak) * (bottom - top)
            pts.append((x0, bottom))
            pts.append((x0, y0))
            pts.append((x0 + bw, y0))
            pts.append((x0 + bw, bottom))
        # outline
        line = []
        for i, c in enumerate(hist):
            line.append((left + (i + 0.5) * bw, bottom - (c / peak) * (bottom - top)))
        draw.line(line, fill=color, width=2)
    legend_x = 70
    for name, _, color in arrays:
        draw.rectangle((legend_x, 460, legend_x + 18, 478), fill=color)
        draw.text((legend_x + 24, 462), name, fill=(20, 20, 20))
        legend_x += 40 + 8 * len(name)
    draw.text((60, 12), "KL(p_orig || p) on the 3500 forget images, T=2", fill=(20, 20, 20))
    draw.text((left, 490), f"{lo:.2f}", fill=(20, 20, 20))
    draw.text((right - 40, 490), f"{hi:.2f}", fill=(20, 20, 20))
    img.save(path)
    print(f"wrote {path}", flush=True)


def strict_check(path: Path):
    model = tt.build_model()
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    bad = [k for k, v in state.items() if v.dtype != torch.float32 or not torch.isfinite(v).all()]
    n_bytes = path.stat().st_size
    return {"path": str(path), "strict_load": True, "bad_tensors": bad, "bytes": n_bytes, "under_450mb": n_bytes < 450 * 1024 * 1024}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "audit"
    t0 = time.time()
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    print("torch", torch.__version__, "cuda", torch.cuda.get_device_name(0), flush=True)
    hashes = protect_baselines()
    self_check()

    images = tt.load_forget_images(U.STARTER / "forget_images.npz")
    x = U.to_norm(images)
    blob = load_teachers()
    final, b8 = blob["final"], blob["b8"]
    y = final.argmax(1)
    p_b8 = torch.softmax(b8, dim=-1)
    p_orig = torch.softmax(final / T, dim=-1)
    print(
        f"orig maxp {p_orig.max(1).values.mean():.4f}  classes {int(y.unique().numel())}  "
        f"b8 maxp {p_b8.max(1).values.mean():.3f}  b8==y {(p_b8.argmax(1) == y).float().mean():.3f}",
        flush=True,
    )

    student = tt.load_model(ROOT / "runs" / "fit_b8.pt")
    params, names = trainable_of(student)
    base_state = copy_state(student)
    anchor_params = [p.detach().clone() for p in params]
    theta_norm = float(torch.stack([a.float().pow(2).sum() for a in anchor_params]).sum().sqrt())

    print("reproducing fit_b8 diagnostics", flush=True)
    base_stats, base_detail, base_trans = diagnose(student, x, p_b8, p_orig, y, with_shift=True)
    base_pred = base_detail["pred"].long()
    print(
        f"fit_b8  KL_b8 {base_stats['kl_b8']['mean']:.4f}  KL_orig {base_stats['kl_orig']['mean']:.4f}  "
        f"py {base_stats['py']['mean']:.3f}  maxp {base_stats['maxp']['mean']:.3f}  "
        f"H {base_stats['entropy']['mean']:.3f}  =y {base_stats['agree_y']:.3f}  "
        f"=b8 {base_stats['agree_b8']:.3f}  classes {base_stats['n_classes']}  "
        f"corr(KL_orig, -log py) {base_stats['kl_orig_vs_nll_y']:.4f}",
        flush=True,
    )
    if not (0.04 <= base_stats["kl_b8"]["mean"] <= 0.09 and 0.45 <= base_stats["agree_y"] <= 0.58):
        raise SystemExit("fit_b8 reproduction does not match the recorded operating point; not training")

    weights, weight_cv, mean_p_b8_y = selective_weights(p_b8, y)
    ones = torch.ones(len(y))
    # Same forward as the diagnostic. log_softmax KL is the training-loss definition.
    kl_ref = kl_rows(p_orig, base_detail.pop("logits"))
    print(
        f"KL(p_orig||fit) {qstats(kl_ref)}  "
        f"inactive if cap=KL_ref {(kl_ref >= kl_ref).float().mean():.3f}  "
        f"weight cv {weight_cv:.3f}  mean p_b8(y) {mean_p_b8_y:.3f}",
        flush=True,
    )

    # A cap sitting on the current KL must be inactive. A cap one margin above must not be.
    print("gradient probe", flush=True)
    # Temporary caps for the probe use a mild placeholder; design() replaces them.
    # Probe the UNBOUNDED repulsion (cap far above) so the gradient is the true -KL gradient,
    # and also the on-cap repulsion to confirm it is identically zero.
    far = kl_ref + 10.0
    g_anchor, g_rep = probe_gradients(student, params, x, p_b8, p_orig, far, ones)
    cos, na, nr = cosine(g_anchor, g_rep)
    print(f"grad anchor {na:.4e}  grad repulsion {nr:.4e}  cosine {cos:.4f}", flush=True)
    on_cap = kl_ref  # KL >= cap on every image, so the min() is the constant
    # Numerical equality is >=, and kl_ref was computed from a separate forward.
    # Recompute on the probe batches inside probe by passing cap = kl of THAT forward.
    # Here we only check a fresh tiny forward.
    idx = torch.arange(0, BATCH)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        z = student(x[idx].cuda())
    live = kl_rows(p_orig[idx].cuda(), z).detach()
    loss_on, _, inactive_on = repulsion(p_orig[idx].cuda(), z, live, ones[idx].cuda())
    if loss_on.requires_grad:
        g_on_norm = float(flat_grad(loss_on, params).norm())
    else:
        g_on_norm = 0.0
    print(f"on-cap inactive {float(inactive_on):.3f}  grad norm {g_on_norm:.3e}", flush=True)
    student.zero_grad(set_to_none=True)

    cfg = design_from_probe(kl_ref, g_anchor, g_rep, cos, theta_norm, weight_cv)
    cap_mild = (kl_ref + cfg["margin_mild"]).detach()
    cap_mod = (kl_ref + cfg["margin_moderate"]).detach()
    inactive_mild = float((kl_ref >= cap_mild).float().mean())
    inactive_mod = float((kl_ref >= cap_mod).float().mean())
    cfg["inactive_at_step0_mild"] = inactive_mild
    cfg["inactive_at_step0_moderate"] = inactive_mod
    cfg["kl_orig_fit"] = qstats(kl_ref)
    cfg["fit_b8_diagnostics"] = {k: base_stats[k] for k in (
        "kl_b8", "kl_orig", "py", "maxp", "entropy", "agree_y", "agree_b8", "n_classes", "kl_orig_vs_nll_y"
    )}
    cfg["hashes"] = hashes
    cfg["trainable"] = "blocks.8-11"
    cfg["n_trainable_tensors"] = len(names)
    (ART / "design.json").write_text(json.dumps(cfg, indent=2))
    print(
        f"design  margin {cfg['margin_mild']:.3f}/{cfg['margin_moderate']:.3f}  "
        f"lambda {cfg['lambda_mild']:.5f}/{cfg['lambda_moderate']:.5f}  "
        f"beta {cfg['beta']:.4g}  step0 inactive {inactive_mild:.3f}/{inactive_mod:.3f}",
        flush=True,
    )
    torch.save(base_detail, ART / "dist_fit_b8.pt")
    row0 = flat_row(
        "fit_b8", -1,
        {"lambda": 0.0, "margin": 0.0, "beta": 0.0, "selective": False},
        base_stats, 0.0, layer_rms(base_state, base_state), 0.0,
        "reference; public score 0.147; not retrained",
    )
    if mode == "audit":
        # Rewrite the csv so a previous partial run cannot mix with this audit.
        csv_path = ART / "metrics.csv"
        if csv_path.exists() and os.environ.get("SCRUB_KEEP_CSV") != "1":
            csv_path.unlink()
        append_csv(csv_path, row0)
        print(f"audit only, {time.time() - t0:.0f}s. design written to {ART / 'design.json'}", flush=True)
        protect_baselines()
        return

    if inactive_mild > 0.05 or inactive_mod > 0.05:
        raise SystemExit("caps are already binding at step 0; refusing to train a dead repulsion term")
    if g_on_norm > 1e-6:
        raise SystemExit("on-cap repulsion gradient is not zero; the bound is implemented incorrectly")

    csv_path = ART / "metrics.csv"
    if csv_path.exists():
        csv_path.unlink()
    append_csv(csv_path, row0)
    registry = []

    variants = [
        ("scrub_b8_anchor", 0.0, cap_mild, ones, False),
        ("scrub_b8_mild", cfg["lambda_mild"], cap_mild, ones, False),
        ("scrub_b8_moderate", cfg["lambda_moderate"], cap_mod, ones, False),
    ]
    results = {}
    for name, lam, cap, w, selective in variants:
        if name == "scrub_b8_moderate" and "scrub_b8_mild" in results:
            mild_rec, mild_stats = results["scrub_b8_mild"]
            if mild_rec["stopped"]:
                print(f"skip {name}: mild run stopped ({mild_rec['stopped']})", flush=True)
                registry.append({
                    "id": name, "promoted": False, "public_score": None,
                    "decision": f"not run because mild stopped: {mild_rec['stopped']}",
                    "checkpoint": "",
                })
                continue
        vcfg = {
            "lambda": lam,
            "margin": cfg["margin_mild"] if "moderate" not in name else cfg["margin_moderate"],
            "beta": cfg["beta"],
            "selective": selective,
            "steps": STEPS,
            "lr": LR,
            "beta_source": "design.json",
        }
        if name == "scrub_b8_anchor":
            vcfg["margin"] = 0.0
        rec, stats = train_variant(
            name, student, params, anchor_params, x, p_b8, p_orig, y,
            cap, w, vcfg, base_stats, base_pred, base_state,
        )
        results[name] = (rec, stats)
        registry.append(rec)

    # Selective only if uniform mild repulsion actually moved KL_orig beyond the
    # anchor-only control while keeping the layer-8 shape.
    run_selective = False
    skip_why = "mild or anchor result missing"
    if "scrub_b8_mild" in results and "scrub_b8_anchor" in results:
        mild_s = results["scrub_b8_mild"][1]
        anch_s = results["scrub_b8_anchor"][1]
        d_kl = mild_s["kl_orig"]["mean"] - anch_s["kl_orig"]["mean"]
        d_shape = mild_s["kl_b8"]["mean"] - base_stats["kl_b8"]["mean"]
        run_selective = (
            not results["scrub_b8_mild"][0]["stopped"]
            and d_kl > 0.02
            and d_shape < 0.03
            and mild_s["n_classes"] >= 25
            and weight_cv >= 0.20
        )
        skip_why = (
            f"delta KL_orig vs anchor {d_kl:.4f}, delta KL_b8 vs fit {d_shape:.4f}, "
            f"weight cv {weight_cv:.3f}"
        )
    if run_selective:
        vcfg = {
            "lambda": cfg["lambda_mild"],
            "margin": cfg["margin_mild"],
            "beta": cfg["beta"],
            "selective": True,
            "steps": STEPS,
            "lr": LR,
            "weight": "1 - p_b8(y), clip [0.25, 2.5], mean 1",
        }
        rec, stats = train_variant(
            "scrub_b8_selective", student, params, anchor_params, x, p_b8, p_orig, y,
            cap_mild, weights, vcfg, base_stats, base_pred, base_state,
        )
        results["scrub_b8_selective"] = (rec, stats)
        registry.append(rec)
    else:
        print(f"skip selective: {skip_why}", flush=True)
        registry.append({
            "id": "scrub_b8_selective",
            "promoted": False,
            "public_score": None,
            "checkpoint": "",
            "decision": f"not run. {skip_why}",
        })

    # Weight-space midpoints. Evaluated, saved only when the endpoint stayed finite
    # and actually moved.
    for name, _ in list(results.items()):
        rec, stats = results[name]
        if not rec["checkpoint"]:
            continue
        if abs(stats["kl_orig"]["mean"] - base_stats["kl_orig"]["mean"]) < 0.01:
            continue
        other = torch.load(rec["checkpoint"], map_location="cpu", weights_only=True)
        mid = {k: 0.5 * (base_state[k] + other[k]) for k in base_state}
        student.load_state_dict(mid, strict=True)
        mstats, mdetail, mtrans = diagnose(student, x, p_b8, p_orig, y, base_pred, with_shift=False)
        rel = float(relative_l2(params, anchor_params).detach().sqrt())
        mid_name = name.replace("scrub_b8_", "scrub_b8_interp50_")
        print(
            f"interp {mid_name}  KL_b8 {mstats['kl_b8']['mean']:.4f}  "
            f"KL_orig {mstats['kl_orig']['mean']:.4f}  py {mstats['py']['mean']:.3f}  "
            f"maxp {mstats['maxp']['mean']:.3f}  classes {mstats['n_classes']}",
            flush=True,
        )
        reason = failure_reason(mstats, rel, base_stats)
        if reason:
            decision = f"interpolation not saved: {reason}"
            path = ""
        else:
            path = ROOT / "runs" / f"{mid_name}.pt"
            safe_save(copy_state(student), path)
            torch.save(mdetail, ART / f"dist_{mid_name}.pt")
            decision = "weight midpoint between fit_b8 and the named endpoint"
        row = flat_row(
            mid_name, STEPS // 2,
            {"lambda": "", "margin": "", "beta": cfg["beta"], "selective": "selective" in name},
            mstats, rel, layer_rms(copy_state(student), base_state), "", decision,
        )
        append_csv(csv_path, row)
        registry.append({
            "id": mid_name, "checkpoint": str(path) if path else "", "promoted": False,
            "public_score": None, "decision": decision, "diagnostics": row,
        })

    # Figure: baseline plus whichever full runs exist.
    series = [("fit_b8", base_detail["kl_orig"].float(), (60, 60, 60))]
    palette = {
        "scrub_b8_anchor": (30, 90, 160),
        "scrub_b8_mild": (20, 130, 80),
        "scrub_b8_moderate": (190, 90, 20),
        "scrub_b8_selective": (120, 40, 140),
    }
    for name, color in palette.items():
        dist_path = ART / f"dist_{name}.pt"
        if dist_path.exists():
            d = torch.load(dist_path, map_location="cpu", weights_only=False)
            series.append((name.replace("scrub_b8_", ""), d["kl_orig"].float(), color))
    save_hist(ART / "kl_orig_hist.png", series)

    checks = []
    for rec in registry:
        ck = rec.get("checkpoint") or ""
        if ck and Path(ck).exists() and Path(ck).name.endswith(".pt") and "interp" not in Path(ck).name:
            checks.append(strict_check(Path(ck)))
        for mid in rec.get("intermediates") or []:
            if Path(mid).exists():
                checks.append(strict_check(Path(mid)))
    end_hashes = protect_baselines()
    (ART / "registry.json").write_text(json.dumps({
        "hashes_end": end_hashes,
        "hashes_start": hashes,
        "compatibility": checks,
        "runs": registry,
        "seconds": time.time() - t0,
    }, indent=2))
    print(f"done in {time.time() - t0:.0f}s  hashes unchanged {end_hashes['fit_b8_pt'] == hashes['fit_b8_pt']}", flush=True)


if __name__ == "__main__":
    main()
