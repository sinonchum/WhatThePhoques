"""Small MU-Mis pilot. Equation (8) of Cheng et al., with original-argmax pseudo-labels.

Not the labeled MU-Mis method: c is the original model's argmax, fixed before
any update. c' is resampled every step from the other 99 classes. Gradients are
with respect to the normalized model input. The model stays in eval mode, as in
the official implementation. No mixed precision.

The script first checks that one Adam step reduces the loss, records which
blocks receive the parameter gradient, then runs the same short schedule from
model.pt and from runs/fit_b8.pt. It does not write those two files.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402
import unlearn_run as U  # noqa: E402
from mumis_diag import other_classes, quantiles, save_hist, save_scatter, sensitivity, sha256_file

ART = ROOT / "artifacts" / "mumis"
RUNS = ROOT / "runs"
# Home directory quota cannot hold another full ViT checkpoint. Large files go
# to this account's scratch; task3/runs gets a symlink only.
CKPT = Path("/p/scratch/training2666/li103/mumis")
FORBIDDEN = {(ROOT / "submission.pt").resolve(), (ROOT / "runs" / "fit_b8.pt").resolve()}
STEPS = 48
PROBE_LRS = (1e-5, 1e-4, 5e-4)


def protect():
    h = {
        "submission": sha256_file(ROOT / "submission.pt"),
        "fit_b8": sha256_file(ROOT / "runs" / "fit_b8.pt"),
    }
    if h["submission"] != h["fit_b8"]:
        raise SystemExit("submission.pt and fit_b8.pt differ")
    return h


def safe_save(state, path: Path):
    path = Path(path)
    if path.resolve() in FORBIDDEN or path.name in {"submission.pt", "fit_b8.pt"}:
        raise RuntimeError(f"refusing to write {path}")
    bad = [k for k, v in state.items() if v.dtype != torch.float32 or not torch.isfinite(v).all()]
    if bad:
        raise RuntimeError(f"bad tensors {bad[:3]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # The zip writer fails when it streams a 328 MB archive onto the home
    # filesystem, and the home quota cannot hold another copy. Write the
    # archive on local disk, then copy the finished file to scratch.
    fd, local = tempfile.mkstemp(suffix=".pt", dir="/tmp")
    os.close(fd)
    torch.save(state, local)
    dest_tmp = path.with_suffix(path.suffix + ".tmp")
    shutil.copy2(local, dest_tmp)
    os.replace(dest_tmp, path)
    os.remove(local)


def publish(saved: Path, name: str):
    link = RUNS / f"{name}.pt"
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"refusing to replace an existing file at {link}")
    link.symlink_to(saved)
    return saved


def state_of(model):
    return {k: v.detach().cpu().float().contiguous().clone() for k, v in model.state_dict().items()}


def group_of(name: str) -> str:
    if name.startswith("blocks."):
        return "block" + name.split(".")[1]
    if name.startswith("head"):
        return "head"
    if "patch_embed" in name:
        return "patch"
    return "other"


def mumis_loss(model, x, target, other):
    xb = x.detach().requires_grad_(True)
    logits = model(xb).float()
    idx = torch.arange(xb.shape[0], device=xb.device)
    g_t = torch.autograd.grad(logits[idx, target].sum(), xb, create_graph=True, retain_graph=True)[0]
    g_o = torch.autograd.grad(logits[idx, other].sum(), xb, create_graph=True)[0]
    tc = g_t.flatten(1).float().pow(2).sum(1)
    oc = g_o.flatten(1).float().pow(2).sum(1)
    return (tc - oc).mean(), tc.detach(), oc.detach()


def grad_shares(model):
    buckets, total = {}, 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        value = p.grad.detach().float().pow(2).sum().item()
        key = group_of(name)
        buckets[key] = buckets.get(key, 0.0) + value
        total += value
    total = max(total, 1e-12)
    shares = {k: v / total for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])}
    return shares, total ** 0.5


def rel_rms(params, anchor):
    num = torch.zeros((), device=params[0].device)
    den = torch.zeros((), device=params[0].device)
    for p, a in zip(params, anchor):
        num = num + (p.detach() - a).float().pow(2).sum()
        den = den + a.float().pow(2).sum()
    return float((num / den.clamp_min(1e-12)).sqrt())


def linear_graph_check():
    layer = torch.nn.Linear(6, 4, bias=False)
    x = torch.randn(2, 6)
    loss, _, _ = mumis_loss(layer, x, torch.tensor([0, 1]), torch.tensor([1, 2]))
    loss.backward()
    grad = layer.weight.grad
    if grad is None or not torch.isfinite(grad).all() or float(grad.abs().sum()) == 0.0:
        raise RuntimeError("higher-order parameter gradient failed the linear check")
    print("linear higher-order check passed", flush=True)


def load_student(path):
    model = tt.load_model(path).eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def trainable(model, allowed):
    params = []
    for name, p in model.named_parameters():
        keep = allowed is None or group_of(name) in allowed
        p.requires_grad_(keep)
        if keep:
            params.append(p)
    return params


@torch.no_grad()
def predict(model, x, batch=64):
    was = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    rows = []
    for i in range(0, len(x), batch):
        rows.append(model(x[i : i + batch].cuda()).float().cpu())
    for p, flag in zip(model.parameters(), was):
        p.requires_grad_(flag)
    return torch.cat(rows)


def prediction_stats(logits, y, p_orig, p_b8):
    p = torch.softmax(logits / 2, dim=-1)
    pred = logits.argmax(1)
    order = torch.sort(logits, dim=-1, descending=True).values
    ent = -(p.clamp_min(1e-8).log() * p).sum(-1)

    def kl(src, dst):
        return (src * (src.clamp_min(1e-8).log() - dst.clamp_min(1e-8).log())).sum(-1)

    return {
        "py_t2": float(p[torch.arange(len(y)), y].mean()),
        "maxp_t2": float(p.max(1).values.mean()),
        "entropy": float(ent.mean()),
        "agree_y": float((pred == y).float().mean()),
        "n_classes": int(pred.unique().numel()),
        "logit_gap12": float((order[:, 0] - order[:, 1]).mean()),
        "logit_gap23": float((order[:, 1] - order[:, 2]).mean()),
        "kl_orig": float(kl(p_orig, p).mean()),
        "kl_b8": float(kl(p_b8, p).mean()),
        "kl_to_b8": float(kl(p, p_b8).mean()),
        "finite": bool(torch.isfinite(logits).all()),
    }


def smoke(model, x, y, base_state):
    model.load_state_dict(base_state, strict=True)
    trainable(model, None)
    other = other_classes(y[:1], seed=123)
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    loss, tc, oc = mumis_loss(model, x[:1].cuda(), y[:1].cuda(), other.cuda())
    loss.backward()
    shares, gnorm = grad_shares(model)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(
        f"smoke grad norm {gnorm:.4g}  peak {peak:.2f} GiB  loss {float(loss.detach()):.4g}  "
        f"TC {float(tc):.4g}  OC {float(oc):.4g}",
        flush=True,
    )
    top = sorted(shares.items(), key=lambda kv: -kv[1])[:8]
    print("grad share", " ".join(f"{k}:{v:.3f}" for k, v in top), flush=True)
    kept = {name for name, share in shares.items() if share >= 0.02}
    if not kept:
        kept = set(shares)
    trials = []
    for lr in PROBE_LRS:
        model.load_state_dict(base_state, strict=True)
        params = trainable(model, kept)
        anchor = [p.detach().clone() for p in params]
        opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
        model.zero_grad(set_to_none=True)
        loss0, tc0, oc0 = mumis_loss(model, x[:1].cuda(), y[:1].cuda(), other.cuda())
        loss0.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        loss1, tc1, oc1 = mumis_loss(model, x[:1].cuda(), y[:1].cuda(), other.cuda())
        moved = rel_rms(params, anchor)
        finite = bool(torch.isfinite(loss1).all() and all(torch.isfinite(p).all() for p in params))
        row = {
            "lr": lr,
            "loss0": float(loss0.detach()),
            "loss1": float(loss1.detach()),
            "tc0": float(tc0),
            "tc1": float(tc1),
            "oc0": float(oc0),
            "oc1": float(oc1),
            "rel_rms": moved,
            "finite": finite,
        }
        trials.append(row)
        print(
            f"  lr {lr:.1e}  loss {row['loss0']:.4g}->{row['loss1']:.4g}  "
            f"TC {row['tc0']:.4g}->{row['tc1']:.4g}  OC {row['oc0']:.4g}->{row['oc1']:.4g}  "
            f"rel {moved:.5f}  finite {finite}",
            flush=True,
        )
        model.zero_grad(set_to_none=True)
    # One Adam step at 1e-4 already moves the whole network by 0.003 relative RMS
    # and then walks into the safeguard within a few dozen steps. Keep the largest
    # rate whose single step stays under 0.001 relative RMS.
    good = [t for t in trials if t["finite"] and t["loss1"] < t["loss0"] and t["rel_rms"] <= 0.001]
    chosen = max(good, key=lambda t: t["lr"])["lr"] if good else None
    model.load_state_dict(base_state, strict=True)
    return {
        "shares": shares,
        "grad_norm": gnorm,
        "peak_gib": peak,
        "trials": trials,
        "kept": sorted(kept),
        "lr": chosen,
    }


def run_pilot(name, init_path, x_sub, y_sub, kept, lr, x_all, y_all, p_orig, p_b8):
    model = load_student(init_path)
    params = trainable(model, set(kept))
    anchor = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
    order = torch.randperm(len(x_sub), generator=torch.Generator().manual_seed(0))
    history = []
    last_good = state_of(model)
    stopped = ""
    t0 = time.time()
    print(f"\n=== {name}  lr {lr:.1e}  groups {len(kept)}  steps {STEPS} ===", flush=True)
    for step in range(STEPS):
        pos = step % len(order)
        idx = order[pos : pos + 1]
        y = y_sub[idx]
        other = other_classes(y, seed=20_000 + step)
        opt.zero_grad(set_to_none=True)
        loss, tc, oc = mumis_loss(model, x_sub[idx].cuda(), y.cuda(), other.cuda())
        if not torch.isfinite(loss):
            stopped = "non-finite loss"
            break
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 10.0))
        opt.step()
        if not all(torch.isfinite(p).all() for p in params):
            stopped = "non-finite parameters"
            model.load_state_dict(last_good, strict=True)
            break
        if step % 8 == 0 or step + 1 == STEPS:
            moved = rel_rms(params, anchor)
            print(
                f"  {step:03d}  loss {float(loss.detach()):.4g}  TC {float(tc):.4g}  "
                f"OC {float(oc):.4g}  |g| {gnorm:.3f}  rel {moved:.5f}",
                flush=True,
            )
            history.append({
                "step": step, "loss": float(loss.detach()), "tc": float(tc), "oc": float(oc),
                "grad_norm": gnorm, "rel_rms": moved,
            })
            if moved > 0.05:
                stopped = f"relative RMS {moved:.4f}"
                history.pop()
                print(f"  stop {stopped}; reverting to the previous checkpoint", flush=True)
                model.load_state_dict(last_good, strict=True)
                break
            last_good = state_of(model)
        model.eval()
    model.load_state_dict(last_good, strict=True)
    path = CKPT / f"{name}.pt"
    safe_save(state_of(model), path)
    publish(path, name)
    print(f"  saved {path} before evaluation  stop '{stopped}'", flush=True)
    torch.cuda.empty_cache()
    sens = []
    for seed in (0, 1):
        tc, oc, finite = sensitivity(model, x_sub, y_sub, other_classes(y_sub, seed=10_000 + seed))
        sens.append((tc, oc, finite))
    tc = torch.stack([item[0] for item in sens]).mean(0)
    oc = torch.stack([item[1] for item in sens]).mean(0)
    logits = predict(model, x_all)
    stats = prediction_stats(logits, y_all, p_orig, p_b8)
    moved = rel_rms(params, anchor)
    path = CKPT / f"{name}.pt"
    safe_save(state_of(model), path)
    publish(path, name)
    probe = tt.build_model()
    probe.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    del probe
    record = {
        "id": name,
        "init": str(init_path),
        "lr": lr,
        "steps_completed": (history[-1]["step"] + 1) if history else 0,
        "stopped": stopped,
        "seconds": time.time() - t0,
        "rel_rms": moved,
        "kept": kept,
        "checkpoint": str(path),
        "predictions": stats,
        "sensitivity": {
            "tc": quantiles(tc),
            "oc": quantiles(oc),
            "gap": quantiles(tc - oc),
            "frac_tc_gt_oc": float((tc > oc).float().mean()),
        },
        "history": history,
        "promoted": False,
        "public_score": None,
    }
    print(
        f"{name}  KL_orig {stats['kl_orig']:.3f}  KL_b8 {stats['kl_b8']:.3f}  "
        f"py {stats['py_t2']:.3f}  H {stats['entropy']:.3f}  =y {stats['agree_y']:.3f}  "
        f"classes {stats['n_classes']}  gap {record['sensitivity']['gap']['mean']:.3f}  "
        f"rel {moved:.5f}  stop '{stopped}'",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()
    return record, logits, tc, oc


def enable_math_attention():
    """Efficient SDPA has no second derivative. MU-Mis needs that derivative."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def main():
    t0 = time.time()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    enable_math_attention()
    hashes = protect()
    linear_graph_check()
    blob = torch.load(ART / "sensitivity.pt", map_location="cpu", weights_only=False)
    images = tt.load_forget_images(U.STARTER / "forget_images.npz")
    x_all = U.to_norm(images)
    y_all = torch.from_numpy(np.load(ROOT / "artifacts" / "clean_preds.npz")["pred"].astype("int64"))
    idx = blob["subsets"][0]
    x_sub, y_sub = x_all[idx], y_all[idx]

    print("reference distributions", flush=True)
    orig_model = load_student(U.STARTER / "model.pt")
    orig_logits = predict(orig_model, x_all)
    del orig_model
    torch.cuda.empty_cache()
    b8_model = load_student(ROOT / "runs" / "fit_b8.pt")
    b8_logits = predict(b8_model, x_all)
    del b8_model
    torch.cuda.empty_cache()
    p_orig = torch.softmax(orig_logits / 2, dim=-1)
    p_b8 = torch.softmax(b8_logits / 2, dim=-1)
    ref_rows = {
        "orig": prediction_stats(orig_logits, y_all, p_orig, p_b8),
        "b8": prediction_stats(b8_logits, y_all, p_orig, p_b8),
    }
    print(
        f"orig py {ref_rows['orig']['py_t2']:.3f}  b8 py {ref_rows['b8']['py_t2']:.3f}  "
        f"b8 KL_orig {ref_rows['b8']['kl_orig']:.3f}",
        flush=True,
    )

    model = load_student(U.STARTER / "model.pt")
    info = smoke(model, x_sub, y_sub, state_of(model))
    del model
    torch.cuda.empty_cache()
    (ART / "smoke.json").write_text(json.dumps(info, indent=2))
    if info["lr"] is None:
        protect()
        print("no learning rate reduced the loss without a large jump; pilot not started", flush=True)
        (ART / "pilot_registry.json").write_text(json.dumps({
            "smoke": info, "hashes": hashes, "pilots": [], "reference": ref_rows,
        }, indent=2))
        return

    pilots = []
    curves = {}
    for name, path in (
        ("mumis_orig_pilot", U.STARTER / "model.pt"),
        ("mumis_b8_pilot", ROOT / "runs" / "fit_b8.pt"),
    ):
        rec, logits, tc, oc = run_pilot(
            name, path, x_sub, y_sub, info["kept"], info["lr"], x_all, y_all, p_orig, p_b8,
        )
        pilots.append(rec)
        curves[name] = {"logits": logits, "tc": tc, "oc": oc}

    base_gap = blob["records_mean"]["orig_0"]["tc"] - blob["records_mean"]["orig_0"]["oc"]
    b8_gap = blob["records_mean"]["b8_0"]["tc"] - blob["records_mean"]["b8_0"]["oc"]
    save_scatter(
        ART / "gap_before_after.png",
        [
            (base_gap, curves["mumis_orig_pilot"]["tc"] - curves["mumis_orig_pilot"]["oc"], (30, 90, 160), "from orig"),
            (b8_gap, curves["mumis_b8_pilot"]["tc"] - curves["mumis_b8_pilot"]["oc"], (190, 90, 20), "from b8"),
        ],
        "Sensitivity gap before and after the pilot",
        "gap before",
        "gap after",
    )

    def entropy_of(logits):
        prob = torch.softmax(logits / 2, dim=-1)
        return -(prob.clamp_min(1e-8).log() * prob).sum(-1)

    save_hist(
        ART / "entropy_compare.png",
        [
            ("orig", entropy_of(orig_logits), (60, 60, 60)),
            ("b8", entropy_of(b8_logits), (190, 90, 20)),
            ("from orig", entropy_of(curves["mumis_orig_pilot"]["logits"]), (30, 90, 160)),
            ("from b8", entropy_of(curves["mumis_b8_pilot"]["logits"]), (20, 130, 80)),
        ],
        "Temperature-2 entropy on all 3500 forget images",
    )
    end = protect()
    (ART / "pilot_registry.json").write_text(json.dumps({
        "hashes_start": hashes,
        "hashes_end": end,
        "smoke": info,
        "reference": ref_rows,
        "pilots": pilots,
        "seconds": time.time() - t0,
    }, indent=2))
    fields = [
        "model", "lr", "steps", "stopped", "rel_rms", "py_t2", "maxp_t2", "entropy", "agree_y",
        "n_classes", "kl_orig", "kl_b8", "gap_mean", "tc_mean", "oc_mean", "frac_tc_gt_oc", "public_score",
    ]
    with (ART / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, stats in ref_rows.items():
            sens_key = "orig_0" if key == "orig" else "b8_0"
            sens = blob["records_mean"][sens_key]
            writer.writerow({
                "model": key, "lr": "", "steps": 0, "stopped": "", "rel_rms": 0,
                "py_t2": stats["py_t2"], "maxp_t2": stats["maxp_t2"], "entropy": stats["entropy"],
                "agree_y": stats["agree_y"], "n_classes": stats["n_classes"],
                "kl_orig": stats["kl_orig"], "kl_b8": stats["kl_b8"],
                "gap_mean": float((sens["tc"] - sens["oc"]).mean()),
                "tc_mean": float(sens["tc"].mean()), "oc_mean": float(sens["oc"].mean()),
                "frac_tc_gt_oc": float((sens["tc"] > sens["oc"]).float().mean()),
                "public_score": 0.147 if key == "b8" else "",
            })
        for rec in pilots:
            st, se = rec["predictions"], rec["sensitivity"]
            writer.writerow({
                "model": rec["id"], "lr": rec["lr"], "steps": rec["steps_completed"], "stopped": rec["stopped"],
                "rel_rms": rec["rel_rms"], "py_t2": st["py_t2"], "maxp_t2": st["maxp_t2"], "entropy": st["entropy"],
                "agree_y": st["agree_y"], "n_classes": st["n_classes"], "kl_orig": st["kl_orig"], "kl_b8": st["kl_b8"],
                "gap_mean": se["gap"]["mean"], "tc_mean": se["tc"]["mean"], "oc_mean": se["oc"]["mean"],
                "frac_tc_gt_oc": se["frac_tc_gt_oc"], "public_score": "",
            })
    print(f"pilot finished in {time.time() - t0:.0f}s  hashes unchanged {end == hashes}", flush=True)


if __name__ == "__main__":
    main()
