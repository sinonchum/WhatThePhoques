"""Input-sensitivity diagnostic for MU-Mis. No parameter updates.

Coordinates: gradients are with respect to the model's normalized input
x = (pixel/255 - 0.5) / 0.5, the same tensor the ViT consumes. They are not
gradients with respect to raw uint8 pixels. A global rescaling would multiply
every squared norm by the same constant, so TC/OC ratios are invariant to that
choice and absolute norms are not. Both checkpoints use this coordinate system.

The target class is the original model's argmax, cached and not updated.
That is a pseudo-label, not a ground-truth label. The other class is drawn
uniformly from the remaining 99 indices. Several fixed seeds are repeated so
the sampling noise is visible.

TC and OC are per-image squared Frobenius norms of the raw-logit input
gradient, which for one image equals the squared Euclidean norm. The official
MU-Mis code squares the Frobenius norm of the whole batch and divides by the
batch size; on a model with no cross-image operations those two definitions match.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402
import unlearn_run as U  # noqa: E402

ART = ROOT / "artifacts" / "mumis"
N = 128
SEEDS = (0, 1, 2, 3)
BATCH = 8


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(v: torch.Tensor) -> dict:
    v = v.float()
    qs = torch.quantile(v, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    return {
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=False)),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
    }


def stratified_indices(y: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    buckets = {}
    for c in y.unique().tolist():
        pool = (y == c).nonzero().view(-1)
        buckets[int(c)] = pool[torch.randperm(pool.numel(), generator=g)]
    ptr = {c: 0 for c in buckets}
    order = []
    while len(order) < n:
        grew = False
        for c in sorted(buckets):
            if ptr[c] < buckets[c].numel():
                order.append(int(buckets[c][ptr[c]]))
                ptr[c] += 1
                grew = True
                if len(order) == n:
                    break
        if not grew:
            break
    return torch.tensor(order, dtype=torch.long)


def other_classes(y: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    draw = torch.randint(0, 99, (y.numel(),), generator=g)
    return draw + (draw >= y).long()


def self_check():
    """A linear map has an input gradient equal to the corresponding row."""
    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 3, bias=False)
    x = torch.randn(4, 8, requires_grad=True)
    logits = layer(x)
    target = torch.tensor([0, 1, 2, 0])
    other = torch.tensor([1, 0, 0, 2])
    g_t = torch.autograd.grad(logits[torch.arange(4), target].sum(), x, retain_graph=True)[0]
    g_o = torch.autograd.grad(logits[torch.arange(4), other].sum(), x)[0]
    expect_t = layer.weight[target]
    expect_o = layer.weight[other]
    if not torch.allclose(g_t, expect_t, atol=1e-5) or not torch.allclose(g_o, expect_o, atol=1e-5):
        raise RuntimeError("input-gradient self-check failed")
    tc = g_t.pow(2).sum(-1)
    official = torch.norm(g_t, p="fro") ** 2 / 4
    if not torch.allclose(tc.mean(), official, rtol=1e-5, atol=1e-6):
        raise RuntimeError("per-image squared norm does not match the official batch formula")
    print("self-check passed", flush=True)


@torch.no_grad()
def forward_logits(model, x, batch=32):
    rows = []
    for i in range(0, len(x), batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            rows.append(model(x[i : i + batch].cuda()).float().cpu())
    return torch.cat(rows)


def sensitivity(model, x, target, other):
    """Per-image TC, OC on normalized inputs. First-order grads only."""
    model.eval()
    tc = torch.empty(len(x))
    oc = torch.empty(len(x))
    finite = True
    for start in range(0, len(x), BATCH):
        sl = slice(start, start + BATCH)
        xb = x[sl].cuda().detach().requires_grad_(True)
        logits = model(xb).float()
        b = xb.shape[0]
        idx = torch.arange(b, device=xb.device)
        g_t = torch.autograd.grad(logits[idx, target[sl].cuda()].sum(), xb, retain_graph=True)[0]
        g_o = torch.autograd.grad(logits[idx, other[sl].cuda()].sum(), xb)[0]
        tc[sl] = g_t.detach().float().flatten(1).pow(2).sum(1).cpu()
        oc[sl] = g_o.detach().float().flatten(1).pow(2).sum(1).cpu()
        finite = finite and bool(torch.isfinite(tc[sl]).all() and torch.isfinite(oc[sl]).all())
        del logits, g_t, g_o, xb
    return tc, oc, finite


def block9_boost(model, x, y):
    boost = []
    for i in range(0, len(x), 32):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            found = U.logits_at_blocks(model, x[i : i + 32].cuda(), (9,))
        final = found["final"].float().cpu()
        mid = found[9].float().cpu()
        yy = y[i : i + 32]
        boost.append((final[torch.arange(len(yy)), yy] - mid[torch.arange(len(yy)), yy]).cpu())
    return torch.cat(boost)


def corr(a, b):
    a, b = a.float(), b.float()
    if float(a.std(unbiased=False)) == 0 or float(b.std(unbiased=False)) == 0:
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def summarize_pair(tc, oc):
    gap = tc - oc
    ratio = tc / oc.clamp_min(1e-12)
    return {
        "tc": quantiles(tc),
        "oc": quantiles(oc),
        "gap": quantiles(gap),
        "ratio": quantiles(ratio),
        "frac_tc_gt_oc": float((tc > oc).float().mean()),
        "finite": bool(torch.isfinite(tc).all() and torch.isfinite(oc).all()),
    }


def save_scatter(path, series, title, xlabel, ylabel):
    from PIL import Image, ImageDraw

    W, H = 880, 640
    img = Image.new("RGB", (W, H), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 70, 840, 40, 560
    xs = np.concatenate([np.asarray(s[0], dtype=np.float64) for s in series])
    ys = np.concatenate([np.asarray(s[1], dtype=np.float64) for s in series])
    lo_x, hi_x = np.quantile(xs, [0.01, 0.99])
    lo_y, hi_y = np.quantile(ys, [0.01, 0.99])
    if hi_x <= lo_x:
        hi_x = lo_x + 1
    if hi_y <= lo_y:
        hi_y = lo_y + 1

    def px(v):
        return left + (v - lo_x) / (hi_x - lo_x) * (right - left)

    def py(v):
        return bottom - (v - lo_y) / (hi_y - lo_y) * (bottom - top)

    draw.line((left, bottom, right, bottom), fill=(30, 30, 30))
    draw.line((left, top, left, bottom), fill=(30, 30, 30))
    if lo_x < 0 < hi_x:
        draw.line((px(0), top, px(0), bottom), fill=(200, 200, 200))
    if lo_y < 0 < hi_y:
        draw.line((left, py(0), right, py(0)), fill=(200, 200, 200))
    for values_x, values_y, color, name in series:
        for a, b in zip(values_x, values_y):
            if not (lo_x <= a <= hi_x and lo_y <= b <= hi_y):
                continue
            cx, cy = px(float(a)), py(float(b))
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
    draw.text((20, 12), title, fill=(20, 20, 20))
    draw.text((left, 575), f"{xlabel}  {lo_x:.3g} .. {hi_x:.3g}", fill=(20, 20, 20))
    draw.text((left, 600), f"{ylabel}  {lo_y:.3g} .. {hi_y:.3g}", fill=(20, 20, 20))
    lx = 70
    for _, _, color, name in series:
        draw.rectangle((lx, 620, lx + 14, 634), fill=color)
        draw.text((lx + 18, 618), name, fill=(20, 20, 20))
        lx += 40 + 8 * len(name)
    img.save(path)


def save_hist(path, series, title):
    from PIL import Image, ImageDraw

    W, H = 880, 560
    img = Image.new("RGB", (W, H), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 60, 840, 40, 470
    arrays = [(name, np.asarray(v, dtype=np.float64), color) for name, v, color in series]
    lo = min(float(np.quantile(a, 0.01)) for _, a, _ in arrays)
    hi = max(float(np.quantile(a, 0.99)) for _, a, _ in arrays)
    if hi <= lo:
        hi = lo + 1
    bins = 30
    hists = []
    for name, a, color in arrays:
        hist, _ = np.histogram(a, bins=bins, range=(lo, hi))
        hists.append((name, hist, color))
    peak = max(int(h.max()) for _, h, _ in hists)
    peak = max(peak, 1)
    bw = (right - left) / bins
    draw.line((left, bottom, right, bottom), fill=(30, 30, 30))
    draw.line((left, top, left, bottom), fill=(30, 30, 30))
    for name, hist, color in hists:
        pts = [(left + (i + 0.5) * bw, bottom - (c / peak) * (bottom - top)) for i, c in enumerate(hist)]
        draw.line(pts, fill=color, width=2)
    draw.text((20, 10), title, fill=(20, 20, 20))
    draw.text((left, 490), f"{lo:.3g}", fill=(20, 20, 20))
    draw.text((right - 60, 490), f"{hi:.3g}", fill=(20, 20, 20))
    lx = 60
    for name, _, color in arrays:
        draw.rectangle((lx, 520, lx + 14, 534), fill=color)
        draw.text((lx + 18, 518), name, fill=(20, 20, 20))
        lx += 36 + 8 * len(name)
    img.save(path)


def per_class_gap(y, gap):
    rows = []
    for c in sorted(y.unique().tolist()):
        m = y == c
        if int(m.sum()) == 0:
            continue
        rows.append({"class": int(c), "n": int(m.sum()), "gap_mean": float(gap[m].mean())})
    rows.sort(key=lambda r: r["gap_mean"])
    return rows


def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    ART.mkdir(parents=True, exist_ok=True)
    hashes = {
        "submission": sha256_file(ROOT / "submission.pt"),
        "fit_b8": sha256_file(ROOT / "runs" / "fit_b8.pt"),
    }
    if hashes["submission"] != hashes["fit_b8"]:
        raise SystemExit("submission.pt and fit_b8.pt differ; refusing to run")
    self_check()
    print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)

    images = tt.load_forget_images(U.STARTER / "forget_images.npz")
    x_all = U.to_norm(images)
    cached = np.load(ROOT / "artifacts" / "clean_preds.npz")
    y_all = torch.from_numpy(cached["pred"].astype("int64"))
    # Spot-check the cached original argmax against a live forward.
    probe = tt.load_model(U.STARTER / "model.pt").eval()
    live = forward_logits(probe, x_all[:64]).argmax(1)
    agree = float((live == y_all[:64]).float().mean())
    print(f"cached original argmax agrees with a live forward on {agree:.3f} of 64 images", flush=True)
    if agree < 0.99:
        raise SystemExit("cached predictions do not match model.pt; not using them as pseudo-labels")
    del probe
    torch.cuda.empty_cache()

    subsets = [stratified_indices(y_all, N, seed) for seed in (0, 1)]
    overlap = len(set(subsets[0].tolist()) & set(subsets[1].tolist()))
    print(f"subset overlap {overlap}/{N}  classes {int(y_all[subsets[0]].unique().numel())}", flush=True)

    records = {}
    amplification = {}
    probs = {}
    for name, path in (("orig", U.STARTER / "model.pt"), ("b8", ROOT / "runs" / "fit_b8.pt")):
        model = tt.load_model(path).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        for s, idx in enumerate(subsets):
            x = x_all[idx]
            y = y_all[idx]
            if name == "orig":
                amplification[s] = block9_boost(model, x, y)
            logits = forward_logits(model, x)
            p_t = torch.softmax(logits, dim=-1)
            p_t2 = torch.softmax(logits / 2, dim=-1)
            probs[(name, s)] = {
                "py": p_t[torch.arange(N), y],
                "py_t2": p_t2[torch.arange(N), y],
                "maxp": p_t.max(1).values,
                "entropy": -(p_t.clamp_min(1e-8).log() * p_t).sum(-1),
                "pred": logits.argmax(1),
                "y": y,
            }
            for seed in SEEDS:
                other = other_classes(y, seed=10_000 + s * 10 + seed)
                tc, oc, finite = sensitivity(model, x, y, other)
                records[(name, s, seed)] = {"tc": tc, "oc": oc, "finite": finite, "other": other}
                print(
                    f"{name} subset {s} seed {seed}  TC {tc.mean():.4g}  OC {oc.mean():.4g}  "
                    f"gap {(tc - oc).mean():.4g}  TC>OC {(tc > oc).float().mean():.3f}  finite {finite}",
                    flush=True,
                )
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"{name} peak allocated {peak:.2f} GiB", flush=True)
        del model
        torch.cuda.empty_cache()

    summary = {"hashes": hashes, "coordinate": "normalized model input [-1, 1]", "n": N, "seeds": list(SEEDS)}
    rows = []
    for s in (0, 1):
        for name in ("orig", "b8"):
            tcs = torch.stack([records[(name, s, seed)]["tc"] for seed in SEEDS])
            ocs = torch.stack([records[(name, s, seed)]["oc"] for seed in SEEDS])
            tc, oc = tcs.mean(0), ocs.mean(0)
            seed_gap_std = (tcs - ocs).std(0, unbiased=False)
            block = summarize_pair(tc, oc)
            block["seed_gap_std_mean"] = float(seed_gap_std.mean())
            block["seed_sign_agreement"] = float(((tcs > ocs).float().mean(0) >= 0.75).float().mean())
            pr = probs[(name, s)]
            block["py_t1"] = quantiles(pr["py"])
            block["entropy"] = quantiles(pr["entropy"])
            block["agree_y"] = float((pr["pred"] == pr["y"]).float().mean())
            summary[f"{name}_subset{s}"] = block
            rows.append({"subset": s, "model": name, **{f"gap_{k}": v for k, v in block["gap"].items()},
                         "frac_tc_gt_oc": block["frac_tc_gt_oc"], "seed_gap_std_mean": block["seed_gap_std_mean"]})
            print(
                f"AVG {name} subset {s}  gap {block['gap']['mean']:.4g}  "
                f"median {block['gap']['p50']:.4g}  TC>OC {block['frac_tc_gt_oc']:.3f}  "
                f"seed std {block['seed_gap_std_mean']:.4g}",
                flush=True,
            )

    comparisons = {}
    for s in (0, 1):
        tc_o = torch.stack([records[("orig", s, seed)]["tc"] for seed in SEEDS]).mean(0)
        oc_o = torch.stack([records[("orig", s, seed)]["oc"] for seed in SEEDS]).mean(0)
        tc_b = torch.stack([records[("b8", s, seed)]["tc"] for seed in SEEDS]).mean(0)
        oc_b = torch.stack([records[("b8", s, seed)]["oc"] for seed in SEEDS]).mean(0)
        gap_o, gap_b = tc_o - oc_o, tc_b - oc_b
        y = probs[("orig", s)]["y"]
        py_b = probs[("b8", s)]["py_t2"]
        disagree = probs[("b8", s)]["pred"] != y
        comp = {
            "frac_b8_reduces_tc": float((tc_b < tc_o).float().mean()),
            "frac_b8_reduces_oc": float((oc_b < oc_o).float().mean()),
            "frac_b8_increases_oc": float((oc_b > oc_o).float().mean()),
            "frac_b8_reduces_gap": float((gap_b < gap_o).float().mean()),
            "mean_tc_ratio_b8_over_orig": float((tc_b / tc_o.clamp_min(1e-12)).mean()),
            "mean_oc_ratio_b8_over_orig": float((oc_b / oc_o.clamp_min(1e-12)).mean()),
            "mean_gap_orig": float(gap_o.mean()),
            "mean_gap_b8": float(gap_b.mean()),
            "corr_gap_orig_vs_b8_py_t2": corr(gap_o, py_b),
            "corr_gap_orig_vs_b8_entropy": corr(gap_o, probs[("b8", s)]["entropy"]),
            "corr_gap_b8_vs_disagree": corr(gap_b, disagree.float()),
            "corr_gap_change_vs_b8_py": corr(gap_o - gap_b, py_b),
            "per_class_orig_gap_low": per_class_gap(y, gap_o)[:5],
            "per_class_orig_gap_high": per_class_gap(y, gap_o)[-5:],
        }
        if s in amplification:
            comp["corr_gap_orig_vs_block9_boost"] = corr(gap_o, amplification[s])
            comp["block9_boost"] = quantiles(amplification[s])
        comparisons[f"subset{s}"] = comp
        print(
            f"COMPARE subset {s}  B8 reduces TC on {comp['frac_b8_reduces_tc']:.3f}  "
            f"reduces gap on {comp['frac_b8_reduces_gap']:.3f}  "
            f"TC ratio {comp['mean_tc_ratio_b8_over_orig']:.3f}  "
            f"OC ratio {comp['mean_oc_ratio_b8_over_orig']:.3f}  "
            f"corr(gap, block9 boost) {comp.get('corr_gap_orig_vs_block9_boost', float('nan')):.3f}  "
            f"corr(gap, b8 py) {comp['corr_gap_orig_vs_b8_py_t2']:.3f}",
            flush=True,
        )
        if s == 0:
            save_hist(
                ART / "gap_hist.png",
                [("orig", gap_o, (30, 90, 160)), ("b8", gap_b, (190, 90, 20))],
                "TC - OC on subset 0, averaged over 4 other-class draws",
            )
            save_scatter(
                ART / "tc_vs_oc.png",
                [(tc_o, oc_o, (30, 90, 160), "orig"), (tc_b, oc_b, (190, 90, 20), "b8")],
                "Target vs other-class sensitivity, subset 0",
                "TC",
                "OC",
            )
            save_scatter(
                ART / "gap_orig_vs_b8.png",
                [(gap_o, gap_b, (20, 20, 20), "images")],
                "Sensitivity gap, original versus B8",
                "orig TC-OC",
                "b8 TC-OC",
            )

    summary["comparisons"] = comparisons
    # Keep the per-image tensors for the later training comparison.
    blob = {
        "subsets": subsets,
        "y": [y_all[idx] for idx in subsets],
        "records_mean": {},
        "amplification": amplification,
        "probs": {f"{n}_{s}": probs[(n, s)] for n in ("orig", "b8") for s in (0, 1)},
    }
    for s in (0, 1):
        for name in ("orig", "b8"):
            tc = torch.stack([records[(name, s, seed)]["tc"] for seed in SEEDS])
            oc = torch.stack([records[(name, s, seed)]["oc"] for seed in SEEDS])
            blob["records_mean"][f"{name}_{s}"] = {"tc": tc.mean(0), "oc": oc.mean(0), "tc_seeds": tc, "oc_seeds": oc}
    torch.save(blob, ART / "sensitivity.pt")
    (ART / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    with (ART / "sensitivity_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    end = {
        "submission": sha256_file(ROOT / "submission.pt"),
        "fit_b8": sha256_file(ROOT / "runs" / "fit_b8.pt"),
    }
    if end != hashes:
        raise SystemExit("protected checkpoints changed during the diagnostic")
    print("diagnostic complete; checkpoints unchanged", flush=True)


if __name__ == "__main__":
    main()
