"""Compare the two MU-Mis pilots with the original model and fit_b8.

Reads the checkpoints the pilot already wrote. Does not train and does not
touch submission.pt or fit_b8.pt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402
import unlearn_run as U  # noqa: E402
from mumis_diag import other_classes, save_hist, sensitivity, sha256_file  # noqa: E402
from mumis_pilot import group_of, predict  # noqa: E402

ART = ROOT / "artifacts" / "mumis"
CKPT = Path("/p/scratch/training2666/li103/mumis")


def js_rows(p, q):
    m = 0.5 * (p + q)
    def kl(src, dst):
        return (src * (src.clamp_min(1e-8).log() - dst.clamp_min(1e-8).log())).sum(-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def quant(v):
    v = v.float()
    qs = torch.quantile(v, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    return {
        "mean": float(v.mean()),
        "std": float(v.std(unbiased=False)),
        "p10": float(qs[0]), "p25": float(qs[1]), "p50": float(qs[2]),
        "p75": float(qs[3]), "p90": float(qs[4]),
    }


def layer_profile(student, reference):
    buckets = {}
    for key, value in student.items():
        group = group_of(key)
        delta = (value.float() - reference[key].float()).pow(2).sum()
        base = reference[key].float().pow(2).sum()
        changed = int((value - reference[key]).abs().sum() > 0)
        slot = buckets.setdefault(group, [0.0, 0.0, 0, 0])
        slot[0] += float(delta)
        slot[1] += float(base)
        slot[2] += value.numel()
        slot[3] += changed
    rows = []
    for name in sorted(buckets, key=lambda n: (n == "other", n)):
        delta, base, numel, n_changed = buckets[name]
        rows.append({
            "group": name,
            "rel_rms": (delta / max(base, 1e-12)) ** 0.5,
            "rms": (delta / max(numel, 1)) ** 0.5,
            "numel": numel,
            "tensors_changed": n_changed,
        })
    return rows


def save_bars(path, rows_a, rows_b, title):
    W, H = 880, 560
    img = Image.new("RGB", (W, H), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    names = [row["group"] for row in rows_a]
    lookup_b = {row["group"]: row["rel_rms"] for row in rows_b}
    vals = [row["rel_rms"] for row in rows_a] + [lookup_b.get(n, 0.0) for n in names]
    peak = max(vals) if vals else 1.0
    peak = max(peak, 1e-8)
    left, right, top, bottom = 70, 840, 36, 470
    draw.line((left, bottom, right, bottom), fill=(30, 30, 30))
    slot = (right - left) / max(len(names), 1)
    for i, name in enumerate(names):
        x0 = left + i * slot + 4
        h1 = rows_a[i]["rel_rms"] / peak * (bottom - top)
        h2 = lookup_b.get(name, 0.0) / peak * (bottom - top)
        mid = x0 + slot * 0.45
        draw.rectangle((x0, bottom - h1, mid - 1, bottom), fill=(30, 90, 160))
        draw.rectangle((mid, bottom - h2, x0 + slot - 8, bottom), fill=(190, 90, 20))
        draw.text((x0, bottom + 6), name.replace("block", "b"), fill=(20, 20, 20))
    draw.text((20, 10), title, fill=(20, 20, 20))
    draw.rectangle((70, 530, 84, 544), fill=(30, 90, 160))
    draw.text((90, 528), "from original", fill=(20, 20, 20))
    draw.rectangle((230, 530, 244, 544), fill=(190, 90, 20))
    draw.text((250, 528), "from b8", fill=(20, 20, 20))
    img.save(path)


def decision_shift(base_logits, new_logits, y):
    base = base_logits.argmax(1)
    new = new_logits.argmax(1)
    order = torch.sort(base_logits, dim=-1, descending=True).indices
    second = order[:, 1]
    top3 = order[:, :3]
    changed = new != base
    in_top3 = (new[:, None] == top3).any(1)
    return {
        "argmax_agreement": float((~changed).float().mean()),
        "n_argmax_changes": int(changed.sum()),
        "frac_changes_to_base_second": float((changed & (new == second)).float().sum() / changed.sum().clamp_min(1)),
        "frac_changes_inside_base_top3": float((changed & in_top3).float().sum() / changed.sum().clamp_min(1)),
        "frac_changes_outside_base_top3": float((changed & ~in_top3).float().sum() / changed.sum().clamp_min(1)),
        "agree_original_class": float((new == y).float().mean()),
        "base_agree_original_class": float((base == y).float().mean()),
    }


def main():
    torch.manual_seed(0)
    hashes = {
        "submission": sha256_file(ROOT / "submission.pt"),
        "fit_b8": sha256_file(ROOT / "runs" / "fit_b8.pt"),
    }
    if hashes["submission"] != hashes["fit_b8"]:
        raise SystemExit("submission.pt and fit_b8.pt differ")
    images = tt.load_forget_images(U.STARTER / "forget_images.npz")
    x_all = U.to_norm(images)
    y_all = torch.from_numpy(np.load(ROOT / "artifacts" / "clean_preds.npz")["pred"].astype("int64"))
    blob = torch.load(ART / "sensitivity.pt", map_location="cpu", weights_only=False)
    idx = blob["subsets"][0]
    x_sub, y_sub = x_all[idx], y_all[idx]
    order = torch.randperm(len(x_sub), generator=torch.Generator().manual_seed(0))
    seen = torch.zeros(len(x_sub), dtype=torch.bool)
    seen[order[:48]] = True

    paths = {
        "orig": U.STARTER / "model.pt",
        "b8": ROOT / "runs" / "fit_b8.pt",
        "mumis_orig_pilot": CKPT / "mumis_orig_pilot.pt",
        "mumis_b8_pilot": CKPT / "mumis_b8_pilot.pt",
    }
    logits = {}
    states = {}
    sens = {}
    for name, path in paths.items():
        model = tt.load_model(path).eval()
        probe = tt.build_model()
        state = torch.load(path, map_location="cpu", weights_only=True)
        probe.load_state_dict(state, strict=True)
        del probe
        if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()):
            raise RuntimeError(f"non-finite parameters in {name}")
        logits[name] = predict(model, x_all)
        acc = []
        for seed in (0, 1):
            tc, oc, finite = sensitivity(model, x_sub, y_sub, other_classes(y_sub, seed=10_000 + seed))
            if not finite:
                raise RuntimeError(f"non-finite sensitivity in {name}")
            acc.append((tc, oc))
        tc = torch.stack([item[0] for item in acc]).mean(0)
        oc = torch.stack([item[1] for item in acc]).mean(0)
        sens[name] = {"tc": tc, "oc": oc}
        states[name] = state
        del model
        torch.cuda.empty_cache()
        print(f"evaluated {name}", flush=True)

    def probs(name):
        return torch.softmax(logits[name] / 2, dim=-1)

    p = {name: probs(name) for name in paths}
    y = y_all

    def kl_rows(src, dst):
        return (src * (src.clamp_min(1e-8).log() - dst.clamp_min(1e-8).log())).sum(-1)

    report = {"hashes": hashes, "paired_seeds": [0, 1], "seen_in_subset": 48, "subset": 128}
    pairs = {
        "orig_to_mumis": ("orig", "mumis_orig_pilot"),
        "b8_to_mumis": ("b8", "mumis_b8_pilot"),
        "b8_vs_orig": ("orig", "b8"),
    }
    report["decisions"] = {}
    report["divergence"] = {}
    for key, (a, b) in pairs.items():
        shift = decision_shift(logits[a], logits[b], y)
        kl_ab = kl_rows(p[a], p[b])
        js = js_rows(p[a], p[b])
        py_a = p[a][torch.arange(len(y)), y]
        py_b = p[b][torch.arange(len(y)), y]
        shift["delta_py_mean"] = float((py_b - py_a).mean())
        shift["frac_py_down"] = float((py_b < py_a - 1e-4).float().mean())
        report["decisions"][key] = shift
        report["divergence"][key] = {
            "kl_a_to_b": quant(kl_ab),
            "js": quant(js),
        }
        print(
            f"{key}  argmax changes {shift['n_argmax_changes']}  "
            f"to 2nd {shift['frac_changes_to_base_second']:.3f}  "
            f"outside top3 {shift['frac_changes_outside_base_top3']:.3f}  "
            f"KL {shift and report['divergence'][key]['kl_a_to_b']['mean']:.4f}",
            flush=True,
        )

    report["parameters"] = {
        "orig_to_mumis": layer_profile(states["mumis_orig_pilot"], states["orig"]),
        "b8_to_mumis": layer_profile(states["mumis_b8_pilot"], states["b8"]),
    }
    report["sensitivity_paired"] = {}
    for name in paths:
        tc, oc = sens[name]["tc"], sens[name]["oc"]
        report["sensitivity_paired"][name] = {
            "tc": float(tc.mean()),
            "oc": float(oc.mean()),
            "gap": float((tc - oc).mean()),
            "frac_tc_gt_oc": float((tc > oc).float().mean()),
        }
    for label, before, after in (
        ("orig_to_mumis", "orig", "mumis_orig_pilot"),
        ("b8_to_mumis", "b8", "mumis_b8_pilot"),
    ):
        tc0, oc0 = sens[before]["tc"], sens[before]["oc"]
        tc1, oc1 = sens[after]["tc"], sens[after]["oc"]
        rows = {}
        for part, mask in (("seen", seen), ("unseen", ~seen), ("all", torch.ones_like(seen))):
            rows[part] = {
                "n": int(mask.sum()),
                "tc_before": float(tc0[mask].mean()),
                "tc_after": float(tc1[mask].mean()),
                "oc_before": float(oc0[mask].mean()),
                "oc_after": float(oc1[mask].mean()),
                "gap_before": float((tc0 - oc0)[mask].mean()),
                "gap_after": float((tc1 - oc1)[mask].mean()),
                "frac_tc_down": float((tc1[mask] < tc0[mask]).float().mean()),
                "frac_oc_up": float((oc1[mask] > oc0[mask]).float().mean()),
                "frac_gap_down": float(((tc1 - oc1)[mask] < (tc0 - oc0)[mask]).float().mean()),
            }
        report["sensitivity_paired"][label] = rows
        print(
            f"{label} gap seen {rows['seen']['gap_before']:.2f}->{rows['seen']['gap_after']:.2f}  "
            f"unseen {rows['unseen']['gap_before']:.2f}->{rows['unseen']['gap_after']:.2f}",
            flush=True,
        )

    py = {name: p[name][torch.arange(len(y)), y] for name in paths}
    ent = {}
    for name in paths:
        prob = p[name]
        ent[name] = -(prob.clamp_min(1e-8).log() * prob).sum(-1)
    save_hist(
        ART / "py_compare.png",
        [
            ("orig", py["orig"], (60, 60, 60)),
            ("b8", py["b8"], (190, 90, 20)),
            ("MU-Mis from orig", py["mumis_orig_pilot"], (30, 90, 160)),
            ("MU-Mis from b8", py["mumis_b8_pilot"], (20, 130, 80)),
        ],
        "Original-class probability at temperature 2",
    )
    save_hist(
        ART / "kl_compare.png",
        [
            ("KL orig to MU-Mis-orig", kl_rows(p["orig"], p["mumis_orig_pilot"]), (30, 90, 160)),
            ("KL b8 to MU-Mis-b8", kl_rows(p["b8"], p["mumis_b8_pilot"]), (20, 130, 80)),
            ("KL orig to b8", kl_rows(p["orig"], p["b8"]), (190, 90, 20)),
        ],
        "Per-image KL at temperature 2",
    )
    save_bars(
        ART / "param_profile.png",
        report["parameters"]["orig_to_mumis"],
        report["parameters"]["b8_to_mumis"],
        "Relative RMS of the parameter change, by block",
    )
    (ART / "compare.json").write_text(json.dumps(report, indent=2))
    end = {
        "submission": sha256_file(ROOT / "submission.pt"),
        "fit_b8": sha256_file(ROOT / "runs" / "fit_b8.pt"),
    }
    if end != hashes:
        raise SystemExit("protected checkpoints changed during evaluation")
    print("compare written", flush=True)


if __name__ == "__main__":
    main()
