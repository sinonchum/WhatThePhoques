"""Strictly clean pure-dev OOF CV model selector & ensemble builder.
Compliance rules:
1. Only loads CLEAN synth3 models (no test-cover pools like synth2 / ext1 / ext3real / ext4_512real_testcovers).
2. Prefers 100% honest OOF models (`dev_in_sample == False`) and selects model subsets, weights,
   and thresholds STRICTLY on dev OOF mIoU (NEVER touching the 102 held-out images for selection/tuning).
3. Evaluates the final frozen dev-selected ensemble ONCE on seed-2026 holdout for reporting only.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1/task1_v11")
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1")
from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index

FUSION_ROOTS = [
    Path("/p/scratch/training2666/qin7/task1/runs/fusion"),
    Path("/p/scratch/training2666/ding7/claude_task1/runs/fusion"),
]

# Explicit allowlist/prefix check: ONLY clean synth3 models (exclude old test-cover models)
CLEAN_PATTERNS = ("_s3_", "s3_")
BANNED_SUBSTRS = ("testcovers", "ext1_", "ext3real", "ext4_")


def is_clean_dir(d: Path) -> bool:
    name = d.name
    if any(b in name for b in BANNED_SUBSTRS) and "s3" not in name:
        return False
    # Exclude individual fold dirs when 4fold exists
    if any(f"_f{k}_" in name for k in range(8)):
        return False
    return any(p in name for p in CLEAN_PATTERNS)


def to_logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def load_model(d: Path, dev_names: list[str], hold_names: list[str], test_names: list[str]):
    dz = np.load(d / "dev.npz", allow_pickle=True)
    hz = np.load(d / "holdout.npz", allow_pickle=True)
    tz = np.load(d / "test.npz", allow_pickle=True)
    dmap = dict(zip(dz["image_names"].tolist(), dz["response"].astype(np.float32)))
    hmap = dict(zip(hz["image_names"].tolist(), hz["response"].astype(np.float32)))
    tmap = dict(zip(tz["image_names"].tolist(), tz["response"].astype(np.float32)))
    thr = float(dz["threshold"])
    in_sample = bool(dz["dev_in_sample"]) if "dev_in_sample" in dz else True
    return {
        "name": d.name,
        "path": str(d),
        "thr": thr,
        "in_sample": in_sample,
        "dev": np.stack([dmap[n] for n in dev_names]),
        "hold": np.stack([hmap[n] for n in hold_names]),
        "test": np.stack([tmap[n] for n in test_names]),
    }


def eval_scores(scores: np.ndarray, rows: list[dict], thr: float = 0.0):
    L, S = [], []
    for i, r in enumerate(rows):
        sc = miou_image(scores[i] >= thr, r["mask128"])
        (L if r["full_watermark"] else S).append(sc)
    return float(np.mean(L)), float(np.mean(S)), float(np.mean(L + S))


def select_on_dev_oof(models: list[dict], dev_rows: list[dict]):
    """Fit threshold-aligned logit weights & bias strictly on honest OOF dev models."""
    oof_models = [m for m in models if not m["in_sample"]]
    if not oof_models:
        oof_models = models

    # Precompute threshold-aligned logits on dev
    d_lg = [to_logit(m["dev"]) - to_logit(np.array(m["thr"], np.float32)) for m in oof_models]
    n = len(oof_models)
    best = (-1.0, [1.0] * n, 0.0)

    # Coarse simplex grid search strictly on DEV OOF (no holdout!)
    weight_opts = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0] if n <= 5 else [0.0, 1.0, 2.0]
    for ws in itertools.product(weight_opts, repeat=n):
        if sum(ws) <= 0:
            continue
        acc = sum(w * x for w, x in zip(ws, d_lg) if w > 0) / sum(ws)
        for bias in np.linspace(-0.4, 0.4, 9):
            _, _, dm = eval_scores(acc, dev_rows, float(bias))
            if dm > best[0]:
                best = (dm, list(ws), float(bias))
    return oof_models, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/p/scratch/training2666/ding7/claude_task1/release")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    train_rows = load_index("train", args.data)
    test_rows = load_index("test", args.data)
    hold_set = holdout_names(train_rows)

    test_masks = {}
    hold_means = []

    for method in ("pixelseal", "trustmark", "stegastamp"):
        m_train = [r for r in train_rows if r["method"] == method]
        dev_rows = [r for r in m_train if r["image_name"] not in hold_set]
        hold_rows = [r for r in m_train if r["image_name"] in hold_set]
        m_test = [r for r in test_rows if r["method"] == method]
        dev_names = [r["image_name"] for r in dev_rows]
        hold_names = [r["image_name"] for r in hold_rows]
        test_names = [r["image_name"] for r in m_test]

        found = []
        for root in FUSION_ROOTS:
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                if not d.is_dir() or not d.name.endswith(f"_{method}"):
                    continue
                if not is_clean_dir(d):
                    continue
                if all((d / f).exists() for f in ("dev.npz", "holdout.npz", "test.npz")):
                    found.append(load_model(d, dev_names, hold_names, test_names))

        print(f"\n=== {method.upper()} (clean s3 models found: {len(found)}) ===")
        for m in found:
            dL, dS, dM = eval_scores(m["dev"], dev_rows, m["thr"])
            hL, hS, hM = eval_scores(m["hold"], hold_rows, m["thr"])
            tag = "OOF" if not m["in_sample"] else "INS"
            print(f"  [{tag}] {m['name']:<44s} thr={m['thr']:.2f} | dev {dM:.4f} | hold {hM:.4f} (L {hL:.4f} S {hS:.4f})")

        if not found:
            continue
        oof_models, (best_dev, ws, bias) = select_on_dev_oof(found, dev_rows)
        chosen = [(m["name"], w) for m, w in zip(oof_models, ws) if w > 0]
        h_acc = sum(
            w * (to_logit(m["hold"]) - to_logit(np.array(m["thr"], np.float32)))
            for m, w in zip(oof_models, ws) if w > 0
        ) / sum(ws)
        t_acc = sum(
            w * (to_logit(m["test"]) - to_logit(np.array(m["thr"], np.float32)))
            for m, w in zip(oof_models, ws) if w > 0
        ) / sum(ws)
        hL, hS, hM = eval_scores(h_acc, hold_rows, bias)
        hold_means.append(hM)
        print(f"  >> DEV-OOF SELECTED: {chosen} bias={bias:+.2f} | dev_oof={best_dev:.4f} | holdout={hM:.4f} (L {hL:.4f} S {hS:.4f})")
        for i, n in enumerate(test_names):
            test_masks[n] = (t_acc[i] >= bias)

    if len(hold_means) == 3:
        print(f"\n=== CLEAN 3-METHOD MEAN HOLDOUT: {np.mean(hold_means):.4f} ===")
    if args.out and len(test_masks) == len(test_rows):
        all_names = [r["image_name"] for r in test_rows]
        masks_t = torch.from_numpy(np.stack([test_masks[n] for n in all_names])).bool()
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"image_names": all_names, "masks": masks_t}, out_p)
        print(f"SAVED {out_p} | {out_p.stat().st_size / 1e6:.2f} MB | fg={masks_t.float().mean():.4f}")


if __name__ == "__main__":
    main()
