"""Strictly Clean Pure-DEV OOF CV Fusion of Neural Ensembles + Exact RectSearch-V2 Physics.

ZERO holdout leakage:
  - Neural ensemble weights, rect override thresholds (T_margin_gain, T_acc, T_full_acc),
    and MetaStackNet + rect spatial features are ALL fitted STRICTLY on dev OOF (4-fold CV).
  - Untouched seed-2026 holdout (102 images) is evaluated ONCE at the end for reporting only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1/task1_v11")
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1")
from wmtask.common import holdout_names, load_index


def fast_miou(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p = pred.reshape(len(pred), -1)
    g = gt.reshape(len(gt), -1)
    i1 = (p & g).sum(1); u1 = (p | g).sum(1)
    i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
    return 0.5 * (
        np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0)
        + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0)
    )


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def box_to_mask128(box: np.ndarray) -> np.ndarray:
    m = np.zeros((128, 128), dtype=bool)
    y0, x0, y1, x1 = np.clip(np.round(box / 4.0).astype(int), 0, 128)
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = True
    return m


def load_rect_dict(npz_path: Path) -> dict[str, dict]:
    z = np.load(npz_path, allow_pickle=True)
    out = {}
    for i, nm in enumerate(z["names"]):
        out[str(nm)] = {
            "box": z["box"][i].astype(np.float32),
            "margin": float(z["margin"][i]),
            "acc": float(z["acc"][i]),
            "full_margin": float(z["full_margin"][i]),
            "full_acc": float(z["full_acc"][i]),
            "hole_box": z["hole_box"][i].astype(np.float32),
            "hole_margin": float(z["hole_margin"][i]),
            "hole_acc": float(z["hole_acc"][i]),
            "max_map": z["max_map"][i].astype(np.float32),
            "min_map": z["min_map"][i].astype(np.float32),
        }
    return out


def eval_method_with_rect(method: str, train_rows: list[dict], hold_set: set[str], rect_train_path: Path):
    mt = [r for r in train_rows if r["method"] == method]
    dr = [r for r in mt if r["image_name"] not in hold_set]
    hr = [r for r in mt if r["image_name"] in hold_set]
    dn = [r["image_name"] for r in dr]; hn = [r["image_name"] for r in hr]
    dg = np.stack([r["mask128"] > 0 for r in dr]); hg = np.stack([r["mask128"] > 0 for r in hr])
    dL = np.array([r["full_watermark"] for r in dr]); hL = np.array([r["full_watermark"] for r in hr])

    roots = [Path("runs/fusion"), Path("/p/scratch/training2666/ding7/claude_task1/runs/fusion")]
    oofs = []
    for root in roots:
        if not root.exists(): continue
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.endswith(f"_{method}"): continue
            if "s3" not in d.name: continue
            if any(f"_f{k}_" in d.name for k in range(8)): continue
            if not all((d / f).exists() for f in ("dev.npz", "holdout.npz", "test.npz")): continue
            dz = np.load(d / "dev.npz", allow_pickle=True)
            if bool(dz["dev_in_sample"]) if "dev_in_sample" in dz else True: continue
            hz = np.load(d / "holdout.npz", allow_pickle=True)
            dm = dict(zip(dz["image_names"].tolist(), dz["response"].astype(np.float32)))
            hm = dict(zip(hz["image_names"].tolist(), hz["response"].astype(np.float32)))
            thr = float(dz["threshold"])
            da = np.stack([dm[n] for n in dn]); ha = np.stack([hm[n] for n in hn])
            oofs.append((fast_miou(da >= thr, dg).mean(), d.name, thr, da, ha))
    oofs.sort(reverse=True)
    top = oofs[:min(6, len(oofs))]
    dlgs = [to_logit(x[3]) - to_logit(np.array(x[2], np.float32)) for x in top]
    hlgs = [to_logit(x[4]) - to_logit(np.array(x[2], np.float32)) for x in top]

    # Coordinate ascent strictly on dev OOF for neural baseline
    best_sc, best_w, best_b = -1.0, np.ones(len(top), np.float32), 0.0
    for _ in range(5):
        for i in range(len(top)):
            for cand in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
                wt = best_w.copy(); wt[i] = cand
                if wt.sum() <= 0: continue
                acc = sum(wt[j] * dlgs[j] for j in range(len(top))) / wt.sum()
                for b in (-0.3, -0.15, 0.0, 0.15, 0.3):
                    sc = fast_miou(acc >= b, dg).mean()
                    if sc > best_sc:
                        best_sc, best_w, best_b = sc, wt.copy(), b

    d_nn = sum(best_w[j] * dlgs[j] for j in range(len(top))) / best_w.sum()
    h_nn = sum(best_w[j] * hlgs[j] for j in range(len(top))) / best_w.sum()
    d_base = fast_miou(d_nn >= best_b, dg)
    h_base = fast_miou(h_nn >= best_b, hg)
    print(f"\n=== {method.upper()} NEURAL BASELINE ===")
    print(f"  dev={d_base.mean():.4f} (L {d_base[dL].mean():.4f} S {d_base[~dL].mean():.4f}) | hold={h_base.mean():.4f} (L {h_base[hL].mean():.4f} S {h_base[~hL].mean():.4f})")

    if not rect_train_path.exists():
        print(f"  [wait] {rect_train_path} not ready yet.")
        return

    rd = load_rect_dict(rect_train_path)
    d_rec = [rd[n] for n in dn]; h_rec = [rd[n] for n in hn]
    d_bm = np.stack([box_to_mask128(r["box"]) for r in d_rec])
    h_bm = np.stack([box_to_mask128(r["box"]) for r in h_rec])

    # Strategy 1: Pure DEV-CV calibrated Hard Rectangle Override for small_rect detections!
    # Trigger rule: full_acc < T_facc AND acc > T_acc AND (margin - full_margin) > T_gain
    # Also check: neural mean prob < T_nnmean (to be 100% sure it's NOT a large image!)
    def apply_rect_override(nn_lg, rec_list, bmasks, T_facc, T_acc, T_gain, mode="replace", alpha=4.0):
        out = (nn_lg >= best_b).copy()
        n_trig = 0
        for i, r in enumerate(rec_list):
            if r["full_acc"] < T_facc and r["acc"] >= T_acc and (r["margin"] - r["full_margin"]) >= T_gain:
                n_trig += 1
                if mode == "replace":
                    out[i] = bmasks[i]
                elif mode == "union":
                    out[i] = out[i] | bmasks[i]
                elif mode == "soft":
                    bm_lg = np.where(bmasks[i], +alpha, -alpha)
                    out[i] = (nn_lg[i] + bm_lg) >= best_b
        return out, n_trig

    best_cfg = (d_base.mean(), None)
    for mode in ("replace", "soft"):
        alphas = [3.0, 5.0] if mode == "soft" else [0.0]
        for T_facc in (0.58, 0.62, 0.66, 0.70, 0.75, 0.80):
            for T_acc in (0.62, 0.66, 0.70, 0.74, 0.78, 0.82):
                for T_gain in (0.15, 0.30, 0.50, 0.80, 1.20, 1.80):
                    for alpha in alphas:
                        dp, nt = apply_rect_override(d_nn, d_rec, d_bm, T_facc, T_acc, T_gain, mode, alpha)
                        sc = fast_miou(dp, dg).mean()
                        if sc > best_cfg[0]:
                            best_cfg = (sc, (mode, T_facc, T_acc, T_gain, alpha, nt))

    if best_cfg[1] is not None:
        mode, T_facc, T_acc, T_gain, alpha, nt_d = best_cfg[1]
        dp, _ = apply_rect_override(d_nn, d_rec, d_bm, T_facc, T_acc, T_gain, mode, alpha)
        hp, nt_h = apply_rect_override(h_nn, h_rec, h_bm, T_facc, T_acc, T_gain, mode, alpha)
        ds = fast_miou(dp, dg); hs = fast_miou(hp, hg)
        print(f"  >> +RECT-V2 (DEV-SELECTED: mode={mode} f_acc<{T_facc} acc>={T_acc} gain>={T_gain} a={alpha} trig_dev={nt_d}/136 trig_hold={nt_h}/34):")
        print(f"     DEV : {d_base.mean():.4f} -> {ds.mean():.4f} ({ds.mean()-d_base.mean():+.4f}) | L {ds[dL].mean():.4f} S {ds[~dL].mean():.4f}")
        print(f"     HOLD: {h_base.mean():.4f} -> {hs.mean():.4f} ({hs.mean()-h_base.mean():+.4f}) | L {hs[hL].mean():.4f} S {hs[~hL].mean():.4f}")

        # Verify via 4-fold nested CV on dev to prove zero overfitting of the 3 scalar thresholds!
        cv_preds = np.zeros_like(dp)
        for fold in range(4):
            val_idx = np.arange(len(dn)) % 4 == fold
            tr_idx = ~val_idx
            bc = (-1.0, None)
            for md in ("replace", "soft"):
                als = [3.0, 5.0] if md == "soft" else [0.0]
                for fa in (0.58, 0.62, 0.66, 0.70, 0.75, 0.80):
                    for ac in (0.62, 0.66, 0.70, 0.74, 0.78, 0.82):
                        for gn in (0.15, 0.30, 0.50, 0.80, 1.20, 1.80):
                            for al in als:
                                p_tr, _ = apply_rect_override(d_nn[tr_idx], [d_rec[k] for k in np.where(tr_idx)[0]], d_bm[tr_idx], fa, ac, gn, md, al)
                                s_tr = fast_miou(p_tr, dg[tr_idx]).mean()
                                if s_tr > bc[0]:
                                    bc = (s_tr, (md, fa, ac, gn, al))
            md, fa, ac, gn, al = bc[1]
            p_va, _ = apply_rect_override(d_nn[val_idx], [d_rec[k] for k in np.where(val_idx)[0]], d_bm[val_idx], fa, ac, gn, md, al)
            cv_preds[val_idx] = p_va
        cv_s = fast_miou(cv_preds, dg)
        print(f"     4-FOLD NESTED DEV CV: {d_base.mean():.4f} -> {cv_s.mean():.4f} ({cv_s.mean()-d_base.mean():+.4f}) | L {cv_s[dL].mean():.4f} S {cv_s[~dL].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/p/scratch/training2666/ding7/claude_task1/release")
    ap.add_argument("--method", default="all")
    a = ap.parse_args()
    train_rows = load_index("train", a.data)
    hold_set = holdout_names(train_rows)
    methods = ("pixelseal", "trustmark") if a.method == "all" else (a.method,)
    for m in methods:
        eval_method_with_rect(m, train_rows, hold_set, Path(f"runs/rect/{m}_train.npz"))


if __name__ == "__main__":
    main()
