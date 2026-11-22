"""Build 100% clean (synth3-only, zero holdout leakage) Task 1 submission.

Pipeline per method (pixelseal, trustmark, stegastamp):
  1. Load all clean synth3 honest-OOF (`dev_in_sample == False`) models from qin7 + ding7.
  2. Fit threshold-aligned logit weights & bias strictly on DEV OOF (coordinate ascent).
  3. Train 4-fold stratified nested-CV MetaStackNet strictly on DEV OOF; keep MetaStackNet
     over linear blend ONLY if nested dev-OOF strictly improves by >= 0.001.
  4. For pixelseal & trustmark, evaluate BOTH `rect` and `rect_nnref` physics overrides
     strictly via 4-fold nested DEV CV; adopt the best physics config ONLY if nested DEV CV
     strictly improves over the neural stage.
  5. Evaluate the frozen dev-selected pipeline ONCE on untouched seed-2026 holdout (102 images).
  6. Save verified submission `.pt` (`uint8`, [1020, 128, 128], <= 20 MB, weights_only=True).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/p/scratch/training2666/qin7/task1")
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1")
from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index

FUSION_ROOTS = [
    Path("/p/scratch/training2666/qin7/task1/runs/fusion"),
    Path("/p/scratch/training2666/ding7/claude_task1/runs/fusion"),
]
CLEAN_PATTERNS = ("_s3_", "s3_")
BANNED_SUBSTRS = ("testcovers", "ext1_", "ext3real", "ext4_")


def is_clean_dir(d: Path) -> bool:
    name = d.name
    if any(b in name for b in BANNED_SUBSTRS) and "s3" not in name:
        return False
    if any(f"_f{k}_" in name for k in range(8)):
        return False
    return any(p in name for p in CLEAN_PATTERNS)


def to_logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p.astype(np.float32), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fast_miou_vec(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p = pred.reshape(len(pred), -1)
    g = gt.reshape(len(gt), -1)
    i1 = (p & g).sum(1)
    u1 = (p | g).sum(1)
    i0 = ((~p) & (~g)).sum(1)
    u0 = ((~p) | (~g)).sum(1)
    return 0.5 * (
        np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0)
        + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0)
    )


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
        }
    return out


class MetaStackNet(nn.Module):
    def __init__(self, n_models: int, init_w: np.ndarray, init_b: float):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(-init_b), dtype=torch.float32))
        ci = n_models + 4
        self.net = nn.Sequential(
            nn.Conv2d(ci, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wn = F.softmax(self.w, dim=0).view(1, -1, 1, 1)
        base = (x * wn).sum(dim=1, keepdim=True) + self.b
        prob = torch.sigmoid(base)
        std = x.std(dim=1, keepdim=True) if x.shape[1] > 1 else torch.zeros_like(base)
        area = prob.mean(dim=(2, 3), keepdim=True).expand_as(base)
        feat = torch.cat(
            [x.clamp(-6, 6) / 3.0, base.clamp(-6, 6) / 3.0, prob, std.clamp(0, 4) / 2.0, area],
            dim=1,
        )
        return (base + self.net(feat))[:, 0]


def sym_miou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    i1 = (p * t).sum(1)
    u1 = (p + t - p * t).sum(1)
    i0 = ((1 - p) * (1 - t)).sum(1)
    u0 = ((1 - p) + (1 - t) - (1 - p) * (1 - t)).sum(1)
    sym = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0)).mean()
    return 0.35 * bce + 0.65 * (1.0 - sym)


def nested_cv_stack(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    fm_dev: np.ndarray,
    X_hold: np.ndarray,
    X_test: np.ndarray,
    init_w: np.ndarray,
    init_b: float,
    device: str,
):
    N, M = X_dev.shape[0], X_dev.shape[1]
    idx_L = np.where(fm_dev)[0]
    idx_S = np.where(~fm_dev)[0]
    folds = [[] for _ in range(4)]
    for k, i in enumerate(idx_L):
        folds[k % 4].append(i)
    for k, i in enumerate(idx_S):
        folds[k % 4].append(i)

    oof_pred = np.zeros((N, 128, 128), np.float32)
    hold_preds, test_preds = [], []
    iw = np.log(np.maximum(init_w / max(float(init_w.sum()), 1e-6), 1e-2))

    X_d_t = torch.from_numpy(X_dev).to(device)
    y_d_t = torch.from_numpy(y_dev.astype(np.float32)).to(device)
    X_h_t = torch.from_numpy(X_hold).to(device)
    X_t_t = torch.from_numpy(X_test).to(device)
    gt_bool = y_dev > 0.5

    for k in range(4):
        val_idx = np.array(folds[k])
        tr_idx = np.array([i for i in range(N) if i not in set(folds[k])])
        torch.manual_seed(42 + k)
        net = MetaStackNet(M, iw, init_b).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2e-2)
        best_sc = -1.0
        best_val = None
        best_state = None
        for ep in range(55):
            net.train()
            xb = X_d_t[tr_idx]
            yb = y_d_t[tr_idx]
            if ep % 2 == 1:
                xb = torch.flip(xb, dims=[-1])
                yb = torch.flip(yb, dims=[-1])
            if ep % 4 == 2:
                xb = torch.flip(xb, dims=[-2])
                yb = torch.flip(yb, dims=[-2])
            opt.zero_grad()
            loss = sym_miou_loss(net(xb), yb)
            loss.backward()
            opt.step()
            if (ep + 1) % 5 == 0:
                net.eval()
                with torch.no_grad():
                    vp = net(X_d_t[val_idx]).cpu().numpy()
                sc = float(fast_miou_vec(vp >= 0.0, gt_bool[val_idx]).mean())
                if sc > best_sc:
                    best_sc = sc
                    best_val = vp.copy()
                    best_state = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
        oof_pred[val_idx] = best_val
        net.load_state_dict({kk: vv.to(device) for kk, vv in best_state.items()})
        net.eval()
        with torch.no_grad():
            hold_preds.append(net(X_h_t).cpu().numpy())
            test_preds.append(net(X_t_t).cpu().numpy())
    return oof_pred, np.mean(hold_preds, axis=0), np.mean(test_preds, axis=0)


def apply_rect_override(nn_lg, rec_list, bmasks, T_facc, T_acc, T_gain, mode="replace", alpha=4.0):
    out = (nn_lg >= 0.0).copy()
    n_trig = 0
    for i, r in enumerate(rec_list):
        if r["full_acc"] < T_facc and r["acc"] >= T_acc and (r["margin"] - r["full_margin"]) >= T_gain:
            n_trig += 1
            if mode == "replace":
                out[i] = bmasks[i]
            elif mode == "soft":
                bm_lg = np.where(bmasks[i], +alpha, -alpha)
                out[i] = (nn_lg[i] + bm_lg) >= 0.0
    return out, n_trig


def fit_rect_cv(d_lg, dg, dL, d_rec, d_bm):
    """Fit rect override strictly via 4-fold nested DEV CV."""
    base_sc = float(fast_miou_vec(d_lg >= 0.0, dg).mean())
    grid = []
    for md in ("replace", "soft"):
        als = (3.0, 5.0) if md == "soft" else (0.0,)
        for fa in (0.58, 0.62, 0.66, 0.70, 0.75, 0.80):
            for ac in (0.62, 0.66, 0.70, 0.74, 0.78, 0.82):
                for gn in (0.15, 0.30, 0.50, 0.80, 1.20, 1.80):
                    for al in als:
                        grid.append((md, fa, ac, gn, al))

    # Precompute per-image mIoU for every candidate on dev -> ultra-fast vectorized CV!
    # Wait: even simpler, only images with (full_acc < 0.80 and acc >= 0.62 and gain >= 0.15) can ever trigger!
    base_vec = fast_miou_vec(d_lg >= 0.0, dg)
    rep_vec = fast_miou_vec(d_bm, dg)
    soft_vecs = {
        al: fast_miou_vec((d_lg + np.where(d_bm, +al, -al)) >= 0.0, dg)
        for al in (3.0, 5.0)
    }
    f_arr = np.array([r["full_acc"] for r in d_rec], np.float32)
    a_arr = np.array([r["acc"] for r in d_rec], np.float32)
    g_arr = np.array([r["margin"] - r["full_margin"] for r in d_rec], np.float32)

    def eval_cfg_on_mask(idx_mask, md, fa, ac, gn, al):
        trig = (f_arr < fa) & (a_arr >= ac) & (g_arr >= gn)
        alt = rep_vec if md == "replace" else soft_vecs[al]
        sc = np.where(trig, alt, base_vec)
        return float(sc[idx_mask].mean())

    # 4-fold nested CV
    cv_vec = np.zeros_like(base_vec)
    all_idx = np.ones(len(d_rec), dtype=bool)
    for fold in range(4):
        val_m = (np.arange(len(d_rec)) % 4) == fold
        tr_m = ~val_m
        best_tr = (float(base_vec[tr_m].mean()), None)
        for cfg in grid:
            s = eval_cfg_on_mask(tr_m, *cfg)
            if s > best_tr[0]:
                best_tr = (s, cfg)
        if best_tr[1] is None:
            cv_vec[val_m] = base_vec[val_m]
        else:
            md, fa, ac, gn, al = best_tr[1]
            trig = (f_arr < fa) & (a_arr >= ac) & (g_arr >= gn)
            alt = rep_vec if md == "replace" else soft_vecs[al]
            cv_vec[val_m] = np.where(trig, alt, base_vec)[val_m]

    nested_cv_sc = float(cv_vec.mean())
    # Fit final config on full dev
    best_full = (base_sc, None)
    for cfg in grid:
        s = eval_cfg_on_mask(all_idx, *cfg)
        if s > best_full[0]:
            best_full = (s, cfg)
    return nested_cv_sc, best_full[0], best_full[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/p/scratch/training2666/ding7/claude_task1/release")
    ap.add_argument("--out", default="runs/submissions/sub_v3_clean_nnref.pt")
    ap.add_argument("--skip-stack", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | data: {a.data}", flush=True)
    train_rows = load_index("train", a.data)
    test_rows = load_index("test", a.data)
    hold_set = holdout_names(train_rows)

    test_masks = {}
    summary = {}

    for method in ("pixelseal", "trustmark", "stegastamp"):
        mt = [r for r in train_rows if r["method"] == method]
        dr = [r for r in mt if r["image_name"] not in hold_set]
        hr = [r for r in mt if r["image_name"] in hold_set]
        tr = [r for r in test_rows if r["method"] == method]
        dn = [r["image_name"] for r in dr]
        hn = [r["image_name"] for r in hr]
        tn = [r["image_name"] for r in tr]
        dg = np.stack([r["mask128"] > 0 for r in dr])
        hg = np.stack([r["mask128"] > 0 for r in hr])
        dL = np.array([r["full_watermark"] for r in dr], bool)
        hL = np.array([r["full_watermark"] for r in hr], bool)

        found = []
        for root in FUSION_ROOTS:
            if not root.exists():
                continue
            for d in sorted(root.iterdir()):
                if not d.is_dir() or not d.name.endswith(f"_{method}") or not is_clean_dir(d):
                    continue
                if not all((d / f).exists() for f in ("dev.npz", "holdout.npz", "test.npz")):
                    continue
                dz = np.load(d / "dev.npz", allow_pickle=True)
                in_s = bool(dz["dev_in_sample"]) if "dev_in_sample" in dz else True
                if in_s:
                    continue
                hz = np.load(d / "holdout.npz", allow_pickle=True)
                tz = np.load(d / "test.npz", allow_pickle=True)
                dmap = dict(zip(dz["image_names"].tolist(), dz["response"].astype(np.float32)))
                hmap = dict(zip(hz["image_names"].tolist(), hz["response"].astype(np.float32)))
                tmap = dict(zip(tz["image_names"].tolist(), tz["response"].astype(np.float32)))
                if not (all(n in dmap for n in dn) and all(n in hmap for n in hn) and all(n in tmap for n in tn)):
                    continue
                thr = float(dz["threshold"])
                da = np.stack([dmap[n] for n in dn])
                ha = np.stack([hmap[n] for n in hn])
                ta = np.stack([tmap[n] for n in tn])
                d_sc = float(fast_miou_vec(da >= thr, dg).mean())
                h_sc = float(fast_miou_vec(ha >= thr, hg).mean())
                found.append((d_sc, h_sc, d.name, thr, da, ha, ta))

        found.sort(reverse=True, key=lambda x: x[0])
        print(f"\n================ {method.upper()} ({len(found)} clean OOF models) ================", flush=True)
        for d_sc, h_sc, nm, thr, *_ in found[:10]:
            print(f"  [OOF] {nm:<44s} thr={thr:.2f} | dev={d_sc:.4f} | hold={h_sc:.4f}", flush=True)

        top = found[:min(6, len(found))]
        Xd = np.stack([to_logit(x[4]) - to_logit(np.array(x[3], np.float32)) for x in top], axis=1)
        Xh = np.stack([to_logit(x[5]) - to_logit(np.array(x[3], np.float32)) for x in top], axis=1)
        Xt = np.stack([to_logit(x[6]) - to_logit(np.array(x[3], np.float32)) for x in top], axis=1)

        # Coordinate ascent strictly on DEV OOF (ultra-fast vectorized on device!)
        Xd_flat = torch.from_numpy(Xd.reshape(len(dg), len(top), -1)).to(device)  # [N, M, P]
        dg_t = torch.from_numpy(dg.reshape(len(dg), -1)).to(device)
        bs_t = torch.tensor([-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30], device=device)
        best_sc, best_w, best_b = -1.0, np.ones(len(top), np.float32), 0.0
        for _ in range(3):
            for i in range(len(top)):
                for cand in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
                    wt = best_w.copy()
                    wt[i] = cand
                    if wt.sum() <= 0:
                        continue
                    wt_t = torch.from_numpy(wt / wt.sum()).to(device).view(1, -1, 1)
                    acc = (Xd_flat * wt_t).sum(dim=1)  # [N, P]
                    for bi in range(len(bs_t)):
                        p = acc >= bs_t[bi]
                        i1 = (p & dg_t).sum(1).float()
                        u1 = (p | dg_t).sum(1).float()
                        i0 = ((~p) & (~dg_t)).sum(1).float()
                        u0 = ((~p) | (~dg_t)).sum(1).float()
                        sc = float((0.5 * (torch.where(u1 > 0, i1 / u1.clamp(min=1), 1.0) + torch.where(u0 > 0, i0 / u0.clamp(min=1), 1.0))).mean().item())
                        if sc > best_sc:
                            best_sc, best_w, best_b = sc, wt.copy(), float(bs_t[bi].item())

        d_lin = np.tensordot(best_w / best_w.sum(), Xd, axes=([0], [1])) - best_b
        h_lin = np.tensordot(best_w / best_w.sum(), Xh, axes=([0], [1])) - best_b
        t_lin = np.tensordot(best_w / best_w.sum(), Xt, axes=([0], [1])) - best_b
        dl_vec = fast_miou_vec(d_lin >= 0.0, dg)
        hl_vec = fast_miou_vec(h_lin >= 0.0, hg)
        chosen = [(top[i][2], float(best_w[i])) for i in range(len(top)) if best_w[i] > 0]
        print(
            f"  [LINEAR] weights={chosen} bias={best_b:+.2f}\n"
            f"           DEV ={dl_vec.mean():.4f} (L {dl_vec[dL].mean():.4f} S {dl_vec[~dL].mean():.4f})\n"
            f"           HOLD={hl_vec.mean():.4f} (L {hl_vec[hL].mean():.4f} S {hl_vec[~hL].mean():.4f})",
            flush=True,
        )

        d_nn, h_nn, t_nn = d_lin, h_lin, t_lin
        stage_name = "LINEAR"

        if not a.skip_stack:
            oof_m, hold_m, test_m = nested_cv_stack(
                Xd, dg.astype(np.float32), dL, Xh, Xt, best_w, best_b, device
            )
            dm_vec = fast_miou_vec(oof_m >= 0.0, dg)
            hm_vec = fast_miou_vec(hold_m >= 0.0, hg)
            print(
                f"  [METASTACK 4f-nested]\n"
                f"           DEV ={dm_vec.mean():.4f} ({dm_vec.mean()-dl_vec.mean():+.4f}) (L {dm_vec[dL].mean():.4f} S {dm_vec[~dL].mean():.4f})\n"
                f"           HOLD={hm_vec.mean():.4f} ({hm_vec.mean()-hl_vec.mean():+.4f}) (L {hm_vec[hL].mean():.4f} S {hm_vec[~hL].mean():.4f})",
                flush=True,
            )
            if dm_vec.mean() >= dl_vec.mean() + 0.001:
                d_nn, h_nn, t_nn = oof_m, hold_m, test_m
                stage_name = "METASTACK"

        # Physics rect override (for pixelseal & trustmark), strictly selected by 4-fold nested DEV CV!
        d_pred = d_nn >= 0.0
        h_pred = h_nn >= 0.0
        t_pred = t_nn >= 0.0
        nn_dev_sc = float(fast_miou_vec(d_pred, dg).mean())

        if method in ("pixelseal", "trustmark"):
            rect_cands = [
                ("rect_nnref", Path(f"runs/rect/{method}_train_nnref.npz"), Path(f"runs/rect/{method}_test_nnref.npz")),
                ("rect_v2", Path(f"runs/rect/{method}_train.npz"), Path(f"runs/rect/{method}_test.npz")),
            ]
            best_rect = (nn_dev_sc + 0.001, None)
            for tag, tr_p, te_p in rect_cands:
                if not (tr_p.exists() and te_p.exists()):
                    print(f"  [RECT {tag}] missing {tr_p} or {te_p}, skip", flush=True)
                    continue
                rd_tr = load_rect_dict(tr_p)
                d_rec = [rd_tr[n] for n in dn]
                d_bm = np.stack([box_to_mask128(r["box"]) for r in d_rec])
                cv_sc, full_sc, cfg = fit_rect_cv(d_nn, dg, dL, d_rec, d_bm)
                print(
                    f"  [RECT {tag}] nested-dev-CV={cv_sc:.4f} ({cv_sc-nn_dev_sc:+.4f}) full-dev={full_sc:.4f} cfg={cfg}",
                    flush=True,
                )
                if cfg is not None and cv_sc > best_rect[0]:
                    best_rect = (cv_sc, (tag, tr_p, te_p, cfg))

            if best_rect[1] is not None:
                tag, tr_p, te_p, (md, fa, ac, gn, al) = best_rect[1]
                rd_tr = load_rect_dict(tr_p)
                rd_te = load_rect_dict(te_p)
                d_rec = [rd_tr[n] for n in dn]
                h_rec = [rd_tr[n] for n in hn]
                t_rec = [rd_te[n] for n in tn]
                d_bm = np.stack([box_to_mask128(r["box"]) for r in d_rec])
                h_bm = np.stack([box_to_mask128(r["box"]) for r in h_rec])
                t_bm = np.stack([box_to_mask128(r["box"]) for r in t_rec])
                d_pred, nt_d = apply_rect_override(d_nn, d_rec, d_bm, fa, ac, gn, md, al)
                h_pred, nt_h = apply_rect_override(h_nn, h_rec, h_bm, fa, ac, gn, md, al)
                t_pred, nt_t = apply_rect_override(t_nn, t_rec, t_bm, fa, ac, gn, md, al)
                stage_name += f"+{tag}"
                print(
                    f"  >> ADOPTED {tag}: trig dev={nt_d}/136 hold={nt_h}/34 test={nt_t}/340",
                    flush=True,
                )

        # Official per-image miou_image check
        d_off = np.array([miou_image(d_pred[i], dr[i]["mask128"]) for i in range(len(dr))])
        h_off = np.array([miou_image(h_pred[i], hr[i]["mask128"]) for i in range(len(hr))])
        print(
            f"  *** FINAL {method.upper()} ({stage_name}) ***\n"
            f"      DEV ={d_off.mean():.4f} (L {d_off[dL].mean():.4f} S {d_off[~dL].mean():.4f})\n"
            f"      HOLD={h_off.mean():.4f} (L {h_off[hL].mean():.4f} S {h_off[~hL].mean():.4f}) | test_fg={t_pred.mean():.4f}",
            flush=True,
        )
        summary[method] = {
            "stage": stage_name,
            "dev": float(d_off.mean()),
            "dev_L": float(d_off[dL].mean()),
            "dev_S": float(d_off[~dL].mean()),
            "hold": float(h_off.mean()),
            "hold_L": float(h_off[hL].mean()),
            "hold_S": float(h_off[~hL].mean()),
            "test_fg": float(t_pred.mean()),
        }
        for i, n in enumerate(tn):
            test_masks[n] = t_pred[i].astype(np.uint8)

    mean_dev = float(np.mean([summary[m]["dev"] for m in summary]))
    mean_hold = float(np.mean([summary[m]["hold"] for m in summary]))
    print("\n============================================================")
    print(f"FINAL 100% CLEAN 3-METHOD DEV  OOF   : {mean_dev:.4f}")
    print(f"FINAL 100% CLEAN 3-METHOD HOLDOUT    : {mean_hold:.4f}")
    print(f"PER-METHOD HOLDOUT: { {m: round(summary[m]['hold'], 4) for m in summary} }")
    print(f"ELAPSED: {time.time() - t0:.1f}s")
    print("============================================================", flush=True)

    all_names = [r["image_name"] for r in test_rows]
    assert len(all_names) == 1020 and len(set(all_names)) == 1020
    masks_u8 = torch.from_numpy(np.stack([test_masks[n] for n in all_names])).to(torch.uint8)
    assert masks_u8.shape == (1020, 128, 128)
    assert set(torch.unique(masks_u8).tolist()).issubset({0, 1})

    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": all_names, "masks": masks_u8}, out_p)
    size_mb = out_p.stat().st_size / (1024 * 1024)
    # Reload verification with weights_only=True
    chk = torch.load(out_p, weights_only=True)
    assert chk["image_names"] == all_names
    assert chk["masks"].shape == (1020, 128, 128) and chk["masks"].dtype == torch.uint8
    assert size_mb < 20.0, f"Submission too large: {size_mb:.2f} MB"
    (out_p.with_suffix(".json")).write_text(
        json.dumps({"dev": mean_dev, "hold": mean_hold, "methods": summary, "size_mb": size_mb}, indent=2)
    )
    print(f"VERIFIED SUBMISSION SAVED: {out_p} ({size_mb:.2f} MB, fg={masks_u8.float().mean():.4f})", flush=True)


if __name__ == "__main__":
    main()
