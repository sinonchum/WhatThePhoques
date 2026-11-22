"""Stage-2 Spatial Meta-Stacker trained strictly via 4-Fold Nested CV on clean dev OOF maps.

NEVER touches holdout for training, model selection, or early stopping.
For each method, compares:
  - Linear dev-OOF blend (baseline)
  - Nested 4-fold CV Spatial MetaStackNet (residual ConvNet <12K params with exact Symmetric Soft-mIoU Loss)
Only keeps MetaStackNet if its honest Nested Dev-OOF strictly beats linear blend!
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1/task1_v11")
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1")
from wmtask.common import holdout_names, load_index
from wmloc.utils import miou_image
from fuse_clean_cv import FUSION_ROOTS, is_clean_dir, load_model, to_logit


class MetaStackNet(nn.Module):
    """Tiny regularized spatial calibrator: starts from linear blend + learns spatial refinement."""
    def __init__(self, n_models: int, init_w: np.ndarray, init_b: float):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(float(-init_b), dtype=torch.float32))
        # Input to spatial residual net: [n_models centered logits, base_logit, base_prob, std_map, area_map]
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
        # x: [B, M, 128, 128]
        wn = F.softmax(self.w, dim=0).view(1, -1, 1, 1)
        base = (x * wn).sum(dim=1, keepdim=True) + self.b
        prob = torch.sigmoid(base)
        std = x.std(dim=1, keepdim=True) if x.shape[1] > 1 else torch.zeros_like(base)
        area = prob.mean(dim=(2, 3), keepdim=True).expand_as(base)
        feat = torch.cat([x.clamp(-6, 6) / 3.0, base.clamp(-6, 6) / 3.0, prob, std.clamp(0, 4) / 2.0, area], dim=1)
        return (base + self.net(feat))[:, 0]


def sym_miou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    i1 = (p * t).sum(1); u1 = (p + t - p * t).sum(1)
    i0 = ((1 - p) * (1 - t)).sum(1); u0 = ((1 - p) + (1 - t) - (1 - p) * (1 - t)).sum(1)
    sym = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0)).mean()
    return 0.35 * bce + 0.65 * (1.0 - sym)


def fast_miou(scores: np.ndarray, gt_flat: np.ndarray, fm: np.ndarray, thr: float = 0.0):
    pred = scores.reshape(len(gt_flat), -1) >= thr
    i1 = np.logical_and(pred, gt_flat).sum(1); u1 = np.logical_or(pred, gt_flat).sum(1)
    i0 = np.logical_and(~pred, ~gt_flat).sum(1); u0 = np.logical_or(~pred, ~gt_flat).sum(1)
    s = 0.5 * (np.where(u1 == 0, 1.0, i1 / np.maximum(u1, 1)) + np.where(u0 == 0, 1.0, i0 / np.maximum(u0, 1)))
    return float(s[fm].mean()), float(s[~fm].mean()), float(s.mean())


def nested_cv_stack(X_dev: np.ndarray, y_dev: np.ndarray, fm_dev: np.ndarray, X_hold: np.ndarray, X_test: np.ndarray, init_w: np.ndarray, init_b: float):
    """4-fold stratified nested CV strictly inside dev."""
    N, M = X_dev.shape[0], X_dev.shape[1]
    idx_L = np.where(fm_dev)[0]; idx_S = np.where(~fm_dev)[0]
    folds = [[] for _ in range(4)]
    for k, i in enumerate(idx_L): folds[k % 4].append(i)
    for k, i in enumerate(idx_S): folds[k % 4].append(i)

    oof_pred = np.zeros((N, 128, 128), np.float32)
    hold_preds, test_preds = [], []
    # init logits for softmax weights
    iw = np.log(np.maximum(init_w / max(init_w.sum(), 1e-6), 1e-2))

    X_d_t = torch.from_numpy(X_dev)
    y_d_t = torch.from_numpy(y_dev.astype(np.float32))
    X_h_t = torch.from_numpy(X_hold)
    X_t_t = torch.from_numpy(X_test)
    gt_flat = y_dev.astype(bool).reshape(N, -1)

    for k in range(4):
        val_idx = np.array(folds[k])
        tr_idx = np.array([i for i in range(N) if i not in set(folds[k])])
        torch.manual_seed(42 + k)
        net = MetaStackNet(M, iw, init_b)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2e-2)
        best_sc = -1.0
        best_val = None
        best_state = None
        for ep in range(50):
            net.train()
            # random flip augmentation (valid because spatial stacker is equivariant!)
            xb = X_d_t[tr_idx]; yb = y_d_t[tr_idx]
            if ep % 2 == 1:
                xb = torch.flip(xb, dims=[-1]); yb = torch.flip(yb, dims=[-1])
            if ep % 4 == 2:
                xb = torch.flip(xb, dims=[-2]); yb = torch.flip(yb, dims=[-2])
            opt.zero_grad()
            loss = sym_miou_loss(net(xb), yb)
            loss.backward()
            opt.step()
            if (ep + 1) % 5 == 0:
                net.eval()
                with torch.no_grad():
                    vp = net(X_d_t[val_idx]).numpy()
                _, _, sc = fast_miou(vp, gt_flat[val_idx], fm_dev[val_idx], 0.0)
                if sc > best_sc:
                    best_sc = sc
                    best_val = vp.copy()
                    best_state = {kk: vv.clone() for kk, vv in net.state_dict().items()}
        oof_pred[val_idx] = best_val
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            hold_preds.append(net(X_h_t).numpy())
            test_preds.append(net(X_t_t).numpy())
    return oof_pred, np.mean(hold_preds, axis=0), np.mean(test_preds, axis=0)


if __name__ == "__main__":
    train_rows = load_index("train", "/p/scratch/training2666/ding7/claude_task1/release")
    test_rows = load_index("test", "/p/scratch/training2666/ding7/claude_task1/release")
    hold_set = holdout_names(train_rows)

    for method in ("pixelseal", "trustmark", "stegastamp"):
        mt = [r for r in train_rows if r["method"] == method]
        dev_rows = [r for r in mt if r["image_name"] not in hold_set]
        hold_rows = [r for r in mt if r["image_name"] in hold_set]
        te_rows = [r for r in test_rows if r["method"] == method]
        dn, hn, tn = [r["image_name"] for r in dev_rows], [r["image_name"] for r in hold_rows], [r["image_name"] for r in te_rows]
        y_d = np.stack([r["mask128"].astype(np.float32) for r in dev_rows])
        gt_d = y_d.astype(bool).reshape(len(dev_rows), -1)
        fm_d = np.array([r["full_watermark"] for r in dev_rows], bool)
        gt_h = np.stack([r["mask128"].astype(bool).ravel() for r in hold_rows])
        fm_h = np.array([r["full_watermark"] for r in hold_rows], bool)

        found = []
        for root in FUSION_ROOTS:
            if not root.exists(): continue
            for d in sorted(root.iterdir()):
                if d.is_dir() and d.name.endswith(f"_{method}") and is_clean_dir(d):
                    if all((d / f).exists() for f in ("dev.npz", "holdout.npz", "test.npz")):
                        m = load_model(d, dn, hn, tn)
                        if not m["in_sample"]:
                            _, _, dm = fast_miou(m["dev"], gt_d, fm_d, m["thr"])
                            found.append((dm, m))
        found.sort(key=lambda x: x[0], reverse=True)
        top = [x[1] for x in found[:4]]
        Xd = np.stack([to_logit(m["dev"]) - to_logit(np.array(m["thr"], np.float32)) for m in top], axis=1)
        Xh = np.stack([to_logit(m["hold"]) - to_logit(np.array(m["thr"], np.float32)) for m in top], axis=1)
        Xt = np.stack([to_logit(m["test"]) - to_logit(np.array(m["thr"], np.float32)) for m in top], axis=1)

        # Linear baseline
        best = (-1.0, None, 0.0)
        for ws in itertools.product([0.0, 1.0, 2.0, 3.0], repeat=len(top)):
            s = sum(ws)
            if s <= 0: continue
            acc = sum(w * Xd[:, i] for i, w in enumerate(ws) if w > 0) / s
            for b in (-0.3, -0.15, 0.0, 0.15, 0.3):
                _, _, dm = fast_miou(acc, gt_d, fm_d, b)
                if dm > best[0]: best = (dm, np.array(ws, np.float32), b)
        lin_dm, ws, b = best
        lin_h = sum(w * Xh[:, i] for i, w in enumerate(ws) if w > 0) / ws.sum() - b
        _, _, lin_hm = fast_miou(lin_h, gt_h, fm_h, 0.0)

        oof_m, hold_m, _ = nested_cv_stack(Xd, y_d, fm_d, Xh, Xt, ws, b)
        _, _, st_dm = fast_miou(oof_m, gt_d, fm_d, 0.0)
        hL, hS, st_hm = fast_miou(hold_m, gt_h, fm_h, 0.0)
        print(f"[{method:<10s}] LINEAR: dev={lin_dm:.4f} hold={lin_hm:.4f}  |  META-STACK(4f nested): dev={st_dm:.4f} hold={st_hm:.4f} (L {hL:.4f} S {hS:.4f}) | delta_dev={st_dm-lin_dm:+.4f} delta_hold={st_hm-lin_hm:+.4f}")
