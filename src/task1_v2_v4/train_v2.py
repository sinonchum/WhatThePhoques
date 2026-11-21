"""Unified V2 trainer for ExtSegNetV2 (pixelseal/trustmark) and StegaExtSegNetV2 (stegastamp).

Features:
- 100% aligned with team seed-2026 holdout (34 held-out images per method never touched for training or selection).
- Unbiased dev validation slice (--dev-val-frac 0.2 when --p-real > 0) for checkpoint & threshold selection.
- EMA (Exponential Moving Average) weights evaluated alongside raw weights at every checkpoint.
- Exports best.pt, pred_train.npz, pred_test.npz, AND runs/fusion/<name>/{dev,holdout,test}.npz.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

DING7_ROOT = Path("/p/scratch/training2666/ding7/claude_task1")
THIS_DIR = Path(__file__).resolve().parent
for p in (str(THIS_DIR), str(DING7_ROOT / "task1_v8"), str(DING7_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from wmloc.utils import miou_image
from wmloc.watermarks import load_watermark
from wmtask.common import holdout_names, load_index, report
from wmtask.model import seg_loss
from wmtask.synth import ShapeBank, SynthDataset

from extnet_v2 import ExtSegNetV2
from stega_extnet_v2 import StegaExtSegNetV2


AMP = {"bf16": dict(dtype=torch.bfloat16), "fp16": dict(dtype=torch.float16), "off": dict(enabled=False)}
AMP_MODE = "fp16"


def load_images(rows, split, data_dir):
    return np.stack([
        np.asarray(Image.open(Path(data_dir) / split / "images" / f"{r['image_name']}.png").convert("RGB"))
        for r in rows
    ])


def predict(model, images: np.ndarray, bits: np.ndarray, bs: int = 16) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(images), bs):
        x = torch.from_numpy(images[s:s + bs]).cuda().permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        b = torch.from_numpy(bits[s:s + bs]).cuda().float()
        with torch.no_grad():
            with torch.autocast("cuda", **AMP[AMP_MODE]):
                logits, _ = model(x, b)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def group_scores(rows, probs, hold, thr=0.5):
    s = {}
    for r, p in zip(rows, probs):
        key = ("holdout" if r["image_name"] in hold else "dev", "large" if r["full_watermark"] else "small")
        s.setdefault(key, []).append(miou_image(p >= thr, r["mask128"]))
    m = {k: float(np.mean(v)) for k, v in s.items()}
    for split in ("dev", "holdout"):
        v = s.get((split, "large"), []) + s.get((split, "small"), [])
        m[(split, "both")] = float(np.mean(v)) if v else float("nan")
    return m


def subset_score(rows, probs, names_set, thr=0.5):
    v = [miou_image(p >= thr, r["mask128"]) for r, p in zip(rows, probs) if r["image_name"] in names_set]
    return float(np.mean(v)) if v else float("nan")


def export_fusion(rows, probs_train, test_rows, probs_test, hold, fusion_dir: Path, dev_in_sample: bool, sel_names: set[str] | None = None):
    names = [r["image_name"] for r in rows]
    row_map = {r["image_name"]: r for r in rows}
    resp = probs_train.astype(np.float32)
    is_hold = np.array([n in hold for n in names])
    dev_idx = np.where(~is_hold)[0]
    hi = np.where(is_hold)[0]
    thr_idx = np.array([i for i in dev_idx if sel_names is None or names[i] in sel_names])

    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    dev_scores = [np.mean([miou_image(resp[i] >= t, row_map[names[i]]["mask128"]) for i in thr_idx]) for t in grid]
    thr = float(grid[int(np.argmax(dev_scores))])

    fusion_dir.mkdir(parents=True, exist_ok=True)
    common = {"threshold": np.float32(thr), "dev_in_sample": np.bool_(dev_in_sample)}
    np.savez_compressed(fusion_dir / "dev.npz", image_names=np.array([names[i] for i in dev_idx]), response=resp[dev_idx], **common)
    np.savez_compressed(fusion_dir / "holdout.npz", image_names=np.array([names[i] for i in hi]), response=resp[hi], **common)
    np.savez_compressed(
        fusion_dir / "test.npz",
        image_names=np.array([r["image_name"] for r in test_rows]),
        response=probs_test.astype(np.float32),
        **common,
    )
    m = group_scores(rows, resp, hold, thr=thr)
    return thr, m


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        d = self.decay
        for k, v in self.shadow.items():
            v.mul_(d).add_(msd[k].detach(), alpha=1.0 - d)

    def apply_to(self, model: torch.nn.Module):
        msd = model.state_dict()
        backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, v in self.shadow.items():
            msd[k].copy_(v)
        return backup

    def restore(self, model: torch.nn.Module, backup: dict):
        msd = model.state_dict()
        for k, v in backup.items():
            msd[k].copy_(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["pixelseal", "trustmark", "stegastamp"])
    ap.add_argument("--pool", default=None)
    ap.add_argument("--covers", default=str(DING7_ROOT / "runs/covers"))
    ap.add_argument("--data", default=str(DING7_ROOT / "release"))
    ap.add_argument("--models-dir", default=str(THIS_DIR / "models"))
    ap.add_argument("--onnx", default=str(THIS_DIR / "models/stegastamp/stega_stamp.onnx"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-dir", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--dec", type=int, default=160)
    ap.add_argument("--enc-res", type=int, default=512)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--p-real", type=float, default=0.0)
    ap.add_argument("--dev-val-frac", type=float, default=0.2)
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="fp16")
    a = ap.parse_args()

    if a.pool is None:
        a.pool = str(DING7_ROOT / "runs/synth2" / a.method)

    global AMP_MODE
    AMP_MODE = a.amp
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    log = open(out / "log.txt", "a")

    def say(*s):
        line = " ".join(str(x) for x in s)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say("args:", json.dumps(vars(a)))
    all_rows = load_index("train", a.data)
    hold = holdout_names(all_rows)
    rows = [r for r in all_rows if r["method"] == a.method]
    real_img = load_images(rows, "train", a.data)
    real_bits = np.stack([r["message"] for r in rows]).astype(np.float32)
    bank = ShapeBank(all_rows, hold, a.data, a.method)
    say(f"[{a.method}] real train {len(rows)} (held-out {sum(r['image_name'] in hold for r in rows)}); "
        f"holes {len(bank.holes)} objects {len(bank.objects)} rect_share {bank.rect_share:.2f}")

    covers = np.load(Path(a.covers) / "covers.npy", mmap_mode="r")
    dev_rows = [(i, r) for i, r in enumerate(rows) if r["image_name"] not in hold]
    sel_names: set[str] | None = None
    real = None
    if a.p_real > 0:
        if a.dev_val_frac > 0:
            rng = np.random.default_rng(a.seed + 999)
            val_set = set()
            for full in (False, True):
                sub = [r["image_name"] for _, r in dev_rows if bool(r["full_watermark"]) == full]
                k = max(1, round(len(sub) * a.dev_val_frac))
                val_set.update(rng.choice(sub, size=k, replace=False).tolist())
            sel_names = val_set
            real = [
                (real_img[i], r["mask128"], int(bool(r["full_watermark"])), real_bits[i])
                for i, r in dev_rows if r["image_name"] not in val_set
            ]
            say(f"mixing {len(real)} real DEV into train (p={a.p_real}); unbiased selection on {len(sel_names)} DEV images")
        else:
            real = [
                (real_img[i], r["mask128"], int(bool(r["full_watermark"])), real_bits[i])
                for i, r in dev_rows
            ]
            say(f"mixing ALL {len(real)} real DEV into train (p={a.p_real})")

    alpha = mp.Value("f", a.alpha0)
    ds = SynthDataset(a.pool, covers, bank, a.steps * a.bs, real=real, p_real=a.p_real, seed=a.seed, alpha=alpha)
    dl = DataLoader(
        ds, batch_size=a.bs, num_workers=a.workers, drop_last=True, pin_memory=True,
        persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None,
    )

    if a.method == "stegastamp":
        model = StegaExtSegNetV2(a.onnx, dec=a.dec).cuda()
    else:
        wm = load_watermark(a.method, models_dir=a.models_dir, device="cuda")
        model = ExtSegNetV2(a.method, wm, dec=a.dec, enc_res=a.enc_res, use_hp=True).cuda()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    say(f"{model.__class__.__name__} trainable params: {n_train / 1e6:.2f} M")

    groups = [
        {"params": list(model.backbone_parameters()), "lr": a.lr_backbone},
        {"params": list(model.head_parameters()), "lr": a.lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    warm = min(300, max(50, a.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp == "fp16")
    ema = ModelEMA(model, decay=a.ema_decay)

    best, best_tag, t0, run_loss, run_n = -1.0, "", time.time(), 0.0, 0
    model.train()
    for step, (x, y, large, bits) in enumerate(dl, 1):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        bits = bits.cuda(non_blocking=True)
        with torch.autocast("cuda", **AMP[a.amp]):
            logits, cls = model(x, bits)
        loss = seg_loss(logits.float(), y) + 0.2 * F.binary_cross_entropy_with_logits(cls.float(), large.cuda().float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema.update(model)

        run_loss += float(loss)
        run_n += 1
        if a.alpha0 > 1.0:
            alpha.value = 1.0 + (a.alpha0 - 1.0) * max(0.0, 1.0 - step / (0.6 * a.steps))

        if step % a.eval_every == 0 or step == a.steps:
            # Evaluate both raw and EMA weights
            probs_raw = predict(model, real_img, real_bits, bs=max(8, a.bs))
            m_raw = group_scores(rows, probs_raw, hold)
            sel_raw = subset_score(rows, probs_raw, sel_names) if sel_names else m_raw[("dev", "both")]

            bak = ema.apply_to(model)
            probs_ema = predict(model, real_img, real_bits, bs=max(8, a.bs))
            m_ema = group_scores(rows, probs_ema, hold)
            sel_ema = subset_score(rows, probs_ema, sel_names) if sel_names else m_ema[("dev", "both")]

            elapsed = time.time() - t0
            say(
                f"[{a.method} step {step}/{a.steps}] loss {run_loss / run_n:.4f} | "
                f"RAW sel {sel_raw:.4f} dev {m_raw[('dev','both')]:.4f} hold {m_raw[('holdout','both')]:.4f} "
                f"(L {m_raw[('holdout','large')]:.4f} S {m_raw[('holdout','small')]:.4f}) | "
                f"EMA sel {sel_ema:.4f} dev {m_ema[('dev','both')]:.4f} hold {m_ema[('holdout','both')]:.4f} "
                f"(L {m_ema[('holdout','large')]:.4f} S {m_ema[('holdout','small')]:.4f}) | "
                f"{step / elapsed:.2f} it/s {elapsed / 60:.1f}m"
            )
            run_loss, run_n = 0.0, 0

            if sel_ema >= sel_raw and sel_ema > best:
                best, best_tag = sel_ema, f"step {step} EMA"
                torch.save({"model": model.state_dict(), "step": step, "ema": True, "sel": sel_ema, "args": vars(a)}, out / "best.pt")
            ema.restore(model, bak)
            if sel_raw > sel_ema and sel_raw > best:
                best, best_tag = sel_raw, f"step {step} RAW"
                torch.save({"model": model.state_dict(), "step": step, "ema": False, "sel": sel_raw, "args": vars(a)}, out / "best.pt")

    ckpt = torch.load(out / "best.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    say(f"best checkpoint: {best_tag} sel={best:.4f}")

    probs_tr = predict(model, real_img, real_bits, bs=max(8, a.bs))
    test_rows = [r for r in load_index("test", a.data) if r["method"] == a.method]
    test_img = load_images(test_rows, "test", a.data)
    test_bits = np.stack([r["message"] for r in test_rows]).astype(np.float32)
    probs_te = predict(model, test_img, test_bits, bs=max(8, a.bs))

    np.savez_compressed(out / "pred_train.npz", names=np.array([r["image_name"] for r in rows]), probs=probs_tr.astype(np.float16))
    np.savez_compressed(out / "pred_test.npz", names=np.array([r["image_name"] for r in test_rows]), probs=probs_te.astype(np.float16))

    thr, mf = export_fusion(rows, probs_tr, test_rows, probs_te, hold, Path(a.fusion_dir), dev_in_sample=(a.p_real > 0), sel_names=sel_names)
    say(
        f"exported fusion {a.fusion_dir} | thr={thr:.2f} | "
        f"dev {mf[('dev','both')]:.4f} (L {mf[('dev','large')]:.4f} S {mf[('dev','small')]:.4f}) | "
        f"holdout {mf[('holdout','both')]:.4f} (L {mf[('holdout','large')]:.4f} S {mf[('holdout','small')]:.4f})"
    )


if __name__ == "__main__":
    main()
