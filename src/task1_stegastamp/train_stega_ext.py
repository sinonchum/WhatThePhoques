"""Train StegaExtSegNet on ding7's lowres StegaStamp synthetic pool + optional real DEV split.

Strictly respects the team's seed-2026 holdout (34 held-out StegaStamp images never enter training
or threshold selection). Exports predictions and fusion-format {dev,holdout,test}.npz.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

# Add ding7's task1_v6 helpers (read-only) so split, SynthDataset, ShapeBank, and scoring are 100% identical
DING7_ROOT = Path("/p/scratch/training2666/ding7/claude_task1")
for p in (str(DING7_ROOT / "task1_v6"), str(DING7_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index, report
from wmtask.model import seg_loss
from wmtask.synth import ShapeBank, SynthDataset
from stega_extnet import StegaExtSegNet


def load_images(rows, split, data_dir):
    return np.stack([
        np.asarray(Image.open(Path(data_dir) / split / "images" / f"{r['image_name']}.png").convert("RGB"))
        for r in rows
    ])


AMP = {"bf16": dict(dtype=torch.bfloat16), "fp16": dict(dtype=torch.float16), "off": dict(enabled=False)}
AMP_MODE = "fp16"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DING7_ROOT / "runs/synth2/stegastamp"))
    ap.add_argument("--covers", default=str(DING7_ROOT / "runs/covers"))
    ap.add_argument("--data", default=str(DING7_ROOT / "release"))
    ap.add_argument("--onnx", default="models/stegastamp/stega_stamp.onnx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-dir", default=None)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--dec", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--p-real", type=float, default=0.0)
    ap.add_argument("--dev-val-frac", type=float, default=0.0, help="if >0 and p-real>0, hold back this fraction of DEV for unbiased checkpoint/threshold selection")
    ap.add_argument("--alpha0", type=float, default=1.0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="fp16")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    global AMP_MODE
    AMP_MODE = a.amp
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
    rows = [r for r in all_rows if r["method"] == "stegastamp"]
    real_img = load_images(rows, "train", a.data)
    real_bits = np.stack([r["message"] for r in rows]).astype(np.float32)
    bank = ShapeBank(all_rows, hold, a.data, "stegastamp")
    say(f"[stegastamp] real train {len(rows)} (held-out {sum(r['image_name'] in hold for r in rows)}); "
        f"holes {len(bank.holes)} objects {len(bank.objects)}")

    covers = np.load(Path(a.covers) / "covers.npy", mmap_mode="r")
    real = None
    sel_names = {r["image_name"] for r in rows if r["image_name"] not in hold}
    if a.p_real > 0:
        dev_indices = [i for i, r in enumerate(rows) if r["image_name"] not in hold]
        if a.dev_val_frac > 0:
            rng_sp = np.random.default_rng(a.seed + 101)
            val_set = set()
            for full_flag in (False, True):
                sub = [i for i in dev_indices if bool(rows[i]["full_watermark"]) == full_flag]
                rng_sp.shuffle(sub)
                n_v = max(1, int(round(len(sub) * a.dev_val_frac)))
                val_set.update(sub[:n_v])
            fit_indices = [i for i in dev_indices if i not in val_set]
            sel_names = {rows[i]["image_name"] for i in val_set}
        else:
            fit_indices = dev_indices
        real = [
            (real_img[i], rows[i]["mask128"], int(bool(rows[i]["full_watermark"])), real_bits[i])
            for i in fit_indices
        ]
        say(f"mixing {len(real)} real DEV images into training with p={a.p_real}; selection uses {len(sel_names)} DEV images")

    alpha = mp.Value("f", a.alpha0)
    ds = SynthDataset(a.pool, covers, bank, a.steps * a.bs, real=real, p_real=a.p_real, seed=a.seed, alpha=alpha)
    dl = DataLoader(
        ds, batch_size=a.bs, num_workers=a.workers, drop_last=True, pin_memory=True,
        persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None,
    )

    model = StegaExtSegNet(a.onnx, dec=a.dec).cuda()
    if a.init:
        ck = torch.load(a.init, map_location="cuda", weights_only=True)
        model.load_state_dict(ck["model"])
        say("initialized from", a.init)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    say(f"StegaExtSegNet trainable params: {n_train / 1e6:.2f} M")

    groups = [
        {"params": list(model.backbone_parameters()), "lr": a.lr_backbone},
        {"params": list(model.head_parameters()), "lr": a.lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    warm = min(300, max(1, a.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp == "fp16")

    best, t0, run_loss, run_n = -1.0, time.time(), 0.0, 0
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
        run_loss += float(loss)
        run_n += 1
        alpha.value = 1.0 + (a.alpha0 - 1.0) * max(0.0, 1.0 - step / (0.6 * a.steps))

        if step % a.eval_every == 0 or step == a.steps:
            pr = predict(model, real_img, real_bits)
            m = group_scores(rows, pr, hold)
            sel_score = float(np.mean([miou_image(p >= 0.5, r["mask128"]) for r, p in zip(rows, pr) if r["image_name"] in sel_names]))
            say(
                f"[stegastamp step {step}/{a.steps}] loss {run_loss / run_n:.4f} | sel {sel_score:.4f} | "
                f"REAL dev {m[('dev', 'both')]:.4f} (large {m[('dev', 'large')]:.4f} small {m[('dev', 'small')]:.4f}) | "
                f"held-out {m[('holdout', 'both')]:.4f} (large {m[('holdout', 'large')]:.4f} small {m[('holdout', 'small')]:.4f}) | "
                f"{step / (time.time() - t0):.1f} it/s {(time.time() - t0) / 60:.1f} min"
            )
            run_loss, run_n = 0.0, 0
            torch.save({"model": model.state_dict(), "step": step, "args": vars(a)}, out / "last.pt")
            if sel_score > best:
                best = sel_score
                torch.save({"model": model.state_dict(), "step": step, "args": vars(a), "dev": best}, out / "best.pt")
        if step >= a.steps:
            break

    ck = torch.load(out / "best.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(ck["model"])
    say(f"best checkpoint: step {ck['step']} dev {ck['dev']:.4f}")
    probs = predict(model, real_img, real_bits)
    np.savez_compressed(out / "pred_train.npz", names=np.array([r["image_name"] for r in rows]), probs=probs.astype(np.float16))
    report(rows, {r["image_name"]: (p >= 0.5).astype(np.uint8) for r, p in zip(rows, probs)}, f"stegastamp {out.name} thr=0.5", hold)

    test_rows = [r for r in load_index("test", a.data) if r["method"] == "stegastamp"]
    test_img = load_images(test_rows, "test", a.data)
    test_bits = np.stack([r["message"] for r in test_rows]).astype(np.float32)
    tp = predict(model, test_img, test_bits)
    np.savez_compressed(out / "pred_test.npz", names=np.array([r["image_name"] for r in test_rows]), probs=tp.astype(np.float16))
    say(f"wrote predictions: train {probs.shape}, test {tp.shape}; test mean wm frac {(tp >= 0.5).mean():.3f}")

    if a.fusion_dir:
        thr, fm = export_fusion(rows, probs, test_rows, tp, hold, Path(a.fusion_dir), dev_in_sample=(a.p_real > 0), sel_names=sel_names)
        say(
            f"FUSION exported to {a.fusion_dir} | thr={thr} (dev only) dev_in_sample={a.p_real > 0} | "
            f"dev {fm[('dev','both')]:.4f} (L {fm[('dev','large')]:.4f} S {fm[('dev','small')]:.4f}) | "
            f"holdout {fm[('holdout','both')]:.4f} (L {fm[('holdout','large')]:.4f} S {fm[('holdout','small')]:.4f})"
        )


if __name__ == "__main__":
    main()
