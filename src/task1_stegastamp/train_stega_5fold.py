"""5-fold stratified CV ensemble of StegaExtSegNet on StegaStamp DEV (seed-2026 holdout untouched).

Each fold trains on synth2/stegastamp + 80% of DEV (p_real=0.25) and validates on the remaining
20% of DEV (strictly out-of-fold). Produces:
  - 100% honest OOF predictions for all 136 DEV images
  - 5-fold mean predictions for holdout (34) and test (340)
  - Fusion-format {dev,holdout,test}.npz with threshold chosen on OOF DEV only
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

DING7_ROOT = Path("/p/scratch/training2666/ding7/claude_task1")
for p in (str(DING7_ROOT / "task1_v6"), str(DING7_ROOT), str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index
from wmtask.model import seg_loss
from wmtask.synth import ShapeBank, SynthDataset
from stega_extnet import StegaExtSegNet
from train_stega_ext import load_images, predict, group_scores


def make_folds(rows, hold, k=5, seed=2026):
    dev_idx = [i for i, r in enumerate(rows) if r["image_name"] not in hold]
    folds = [[] for _ in range(k)]
    rng = np.random.default_rng(seed)
    for full_flag in (False, True):
        sub = [i for i in dev_idx if bool(rows[i]["full_watermark"]) == full_flag]
        sub.sort(key=lambda i: (rows[i]["mask128"].mean(), rows[i]["image_name"]))
        # stratified chunks of size k
        for start in range(0, len(sub), k):
            chunk = list(sub[start:start + k])
            rng.shuffle(chunk)
            for f, idx in enumerate(chunk):
                folds[f].append(idx)
    return dev_idx, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(DING7_ROOT / "runs/synth2/stegastamp"))
    ap.add_argument("--covers", default=str(DING7_ROOT / "runs/covers"))
    ap.add_argument("--data", default=str(DING7_ROOT / "release"))
    ap.add_argument("--onnx", default="models/stegastamp/stega_stamp.onnx")
    ap.add_argument("--out", default="runs/stegaext_5fold")
    ap.add_argument("--fusion-dir", default="runs/fusion/qin7_stegaext_5fold_stegastamp")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--dec", type=int, default=160)
    ap.add_argument("--p-real", type=float, default=0.25)
    ap.add_argument("--alpha0", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=250)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.txt", "a")

    def say(*s):
        line = " ".join(str(x) for x in s)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    all_rows = load_index("train", a.data)
    hold = holdout_names(all_rows)
    rows = [r for r in all_rows if r["method"] == "stegastamp"]
    real_img = load_images(rows, "train", a.data)
    real_bits = np.stack([r["message"] for r in rows]).astype(np.float32)
    bank = ShapeBank(all_rows, hold, a.data, "stegastamp")
    covers = np.load(Path(a.covers) / "covers.npy", mmap_mode="r")

    test_rows = [r for r in load_index("test", a.data) if r["method"] == "stegastamp"]
    test_img = load_images(test_rows, "test", a.data)
    test_bits = np.stack([r["message"] for r in test_rows]).astype(np.float32)

    dev_idx, folds = make_folds(rows, hold, k=5, seed=2026)
    hold_idx = [i for i, r in enumerate(rows) if r["image_name"] in hold]
    say(f"5-fold CV: dev={len(dev_idx)}, holdout={len(hold_idx)}, test={len(test_rows)}, fold sizes={[len(f) for f in folds]}")

    oof_probs = np.zeros((len(rows), 128, 128), np.float32)
    hold_sum = np.zeros((len(hold_idx), 128, 128), np.float32)
    test_sum = np.zeros((len(test_rows), 128, 128), np.float32)

    torch.backends.cudnn.benchmark = True
    for fold_id, val_ids in enumerate(folds):
        val_set = set(val_ids)
        fit_ids = [i for i in dev_idx if i not in val_set]
        seed = 100 + fold_id
        torch.manual_seed(seed)
        np.random.seed(seed)

        real = [(real_img[i], rows[i]["mask128"], int(bool(rows[i]["full_watermark"])), real_bits[i]) for i in fit_ids]
        alpha = mp.Value("f", a.alpha0)
        ds = SynthDataset(a.pool, covers, bank, a.steps * a.bs, real=real, p_real=a.p_real, seed=seed, alpha=alpha)
        dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True, pin_memory=True,
                        persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)

        model = StegaExtSegNet(a.onnx, dec=a.dec).cuda()
        groups = [
            {"params": list(model.backbone_parameters()), "lr": a.lr_backbone},
            {"params": list(model.head_parameters()), "lr": a.lr},
        ]
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        warm = min(300, a.steps // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)

        best_val, best_state = -1.0, None
        t0, run_loss, run_n = time.time(), 0.0, 0
        model.train()
        for step, (x, y, large, bits) in enumerate(dl, 1):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            bits = bits.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
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
                val_s = float(np.mean([miou_image(pr[i] >= 0.5, rows[i]["mask128"]) for i in val_ids]))
                ho_s = float(np.mean([miou_image(pr[i] >= 0.5, rows[i]["mask128"]) for i in hold_idx]))
                say(f"[fold {fold_id} step {step}/{a.steps}] loss {run_loss/run_n:.4f} | oof_val {val_s:.4f} | holdout {ho_s:.4f} | {(time.time()-t0)/60:.1f}m")
                run_loss, run_n = 0.0, 0
                if val_s > best_val:
                    best_val = val_s
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step >= a.steps:
                break

        torch.save({"model": best_state, "val": best_val}, out / f"fold{fold_id}_best.pt")
        model.load_state_dict(best_state)
        pr = predict(model, real_img, real_bits)
        tp = predict(model, test_img, test_bits)
        for i in val_ids:
            oof_probs[i] = pr[i]
        hold_sum += pr[hold_idx]
        test_sum += tp
        ho_fold = float(np.mean([miou_image(pr[i] >= 0.5, rows[i]["mask128"]) for i in hold_idx]))
        say(f"FOLD {fold_id} DONE | best oof_val {best_val:.4f} | fold holdout@0.5 {ho_fold:.4f}")
        del model, opt, sched, scaler, dl, ds
        torch.cuda.empty_cache()

    hold_ens = hold_sum / 5.0
    test_ens = test_sum / 5.0
    for k, i in enumerate(hold_idx):
        oof_probs[i] = hold_ens[k]

    # Pick threshold strictly on 100% honest OOF DEV predictions
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    dev_scores = [np.mean([miou_image(oof_probs[i] >= t, rows[i]["mask128"]) for i in dev_idx]) for t in grid]
    thr = float(grid[int(np.argmax(dev_scores))])
    m = group_scores(rows, oof_probs, hold, thr=thr)
    m05 = group_scores(rows, oof_probs, hold, thr=0.5)
    say(f"5-FOLD ENSEMBLE @0.5 | OOF dev {m05[('dev','both')]:.4f} (L {m05[('dev','large')]:.4f} S {m05[('dev','small')]:.4f}) | "
        f"holdout {m05[('holdout','both')]:.4f} (L {m05[('holdout','large')]:.4f} S {m05[('holdout','small')]:.4f})")
    say(f"5-FOLD ENSEMBLE @thr={thr} | OOF dev {m[('dev','both')]:.4f} (L {m[('dev','large')]:.4f} S {m[('dev','small')]:.4f}) | "
        f"holdout {m[('holdout','both')]:.4f} (L {m[('holdout','large')]:.4f} S {m[('holdout','small')]:.4f})")

    fdir = Path(a.fusion_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    # OOF dev predictions are out-of-sample for each dev image!
    common = {"threshold": np.float32(thr), "dev_in_sample": np.bool_(False)}
    np.savez_compressed(fdir / "dev.npz", image_names=np.array([rows[i]["image_name"] for i in dev_idx]), response=oof_probs[dev_idx], **common)
    np.savez_compressed(fdir / "holdout.npz", image_names=np.array([rows[i]["image_name"] for i in hold_idx]), response=hold_ens, **common)
    np.savez_compressed(fdir / "test.npz", image_names=np.array([r["image_name"] for r in test_rows]), response=test_ens, **common)
    say(f"EXPORTED 5-fold fusion to {fdir} | test mean wm frac {(test_ens >= thr).mean():.3f}")


if __name__ == "__main__":
    main()
