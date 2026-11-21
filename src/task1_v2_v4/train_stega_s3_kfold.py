"""Clean K-fold OOF trainer for StegaStamp V1 (StegaExtSegNet) and V2 (StegaExtSegNetV2)
strictly on clean synth3/stegastamp (train-only non-holdout covers + lowres 256 embed delta upsample).
Zero holdout leakage: fold validation selects checkpoint & threshold; exports OOF fold preds for honest dev CV.
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1/task1_v8")
sys.path.insert(0, "/p/scratch/training2666/ding7/claude_task1")

from wmloc.utils import miou_image
from wmtask.common import holdout_names, load_index
from wmtask.synth import SynthDataset, SynthPool
from wmtask.train_seg import seg_loss
from train_v2 import MixedDataset, load_img_m11
from train_v3_attr import make_stratified_kfold_split
from stega_extnet import StegaExtSegNet
from stega_extnet_v2 import StegaExtSegNetV2


@torch.no_grad()
def predict_rows(model, rows, split_dir: Path, device: str, bs: int = 16, tta: bool = True):
    model.eval()
    probs = {}
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        xs = torch.stack([load_img_m11(split_dir / f"{r['image_name']}.png") for r in batch]).to(device)
        b_t = torch.from_numpy(np.stack([r["bits"].astype(np.float32) for r in batch])).to(device)
        with torch.enable_grad():
            p = torch.sigmoid(model(xs, b_t)[0])
            if tta:
                p = 0.5 * (p + torch.sigmoid(model(xs.flip(-1), b_t)[0]).flip(-1))
        for r, pi in zip(batch, p.detach().cpu().numpy().astype(np.float32)):
            probs[r["image_name"]] = pi
    return probs


def eval_probs(probs, rows, thr: float = 0.5):
    L, S = [], []
    for r in rows:
        sc = miou_image(probs[r["image_name"]] >= thr, r["mask128"])
        (L if r["full_watermark"] else S).append(sc)
    return float(np.mean(L)), float(np.mean(S)), float(np.mean(L + S))


def best_thr(probs, rows):
    grid = np.linspace(0.05, 0.90, 18)
    best = (0.5, -1.0)
    for t in grid:
        _, _, m = eval_probs(probs, rows, float(t))
        if m > best[1]:
            best = (float(t), m)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["v1", "v2"], default="v2")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--covers", default="/p/scratch/training2666/ding7/claude_task1/runs/covers")
    ap.add_argument("--data", default="/p/scratch/training2666/ding7/claude_task1/release")
    ap.add_argument("--onnx", default="/p/scratch/training2666/qin7/task1/models/stega_stamp.onnx")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fusion-dir", required=True)
    ap.add_argument("--steps", type=int, default=4500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-real", type=float, default=0.25)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ema-decay", type=float, default=0.995)
    ap.add_argument("--seed", type=int, default=801)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_f = open(out / "log.txt", "a", buffering=1)

    def log(msg: str):
        print(msg, flush=True)
        log_f.write(msg + "\n")

    rows = load_index("train", args.data)
    hold_set = holdout_names(rows)
    m_rows = [r for r in rows if r["method"] == "stegastamp"]
    dev_all = [r for r in m_rows if r["image_name"] not in hold_set]
    hold_rows = [r for r in m_rows if r["image_name"] in hold_set]
    test_rows = [r for r in load_index("test", args.data) if r["method"] == "stegastamp"]
    train_dir = Path(args.data) / "train" / "images"
    test_dir = Path(args.data) / "test" / "images"

    real_train_rows, sel_rows = make_stratified_kfold_split(dev_all, args.n_folds, args.fold)
    log(f"[stega {args.arch} clean-s3 fold {args.fold}/{args.n_folds}] real_train={len(real_train_rows)} sel={len(sel_rows)} hold={len(hold_rows)}")

    pool = SynthPool(args.pool, args.covers, "stegastamp")
    synth_ds = SynthDataset(pool, steps=args.steps * args.bs + 256, seed=args.seed)
    ds = MixedDataset(synth_ds, real_train_rows, train_dir, args.p_real, args.seed + 77)
    loader = DataLoader(ds, batch_size=args.bs, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, drop_last=True)

    if args.arch == "v2":
        model = StegaExtSegNetV2(onnx_path=args.onnx, dec=128).to(device)
    else:
        model = StegaExtSegNet(onnx_path=args.onnx, dec=128).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_sel = -1.0
    best_step = 0
    best_tag = "RAW"
    running = 0.0
    t0 = time.time()

    for step, (x, mask, bits, full) in enumerate(loader, 1):
        model.train()
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        bits = bits.to(device, non_blocking=True)
        full = full.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, cls_l = model(x, bits)
            loss = seg_loss(logits, mask) + 0.1 * torch.nn.functional.binary_cross_entropy_with_logits(cls_l, full)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        with torch.no_grad():
            d = args.ema_decay
            for pe, pm in zip(ema.parameters(), model.parameters()):
                if pm.requires_grad:
                    pe.data.mul_(d).add_(pm.data, alpha=1.0 - d)

        running += float(loss.item())
        if step % args.eval_every == 0 or step == args.steps:
            raw_sel = eval_probs(predict_rows(model, sel_rows, train_dir, device, tta=True), sel_rows)[2]
            ema_sel = eval_probs(predict_rows(ema, sel_rows, train_dir, device, tta=True), sel_rows)[2]
            raw_hold = eval_probs(predict_rows(model, hold_rows, train_dir, device, tta=True), hold_rows)
            ema_hold = eval_probs(predict_rows(ema, hold_rows, train_dir, device, tta=True), hold_rows)
            dt = time.time() - t0
            log(f"[stega {args.arch} f{args.fold} step {step}/{args.steps}] loss {running/args.eval_every:.4f} | "
                f"RAW sel {raw_sel:.4f} hold {raw_hold[2]:.4f} (L {raw_hold[0]:.4f} S {raw_hold[1]:.4f}) | "
                f"EMA sel {ema_sel:.4f} hold {ema_hold[2]:.4f} (L {ema_hold[0]:.4f} S {ema_hold[1]:.4f}) | {step/dt:.2f} it/s")
            running = 0.0
            for tag, sc, net in [("RAW", raw_sel, model), ("EMA", ema_sel, ema)]:
                if sc > best_sel:
                    best_sel = sc; best_step = step; best_tag = tag
                    torch.save({"step": step, "tag": tag, "sel": sc, "state_dict": net.state_dict()}, out / "best.pt")
        if step >= args.steps:
            break

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    log(f"best checkpoint: step {best_step} {best_tag} sel={best_sel:.4f}")

    dev_p = predict_rows(model, dev_all, train_dir, device, tta=True)
    hold_p = predict_rows(model, hold_rows, train_dir, device, tta=True)
    test_p = predict_rows(model, test_rows, test_dir, device, tta=True)
    thr, _ = best_thr(dev_p, sel_rows)

    sel_names = [r["image_name"] for r in sel_rows]
    np.savez_compressed(
        out / "oof_fold.npz",
        image_names=np.array(sel_names, dtype=object),
        response=np.stack([dev_p[n] for n in sel_names]).astype(np.float32),
        hold_names=np.array([r["image_name"] for r in hold_rows], dtype=object),
        hold_response=np.stack([hold_p[r["image_name"]] for r in hold_rows]).astype(np.float32),
        test_names=np.array([r["image_name"] for r in test_rows], dtype=object),
        test_response=np.stack([test_p[r["image_name"]] for r in test_rows]).astype(np.float32),
        threshold=np.float32(thr),
        fold=np.int32(args.fold),
        n_folds=np.int32(args.n_folds),
    )

    fdir = Path(args.fusion_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    for split, sp_rows, sp_p in [("dev", dev_all, dev_p), ("holdout", hold_rows, hold_p), ("test", test_rows, test_p)]:
        names = [r["image_name"] for r in sp_rows]
        np.savez_compressed(
            fdir / f"{split}.npz",
            image_names=np.array(names, dtype=object),
            response=np.stack([sp_p[n] for n in names]).astype(np.float32),
            threshold=np.float32(thr),
            dev_in_sample=np.bool_(True),
        )
    hL, hS, hM = eval_probs(hold_p, hold_rows, thr)
    sL, sS, sM = eval_probs(dev_p, sel_rows, thr)
    log(f"exported {fdir} | thr={thr:.2f} | OOF-fold {sM:.4f} (L {sL:.4f} S {sS:.4f}) | holdout {hM:.4f} (L {hL:.4f} S {hS:.4f})")


if __name__ == "__main__":
    main()
