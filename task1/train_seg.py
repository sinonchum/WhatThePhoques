"""Train the segmentation net of one method on synthetic composites; validate on the REAL train images of that
method (never used for synthetic training, so all 170 are a fair check; with --p-real > 0 the dev part is mixed
into training and only the held-out 34 remain a fair check).

    python task1/train_seg.py --method trustmark --pool runs/synth/trustmark --out runs/seg/trustmark_v1

Writes best.pt (best dev mIoU), log.txt, pred_train.npz and pred_test.npz (probabilities, 128 x 128)."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from wmloc.utils import miou_image
from wmtask.common import DATA_DIR, holdout_names, load_index, report
from wmtask.model import WMSegNet, seg_loss
from wmtask.synth import ShapeBank, SynthDataset


def load_images(rows, split, data_dir):
    return np.stack([np.asarray(Image.open(Path(data_dir) / split / "images" / f"{r['image_name']}.png").convert("RGB")) for r in rows])


AMP = {"bf16": dict(dtype=torch.bfloat16), "fp16": dict(dtype=torch.float16), "off": dict(enabled=False)}
AMP_MODE = "fp16"


def run_model(model, x, bits):
    """Blind nets take the image only; extractor-based nets also take the known message."""
    return model(x, bits) if getattr(model, "method", None) else model(x)


@torch.no_grad()
def predict(model, images: np.ndarray, bits: np.ndarray, bs: int = 32) -> np.ndarray:
    model.eval(); out = []
    for s in range(0, len(images), bs):
        x = torch.from_numpy(images[s:s + bs]).cuda().permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        b = torch.from_numpy(bits[s:s + bs]).cuda().float()
        with torch.autocast("cuda", **AMP[AMP_MODE]):
            logits, _ = run_model(model, x.contiguous(memory_format=torch.channels_last), b)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    model.train()
    return np.concatenate(out)


def group_scores(rows, probs, hold, thr=0.5):
    s = {}
    for r, p in zip(rows, probs):
        key = ("holdout" if r["image_name"] in hold else "dev", "large" if r["full_watermark"] else "small")
        s.setdefault(key, []).append(miou_image(p > thr, r["mask128"]))
    m = {k: float(np.mean(v)) for k, v in s.items()}
    for split in ("dev", "holdout"):
        v = s.get((split, "large"), []) + s.get((split, "small"), [])
        m[(split, "both")] = float(np.mean(v)) if v else float("nan")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--covers", default="runs/covers")
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--p-real", type=float, default=0.0, help="share of real DEV images mixed into training")
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hp", action="store_true", help="fixed high-pass front end (see wmtask.model.HighPass)")
    ap.add_argument("--fold", type=int, default=-1, help="with --p-real: leave this fold of the dev images out of training (out-of-fold predictions)")
    ap.add_argument("--nfolds", type=int, default=4)
    ap.add_argument("--select", choices=["dev", "last"], default="dev", help="checkpoint choice; use 'last' when dev images are trained on")
    ap.add_argument("--msg-crops", action="store_true", help="ext only: add the 16 x 16 message map from overlapping windows")
    ap.add_argument("--enc-res", type=int, default=256, help="ext only: input size of the trainable extractor copy")
    ap.add_argument("--alpha0", type=float, default=1.0, help="strength curriculum: residual amplified by alpha0 at the start, decaying to 1 at 60 %% of the steps")
    ap.add_argument("--arch", choices=["blind", "ext"], default="blind",
                    help="blind: WMSegNet from scratch; ext: ExtSegNet on the official extractor + known message")
    ap.add_argument("--lr-backbone", type=float, default=1e-4, help="ext only: learning rate of the extractor copy")
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="fp16", help="mixed precision below the float32 stem")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    global AMP_MODE
    AMP_MODE = a.amp
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    log = open(out / "log.txt", "a")

    def say(*s):
        line = " ".join(str(x) for x in s); print(line, flush=True); log.write(line + "\n"); log.flush()

    say("args:", json.dumps(vars(a)))
    all_rows = load_index("train", a.data)
    hold = holdout_names(all_rows)
    rows = [r for r in all_rows if r["method"] == a.method]
    real_img = load_images(rows, "train", a.data)
    real_bits = np.stack([r["message"] for r in rows]).astype(np.float32)
    bank = ShapeBank(all_rows, hold, a.data, a.method)
    say(f"[{a.method}] real train images {len(rows)} (held-out {sum(r['image_name'] in hold for r in rows)}); "
        f"shape bank: holes {len(bank.holes)} objects {len(bank.objects)} rect sizes {len(bank.hw)} rect share {bank.rect_share:.2f}")

    covers = np.load(Path(a.covers) / "covers.npy", mmap_mode="r")
    real = None
    if a.p_real > 0:
        dev_i = [i for i, r in enumerate(rows) if r["image_name"] not in hold]
        oof = {i for j, i in enumerate(dev_i) if a.fold >= 0 and j % a.nfolds == a.fold}
        real = [(real_img[i], rows[i]["mask128"], int(bool(rows[i]["full_watermark"])), real_bits[i]) for i in dev_i if i not in oof]
        (out / "oof_names.json").write_text(json.dumps([rows[i]["image_name"] for i in sorted(oof)]))
        if oof:
            say(f"fold {a.fold}/{a.nfolds}: {len(oof)} dev images left out of training")
        say(f"mixing {len(real)} real DEV images into training with p={a.p_real}: only held-out scores are a fair check now")
    import multiprocessing as mp
    alpha = mp.Value("f", a.alpha0)
    ds = SynthDataset(a.pool, covers, bank, a.steps * a.bs, real=real, p_real=a.p_real, seed=a.seed, alpha=alpha)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True, pin_memory=True,
                    persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)

    if a.arch == "ext":
        from wmloc.watermarks import load_watermark
        from wmtask.extnet import ExtSegNet
        if a.method == "stegastamp":
            import os
            from wmtask.extnet import OrtFeatSegNet
            model = OrtFeatSegNet(os.path.join(os.environ["WMLOC_MODELS"], "stegastamp", "stega_stamp_feats.onnx")).cuda()
            groups = [{"params": model.head_parameters(), "lr": a.lr}]
        else:
            model = ExtSegNet(a.method, load_watermark(a.method, device="cuda"), enc_res=a.enc_res, msg_crops=a.msg_crops).cuda()
            groups = [{"params": list(model.backbone_parameters()), "lr": a.lr_backbone}, {"params": model.head_parameters(), "lr": a.lr}]
    else:
        model = WMSegNet(a.base, hp=a.hp).cuda().to(memory_format=torch.channels_last)
        groups = [{"params": list(model.parameters()), "lr": a.lr}]
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cuda", weights_only=True)["model"])
        say("initialised from", a.init)
    say(f"arch {a.arch}; trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M")
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    warm = min(300, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))))

    scaler = torch.amp.GradScaler("cuda", enabled=a.amp == "fp16")
    best, t0, run_loss, run_n = -1.0, time.time(), 0.0, 0
    model.train()
    for step, (x, y, large, bits) in enumerate(dl, 1):
        x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last); y = y.cuda(non_blocking=True)
        with torch.autocast("cuda", **AMP[a.amp]):
            logits, cls = run_model(model, x, bits.cuda(non_blocking=True))
        loss = seg_loss(logits.float(), y) + 0.2 * F.binary_cross_entropy_with_logits(cls.float(), large.cuda().float())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        run_loss += float(loss); run_n += 1
        alpha.value = 1.0 + (a.alpha0 - 1.0) * max(0.0, 1.0 - step / (0.6 * a.steps))
        if step % a.eval_every == 0 or step == a.steps:
            m = group_scores(rows, predict(model, real_img, real_bits), hold)
            say(f"[{a.method} step {step}/{a.steps}] loss {run_loss / run_n:.4f} | REAL dev {m[('dev', 'both')]:.4f} "
                f"(large {m[('dev', 'large')]:.4f} small {m[('dev', 'small')]:.4f}) | held-out {m[('holdout', 'both')]:.4f} "
                f"(large {m[('holdout', 'large')]:.4f} small {m[('holdout', 'small')]:.4f}) | "
                f"{step / (time.time() - t0):.1f} it/s {(time.time() - t0) / 60:.1f} min")
            run_loss, run_n = 0.0, 0
            torch.save({"model": model.state_dict(), "step": step, "args": vars(a)}, out / "last.pt")
            if a.select == "last" or m[("dev", "both")] > best:           # model selection on DEV only, never on held-out
                best = m[("dev", "both")]
                torch.save({"model": model.state_dict(), "step": step, "args": vars(a), "dev": best}, out / "best.pt")
        if step >= a.steps:
            break

    ck = torch.load(out / "best.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(ck["model"])
    say(f"best checkpoint: step {ck['step']} dev {ck['dev']:.4f}")
    probs = predict(model, real_img, real_bits)
    np.savez_compressed(out / "pred_train.npz", names=np.array([r["image_name"] for r in rows]), probs=probs.astype(np.float16))
    report(rows, {r["image_name"]: (p > 0.5).astype(np.uint8) for r, p in zip(rows, probs)},
           f"{a.method} seg net ({out.name}), threshold 0.5, real train images", hold)
    test_rows = [r for r in load_index("test", a.data) if r["method"] == a.method]
    tp = predict(model, load_images(test_rows, "test", a.data), np.stack([r["message"] for r in test_rows]).astype(np.float32))
    np.savez_compressed(out / "pred_test.npz", names=np.array([r["image_name"] for r in test_rows]), probs=tp.astype(np.float16))
    say(f"wrote predictions: train {probs.shape}, test {tp.shape}; test mean watermarked fraction {(tp > 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
