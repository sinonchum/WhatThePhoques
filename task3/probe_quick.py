"""Quick read of the original model on the forget set: confidence, classes, a few views."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

STARTER = Path(__file__).resolve().parent / "starter"
sys.path.insert(0, str(STARTER))

import task_template as tt  # noqa: E402

OUT = Path(__file__).resolve().parent / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)


def logit_lens_argmax(model, x):
    """Argmax after each block, read out with the final norm + head. x is preprocessed."""
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.patch_drop(h)
    h = model.norm_pre(h)
    preds = []
    for blk in model.blocks:
        h = blk(h)
        cls = model.norm(h[:, 0])
        cls = model.fc_norm(cls)
        logits = model.head(cls)
        preds.append(logits.argmax(dim=-1))
    return torch.stack(preds, dim=1)


def main():
    tt.MODEL_FILE = STARTER / "model.pt"
    tt.FORGET_IMAGES_FILE = STARTER / "forget_images.npz"
    model = tt.load_model(tt.MODEL_FILE)
    images = tt.load_forget_images(tt.FORGET_IMAGES_FILE)
    print("images", tuple(images.shape), images.dtype, "min", int(images.min()), "max", int(images.max()))

    logits = tt.predict(model, images, batch_size=128)
    prob = torch.softmax(logits, dim=-1)
    top2 = prob.topk(2, dim=-1)
    pred = top2.indices[:, 0]
    top1 = top2.values[:, 0]
    top2p = top2.values[:, 1]
    entropy = -(prob.clamp_min(1e-12).log() * prob).sum(-1)
    print(f"top1 mean {top1.mean():.4f}  p50 {top1.median():.4f}  p10 {top1.quantile(0.1):.4f}  p90 {top1.quantile(0.9):.4f}")
    for t in (0.5, 0.8, 0.9, 0.95, 0.99, 0.999):
        print(f"  frac top1>{t:.3f}: {(top1 > t).float().mean():.3f}")
    print(f"margin mean {(top1-top2p).mean():.4f}  p50 {(top1-top2p).median():.4f}")
    print(f"entropy mean {entropy.mean():.3f}  (uniform {np.log(100):.3f})")
    counts = torch.bincount(pred, minlength=100)
    print(f"classes used {(counts>0).sum().item()} / 100  max {counts.max().item()}  min {counts.min().item()}  "
          f"p10 {counts.float().quantile(0.1):.0f}  p50 {counts.float().median():.0f}")
    print("largest classes", counts.topk(8).indices.tolist(), counts.topk(8).values.tolist())
    print("smallest classes", counts.topk(8, largest=False).indices.tolist(), counts.topk(8, largest=False).values.tolist())

    # logit lens on 512 spread-out images
    idx = torch.linspace(0, len(images) - 1, 512).long()
    lens = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx), 64):
            sl = idx[i:i+64]
            x = tt.preprocess(images[sl])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                lens.append(logit_lens_argmax(model, x).cpu())
    lens = torch.cat(lens)
    final = pred[idx]
    print("logit lens agreement with final, by block:")
    for b in range(lens.shape[1]):
        print(f"  block {b:02d}: {(lens[:, b] == final).float().mean():.3f}")
    # first block where prediction matches final and stays
    print("block0==final", (lens[:, 0] == final).float().mean().item())
    print("block6==final", (lens[:, 6] == final).float().mean().item())
    print("any early (0-5) differs", (lens[:, :6] != final[:, None]).any(1).float().mean().item())

    # save a contact sheet: 4 high conf, 4 low conf, 4 median
    order = top1.argsort()
    pick = torch.cat([order[:4], order[len(order)//2 - 2: len(order)//2 + 2], order[-4:]])
    tiles = []
    for i in pick.tolist():
        im = images[i].numpy()
        tiles.append(im)
    grid = np.concatenate([
        np.concatenate(tiles[0:4], axis=1),
        np.concatenate(tiles[4:8], axis=1),
        np.concatenate(tiles[8:12], axis=1),
    ], axis=0)
    Image.fromarray(grid).save(OUT / "forget_examples.png")
    meta = np.stack([pick.numpy(), pred[pick].numpy(), (top1[pick].numpy()*1000).astype(int)], axis=1)
    print("sheet rows: low conf / mid / high conf. columns of (index, pred, top1*1000):")
    print(meta)

    np.savez_compressed(
        OUT / "clean_preds.npz",
        logits=logits.half().numpy(),
        pred=pred.numpy().astype(np.int16),
        top1=top1.numpy().astype(np.float32),
        top2=top2.indices[:, 1].numpy().astype(np.int16),
        top2p=top2p.numpy().astype(np.float32),
        entropy=entropy.numpy().astype(np.float32),
    )
    print("saved", OUT)


if __name__ == "__main__":
    main()
