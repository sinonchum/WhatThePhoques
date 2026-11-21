"""Is the intermediate ViT readout a sharp class, or noise?"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

STARTER = Path(__file__).resolve().parent / "starter"
sys.path.insert(0, str(STARTER))
import task_template as tt  # noqa: E402

OUT = Path(__file__).resolve().parent / "artifacts"


@torch.no_grad()
def lens_logits(model, x):
    h = model.patch_embed(x)
    h = model._pos_embed(h)
    h = model.patch_drop(h)
    h = model.norm_pre(h)
    logits = []
    for blk in model.blocks:
        h = blk(h)
        cls = model.fc_norm(model.norm(h[:, 0]))
        logits.append(model.head(cls))
    return torch.stack(logits, dim=1)  # B, L, C


def main():
    model = tt.load_model(STARTER / "model.pt")
    images = tt.load_forget_images(STARTER / "forget_images.npz")
    model.eval()

    chunks = []
    bs = 64
    for i in range(0, len(images), bs):
        x = tt.preprocess(images[i:i + bs])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            chunks.append(lens_logits(model, x).float().cpu())
        if i % 512 == 0:
            print(f"forward {i}/{len(images)}", flush=True)
    logits = torch.cat(chunks)  # N, L, C
    prob = torch.softmax(logits, dim=-1)
    top2v, top2i = prob.topk(2, dim=-1)
    pred = top2i[..., 0]
    top1 = top2v[..., 0]
    margin = top1 - top2v[..., 1]
    ent = -(prob.clamp_min(1e-8).log() * prob).sum(-1)
    final = pred[:, -1]
    L = pred.shape[1]
    print(f"N={len(images)} blocks={L}")
    print(f"{'blk':>4} {'agree_final':>11} {'agree_prev':>10} {'top1':>8} {'margin':>8} {'H':>8} {'top1|disagree':>14}")
    for b in range(L):
        agree = (pred[:, b] == final).float().mean().item()
        prev = 1.0 if b == 0 else (pred[:, b] == pred[:, b - 1]).float().mean().item()
        dis = pred[:, b] != final
        top1_dis = top1[dis, b].mean().item() if dis.any() else float("nan")
        print(f"{b:4d} {agree:11.3f} {prev:10.3f} {top1[:, b].mean():8.3f} {margin[:, b].mean():8.3f} {ent[:, b].mean():8.3f} {top1_dis:14.3f}")

    # Where does the class settle, relative to the final label?
    # last block that still disagrees with final
    disagree = pred != final[:, None]
    # stable class: mode of blocks 4,5,6,7
    early = pred[:, 4:8]
    # mode
    mode = []
    mode_frac = []
    for n in range(len(images)):
        vals, cnts = torch.unique(early[n], return_counts=True)
        j = cnts.argmax()
        mode.append(vals[j])
        mode_frac.append(cnts[j].float() / early.shape[1])
    mode = torch.stack(mode)
    mode_frac = torch.stack(mode_frac)
    print(f"mode of blocks 4-7 == final: {(mode == final).float().mean():.3f}")
    print(f"mode frac mean {mode_frac.mean():.3f}  frac mode_frac==1 {(mode_frac==1).float().mean():.3f}")
    print(f"when mode!=final, mode frac {mode_frac[mode!=final].mean():.3f} n={(mode!=final).sum().item()}")

    # confidence of that mode averaged over blocks 4-7 (mean prob of the modal class)
    # rebuild
    modal_prob = []
    for b in range(4, 8):
        modal_prob.append(prob[:, b].gather(1, mode[:, None]).squeeze(1))
    modal_prob = torch.stack(modal_prob, 1).mean(1)
    print(f"mean prob of blocks4-7 mode, averaged over those blocks: {modal_prob.mean():.3f}")
    print(f"  when mode!=final: {modal_prob[mode!=final].mean():.3f}")
    print(f"  when mode==final: {modal_prob[mode==final].mean():.3f}")

    torch.save(
        {
            "pred": pred.half() if False else pred.to(torch.int16),
            "top1": top1.to(torch.float16),
            "final": final.to(torch.int16),
            "mode47": mode.to(torch.int16),
        },
        OUT / "lens.pt",
    )
    print("saved lens.pt")


if __name__ == "__main__":
    main()
