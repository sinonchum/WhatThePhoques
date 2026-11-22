"""What class sits under the memorized spike?

Top-2 stability under tiny noise, block-9 centroid margin, and a short PGD flip.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

STARTER = Path(__file__).resolve().parent / "starter"
sys.path.insert(0, str(STARTER))
import task_template as tt  # noqa: E402

OUT = Path(__file__).resolve().parent / "artifacts"


def to_model(img_uint8):
    return tt.preprocess(img_uint8)


@torch.no_grad()
def logits_of(model, x):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        return model(x).float()


def main():
    model = tt.load_model(STARTER / "model.pt").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    images = tt.load_forget_images(STARTER / "forget_images.npz")
    N = len(images)

    # --- top2 stability under tiny gaussian noise and hflip ---
    clean_top2 = []
    noise_top2 = []
    flip_top2 = []
    clean_gap = []
    bs = 64
    for i in range(0, N, bs):
        x = to_model(images[i:i + bs])
        z = logits_of(model, x)
        top = z.topk(2, dim=-1)
        clean_top2.append(top.indices.cpu())
        clean_gap.append((top.values[:, 0] - top.values[:, 1]).cpu())
        noise = (torch.randn_like(x) * (2 / 255)).clamp(-1, 1)  # will reclamp x
        zn = logits_of(model, (x + torch.randn_like(x) * (4 / 255)).clamp(-1, 1))
        noise_top2.append(zn.topk(2, dim=-1).indices.cpu())
        zf = logits_of(model, x.flip(-1))
        flip_top2.append(zf.topk(2, dim=-1).indices.cpu())
    clean_top2 = torch.cat(clean_top2)
    noise_top2 = torch.cat(noise_top2)
    flip_top2 = torch.cat(flip_top2)
    clean_gap = torch.cat(clean_gap)
    y = clean_top2[:, 0]
    y2 = clean_top2[:, 1]
    print(f"logit gap top1-top2: mean {clean_gap.mean():.2f} p10 {clean_gap.quantile(0.1):.2f} p50 {clean_gap.median():.2f} p90 {clean_gap.quantile(0.9):.2f}")
    print(f"top1 stable under noise σ=4/255: {(noise_top2[:,0]==y).float().mean():.3f}")
    print(f"top2 stable under noise: {(noise_top2[:,1]==y2).float().mean():.3f}")
    print(f"top1 stable under hflip: {(flip_top2[:,0]==y).float().mean():.3f}")
    print(f"top2 stable under hflip: {(flip_top2[:,1]==y2).float().mean():.3f}")
    print(f"hflip top1 equals clean top2: {(flip_top2[:,0]==y2).float().mean():.3f}")

    # --- block 9 CLS centroid margin ---
    feats = []
    with torch.no_grad():
        for i in range(0, N, bs):
            x = to_model(images[i:i + bs])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                h = model.patch_embed(x)
                h = model._pos_embed(h)
                h = model.patch_drop(h)
                h = model.norm_pre(h)
                for bi, blk in enumerate(model.blocks):
                    h = blk(h)
                    if bi == 8:
                        feats.append(h[:, 0].float().cpu())
                        break
    feat = F.normalize(torch.cat(feats), dim=-1)
    uniq = y.unique()
    cent = torch.stack([F.normalize(feat[y == int(c)].mean(0), dim=0) for c in uniq.tolist()])
    sim = feat @ cent.T
    own_i = torch.tensor([int((uniq == int(c)).nonzero()[0]) for c in y.tolist()])
    # LOO adjust own similarity roughly: subtract self from sum
    for ci, c in enumerate(uniq.tolist()):
        m = y == c
        n = int(m.sum())
        s = feat[m].sum(0)
        loo = F.normalize(s - feat[m], dim=-1)
        sim[m, ci] = (feat[m] * loo).sum(-1)
    own_sim = sim.gather(1, own_i[:, None]).squeeze(1)
    sim2 = sim.clone()
    sim2.scatter_(1, own_i[:, None], -2)
    other_sim, other_j = sim2.max(dim=1)
    other_c = uniq[other_j]
    margin = own_sim - other_sim
    print(f"block8 LOO centroid margin own-other: mean {margin.mean():.3f} p10 {margin.quantile(0.1):.3f} p50 {margin.median():.3f}")
    print(f"fraction other centroid closer: {(margin<0).float().mean():.3f}")
    print(f"fraction margin < -0.05: {(margin<-0.05).float().mean():.3f}  < -0.02 {(margin<-0.02).float().mean():.3f}")

    # --- PGD flip: minimize z[y], see when argmax changes ---
    # report at eps in normalized units corresponding to 2, 4, 8, 16 px/255
    # one run at eps=16/255 with step records
    eps_list = [2 / 255, 4 / 255, 8 / 255, 16 / 255, 32 / 255]
    # store flip class at the smallest eps where it flipped;  -1 if never
    flipped_at = torch.full((N,), 999, dtype=torch.int16)
    flip_class = torch.full((N,), -1, dtype=torch.int16)
    flip_conf = torch.zeros(N)
    steps = 8
    for i in range(0, N, bs):
        x0 = to_model(images[i:i + bs])
        yy = y[i:i + bs].to(x0.device)
        # start from x0, ascend loss of true pseudo-label == descend logit y, bounded
        delta = torch.zeros_like(x0)
        # use the largest eps and record intermediate
        max_eps = eps_list[-1]
        records = []
        for t in range(steps):
            delta.requires_grad_(True)
            z = model((x0 + delta).clamp(-1, 1)).float()
            # loss = logit of memorized class (we descend it)
            loss = z.gather(1, yy[:, None]).mean()
            g = torch.autograd.grad(loss, delta)[0]
            delta = (delta - (max_eps / 2) * g.sign()).detach()
            delta = delta.clamp(-max_eps, max_eps)
            with torch.no_grad():
                z = model((x0 + delta).clamp(-1, 1)).float()
                pred = z.argmax(1)
                conf = torch.softmax(z, dim=-1).max(1).values
                records.append((pred.cpu(), conf.cpu(), delta.detach().abs().amax(dim=(1, 2, 3)).cpu()))
        # assign smallest eps
        for local, (pred, conf, linf) in enumerate(zip(*[torch.stack([r[k] for r in records], 1) for k in range(3)])):
            # pred: steps
            pass
        preds = torch.stack([r[0] for r in records], 1)  # B, steps
        confs = torch.stack([r[1] for r in records], 1)
        linfs = torch.stack([r[2] for r in records], 1)
        changed = preds != yy.cpu()[:, None]
        for b in range(x0.shape[0]):
            idx = i + b
            if changed[b].any():
                t = int(changed[b].nonzero()[0])
                flipped_at[idx] = t
                flip_class[idx] = int(preds[b, t])
                flip_conf[idx] = float(confs[b, t])
        if i % 512 == 0:
            print(f"pgd {i}/{N}", flush=True)

    never = (flip_class < 0).float().mean()
    print(f"never flipped in {steps} steps eps={eps_list[-1]:.4f}: {never:.3f}")
    print(f"flip class == top2: {(flip_class.long()==y2).float().mean():.3f} among flipped {((flip_class.long()==y2)&(flip_class>=0)).float().sum() / (flip_class>=0).float().sum():.3f}")
    print(f"flip conf mean {flip_conf[flip_class>=0].mean():.3f}")
    # confidence of new class is low or high?
    print(f"flip conf > 0.5: {(flip_conf[flip_class>=0] > 0.5).float().mean():.3f}  >0.8 {(flip_conf[flip_class>=0] > 0.8).float().mean():.3f}")
    print("step of first flip histogram", torch.bincount(flipped_at.clamp(max=steps).long(), minlength=steps+1).tolist())

    torch.save({
        "y": y.to(torch.int16),
        "y2": y2.to(torch.int16),
        "gap": clean_gap,
        "margin_b8": margin,
        "other_c": other_c.to(torch.int16),
        "flip_class": flip_class,
        "flip_conf": flip_conf,
        "flipped_at": flipped_at,
    }, OUT / "flip.pt")
    print("saved flip.pt")


if __name__ == "__main__":
    main()
