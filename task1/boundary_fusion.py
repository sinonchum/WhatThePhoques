"""Frozen semantic tokens + native-decoder evidence + forensic cues. Learn a method-conditioned boundary and residual-watermark head."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_grid import METHODS, read, split_ids
from train_cue_net import data
from ensemble_views import mask_for, model_score


def block(i, o, k=3):
    return nn.Sequential(
        nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.GroupNorm(4, o), nn.SiLU()
    )


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def edges(y):
    return (F.max_pool2d(y, 3, 1, 1) + F.max_pool2d(-y, 3, 1, 1)).clamp(0, 1)


class BoundaryFusion(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        w = 32
        self.a = nn.Sequential(block(c + 3, w, 1), block(w, w))
        self.s = nn.Sequential(block(d, w, 1), block(w, w))
        self.trust = nn.Conv2d(2 * w, w, 1)
        self.mix = block(2 * w, w)
        self.b = nn.Sequential(block(w, 48), block(48, 48))
        self.c = nn.Sequential(block(48, 64), block(64, 64))
        self.d = nn.Sequential(block(64, 80), block(80, 80))
        self.gate = nn.Sequential(
            nn.Linear(160 + 3, 48), nn.SiLU(), nn.Dropout(0.15), nn.Linear(48, 1)
        )
        self.agg = nn.Sequential(block(w + 48 + 64 + 80, 48, 1), block(48, 32))
        self.edge = nn.Conv2d(32, 4, 1)
        self.out = nn.Conv2d(33, 8, 1)

    def forward(self, x, t, m):
        h = x.shape[-2:]
        tag = F.one_hot(m, 3).float()
        a = self.a(torch.cat([x, tag[:, :, None, None].expand(-1, -1, *h)], 1))
        s = resize(self.s(t), h)
        a = self.mix(torch.cat([a, s * self.trust(torch.cat([a, s], 1)).sigmoid()], 1))
        pool = lambda z: F.avg_pool2d(z, 2)
        b = self.b(pool(a))
        c = self.c(pool(b))
        d = self.d(pool(c))
        gate = self.gate(torch.cat([d.mean((2, 3)), d.amax((2, 3)), tag], 1)).flatten()
        v = self.agg(torch.cat([a, resize(b, h), resize(c, h), resize(d, h)], 1))
        e = F.pixel_shuffle(self.edge(v), 2)
        mask = F.pixel_shuffle(self.out(torch.cat([v, pool(e.sigmoid())], 1)), 2)
        return gate, mask, e


def token_data(root, split, rows):
    out = np.empty((len(rows), 128, 32, 32), dtype=np.float16)
    deadline = time.monotonic() + 1800
    for i, r in enumerate(rows):
        p = root / f"dino-patch-v1-{split}" / (r["image_name"] + ".npy")
        while not p.is_file():
            if time.monotonic() > deadline:
                raise TimeoutError(p)
            time.sleep(2)
        out[i] = np.load(p, allow_pickle=False)
    return torch.from_numpy(out)


def run(root, epochs, seed, semantic):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, rows, ms, y = data(root, "train", True, True)
    assert y is not None
    t = (
        token_data(root, "train", rows)
        if semantic
        else torch.zeros(1, 128, 32, 32, dtype=torch.float16).expand(
            len(rows), -1, -1, -1
        )
    )
    tr, d, h = split_ids(rows)
    full = torch.tensor([r["full_watermark"] for r in rows]).float()
    sample = x[tr[::3], :, ::4, ::4].float()
    mu = sample.mean((0, 2, 3))[None, :, None, None].to(device)
    sd = sample.std((0, 2, 3))[None, :, None, None].clamp_min(0.005).to(device)
    net = BoundaryFusion(x.shape[1], 128).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0008, weight_decay=0.005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=0.00008)
    out = root / f"boundary-v1-{seed}-{epochs}-s{int(semantic)}"
    out.mkdir(exist_ok=True)
    best = {m: -1 for m in METHODS}
    rules = {}
    print(
        "START",
        device,
        "semantic",
        semantic,
        "train/dev/hold",
        len(tr),
        len(d),
        len(h),
        "trainable",
        sum(p.numel() for p in net.parameters() if p.requires_grad),
        flush=True,
    )

    def inputs(x, t, ms, idx):
        return (
            ((x[idx].to(device).float() - mu) / sd).clamp(-7, 7),
            t[idx].to(device).float(),
            ms[idx].to(device),
        )

    @torch.inference_mode()
    def preds(x, t, ms, idx):
        net.eval()
        gs = []
        qs = []
        for k in range(0, len(idx), 8):
            a, b, m = inputs(x, t, ms, idx[k : k + 8])
            g, q, _ = net(a, b, m)
            gs.append(g.sigmoid().cpu())
            qs.append(q.sigmoid().cpu())
        return torch.cat(gs), torch.cat(qs)

    for ep in range(1, epochs + 1):
        net.train()
        order = rng.permutation(tr)
        losses = []
        for j in range(0, len(order), 8):
            idx = order[j : j + 8].tolist()
            a, b, m = inputs(x, t, ms, idx)
            yy = y[idx].to(device)
            f = full[idx].to(device)
            for axis in (-2, -1):
                if rng.random() < 0.5:
                    a, b, yy = (z.flip((axis,)) for z in (a, b, yy))
            gate, regions, edge = net(a, b, m)
            f4 = f[:, None, None, None]
            rare = yy * (1 - f4) + (1 - yy) * f4
            z = regions[:, :1] * (1 - f4) + regions[:, 1:] * f4
            prob = z.sigmoid()
            dims = (2, 3)
            dice = (2 * (prob * rare).sum(dims) + 1) / (
                prob.sum(dims) + rare.sum(dims) + 1
            )
            bo = edges(rare)
            weight = (1 - bo.mean()) / bo.mean().clamp_min(0.015)
            loss = (
                F.binary_cross_entropy_with_logits(z, rare)
                + (1 - dice.mean())
                + 0.5 * F.binary_cross_entropy_with_logits(gate, f)
                + 0.2
                * F.binary_cross_entropy_with_logits(
                    edge, bo, pos_weight=weight.clamp(1, 12)
                )
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2)
            opt.step()
            losses.append(loss.detach())
        sched.step()
        if ep == 1 or ep % 4 == 0 or ep == epochs:
            g, q = preds(x, t, ms, d)
            for k, m in enumerate(METHODS):
                take = ms[d] == k
                r = max(
                    (
                        model_score(
                            mask_for(g[take], q[take], r, th), y[d, 0][take] > 0.5
                        ),
                        r,
                        th,
                    )
                    for r in ("hard", "soft")
                    for th in (0.35, 0.5, 0.65)
                )
                if r[0] > best[m]:
                    best[m] = r[0]
                    rules[m] = (r[1], r[2], ep)
                    torch.save(net.state_dict(), out / (m + ".pth"))
            print(
                "EPOCH", ep, float(torch.stack(losses).mean().cpu()), best, flush=True
            )
    reports = {}
    for k, m in enumerate(METHODS):
        net.load_state_dict(
            torch.load(out / (m + ".pth"), map_location="cpu", weights_only=True)
        )
        idx = [i for i in h if rows[i]["method"] == m]
        g, q = preds(x, t, ms, idx)
        r, th, ep = rules[m]
        p = mask_for(g, q, r, th)
        reports[m] = {
            "method": m,
            "dev": best[m],
            "hold": model_score(p, y[idx, 0] > 0.5),
            "n": len(idx),
            "rule": r,
            "threshold": th,
            "epoch": ep,
        }
        np.savez_compressed(out / f"{m}-hold.npz", ids=idx, masks=p.numpy())
        print("HOLD", reports[m], flush=True)
    del x, t
    x, rows, ms, _ = data(root, "test", True, True)
    t = (
        token_data(root, "test", rows)
        if semantic
        else torch.zeros(1, 128, 32, 32, dtype=torch.float16).expand(
            len(rows), -1, -1, -1
        )
    )
    for m in METHODS:
        net.load_state_dict(
            torch.load(out / (m + ".pth"), map_location="cpu", weights_only=True)
        )
        idx = [i for i, r in enumerate(rows) if r["method"] == m]
        g, q = preds(x, t, ms, idx)
        r, th, _ = rules[m]
        p = mask_for(g, q, r, th).numpy()
        assert p.dtype == bool and p.shape == (340, 128, 128)
        tmp = out / f"{m}.partial.npz"
        np.savez_compressed(tmp, ids=idx, masks=p)
        tmp.replace(out / f"{m}-test.npz")
        (out / f"{m}.json").write_text(json.dumps(reports[m]))
        print("TEST", m, len(idx), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=2357)
    p.add_argument("--no-semantic", action="store_true")
    a = p.parse_args()
    run(a.root, a.epochs, a.seed, not a.no_semantic)
