"""Multi-input shared segmenter: predict polarity and minority-region mask separately."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_grid import METHODS, layer, read, split_ids


def data(root, split):
    x, rows, g = read(root, split)
    tmp = []
    deadline = time.monotonic() + 1800
    for i, r in enumerate(rows):
        cam = root / f"cam-v1-{split}-{r['method']}" / (r["image_name"] + ".npy")
        sem = root / f"semantic-v1-{split}" / (r["image_name"] + ".npy")
        while not (cam.exists() and sem.exists()):
            if time.monotonic() > deadline:
                raise TimeoutError(str(cam))
            time.sleep(3)
        c = np.load(cam)
        s = np.load(sem)
        extra = np.zeros((24, 64, 64), np.float16)
        extra[: len(c)] = c
        tmp.append(np.concatenate((x[i, ::3], extra, s)))
    methods = torch.tensor([METHODS.index(r["method"]) for r in rows])
    return torch.tensor(np.stack(tmp)), rows, methods, g


class Region(nn.Module):
    def __init__(self, c, w=24):
        super().__init__()
        self.inp = layer(c + 3, w, 1)
        self.e1 = layer(w, w)
        self.e2 = nn.Sequential(layer(w, 2 * w), layer(2 * w, 2 * w))
        self.e3 = nn.Sequential(layer(2 * w, 3 * w), layer(3 * w, 3 * w))
        self.e4 = nn.Sequential(layer(3 * w, 3 * w), layer(3 * w, 3 * w))
        self.d3 = layer(6 * w, 3 * w)
        self.d2 = layer(5 * w, 2 * w)
        self.d1 = layer(3 * w, w)
        self.out = nn.Conv2d(w, 2, 1)
        self.gate = nn.Sequential(
            nn.Linear(3 * w, 32), nn.SiLU(), nn.Dropout(0.18), nn.Linear(32, 1)
        )

    def forward(self, x, k):
        p = lambda a: F.avg_pool2d(a, 2)
        u = lambda a: F.interpolate(
            a, scale_factor=2, mode="bilinear", align_corners=False
        )
        tag = F.one_hot(k, 3).float()[:, :, None, None].expand(-1, -1, 64, 64)
        a = self.e1(self.inp(torch.cat((x, tag), 1)))
        b = self.e2(p(a))
        c = self.e3(p(b))
        d = self.e4(p(c))
        g = self.gate(d.mean((2, 3)))[:, :, None, None]
        o = self.d3(torch.cat((c, u(d)), 1))
        o = self.d2(torch.cat((b, u(o)), 1))
        o = self.d1(torch.cat((a, u(o)), 1))
        return g, u(self.out(o))


def run(root, epochs, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "mps"
    x, rows, ms, y = data(root, "train")
    tr, d, h = split_ids(rows)
    assert y is not None
    gt = torch.tensor([float(r["full_watermark"]) for r in rows])[:, None, None, None]
    sample = x[tr[::3], :, ::4, ::4].float()
    mu = sample.mean((0, 2, 3))[None, :, None, None].to(dev)
    sd = sample.std((0, 2, 3))[None, :, None, None].clamp_min(0.005).to(dev)
    net = Region(x.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=0.0001
    )
    out = root / (
        "neural-message-v1"
        if (epochs, seed) == (40, 203)
        else f"neural-message-v1-{seed}-{epochs}"
    )
    out.mkdir(exist_ok=True)

    def norm(x):
        return ((x.to(dev).float() - mu) / sd).clamp(-7, 7)

    def probs(g, log):
        z = g.sigmoid()
        return (1 - z) * log[:, :1].sigmoid() + z * (1 - log[:, 1:].sigmoid())

    @torch.inference_mode()
    def predict(xx, mm, ids):
        net.eval()
        acc = []
        for k in range(0, len(ids), 12):
            u = ids[k : k + 12]
            g, l = net(norm(xx[u]), mm[u].to(dev))
            acc.append(probs(g, l).cpu()[:, 0])
        return torch.cat(acc)

    def score(mask, labels):
        val = 0
        for a, b in ((mask, labels), (~mask, ~labels)):
            union = (a | b).sum((1, 2))
            val = val + 0.5 * torch.where(
                union > 0, (a & b).sum((1, 2)) / union.clamp_min(1), 1.0
            )
        return float(val.mean())

    best = {m: -1 for m in METHODS}
    settings = {}
    for ep in range(1, epochs + 1):
        net.train()
        order = rng.permutation(tr)
        for k in range(0, len(order), 12):
            ids = order[k : k + 12].tolist()
            xx = norm(x[ids])
            yy = y[ids].to(dev)
            ff = gt[ids].to(dev)
            mm = ms[ids].to(dev)
            for axis in (-2, -1):
                if rng.random() < 0.5:
                    xx = xx.flip((axis,))
                    yy = yy.flip((axis,))
            g, l = net(xx, mm)
            rare = (1 - ff) * yy + ff * (1 - yy)
            log = (1 - ff) * l[:, :1] + ff * l[:, 1:]
            p = log.sigmoid()
            dice = (2 * (p * rare).sum((2, 3)) + 1) / (
                p.sum((2, 3)) + rare.sum((2, 3)) + 1
            )
            loss = (
                0.45 * F.binary_cross_entropy_with_logits(g, ff)
                + F.binary_cross_entropy_with_logits(log, rare)
                + 0.95 * (1 - dice.mean())
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        schedule.step()
        if ep % 4 == 0 or ep == epochs:
            val = predict(x, ms, d)
            for j, m in enumerate(METHODS):
                mask = ms[d] == j
                v = val[mask]
                lab = y[d, 0][mask] > 0.5
                vsc, t = max((score(v > t, lab), t) for t in (0.35, 0.5, 0.65))
                if vsc > best[m]:
                    best[m] = vsc
                    settings[m] = (t, ep)
                    torch.save(net.state_dict(), out / (m + ".pth"))
            print(ep, best, flush=True)

    def select(m):
        net.load_state_dict(
            torch.load(out / (m + ".pth"), map_location=dev, weights_only=True)
        )

    for j, m in enumerate(METHODS):
        select(m)
        t, ep = settings[m]
        ids = [i for i in h if ms[i].item() == j]
        pred = predict(x, ms, ids) > t
        sc = score(pred, y[ids, 0] > 0.5)
        r = {
            "method": m,
            "dev": best[m],
            "hold": sc,
            "threshold": t,
            "epoch": ep,
            "n": len(ids),
        }
        (out / (m + ".json")).write_text(json.dumps(r))
        np.savez_compressed(out / (m + "-hold.npz"), ids=ids, masks=pred.numpy())
        print("RESULT", r, flush=True)
    del x
    xx, rr, mm, _ = data(root, "test")
    for j, m in enumerate(METHODS):
        select(m)
        ids = [i for i, r in enumerate(rr) if r["method"] == m]
        p = predict(xx, mm, ids) > settings[m][0]
        np.savez_compressed(out / (m + "-test.npz"), ids=ids, masks=p.numpy())
        print("TEST", m, len(ids), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=203)
    a = p.parse_args()
    run(a.root, a.epochs, a.seed)
