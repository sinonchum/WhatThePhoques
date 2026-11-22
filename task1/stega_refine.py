"""Calibrate a frozen, published StegaStamp model with native known-message attribution maps.
The zero correction is a first-class dev candidate, and selection never reads holdout/test labels.
"""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_cue_net import Region
from train_grid import read, split_ids, METHODS
from ensemble_views import feature, model_score


class Correction(nn.Module):
    def __init__(self, cin, w=24):
        super().__init__()
        block = lambda a, b, d: nn.Sequential(
            nn.Conv2d(a, b, 3, padding=d, dilation=d, bias=False),
            nn.GroupNorm(4, b),
            nn.SiLU(),
        )
        self.stem = block(cin + 2, w, 1)
        self.context = nn.Sequential(block(w, w, 1), block(w, w, 2), block(w, w, 3))
        self.out = nn.Conv2d(w * 2, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, base, gate):
        low = F.avg_pool2d(base, 2)
        a = self.stem(torch.cat((x, low, gate.expand_as(low)), 1))
        b = self.context(F.avg_pool2d(a, 2))
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        delta = F.interpolate(
            self.out(torch.cat([a, b], 1)),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return base + 5 * torch.tanh(delta / 5)


def run(root, seed, epochs, arm):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    gb, rows, y = read(root, "train")
    tr, d, h = split_ids(rows)
    ti, di, hi = (
        [i for i in ids if rows[i]["method"] == "stegastamp"] for ids in (tr, d, h)
    )
    out = root / f"stega-native-refine-v1-{arm}-{seed}-{epochs}"
    out.mkdir(exist_ok=True)
    modelroot = root / "neural-message-v1-1451-120-r1t1"
    stats = torch.from_numpy(
        np.stack([feature(root, "train", gb, rows, i, 2)[:, ::4, ::4] for i in tr[::3]])
    ).float()
    mu = stats.mean((0, 2, 3))[None, :, None, None].to(device)
    sd = stats.std((0, 2, 3))[None, :, None, None].clamp_min(0.005).to(device)
    net = Region(stats.shape[1])
    net.load_state_dict(
        torch.load(modelroot / "stegastamp.pth", map_location="cpu", weights_only=True)
    )
    net.eval().to(device)
    tag = METHODS.index("stegastamp")

    @torch.no_grad()
    def get(split, block, rr, ids):
        gates = []
        base = []
        cams = []
        deadline = time.monotonic() + 7200
        for j in range(0, len(ids), 8):
            ix = ids[j : j + 8]
            x = (
                torch.from_numpy(
                    np.stack([feature(root, split, block, rr, i, 2) for i in ix])
                )
                .to(device)
                .float()
            )
            g, l = net(
                ((x - mu) / sd).clamp(-7, 7), torch.full((len(ix),), tag, device=device)
            )
            p = (1 - g.sigmoid()) * l[:, :1].sigmoid() + g.sigmoid() * (
                1 - l[:, 1:].sigmoid()
            )
            base.append(torch.logit(p.clamp(0.0001, 0.9999)).cpu())
            gates.append(g.sigmoid().cpu())
            for i in ix:
                path = (
                    root
                    / f"stega-native-cam-v1-{split}"
                    / (rr[i]["image_name"] + ".npy")
                )
                while not path.is_file():
                    if time.monotonic() > deadline:
                        raise TimeoutError(path)
                    time.sleep(3)
                a = np.load(path, allow_pickle=False)
                assert a.shape == (36, 64, 64) and np.isfinite(a).all()
                ind = [
                    k
                    for k in range(36)
                    if (arm != "magnitude" or k % 12 in (1, 3, 5, 7, 9))
                    and (arm != "global" or k < 12)
                ]
                cams.append(torch.from_numpy(a[ind]))
        return torch.stack(cams), torch.cat(base), torch.cat(gates)

    trm, b, g = get("train", gb, rows, ti + di + hi)
    nt, nd = len(ti), len(di)
    yv = y[ti + di + hi].float()
    loc_tr = np.arange(nt)
    loc_d = list(range(nt, nt + nd))
    loc_h = list(range(nt + nd, len(trm)))
    sm = trm[:nt, :, ::3, ::3].float()
    mean = sm.mean((0, 2, 3))[None, :, None, None].to(device)
    std = sm.std((0, 2, 3))[None, :, None, None].clamp_min(0.015).to(device)
    n = Correction(trm.shape[1]).to(device)
    opt = torch.optim.AdamW(n.parameters(), lr=0.0007, weight_decay=0.002)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=0.00007)
    ck = out / "stegastamp.pth"

    def inputs(z, bl, gg, ids):
        return (
            ((z[ids].to(device).float() - mean) / std).clamp(-8, 8),
            bl[ids].to(device),
            gg[ids].to(device),
        )

    @torch.inference_mode()
    def pred(z, bl, gg, ids):
        n.eval()
        return torch.cat(
            [
                n(*inputs(z, bl, gg, ids[k : k + 8])).sigmoid().cpu()[:, 0]
                for k in range(0, len(ids), 8)
            ]
        )

    t = 0.5
    best = model_score(b[loc_d, 0] > 0, yv[loc_d, 0] > 0.5)
    epbest = 0
    torch.save(n.state_dict(), ck)
    print("CONTROL", best, flush=True)
    assert (
        abs(best - json.loads((modelroot / "stegastamp.json").read_text())["dev"])
        < 1e-5
    )
    for ep in range(1, epochs + 1):
        n.train()
        order = rng.permutation(loc_tr)
        for k in range(0, nt, 8):
            ids = order[k : k + 8].tolist()
            a, bl, gg = inputs(trm, b, g, ids)
            target = yv[ids].to(device)
            for dim in (-1, -2):
                if rng.random() < 0.5:
                    a = a.flip((dim,))
                    bl = bl.flip((dim,))
                    target = target.flip((dim,))
            log = n(a, bl, gg)
            p = log.sigmoid()
            loss = (
                F.binary_cross_entropy_with_logits(log, target)
                + 0.012 * (log - bl).abs().mean()
            )
            for u, v in [(p, target), (1 - p, 1 - target)]:
                loss = (
                    loss
                    + 0.5
                    * (
                        1
                        - ((u * v).sum((2, 3)) + 1) / ((u + v - u * v).sum((2, 3)) + 1)
                    ).mean()
                )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(n.parameters(), 1)
            opt.step()
        sched.step()
        if ep % 4 == 0 or ep == epochs:
            p = pred(trm, b, g, loc_d)
            sc, th = max(
                (model_score(p > th, yv[loc_d, 0] > 0.5), th) for th in (0.4, 0.5, 0.6)
            )
            if sc > best:
                best, t, epbest = sc, th, ep
                torch.save(n.state_dict(), ck)
            print("DEV", ep, best, epbest, flush=True)
    n.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    hm = pred(trm, b, g, loc_h) > t
    hold = model_score(hm, yv[loc_h, 0] > 0.5)
    info = {
        "method": "stegastamp",
        "dev": best,
        "hold": hold,
        "n": len(hi),
        "threshold": t,
        "epoch": epbest,
        "arm": arm,
        "seed": seed,
    }
    np.savez_compressed(out / "stegastamp-hold.npz", ids=hi, masks=hm.numpy())
    print("HOLD", info, flush=True)
    tb, rr, _ = read(root, "test")
    ids = [i for i, r in enumerate(rr) if r["method"] == "stegastamp"]
    tc, bc, gc = get("test", tb, rr, ids)
    tm = pred(tc, bc, gc, list(range(len(ids)))) > t
    assert tm.shape == (340, 128, 128)
    np.savez_compressed(out / "stegastamp-test.npz", ids=ids, masks=tm.numpy())
    (out / "stegastamp.json").write_text(json.dumps(info))
    print("TEST", len(ids), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--seed", type=int, default=3701)
    p.add_argument("--epochs", type=int, default=56)
    p.add_argument("--arm", choices=["all", "magnitude", "global"], default="all")
    a = p.parse_args()
    run(a.root, a.seed, a.epochs, a.arm)
