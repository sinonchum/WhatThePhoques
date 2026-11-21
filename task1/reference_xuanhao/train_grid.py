"""Supervised fixed-spectrum U-Net. Writes local checkpoints and competition-format predictions only."""

import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("pixelseal", "trustmark", "stegastamp")


def read(root, split):
    x = np.load(root / f"forensic-v1-{split}.npy", mmap_mode="r")
    with zipfile.ZipFile(root / f"{split}.zip") as z:
        rows = json.loads(z.read(f"{split}/metadata.json"))
        y = None
        if split == "train":
            with np.load(
                io.BytesIO(z.read("train/masks_128.npz")), allow_pickle=False
            ) as f:
                masks = dict(zip(f["names"].tolist(), f["masks"]))
                y = torch.tensor(
                    np.stack([masks[r["image_name"]] for r in rows]),
                    dtype=torch.float32,
                )[:, None]
    assert len(rows) == len(x)
    return x, rows, y


def split_ids(rows):
    rank = lambda n: hashlib.sha256(("paris-residual-v1:" + n).encode()).digest()
    tr = []
    dev = []
    hold = []
    for m in METHODS:
        for full in (True, False):
            ids = sorted(
                (
                    i
                    for i, r in enumerate(rows)
                    if r["method"] == m and r["full_watermark"] == full
                ),
                key=lambda i: rank(rows[i]["image_name"]),
            )
            hold.extend(ids[:17])
            dev.extend(ids[17:25])
            tr.extend(ids[25:])
    assert (len(tr), len(dev), len(hold)) == (360, 48, 102)
    return tr, dev, hold


def layer(a, b, k=3):
    return nn.Sequential(
        nn.Conv2d(a, b, k, padding=k // 2, bias=False), nn.BatchNorm2d(b), nn.SiLU()
    )


class Net(nn.Module):
    def __init__(self, c, w):
        super().__init__()
        self.a = nn.Sequential(layer(c, w, 1), layer(w, w))
        self.b = nn.Sequential(layer(w, 2 * w), layer(2 * w, 2 * w))
        self.c = nn.Sequential(layer(2 * w, 3 * w), layer(3 * w, 3 * w))
        self.d = nn.Sequential(layer(3 * w, 4 * w), layer(4 * w, 4 * w))
        self.e = layer(7 * w, 3 * w)
        self.f = layer(5 * w, 2 * w)
        self.g = layer(3 * w, w)
        self.out = nn.Conv2d(w, 3, 1)

    def forward(self, x, m):
        def up(t):
            return F.interpolate(
                t, scale_factor=2, mode="bilinear", align_corners=False
            )

        pool = lambda t: F.avg_pool2d(t, 2)
        a = self.a(x)
        b = self.b(pool(a))
        c = self.c(pool(b))
        d = self.d(pool(c))
        h = self.e(torch.cat([c, up(d)], 1))
        h = self.f(torch.cat([b, up(h)], 1))
        h = self.g(torch.cat([a, up(h)], 1))
        logits = self.out(h).gather(1, m[:, None, None, None].expand(-1, 1, 64, 64))
        return up(logits)


def main(args):
    root = args.root
    x, rows, y = read(root, "train")
    train, dev, hold = split_ids(rows)
    method = torch.tensor([METHODS.index(r["method"]) for r in rows])
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = root / f"net-{args.seed}-{args.width}"
    out.mkdir(exist_ok=True)
    if args.full:
        train = list(range(len(rows)))
    sample = np.asarray(x[train[::4], :, ::4, ::4], np.float32)
    mu = torch.tensor(sample.mean((0, 2, 3)), device=device)[None, :, None, None]
    sd = torch.tensor(sample.std((0, 2, 3)).clip(0.005), device=device)[
        None, :, None, None
    ]

    def batch(data, ind):
        return (
            (torch.from_numpy(np.array(data[ind], dtype=np.float32)).to(device) - mu)
            / sd
        ).clamp(-10, 10)

    net = Net(x.shape[1], args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0018, weight_decay=0.0004)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=0.00012
    )

    def predict(data, methods, ids):
        net.eval()
        p = []
        with torch.no_grad():
            for i in range(0, len(ids), args.batch):
                ix = ids[i : i + args.batch]
                p.append(
                    torch.sigmoid(net(batch(data, ix), methods[ix].to(device))).cpu()[
                        :, 0
                    ]
                )
        return torch.cat(p)

    def scores(probs, ids):
        a = probs > 0.5
        b = y[ids, 0] > 0.5
        sc = 0.0
        for v, t in ((a, b), (~a, ~b)):
            union = (v | t).sum((1, 2))
            sc = (
                sc
                + torch.where(union > 0, (v & t).sum((1, 2)) / union.clamp_min(1), 1.0)
                * 0.5
            )
        return {
            "overall": float(sc.mean()),
            **{m: float(sc[method[ids] == k].mean()) for k, m in enumerate(METHODS)},
        }

    best = -1.0
    ck = out / "development.pth"
    epochbest = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(train)
        for i in range(0, len(order), args.batch):
            ids = order[i : i + args.batch].tolist()
            xx = batch(x, ids)
            yy = y[ids].to(device)
            mm = method[ids].to(device)
            for axis in (-1, -2):
                if rng.random() < 0.5:
                    xx = xx.flip((axis,))
                    yy = yy.flip((axis,))
            logit = net(xx, mm)
            pp = logit.sigmoid()
            dice = [
                (2 * (v * t).mean((2, 3)) + 0.003)
                / (v.mean((2, 3)) + t.mean((2, 3)) + 0.003)
                for v, t in ((pp, yy), (1 - pp, 1 - yy))
            ]
            loss = F.binary_cross_entropy_with_logits(logit, yy) + 0.65 * (
                1 - (dice[0] + dice[1]).mean() / 2
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        sched.step()
        if not args.full and (epoch % 2 == 0 or epoch == args.epochs):
            sc = scores(predict(x, method, dev), dev)
            print("dev", epoch, sc, flush=True)
            if sc["overall"] > best:
                best = sc["overall"]
                epochbest = epoch
                torch.save(net.state_dict(), ck)
        elif epoch % 4 == 0:
            print("full epoch", epoch, flush=True)
    if not args.full:
        net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
        pred = predict(x, method, hold)
        res = {
            "epoch": epochbest,
            "dev": best,
            "hold": scores(pred, hold),
            "hold_n": len(hold),
        }
        print("RESULT", res, flush=True)
        (out / "result.json").write_text(json.dumps(res))
        np.save(out / "hold.npy", pred.numpy().astype(np.float16))
        return
    torch.save(
        {
            "weights": net.state_dict(),
            "mu": mu.cpu(),
            "sd": sd.cpu(),
            "width": args.width,
        },
        out / "full.pth",
    )
    tx, rr, _ = read(root, "test")
    m = torch.tensor([METHODS.index(r["method"]) for r in rr])
    pr = predict(tx, m, list(range(len(rr))))
    assert torch.isfinite(pr).all()
    names = [r["image_name"] for r in rr]
    mask = (pr > 0.5).to(torch.uint8)
    assert len(names) == len(set(names)) == 1020 and mask.shape == (1020, 128, 128)
    file = out / "submission.pt"
    torch.save({"image_names": names, "masks": mask}, file)
    assert file.stat().st_size < 20_000_000
    v = torch.load(file, weights_only=True)
    assert torch.equal(v["masks"], mask) and v["image_names"] == names
    print("VALID SUBMISSION", file, file.stat().st_size, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--seed", type=int, default=173)
    p.add_argument("--device", default="mps")
    p.add_argument("--full", action="store_true")
    main(p.parse_args())
