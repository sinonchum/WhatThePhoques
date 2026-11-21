"""Train a StegaStamp specialist with pairs differing only by the native writer's residual.
Bank sources, background images and shape templates come ONLY from the fixed fitting split.
No competition labels for test-time polarity; no changes to the pinned teammate bundle.
"""

import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from train_grid import split_ids
from ensemble_views import model_score
from stega_pair import make


def load(root, split):
    with zipfile.ZipFile(root / f"{split}.zip") as z:
        rows = json.loads(z.read(f"{split}/metadata.json"))
        ims = np.stack(
            [
                np.asarray(
                    Image.open(
                        io.BytesIO(z.read(f"{split}/images/{r['image_name']}.png"))
                    ).convert("RGB")
                )
                for r in rows
            ]
        )
        y = None
        if split == "train":
            with np.load(
                io.BytesIO(z.read("train/masks_128.npz")), allow_pickle=False
            ) as f:
                idx = {n: i for i, n in enumerate(f["names"])}
                y = torch.from_numpy(
                    f["masks"][[idx[r["image_name"]] for r in rows]].copy()
                ).bool()
    return rows, torch.from_numpy(ims).permute(0, 3, 1, 2), y


def prepare(root, per_source=3):
    rows, x, y = load(root, "train")
    tr, d, h = split_ids(rows)
    sources = [i for i in tr if rows[i]["method"] != "stegastamp"]
    rng = np.random.default_rng(781)
    sess = make(root)
    out = root / "synth-stega-bank-v1"
    out.mkdir(exist_ok=True)
    for k in range(per_source):
        for j, i in enumerate(sources):
            bits = rng.integers(0, 2, (1, 100)).astype(np.float32)
            dest = out / f"{i}-{k}.npy"
            if dest.exists():
                continue
            im = Image.fromarray(x[i].permute(1, 2, 0).numpy())
            inp = (
                np.asarray(im.resize((400, 400), Image.Resampling.BICUBIC), np.float32)[
                    None
                ]
                / 255
            )
            res = sess.run(["residual"], {"image": inp, "secret": bits})[0][0]
            res = (res - res.mean((0, 1), keepdims=True)) * 255
            assert (
                res.shape == (400, 400, 3)
                and np.isfinite(res).all()
                and np.std(res) > 0.01
            )
            # Quantize to 1/4 intensity level, preserving the subtle native signal.
            a = np.clip(np.rint(res * 4), -32767, 32767).astype(np.int16)
            tmp = dest.with_suffix(".partial.npy")
            np.save(tmp, a)
            tmp.replace(dest)
            if j % 50 == 0:
                print("BANK", k, j, len(sources), flush=True)
    return out


class Specialist(nn.Module):
    def __init__(self, w=32):
        super().__init__()

        def block(a, b, k=3, d=1):
            return nn.Sequential(
                nn.Conv2d(a, b, k, padding=(k // 2) * d, dilation=d, bias=False),
                nn.GroupNorm(4, b),
                nn.SiLU(),
            )

        self.fe = nn.Sequential(block(51, w, 1), block(w, w))
        self.mid = nn.Sequential(block(w, w * 2), block(w * 2, w * 2))
        self.low = nn.Sequential(
            block(w * 2, w * 2), block(w * 2, w * 2, 3, 2), block(w * 2, w * 2, 3, 3)
        )
        self.head = nn.Sequential(block(5 * w, w), nn.Conv2d(w, 1, 1))

    def forward(self, x):
        hp = x - F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, stride=1)
        a = self.fe(
            torch.cat(
                [
                    F.pixel_unshuffle((hp / 5).clamp(-8, 8), 4),
                    F.avg_pool2d(x / 128 - 1, 4),
                ],
                1,
            )
        )
        b = self.mid(F.avg_pool2d(a, 2))
        c = self.low(F.avg_pool2d(b, 2))
        up = lambda v: F.interpolate(
            v, size=a.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.head(torch.cat([a, up(b), up(c)], 1))[:, 0]


def run(args):
    root = args.root
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows, x, y = load(root, "train")
    tr, d, h = split_ids(rows)
    spec = lambda ids: [i for i in ids if rows[i]["method"] == "stegastamp"]
    tr_s, d, h = map(spec, (tr, d, h))
    donors = [i for i in tr if rows[i]["method"] != "stegastamp"]
    bank = root / "synth-stega-bank-v1"
    files = {
        int(f.stem.split("-")[0]): []
        for f in bank.glob("*.npy")
        if ".partial." not in f.name
    }
    for f in sorted(bank.glob("*.npy")):
        if ".partial." not in f.name:
            files[int(f.stem.split("-")[0])].append(f)
    if args.pre:
        assert set(files) == set(donors), (
            "run --prepare first; do not fit on heldout sources"
        )
    net = Specialist().to(device)
    out = root / f"stega-synth-v1-{args.seed}-p{args.pre}-e{args.epochs}"
    out.mkdir(exist_ok=True)
    print(
        "START",
        device,
        len(tr_s),
        len(d),
        len(h),
        "synthetic steps",
        args.pre,
        flush=True,
    )

    def synthetic(n):
        xx = []
        yy = []
        for _ in range(n):
            i = int(rng.choice(donors))
            a = x[i : i + 1].to(device).float()
            path = files[i][rng.integers(len(files[i]))]
            r = (
                torch.from_numpy(np.load(path).astype(np.float32))
                .permute(2, 0, 1)[None]
                .to(device)
                * 0.25
            )
            force_patch = rng.random() < 0.4
            if force_patch:
                ww = int(rng.integers(4, 116))
                hh = int(rng.integers(max(2, ww // 16), min(124, 16 * ww)))
                px = int(rng.integers(129 - ww))
                py = int(rng.integers(129 - hh))
                m = torch.zeros((1, 1, 128, 128), device=device)
                m[:, :, py : py + hh, px : px + ww] = 1
                tile = F.interpolate(
                    r, size=(4 * hh, 4 * ww), mode="bilinear", align_corners=False
                )
                delta = torch.zeros_like(a)
                delta[:, :, 4 * py : 4 * (py + hh), 4 * px : 4 * (px + ww)] = tile
            else:
                j = int(rng.choice(tr_s))
                m = y[j].to(device).float()[None, None]
                m = m if rng.random() < 0.5 else 1 - m
                if rng.random() < 0.5:
                    m = m.flip((-1,))
                delta = F.interpolate(
                    r, size=(512, 512), mode="bilinear", align_corners=False
                ) * F.interpolate(m, scale_factor=4, mode="nearest")
            xx.append(
                torch.floor(a + delta * float(rng.uniform(0.65, 1.3))).clamp(0, 255)
            )
            yy.append(m[:, 0])
        return torch.cat(xx), torch.cat(yy)

    def loss_fn(log, target):
        p = log.sigmoid()
        loss = F.binary_cross_entropy_with_logits(log, target)
        for a, b in [(p, target), (1 - p, 1 - target)]:
            loss = (
                loss
                + 0.5
                * (
                    1 - ((a * b).sum((1, 2)) + 1) / ((a + b - a * b).sum((1, 2)) + 1)
                ).mean()
            )
        return loss

    def update(xx, yy):
        if rng.random() < 0.5:
            xx = xx.flip((-1,))
            yy = yy.flip((-1,))
        if rng.random() < 0.5:
            xx = xx.flip((-2,))
            yy = yy.flip((-2,))
        opt.zero_grad()
        loss = loss_fn(net(xx), yy)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 2)
        opt.step()
        return float(loss.detach().cpu())

    opt = torch.optim.AdamW(net.parameters(), lr=0.001)
    for s in range(args.pre):
        val = update(*synthetic(args.batch))
        if (s + 1) % 50 == 0:
            print("SYN", s + 1, val, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=0.0005)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, args.epochs, eta_min=0.00004
    )

    @torch.inference_mode()
    def predict(imgs, ids):
        net.eval()
        pred = []
        for j in range(0, len(ids), args.batch):
            a = imgs[ids[j : j + args.batch]].float().to(device)
            p = 0.5 * (net(a).sigmoid() + net(a.flip((-1,))).sigmoid().flip((-1,)))
            pred.append(p.cpu())
        return torch.cat(pred)

    best = -1
    ck = out / "stegastamp.pth"
    for ep in range(1, args.epochs + 1):
        net.train()
        ix = rng.permutation(tr_s)
        for j in range(0, len(ix), args.batch):
            ids = ix[j : j + args.batch].tolist()
            update(x[ids].to(device).float(), y[ids].to(device).float())
        sched.step()
        if ep % 4 == 0 or ep == args.epochs:
            p = predict(x, d)
            sc, t = max((model_score(p > t, y[d]), t) for t in (0.3, 0.45, 0.55, 0.7))
            if sc > best:
                best, th, epoch = sc, t, ep
                torch.save(net.state_dict(), ck)
            print("DEV", ep, sc, best, flush=True)
    net.load_state_dict(torch.load(ck, map_location=device, weights_only=True))
    hm = predict(x, h) > th
    hs = model_score(hm, y[h])
    info = {
        "method": "stegastamp",
        "dev": best,
        "hold": hs,
        "n": len(h),
        "epoch": epoch,
        "threshold": th,
        "synthetic_pretrain_steps": args.pre,
    }
    print("HOLD", info, flush=True)
    np.savez_compressed(out / "stegastamp-hold.npz", ids=h, masks=hm.numpy())
    del x
    rr, xt, _ = load(root, "test")
    ti = [i for i, r in enumerate(rr) if r["method"] == "stegastamp"]
    tm = predict(xt, ti) > th
    assert tm.shape == (340, 128, 128)
    np.savez_compressed(out / "stegastamp-test.npz", ids=ti, masks=tm.numpy())
    (out / "stegastamp.json").write_text(json.dumps(info))
    print("TEST", len(ti), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--pre", type=int, default=400)
    p.add_argument("--seed", type=int, default=3301)
    p.add_argument("--epochs", type=int, default=48)
    p.add_argument("--batch", type=int, default=6)
    a = p.parse_args()
    prepare(a.root) if a.prepare else run(a)
