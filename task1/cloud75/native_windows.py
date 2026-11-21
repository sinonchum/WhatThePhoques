"""CUDA: locate residual watermarks using calibrated native decoder responses on overlapping boxes.
Only the inner training split provides gradients; compare correctly signed, shuffled and absent payloads.
"""

import argparse, hashlib, io, json, os, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from train_grid import split_ids
from train_cue_net import Region
from forensic_features import features

SPECS = (
    (1.0, 1.0, 1, 1),
    (0.72, 0.72, 3, 3),
    (0.40, 0.40, 4, 4),
    (0.20, 0.20, 5, 5),
    (0.18, 0.55, 4, 4),
    (0.55, 0.18, 4, 4),
    (0.10, 0.36, 5, 5),
    (0.36, 0.10, 5, 5),
)


def bank():
    regions = []
    groups = []
    for w, h, nx, ny in SPECS:
        y, x = np.meshgrid(
            np.linspace(0, 1 - h, ny), np.linspace(0, 1 - w, nx), indexing="ij"
        )
        arr = np.stack([x, y, x + w, y + h], axis=-1).reshape(-1, 4)
        groups.append(len(arr))
        regions.extend(arr)
    return torch.tensor(np.array(regions), dtype=torch.float32), groups


def crops(img, bb, side):
    t = (torch.arange(side, device=img.device) + 0.5) / side
    x = bb[:, 0, None] + (bb[:, 2] - bb[:, 0])[:, None] * t
    y = bb[:, 1, None] + (bb[:, 3] - bb[:, 1])[:, None] * t
    grid = (
        torch.stack(
            (x[:, None].expand(-1, side, -1), y[:, :, None].expand(-1, -1, side)), -1
        )
        * 2
        - 1
    )
    return F.grid_sample(
        img.expand(len(bb), -1, -1, -1),
        grid,
        mode="bilinear",
        align_corners=False,
        padding_mode="border",
    )


def paint(q, bb, groups):
    # One center-interpolated map, overlapping-window vote, best rectangle, and confidence per scale.
    v = (torch.arange(64, device=q.device) + 0.5) / 64
    X = v[None, None, :]
    Y = v[None, :, None]
    zz = bb[:, :, None, None]
    mask = (X >= zz[:, 0]) & (X < zz[:, 2]) & (Y >= zz[:, 1]) & (Y < zz[:, 3])
    out = []
    st = 0
    for (_, _, nx, ny), n in zip(SPECS, groups):
        a = q[st : st + n]
        m = mask[st : st + n].float()
        b = bb[st : st + n]
        st += n
        # Interpolate scores at physical box centers, extending the nearest score to each image edge.
        if n == 1:
            field = a[0].expand(64, 64)
        else:
            cx = (b[:, 0] + b[:, 2]) * 0.5
            cy = (b[:, 1] + b[:, 3]) * 0.5
            gx = 2 * (v - cx.min()) / (cx.max() - cx.min()) - 1
            gy = 2 * (v - cy.min()) / (cy.max() - cy.min()) - 1
            yy, xx = torch.meshgrid(gy, gx, indexing="ij")
            field = F.grid_sample(
                a.view(1, 1, ny, nx),
                torch.stack([xx, yy], -1)[None],
                padding_mode="border",
                align_corners=True,
            )[0, 0]
        out.extend(
            [
                field,
                (m * a[:, None, None]).sum(0) / m.sum(0).clamp_min(1),
                m[a.argmax()] * (a.max() - a.mean()),
                a.max().expand_as(field),
            ]
        )
    return torch.stack(out)


@torch.inference_mode()
def extract(root, rows, ids, split, method, dev):
    folder = root / f"native75-cues-v1-{method}-{split}"
    folder.mkdir(exist_ok=True)
    net = None
    bb, groups = bank()
    bb = bb.to(dev)
    if method == "stegastamp":
        side = 400
    else:
        side = 245 if method == "trustmark" else 256
    precomputed = root / f"forensic-v1-{split}.npy"
    grid = np.load(precomputed, mmap_mode="r") if precomputed.exists() else None
    stored = []
    with zipfile.ZipFile(root / f"{split}.zip") as z:
        for j, i in enumerate(ids):
            r = rows[i]
            n = r["image_name"]
            p = folder / (n + ".npz")
            if not p.exists():
                if net is None:
                    if method == "stegastamp":
                        from stega_native_cam import FixedDecoder

                        net = FixedDecoder(root).eval().to(dev)
                    else:
                        from message_cam import Decoder

                        net = Decoder(method, root).eval().to(dev)
                im = np.asarray(
                    Image.open(io.BytesIO(z.read(f"{split}/images/{n}.png"))).convert(
                        "RGB"
                    )
                )
                a = (
                    torch.from_numpy(im.copy()).permute(2, 0, 1)[None].to(dev).float()
                    / 255
                )
                key = torch.tensor(
                    [2 * int(k) - 1 for k in r["message"]],
                    dtype=torch.float32,
                    device=dev,
                )
                gen = np.random.default_rng(
                    int.from_bytes(hashlib.sha256(n.encode()).digest()[:8], "little")
                )
                wrong = key[torch.from_numpy(gen.permutation(len(key))).to(dev)]
                ks = torch.stack([key, wrong])
                allsc = []
                for start in range(0, len(bb), 16):
                    x = crops(a, bb[start : start + 16], side)
                    if method == "stegastamp":
                        logit = net(x)[0]
                    elif method == "trustmark":
                        logit = net.n(x * 2 - 1)
                    else:
                        h = net.pixel_decoder.output_upscaling(net.convnext(x * 2 - 1))
                        linear = net.pixel_decoder.linear
                        logit = F.linear(
                            h.mean((-2, -1)), linear.weight[1:], linear.bias[1:]
                        )
                    zbit = logit.float()[:, None] * ks[None]
                    sc = 0.5 * zbit.sign().mean(-1) + 0.5 * zbit.mean(
                        -1
                    ) / logit.float().square().mean(-1, keepdim=True).sqrt().clamp_min(
                        1e-5
                    )
                    allsc.append(sc)
                scores = torch.cat(allsc)
                native = (
                    torch.stack([paint(scores[:, u], bb, groups) for u in (0, 1)])
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                geo = (grid[i, ::3] if grid is not None else features(im)[::3]).astype(
                    np.float16
                )
                assert np.isfinite(native).all() and np.isfinite(geo).all()
                tmp = p.with_suffix(".partial.npz")
                np.savez_compressed(tmp, geo=geo, native=native)
                tmp.replace(p)
            with np.load(p, allow_pickle=False) as f:
                stored.append((f["geo"], f["native"]))
            if j % 25 == 0:
                print("CUES", split, j + 1, len(ids), flush=True)
    return torch.from_numpy(np.stack([a for a, b in stored])), torch.from_numpy(
        np.stack([b for a, b in stored])
    )


def miou(a, b):
    out = []
    for x, y in ((a, b), (~a, ~b)):
        u = (x | y).sum((-1, -2))
        out.append(torch.where(u > 0, (x & y).sum((-1, -2)) / u.clamp_min(1), 1.0))
    return float(torch.stack(out).mean())


def run(root, m, epochs, seed):
    dev = "cuda"
    assert torch.cuda.is_available()
    print(
        "DEVICE",
        torch.cuda.get_device_name(),
        os.environ.get("SLURM_JOB_ID"),
        "method",
        m,
        flush=True,
    )
    torch.set_num_threads(3)
    with zipfile.ZipFile(root / "train.zip") as z:
        rows = json.loads(z.read("train/metadata.json"))
        tr, dd, hh = split_ids(rows)
        tr, dd, hh = (
            [i for i in ids if rows[i]["method"] == m] for ids in (tr, dd, hh)
        )
        idx = tr + dd + hh
        with np.load(
            io.BytesIO(z.read("train/masks_128.npz")), allow_pickle=False
        ) as f:
            lookup = dict(zip(f["names"], f["masks"]))
            Y = torch.tensor(
                np.stack([lookup[rows[i]["image_name"]] for i in idx]),
                dtype=torch.float32,
            )[:, None]
            G = torch.tensor([float(rows[i]["full_watermark"]) for i in idx])[
                :, None, None, None
            ]
    n = len(tr)
    d = list(range(n, n + len(dd)))
    h = list(range(n + len(dd), len(idx)))
    assert (n, len(d), len(h)) == (120, 16, 34)
    geo, cues = extract(root, rows, idx, "train", m, dev)
    out = root / "native75-v1"
    out.mkdir(exist_ok=True)

    def data(a, b, arm):
        return torch.cat(
            [
                a,
                b[:, 0]
                if arm == "correct"
                else b[:, 1]
                if arm == "shuffled"
                else torch.zeros_like(b[:, 0]),
            ],
            1,
        )

    def predict(net, x, mu, sd, ids):
        net.eval()
        ps = []
        with torch.inference_mode():
            for j in range(0, len(ids), 12):
                t = x[ids[j : j + 12]].float().to(dev)
                t = ((t - mu) / sd).clamp(-7, 7)
                v = []
                for flip in (False, True):
                    a = t.flip((-1,)) if flip else t
                    g, l = net(
                        a,
                        torch.full(
                            (len(a),),
                            ("pixelseal", "trustmark", "stegastamp").index(m),
                            device=dev,
                        ),
                    )
                    p = (1 - g.sigmoid()) * l[:, :1].sigmoid() + g.sigmoid() * (
                        1 - l[:, 1:].sigmoid()
                    )
                    v.append(p.flip((-1,)) if flip else p)
                ps.append(torch.stack(v).mean(0)[:, 0].cpu())
        return torch.cat(ps)

    candidates = {}
    for arm in ("geometry", "shuffled", "correct"):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        x = data(geo, cues, arm)
        stats = x[:n, :, ::3, ::3].float()
        mu = stats.mean((0, 2, 3))[None, :, None, None].to(dev)
        sd = stats.std((0, 2, 3))[None, :, None, None].clamp_min(0.01).to(dev)
        net = Region(x.shape[1], 32).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.002)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=0.0001)
        best = -1.0
        ck = out / f"{m}-{arm}.pth"
        for ep in range(1, epochs + 1):
            net.train()
            order = rng.permutation(n)
            for st in range(0, n, 8):
                ii = order[st : st + 8].tolist()
                a = ((x[ii].float().to(dev) - mu) / sd).clamp(-7, 7)
                y = Y[ii].to(dev)
                g = G[ii].to(dev)
                for ax in (-1, -2):
                    if rng.random() < 0.5:
                        a = a.flip((ax,))
                        y = y.flip((ax,))
                gate, logit = net(
                    a,
                    torch.full(
                        (len(a),),
                        ("pixelseal", "trustmark", "stegastamp").index(m),
                        device=dev,
                    ),
                )
                rare = (1 - g) * y + g * (1 - y)
                l = (1 - g) * logit[:, :1] + g * logit[:, 1:]
                p = l.sigmoid()
                soft = (p * rare).sum((2, 3))
                union = (p + rare - p * rare).sum((2, 3))
                loss = (
                    0.55 * F.binary_cross_entropy_with_logits(gate, g)
                    + F.binary_cross_entropy_with_logits(l, rare)
                    + (1 - (soft + 1) / (union + 1)).mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2)
                opt.step()
            sch.step()
            if ep % 8 == 0 or ep == epochs:
                pr = predict(net, x, mu, sd, d)
                sc, t = max(
                    (miou(pr > t, Y[d, 0] > 0.5), t) for t in (0.35, 0.50, 0.65)
                )
                if sc > best:
                    best = sc
                    candidates[arm] = (sc, t, ep)
                    torch.save(
                        {"weights": net.state_dict(), "mu": mu.cpu(), "sd": sd.cpu()},
                        ck,
                    )
                print("DEV", m, arm, ep, round(best, 5), flush=True)
    arm = max(candidates, key=lambda k: candidates[k][0])
    score, t, ep = candidates[arm]
    ck = torch.load(out / f"{m}-{arm}.pth", map_location=dev, weights_only=True)
    net.load_state_dict(ck["weights"])
    mu, sd = ck["mu"], ck["sd"]
    ph = predict(net, data(geo, cues, arm), mu, sd, h) > t
    hold = miou(ph, Y[h, 0] > 0.5)
    np.savez_compressed(out / f"{m}-hold.npz", ids=hh, masks=ph.numpy())
    print("HOLD", m, arm, hold, flush=True)
    with zipfile.ZipFile(root / "test.zip") as z:
        rr = json.loads(z.read("test/metadata.json"))
    ti = [i for i, r in enumerate(rr) if r["method"] == m]
    tg, tw = extract(root, rr, ti, "test", m, dev)
    pred = predict(net, data(tg, tw, arm), mu, sd, list(range(len(ti)))) > t
    assert pred.shape == (340, 128, 128)
    np.savez_compressed(out / f"{m}-test.npz", ids=ti, masks=pred.numpy())
    report = {
        "method": m,
        "arm": arm,
        "dev": score,
        "hold": hold,
        "n": len(h),
        "threshold": t,
        "epoch": ep,
        "dev_ablations": candidates,
    }
    (out / f"{m}.json").write_text(json.dumps(report))
    print("DONE", report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("method", choices=("pixelseal", "trustmark", "stegastamp"))
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--seed", type=int, default=5101)
    a = p.parse_args()
    run(a.root, a.method, a.epochs, a.seed)
