"""Supervised RGB phase-residual segmenter. Never uses test masks or test metadata beyond method and name."""

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from train_grid import METHODS, read, split_ids
from ensemble_views import mask_for, model_score


def phase_tensor(rgb):
    """Keep all 16 positions in a scoring cell; local second differences reject DC."""
    if (
        rgb.ndim != 4
        or rgb.shape[1] != 3
        or min(rgb.shape[-2:]) < 4
        or any(n % 4 for n in rgb.shape[-2:])
    ):
        raise ValueError("Expected BCHW RGB, dimensions divisible by 4")
    v = rgb.float()
    p = F.pad(v, (1, 1, 1, 1), mode="replicate")
    dx = p[:, :, 1:-1, :-2] + p[:, :, 1:-1, 2:] - 2 * v
    dy = p[:, :, :-2, 1:-1] + p[:, :, 2:, 1:-1] - 2 * v
    a = torch.cat([dx, dy], 1)
    return F.pixel_unshuffle(a.sign() * torch.log1p(a.abs()), 4)


def conv(a, b, k=3, s=1):
    return nn.Sequential(
        nn.Conv2d(a, b, k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(b),
        nn.SiLU(),
    )


class PhaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.phase = nn.Sequential(conv(96, 24, 1), conv(24, 24))
        self.pre = conv(24 + 3 + 3 + 6, 32, 1)
        self.a = conv(32, 32, 3, 2)
        self.b = nn.Sequential(conv(32, 48, 3, 2), conv(48, 48))
        self.c = nn.Sequential(conv(48, 64, 3, 2), conv(64, 64))
        self.d = conv(64, 64, 3, 2)
        self.gate = nn.Sequential(
            nn.Linear(128, 32), nn.SiLU(), nn.Dropout(0.15), nn.Linear(32, 1)
        )
        self.out = nn.Sequential(
            conv(32 + 32 + 48 + 64 + 64, 32, 1), conv(32, 24), nn.Conv2d(24, 2, 1)
        )

    def forward(self, im, cue, m):
        z = self.phase(phase_tensor(im))
        avg = F.avg_pool2d(im.float(), 4) / 127.5 - 1
        size = z.shape[-2:]
        rs = lambda x: F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        tag = F.one_hot(m, 3).to(z.dtype)[:, :, None, None].expand(-1, -1, *size)
        x = self.pre(torch.cat([z, avg, rs(cue), tag], 1))
        a = self.a(x)
        b = self.b(a)
        c = self.c(b)
        d = self.d(c)
        return self.gate(
            torch.cat([d.mean((2, 3)), d.amax((2, 3))], 1)
        ).flatten(), self.out(torch.cat([x, rs(a), rs(b), rs(c), rs(d)], 1))


class Images:
    def __init__(self, root, split, rows):
        self.rows = rows
        self.methods = torch.tensor([METHODS.index(r["method"]) for r in rows])
        self.cues = []
        self.im = np.empty((len(rows), 3, 512, 512), np.uint8)
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            for i, r in enumerate(rows):
                n = r["image_name"]
                m = r["method"]
                self.im[i] = np.asarray(
                    Image.open(io.BytesIO(z.read(f"{split}/images/{n}.png"))).convert(
                        "RGB"
                    )
                ).transpose(2, 0, 1)
                cam = np.load(
                    root / f"cam-v1-{split}-{m}" / (n + ".npy"), allow_pickle=False
                ).astype(np.float32)
                sem = np.load(
                    root / f"semantic-v1-{split}" / (n + ".npy"), allow_pickle=False
                )
                wr = np.load(
                    root / f"rewrite-v1-{split}-{m}" / (n + ".npy"), allow_pickle=False
                )
                selected = [cam[min(j, len(cam) - 1)] for j in (1, 5, 9)] + [
                    sem[0],
                    wr[0],
                    wr[-1],
                ]
                self.cues.append(np.clip(np.stack(selected), -4, 4).astype(np.float16))
        self.cues = np.stack(self.cues)

    def batch(self, ids, device):
        return (
            torch.from_numpy(self.im[ids]).to(device),
            torch.from_numpy(self.cues[ids]).float().to(device),
            self.methods[ids].to(device),
        )


def run(root, seed, epochs):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _, rows, y = read(root, "train")
    assert y is not None
    tr, d, h = split_ids(rows)
    data = Images(root, "train", rows)
    gtruth = torch.tensor([r["full_watermark"] for r in rows]).float()
    y = y[:, 0]
    model = PhaseNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=0.00015)
    out = root / f"phase-v1-{seed}-{epochs}"
    out.mkdir(exist_ok=True)
    best = {m: -1.0 for m in METHODS}
    cfg = {}
    print("START", device, len(tr), len(d), len(h), flush=True)

    @torch.inference_mode()
    def pred(ds, ids):
        model.eval()
        gs = []
        qs = []
        for j in range(0, len(ids), 6):
            im, cu, me = ds.batch(ids[j : j + 6], device)
            g, q = model(im, cu, me)
            gs.append(g.sigmoid().cpu())
            qs.append(q.sigmoid().cpu())
        return torch.cat(gs), torch.cat(qs)

    for ep in range(1, epochs + 1):
        model.train()
        ids = rng.permutation(tr)
        losses = []
        for j in range(0, len(ids), 6):
            take = ids[j : j + 6].tolist()
            im, cu, me = data.batch(take, device)
            gt = y[take].to(device)
            full = gtruth[take].to(device)
            for axis in (-1, -2):
                if rng.random() < 0.5:
                    im, cu, gt = (x.flip((axis,)) for x in (im, cu, gt))
            g, q = model(im, cu, me)
            f = full[:, None, None]
            rare = gt * (1 - f) + (1 - gt) * f
            log = q[:, 0] * (1 - f) + q[:, 1] * f
            prob = log.sigmoid()
            dims = (-2, -1)
            dice = (2 * (prob * rare).sum(dims) + 1) / (
                prob.sum(dims) + rare.sum(dims) + 1
            )
            loss = (
                0.5 * F.binary_cross_entropy_with_logits(g, full)
                + F.binary_cross_entropy_with_logits(log, rare)
                + 1
                - dice.mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            losses.append(loss.detach())
        schedule.step()
        print("epoch", ep, "loss", float(torch.stack(losses).mean().cpu()), flush=True)
        if ep % 4 == 0 or ep == epochs:
            for m in METHODS:
                ids = [i for i in d if rows[i]["method"] == m]
                g, q = pred(data, ids)
                res = max(
                    (model_score(mask_for(g, q, rule, t), y[ids] > 0.5), rule, t)
                    for rule in ("hard", "soft")
                    for t in (0.3, 0.45, 0.6)
                )
                if res[0] > best[m]:
                    best[m] = res[0]
                    cfg[m] = (res[1], res[2], ep)
                    torch.save(model.state_dict(), out / (m + ".pth"))
            print("dev", best, flush=True)
    _, testrows, _ = read(root, "test")
    td = Images(root, "test", testrows)
    for m in METHODS:
        model.load_state_dict(
            torch.load(out / (m + ".pth"), map_location="cpu", weights_only=True)
        )
        rule, t, ep = cfg[m]
        hi = [i for i in h if rows[i]["method"] == m]
        g, q = pred(data, hi)
        mask = mask_for(g, q, rule, t)
        hold = model_score(mask, y[hi] > 0.5)
        np.savez_compressed(out / f"{m}-hold.npz", ids=hi, masks=mask.numpy())
        ti = [i for i, r in enumerate(testrows) if r["method"] == m]
        g, q = pred(td, ti)
        mask = mask_for(g, q, rule, t).numpy()
        assert mask.dtype == bool and mask.shape == (340, 128, 128)
        tmp = out / f"{m}.partial.npz"
        np.savez_compressed(tmp, ids=ti, masks=mask)
        tmp.replace(out / f"{m}-test.npz")
        stats = {
            "method": m,
            "dev": best[m],
            "hold": hold,
            "n": len(hi),
            "epoch": ep,
            "rule": rule,
            "threshold": t,
        }
        (out / f"{m}.json").write_text(json.dumps(stats))
        print("RESULT", stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--seed", type=int, default=1907)
    p.add_argument("--epochs", type=int, default=48)
    a = p.parse_args()
    run(a.root, a.seed, a.epochs)
