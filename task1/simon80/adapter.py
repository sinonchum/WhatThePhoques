"""Train a small image-conditioned correction to Simon's frozen CLEAN OOF logit maps.
No mask or full_watermark at inference. Cross-fitted promotion versus identity; no use of holdout for fitting.
"""

from pathlib import Path
import argparse, io, json, zipfile, hashlib, os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from refine_oof import (
    SOURCES,
    ROOTS,
    ARCHIVE_HASH,
    SIMON_COMMIT,
    load,
    metric,
    team_hold,
    logit,
    fit_mix,
)
from forensic_features import features


class Adapter(nn.Module):
    def __init__(self, c):
        super().__init__()
        b = lambda a, w, d: nn.Sequential(
            nn.Conv2d(a, w, 3, padding=d, dilation=d, bias=False),
            nn.GroupNorm(4, w),
            nn.SiLU(),
        )
        self.stem = b(c + 3, 24, 1)
        self.context = nn.Sequential(b(24, 32, 1), b(32, 32, 2), b(32, 32, 3))
        self.fuse = b(56, 24, 1)
        self.head = nn.Conv2d(24, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, z):
        v = F.avg_pool2d(z, 2)
        p = torch.sigmoid(v)
        desc = p.mean((2, 3), keepdim=True).expand_as(p)
        s = self.stem(torch.cat((x, v.clamp(-8, 8), p, desc), 1))
        h = F.interpolate(
            self.context(F.avg_pool2d(s, 2)),
            size=s.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        r = F.interpolate(
            self.head(self.fuse(torch.cat((s, h), 1))),
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        return z + 3 * torch.tanh(r / 3)


def get_cues(root, split, idx, rr):
    raw = root / f"forensic-v1-{split}.npy"
    mem = np.load(raw, mmap_mode="r") if raw.is_file() else None
    out = []
    with zipfile.ZipFile(root / f"{split}.zip") as z:
        for i in idx:
            im = np.asarray(
                Image.open(
                    io.BytesIO(z.read(f"{split}/images/{rr[i]['image_name']}.png"))
                ).convert("RGB")
            )
            grid = mem[i] if mem is not None else features(im)
            rgb = (
                im.reshape(64, 8, 64, 8, 3).mean((1, 3)).transpose(2, 0, 1) / 127.5 - 1
            )
            out.append(np.concatenate([grid[::3], rgb]).astype(np.float16))
    return torch.from_numpy(np.stack(out))


def fit(x, z, y, fit, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu = x[fit, :, ::3, ::3].float().mean((0, 2, 3))[None, :, None, None].cuda()
    sd = (
        x[fit, :, ::3, ::3]
        .float()
        .std((0, 2, 3))[None, :, None, None]
        .clamp_min(0.02)
        .cuda()
    )
    net = Adapter(x.shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.00065, weight_decay=0.002)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 28, eta_min=0.00004)
    for _ in range(28):
        for group in np.array_split(rng.permutation(fit), 13):
            ii = group.tolist()
            a = ((x[ii].cuda().float() - mu) / sd).clamp(-7, 7)
            b = z[ii].cuda()
            t = y[ii].cuda()
            for axis in (-1, -2):
                if rng.random() < 0.5:
                    a = a.flip((axis,))
                    b = b.flip((axis,))
                    t = t.flip((axis,))
            l = net(a, b)
            p = l.sigmoid()
            loss = (
                0.5 * F.binary_cross_entropy_with_logits(l, t)
                + 0.018 * (l - b).abs().mean()
            )
            for u, v in [(p, t), (1 - p, 1 - t)]:
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
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
        sch.step()
    return net.eval(), mu, sd


@torch.no_grad()
def predict(model, x, z):
    net, mu, sd = model
    v = []
    for j in range(0, len(x), 8):
        a = ((x[j : j + 8].cuda().float() - mu) / sd).clamp(-7, 7)
        b = z[j : j + 8].cuda()
        l = (net(a, b) + net(a.flip((-1,)), b.flip((-1,))).flip((-1,))) * 0.5
        v.append(l[:, 0].cpu().numpy())
    return np.concatenate(v)


def run(root, out, archive):
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    print(
        "GPU_ADAPTER",
        os.environ.get("SLURM_JOB_ID"),
        torch.cuda.get_device_name(),
        flush=True,
    )
    out.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(root / "train.zip") as z:
        rr = json.loads(z.read("train/metadata.json"))
        with np.load(io.BytesIO(z.read("train/masks_128.npz"))) as f:
            truth = dict(zip(f["names"], f["masks"].astype(bool)))
    with zipfile.ZipFile(root / "test.zip") as z:
        tt = json.loads(z.read("test/metadata.json"))
    names = [r["image_name"] for r in tt]
    H = team_hold(rr, truth)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_HASH
    ref = torch.load(archive, weights_only=True, map_location="cpu")
    bn = ref["image_names"]
    assert len(set(bn)) == len(bn) == 1020 and set(bn) == set(names)
    lookup = {n: i for i, n in enumerate(bn)}
    base = ref["masks"][[lookup[n] for n in names]].bool()
    result = base.clone()
    info = {
        "simon_commit": SIMON_COMMIT,
        "archive_sha256": ARCHIVE_HASH,
        "methods": {},
        "public_score": None,
    }
    hold = []
    for m, (source, _) in SOURCES.items():
        folder = ROOTS[0] / source
        ti = [i for i, r in enumerate(tt) if r["method"] == m]
        if not all(
            (folder / (s + ".npz")).is_file() for s in ("dev", "holdout", "test")
        ):
            assert torch.equal(result[ti], base[ti])
            info["methods"][m] = {"source": "unchanged archive", "hold": None}
            print("KEEP", m, flush=True)
            continue
        ids = [
            i for i, r in enumerate(rr) if r["method"] == m and r["image_name"] not in H
        ]
        hi = [i for i, r in enumerate(rr) if r["method"] == m and r["image_name"] in H]
        nnames = [rr[i]["image_name"] for i in ids]
        A = logit(load(folder, "dev", nnames))
        Y = np.stack([truth[n] for n in nnames])
        X = get_cues(root, "train", ids, rr)
        y = torch.from_numpy(Y[:, None]).float()
        assert (len(ids), len(hi), len(ti)) == (136, 34, 340)
        models = []
        sc = []
        for f, (train, val) in enumerate(
            StratifiedKFold(4, shuffle=True, random_state=4819).split(
                ids, [int(rr[i]["full_watermark"]) for i in ids]
            )
        ):
            _, t = fit_mix(A, None, Y, train)
            z = torch.from_numpy((A - t)[:, None])
            model = fit(X, z, y, train, 7001 + f)
            pred = predict(model, X[val], z[val])
            ref_score = metric(A[val] >= t, Y[val])
            ad_score = metric(pred >= 0, Y[val])
            sc.append([float(ref_score.mean()), float(ad_score.mean())])
            models.append((model, t))
            print("FOLD", m, f, sc[-1], flush=True)
        raw, adj = np.array(sc).T
        use = bool((adj - raw).mean() > 0.003 and (adj > raw).sum() >= 3)
        _, tall = fit_mix(A, None, Y, np.arange(len(Y)))
        scores = {
            "source": source,
            "use_adapter": use,
            "cv_base": float(raw.mean()),
            "cv_adapter": float(adj.mean()),
            "folds": sc,
        }
        for split, ii, meta in [("holdout", hi, rr), ("test", ti, tt)]:
            nnames = [meta[i]["image_name"] for i in ii]
            L = logit(load(folder, split, nnames))
            p = L >= tall
            if use:
                xx = get_cues(root, "train" if split == "holdout" else "test", ii, meta)
                delta = 0.0
                for model, t in models:
                    z = torch.from_numpy((L - t)[:, None])
                    delta += (predict(model, xx, z) - (L - t)) / len(models)
                p = (L - tall + delta) >= 0
            if split == "holdout":
                yy = np.stack([truth[n] for n in nnames])
                scores["hold_base"] = float(metric(L >= tall, yy).mean())
                scores["hold"] = float(metric(p, yy).mean())
                hold.append(scores["hold"])
            else:
                result[ii] = torch.from_numpy(p)
            np.savez_compressed(
                out / (m + "-" + split + ".npz"), image_names=np.array(nnames), masks=p
            )
        info["methods"][m] = scores
        print("METHOD", m, scores, flush=True)
    assert hold
    dst = out / "submission.pt"
    torch.save({"image_names": names, "masks": result}, dst)
    vv = torch.load(dst, weights_only=True)
    assert (
        vv["image_names"] == names
        and torch.equal(vv["masks"], result)
        and result.shape == (1020, 128, 128)
        and dst.stat().st_size < 20_000_000
    )
    info["local_overall"] = float(np.mean(hold)) if len(hold) == 3 else None
    info["test_n"] = len(names)
    info["sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(info, indent=2))
    print("DONE_ADAPTER", info["local_overall"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--base",
        type=Path,
        default=Path("/p/scratch/training2666/qin7/task1/runs/submissions/sub_v1.pt"),
    )
    a = p.parse_args()
    run(a.root, a.out, a.base)
