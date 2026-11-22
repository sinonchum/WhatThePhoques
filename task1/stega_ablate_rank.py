"""Rank object-aligned repairs with native known-bit decoding and localized high-frequency ablations.
Keep-v1 is an explicit candidate. Cache contains no labels. Fit ranking weights to training images only.
"""

import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.ensemble import ExtraTreesRegressor
from scipy.ndimage import gaussian_filter, binary_dilation
from PIL import Image
from train_grid import read, split_ids, METHODS
from train_cue_net import Region
from ensemble_views import feature, model_score
from crop_verify import propose, overlap_prior
from stega_native_cam import FixedDecoder


def score_each(ms, gt):
    m = ms.astype(bool)
    y = gt.astype(bool)
    val = 0.0
    for a, b in [(m, y), (~m, ~y)]:
        un = (a | b).sum((-2, -1))
        val = val + 0.5 * np.where(un > 0, (a & b).sum((-2, -1)) / un.clip(min=1), 1.0)
    return val


def run(root):
    seed = 4079
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    dev = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    gb, rows, y = read(root, "train")
    tb, test, _ = read(root, "test")
    tr, dd, hh = split_ids(rows)
    tr_s, dd, hh = (
        [i for i in g if rows[i]["method"] == "stegastamp"] for g in (tr, dd, hh)
    )
    stats = torch.from_numpy(
        np.stack([feature(root, "train", gb, rows, i, 2)[:, ::4, ::4] for i in tr[::3]])
    ).float()
    mu = stats.mean((0, 2, 3))[None, :, None, None].to(dev)
    sd = stats.std((0, 2, 3))[None, :, None, None].clamp_min(0.005).to(dev)
    net = Region(stats.shape[1]).to(dev).eval()
    net.load_state_dict(
        torch.load(
            root / "neural-message-v1-1451-120-r1t1/stegastamp.pth",
            weights_only=True,
            map_location=dev,
        )
    )
    decoder = FixedDecoder(root).to(dev)
    cache = root / "stega-ablate-cache-v1"
    cache.mkdir(exist_ok=True)
    out = root / "stega-ablate-rank-v1"
    out.mkdir(exist_ok=True)

    @torch.inference_mode()
    def item(split, block, rr, i, z):
        r = rr[i]
        name = r["image_name"]
        f = cache / (split + "-" + name + ".npz")
        if f.exists():
            with np.load(f, allow_pickle=False) as obj:
                return np.unpackbits(obj["M"], axis=1).reshape(-1, 128, 128).astype(
                    bool
                ), obj["X"].astype(np.float32)
        x = (
            torch.from_numpy(feature(root, split, block, rr, i, 2))
            .to(dev)
            .float()[None]
        )
        g, l = net(
            ((x - mu) / sd).clamp(-7, 7),
            torch.tensor([METHODS.index("stegastamp")], device=dev),
        )
        gate = g.sigmoid().item()
        prob = (
            ((1 - gate) * l[:, :1].sigmoid() + gate * (1 - l[:, 1:].sigmoid()))[0, 0]
            .cpu()
            .numpy()
        )
        full = gate > 0.5
        rare = 1 - prob if full else prob
        base = rare > 0.5
        with np.load(root / f"sam-v1-{split}" / (name + ".npz")) as o:
            objects = (
                np.unpackbits(o["masks"], axis=1).reshape(-1, 128, 128).astype(bool)
            )
        boxes, ms = propose(rare, objects, [])
        pr = overlap_prior(ms, rare)
        chosen = list(np.argsort(pr)[::-1][:16])
        A = [base]
        seen = {np.packbits(base).tobytes()}
        for j in chosen:
            a = ms[j]
            key = np.packbits(a).tobytes()
            if key not in seen:
                A.append(a)
                seen.add(key)
        # Tiny and empty edits must not be forced to become the nearest SAM object.
        A.extend([rare > 0.35, rare > 0.65, np.zeros_like(base)])
        A = np.stack(A)
        n = len(A)
        img = Image.open(io.BytesIO(z.read(f"{split}/images/{name}.png"))).convert(
            "RGB"
        )
        rgb = (
            np.asarray(img.resize((400, 400), Image.Resampling.BICUBIC), np.float32)
            / 255.0
        )
        bl = gaussian_filter(rgb, (2, 2, 0))
        inputs = [rgb]
        bits = torch.tensor([float(b) * 2 - 1 for b in r["message"]], device=dev)
        for a in A:
            yy, xx = np.where(a)
            box = (
                (
                    int(xx.min()) * 4,
                    int(yy.min()) * 4,
                    int(xx.max() + 1) * 4,
                    int(yy.max() + 1) * 4,
                )
                if len(yy)
                else (0, 0, 512, 512)
            )
            crop = (
                np.asarray(
                    img.crop(box).resize((400, 400), Image.Resampling.BICUBIC),
                    np.float32,
                )
                / 255.0
            )
            alpha = np.asarray(
                Image.fromarray(a.astype(np.float32)).resize(
                    (400, 400), Image.Resampling.BILINEAR
                )
            )[:, :, None]
            inputs.extend([crop, rgb + alpha * (bl - rgb), bl + alpha * (rgb - bl)])
        scores = []
        for st in range(0, len(inputs), 6):
            batch = (
                torch.from_numpy(np.stack(inputs[st : st + 6]))
                .permute(0, 3, 1, 2)
                .to(dev)
            )
            l, _ = decoder(batch)
            s = l * bits
            sc = torch.stack(
                [
                    (s > 0).float().mean(1),
                    s.mean(1),
                    s.mean(1) / s.square().mean(1).sqrt().clamp_min(1e-6),
                    l.abs().mean(1),
                ],
                1,
            )
            scores.extend(sc.cpu().numpy())
        scores = np.asarray(scores)
        scene = scores[0]
        sc = scores[1:].reshape(n, 3, -1)
        cam = np.load(root / f"rewrite-v1-{split}-stegastamp" / (name + ".npy")).astype(
            np.float32
        )[::2]
        cam = np.repeat(np.repeat(cam, 2, 1), 2, 2)
        fs = []
        for j, a in enumerate(A):
            yy, xx = np.where(a)
            area = a.mean()
            ww = (np.ptp(xx) + 1) / 128 if len(xx) else 0
            ht = (np.ptp(yy) + 1) / 128 if len(yy) else 0
            inside = rare[a].mean() if a.any() else 0
            ring = binary_dilation(a, iterations=3) & ~a
            around = rare[ring].mean() if ring.any() else 0
            cm = cam[:, a].mean(1) if a.any() else np.zeros(len(cam))
            features = [
                float(j == 0),
                gate,
                area,
                ww,
                ht,
                area / (ww * ht + 1e-8),
                inside,
                around,
                inside - around,
                base.mean(),
                abs(area - base.mean()),
                float(score_each(a[None], base)[0]),
                rare.mean(),
                rare.std(),
            ]
            fs.append(
                np.r_[features, scene, sc[j].ravel(), (sc[j] - scene).ravel(), cm]
            )
        X = np.nan_to_num(np.array(fs, dtype=np.float32))
        M = A ^ full
        tmp = f.with_suffix(".partial.npz")
        np.savez_compressed(tmp, X=X, M=np.packbits(M.reshape(n, -1), axis=1))
        tmp.replace(f)
        return M, X

    def collect(split, block, rr, ids):
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            data = []
            for k, i in enumerate(ids):
                data.append(item(split, block, rr, i, z))
                if k % 40 == 0:
                    print("CACHE", split, k, len(ids), flush=True)
        return data

    ds = collect("train", gb, rows, tr_s + dd)
    train = ds[: len(tr_s)]
    devel = ds[len(tr_s) :]
    labels = y[:, 0].numpy() > 0.5
    target = [
        score_each(M, labels[i]) - score_each(M[:1], labels[i])[0]
        for (M, _), i in zip(train, tr_s)
    ]
    # Baseline / geometry only / geometry + native decoder (with and without shrinkage toward keep-original).
    models = {}
    for arm in ("geometry", "message"):
        cut = 14 if arm == "geometry" else ds[0][1].shape[1]
        reg = ExtraTreesRegressor(
            n_estimators=160,
            max_features=0.65,
            min_samples_leaf=6,
            random_state=seed,
            n_jobs=2,
        )
        X = np.concatenate([v[:, :cut] for _, v in train])
        reg.fit(
            X,
            np.concatenate(target),
            sample_weight=np.concatenate([np.full(len(t), 1 / len(t)) for t in target]),
        )
        models[arm] = (cut, reg)

    def choose(row, arm, margin):
        ms, X = row
        if arm == "baseline":
            return ms[0]
        cut, reg = models[arm]
        v = reg.predict(X[:, :cut])
        j = np.argmax(v)
        return ms[j if v[j] > max(v[0], 0) + margin else 0]

    configs = [("baseline", 0)] + [
        (arm, margin)
        for arm in ("geometry", "message")
        for margin in (0.0, 0.015, 0.045)
    ]
    results = [
        float(
            np.mean(
                [
                    score_each(choose(row, *c)[None], labels[i])[0]
                    for row, i in zip(devel, dd)
                ]
            )
        )
        for c in configs
    ]
    pick = int(np.argmax(results))
    c = configs[pick]
    print("DEV", dict(zip(map(str, configs), results)), c, flush=True)
    pred = np.stack([choose(row, *c) for row in collect("train", gb, rows, hh)])
    hold = float(score_each(pred, labels[hh]).mean())
    info = {
        "method": "stegastamp",
        "dev": results[pick],
        "hold": hold,
        "n": len(hh),
        "selection": c,
        "inner_ablations": dict(zip(map(str, configs), results)),
    }
    print("HOLD", info, flush=True)
    np.savez_compressed(out / "stegastamp-hold.npz", ids=hh, masks=pred)
    idx = [i for i, r in enumerate(test) if r["method"] == "stegastamp"]
    tm = np.stack([choose(row, *c) for row in collect("test", tb, test, idx)])
    assert tm.shape == (340, 128, 128)
    np.savez_compressed(out / "stegastamp-test.npz", ids=idx, masks=tm)
    (out / "stegastamp.json").write_text(json.dumps(info))
    print("TEST", len(idx), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    run(a.root)
