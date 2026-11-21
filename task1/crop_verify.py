"""Refine candidate masks by decoding the known message in object-aligned crops. Dev-only selection, with an explicit no-op control."""

import argparse, io, json, zipfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label
from train_grid import METHODS, read, split_ids
from train_cue_net import Region
from ensemble_views import feature, multiview, model_score
from message_cam import Decoder


def propose(p, objects, boxes):
    n = p.shape[0]
    assert p.shape == (128, 128)
    boxes_out = []
    masks = []
    seen = set()
    rng = np.indices(p.shape)

    def add(a, box=None):
        if box is not None:
            x, y, r, b = np.asarray(np.clip(np.rint(box), 0, n), int)
            a = np.zeros_like(p, bool)
            a[y:b, x:r] = 1
        else:
            ys, xs = np.where(a)
            if not len(xs):
                return
            x, y, r, b = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        if min(r - x, b - y) < 2 or a.sum() < 4 or a.mean() > 0.93:
            return
        k = np.packbits(a).tobytes()
        if k in seen:
            return
        seen.add(k)
        boxes_out.append(tuple(map(int, (x, y, r, b))))
        masks.append(a)

    for t in (0.22, 0.35, 0.5, 0.67):
        cc, count = label(p > t)
        for j in np.argsort(np.bincount(cc.ravel())[1:])[-2:] + 1:
            add(cc == j)
            yy, xx = np.where(cc == j)
            add(None, (xx.min(), yy.min(), xx.max() + 1, yy.max() + 1))
    for box in boxes:
        add(None, box)
    for a in objects:
        add(a)
    if not masks:
        y, x = np.unravel_index(p.argmax(), p.shape)
        add(None, (max(0, x - 4), max(0, y - 4), min(n, x + 5), min(n, y + 5)))
    return boxes_out, masks


def overlap_prior(masks, p):
    a = np.asarray(masks)
    hard = p > 0.35
    inter = (a & hard).sum((1, 2))
    union = (a | hard).sum((1, 2))
    soft = (a * p).sum((1, 2))
    return 0.55 * inter / np.maximum(union, 1) + 0.45 * 2 * soft / (
        a.sum((1, 2)) + p.sum() + 1e-5
    )


@torch.inference_mode()
def run(root, method):
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    size = 245 if method == "trustmark" else 256
    gb, rows, y = read(root, "train")
    tb, trows, _ = read(root, "test")
    tr, dev, hold = split_ids(rows)
    assert y is not None
    folder = root / "neural-message-v1-1451-120-r1t1"
    stats = torch.from_numpy(
        np.stack([feature(root, "train", gb, rows, i, 2)[:, ::4, ::4] for i in tr[::3]])
    ).float()
    mu = stats.mean((0, 2, 3))[None, :, None, None].to(device)
    sd = stats.std((0, 2, 3))[None, :, None, None].clamp_min(0.005).to(device)
    model = Region(stats.shape[1])
    model.load_state_dict(
        torch.load(folder / (method + ".pth"), map_location="cpu", weights_only=True)
    )
    model.eval().to(device)
    decoder = Decoder(method, root).eval().to(device)
    tag = torch.tensor([METHODS.index(method)], device=device)
    out = root / "object-crop-v1"
    out.mkdir(exist_ok=True)

    def masks_for(split, block, meta, i, z):
        row = meta[i]
        name = row["image_name"]
        x = (
            torch.from_numpy(feature(root, split, block, meta, i, 2))
            .to(device)
            .float()[None]
        )
        g, v = multiview(model, ((x - mu) / sd).clamp(-7, 7), tag)
        full = bool(g.item() >= 0.5)
        p = v[0, int(full)].cpu().numpy()
        base = (p > 0.35) ^ full
        sam = root / f"sam-v1-{split}" / (name + ".npz")
        deadline = time.monotonic() + 2400
        while not sam.exists():
            if time.monotonic() > deadline:
                raise TimeoutError(sam)
            time.sleep(2)
        with np.load(sam) as f:
            obj = np.unpackbits(f["masks"], axis=1).reshape(-1, 128, 128).astype(bool)
        boxes = []
        if method == "trustmark":
            with np.load(root / f"bbox-v1-{split}" / (name + ".npz")) as f:
                boxes = f["boxes"]
        b, a = propose(p, obj, boxes)
        a = np.stack(a)
        prior = overlap_prior(a, p)
        order = np.argsort(-prior)
        ids = list(order[:14])
        # Always retain independent detector boxes as well: they can rescue a poor prior.
        for j, box in enumerate(b):
            if (
                boxes is not None
                and any(np.max(abs(np.asarray(box) - raw)) < 2 for raw in boxes)
                and j not in ids
            ):
                ids.append(j)
        ids = ids[:24]
        a = a[ids]
        prior = prior[ids]
        b = [b[j] for j in ids]
        im = Image.open(io.BytesIO(z.read(f"{split}/images/{name}.png"))).convert("RGB")
        scene = np.asarray(im.resize((size, size), Image.Resampling.BILINEAR))
        photos = [scene] + [
            np.asarray(
                im.crop(tuple(c * 4 for c in box)).resize(
                    (size, size), Image.Resampling.BILINEAR
                )
            )
            for box in b
        ]
        bits = torch.tensor(
            [int(c) * 2 - 1 for c in row["message"]], dtype=torch.float32, device=device
        )[None]
        scores = []
        for j in range(0, len(photos), 8):
            tt = (
                torch.tensor(np.stack(photos[j : j + 8]), device=device)
                .float()
                .permute(0, 3, 1, 2)
                / 127.5
                - 1
            )
            if method == "trustmark":
                logit = decoder.n(tt)
            else:
                h = decoder.pixel_decoder.output_upscaling(decoder.convnext(tt))
                w = decoder.pixel_decoder.linear
                logit = torch.nn.functional.linear(
                    h.mean((-2, -1)), w.weight[1:], w.bias[1:]
                )
            signed = bits * logit
            sc = 0.65 * (signed > 0).float().mean(1) + 0.35 * (
                0.5
                + 0.5 * signed.mean(1) / signed.square().mean(1).sqrt().clamp_min(1e-6)
            )
            scores.extend(sc.cpu().tolist())
        contrast = (np.asarray(scores[1:]) - scores[0]) * (-1 if full else 1)

        def snap(k):
            j = np.argmax(prior + k * contrast)
            return ((0.45 * p + 0.55 * a[j]) > 0.46) ^ full

        return np.stack([base, snap(0), snap(0.6), snap(1.8)])

    def make(split, block, meta, ids):
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            return np.stack([masks_for(split, block, meta, i, z) for i in ids])

    di = [i for i in dev if rows[i]["method"] == method]
    hi = [i for i in hold if rows[i]["method"] == method]
    ds = make("train", gb, rows, di)
    vals = [
        model_score(torch.from_numpy(ds[:, j]), y[di, 0] > 0.5)
        for j in range(ds.shape[1])
    ]
    pick = int(np.argmax(vals))
    print("DEV control/geometry/message", method, vals, "selected", pick, flush=True)
    hs = make("train", gb, rows, hi)[:, pick]
    hr = model_score(torch.from_numpy(hs), y[hi, 0] > 0.5)
    np.savez_compressed(out / f"{method}-hold.npz", ids=hi, masks=hs)
    report = {
        "method": method,
        "dev": vals[pick],
        "hold": hr,
        "n": len(hi),
        "config": pick,
        "development_ablation": vals,
    }
    print("HOLD", report, flush=True)
    ti = [i for i, r in enumerate(trows) if r["method"] == method]
    assert len(ti) == 340
    ps = make("test", tb, trows, ti)[:, pick]
    assert ps.shape == (340, 128, 128) and ps.dtype == bool
    tmp = out / f"{method}.partial.npz"
    np.savez_compressed(tmp, ids=ti, masks=ps)
    tmp.replace(out / f"{method}-test.npz")
    (out / f"{method}.json").write_text(json.dumps(report))
    print("TEST", method, len(ti), flush=True)


if __name__ == "__main__":
    par = argparse.ArgumentParser()
    par.add_argument("root", type=Path)
    par.add_argument("method", choices=["trustmark", "pixelseal"])
    args = par.parse_args()
    run(args.root, args.method)
