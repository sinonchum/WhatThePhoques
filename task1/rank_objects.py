"""Image-only SAM/box hypotheses ranked with local watermark evidence. No test labels."""

from pathlib import Path
import argparse, io, json, zipfile, time
import numpy as np
from scipy.ndimage import label, gaussian_filter, binary_dilation
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from train_grid import METHODS, read, split_ids
from residual_probe import miou


def box_mask(bb, side=64):
    x, y, xx, yy = np.clip(np.rint(bb), 0, side).astype(int)
    a = np.zeros((side, side), bool)
    a[y:yy, x:xx] = 1
    return a


def make(root, split, r, block):
    m, n = r["method"], r["image_name"]
    sam = root / f"sam-v1-{split}" / (n + ".npz")
    cam = root / f"cam-v1-{split}-{m}" / (n + ".npy")
    wr = root / f"rewrite-v1-{split}-{m}" / (n + ".npy")
    tile = root / f"tile-v1-{split}-{m}" / (n + ".npy")
    bbox = root / f"bbox-v1-{split}" / (n + ".npz")
    deadline = time.monotonic() + 3600
    while not all(p.exists() for p in (sam, cam, wr, tile)) or (
        m == "trustmark" and not bbox.exists()
    ):
        if time.monotonic() > deadline:
            raise TimeoutError("waiting for evidence")
        time.sleep(4)
    z = np.load(sam)
    sam128 = np.unpackbits(z["masks"], axis=1).reshape(-1, 128, 128).astype(bool)
    sam64 = sam128[:, ::2, ::2]
    maps = np.load(cam).astype(np.float32)
    rew = np.load(wr).astype(np.float32)
    tiles = np.load(tile).astype(np.float32)
    sem = np.load(root / f"semantic-v1-{split}" / (n + ".npy")).astype(np.float32)
    bank = np.concatenate((maps, rew, tiles[-8:], sem, block[::6].astype(np.float32)))
    base = bank.reshape(len(bank), -1)
    avg = base.mean(1)
    sd = base.std(1)
    gf = np.r_[
        avg, sd, np.quantile(base, [0.02, 0.10, 0.90, 0.98], axis=1).ravel()
    ].astype(np.float32)
    bank = (bank - avg[:, None, None]) / (sd[:, None, None] + 0.015)
    base = bank.reshape(len(bank), -1)
    masks = []
    attrs = []
    seen = set()

    def add(a, typ, score=0):
        area = a.sum()
        if area < 2 or area > 3200:
            return
        key = np.packbits(a).tobytes()
        if key in seen:
            return
        seen.add(key)
        masks.append(a)
        attrs.append((typ, score))

    for j, a in enumerate(sam64):
        add(a, 1, float(z["iou"][j]))
        ys, xs = np.where(a)
        if len(xs) > 0:
            add(
                box_mask((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)),
                2,
                float(z["iou"][j]),
            )
    if m == "trustmark":
        with np.load(bbox) as zbox:
            for bb, s in zip(zbox["boxes"], zbox["scores"]):
                add(box_mask(bb / 2), 3, s)
    # Select plausible rectangles at multiple sizes by contrast to the image mean.
    indices = [1, 5, 9, 17, 21] if m != "stegastamp" else [0, 1, 3, 5]
    probes = [
        bank[indices].mean(0),
        bank[len(maps) : len(maps) + 12].mean(0),
        sem[0],
        bank[len(maps) + 12 : len(maps) + 20].mean(0),
    ]
    for f in probes:
        f = gaussian_filter(f, 1)
        ii = np.pad(f, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        for w, h in (
            (3, 5),
            (6, 6),
            (9, 9),
            (14, 12),
            (16, 6),
            (6, 16),
            (20, 21),
            (27, 12),
            (12, 27),
            (34, 27),
            (48, 36),
        ):
            sums = ii[h:, w:] - ii[h:, :-w] - ii[:-h, w:] + ii[:-h, :-w]
            for idx in (np.argmax(sums), np.argmin(sums)):
                y, x = np.unravel_index(idx, sums.shape)
                add(box_mask((x, y, x + w, y + h)), 4)
        for q in (0.05, 0.16, 0.84, 0.95):
            b = f < np.quantile(f, q) if q < 0.5 else f > np.quantile(f, q)
            cc, num = label(b)
            if num:
                freq = np.bincount(cc.ravel())
                freq[0] = 0
                for j in np.argsort(freq)[-2:]:
                    if j > 0:
                        add(cc == j, 5)
    if not masks:
        add(box_mask((24, 24, 40, 40)), 4)
    A = np.stack(masks)
    v = A.reshape(len(A), -1).astype(np.float32)
    w = v.sum(1)
    ring = (
        np.stack([binary_dilation(a, iterations=3) ^ a for a in A])
        .reshape(len(A), -1)
        .astype(np.float32)
    )
    q = np.maximum(ring.sum(1), 1)
    inner = (v @ base.T) / w[:, None]
    diff = inner - (ring @ base.T) / q[:, None]
    z2 = (v @ (base * base).T) / w[:, None]
    shape = []
    for a, (typ, s) in zip(A, attrs):
        y, x = np.where(a)
        width = np.ptp(x) + 1
        height = np.ptp(y) + 1
        shape.append(
            [
                typ,
                s,
                width / 64,
                height / 64,
                len(x) / 4096,
                len(x) / (width * height),
                (x.mean() - 31.5) / 32,
                (y.mean() - 31.5) / 32,
            ]
        )
    return A, np.column_stack([shape, inner, diff, z2]).astype(np.float32), gf


def run(root, m):
    x, rr, y = read(root, "train")
    tx, trows, _ = read(root, "test")
    tr, dv, h = split_ids(rr)
    tr, dv, h = ([i for i in ids if rr[i]["method"] == m] for ids in (tr, dv, h))
    ids = tr + dv + h
    targets = y[:, 0].numpy() > 0.5
    pol = np.array([bool(r["full_watermark"]) for r in rr])
    out = root / "rank-v1"
    out.mkdir(exist_ok=True)
    cache = root / "rank-cache-v1"
    cache.mkdir(exist_ok=True)

    def item(split, r, b):
        p = cache / (split + "-" + r["image_name"] + ".npz")
        if p.exists():
            with np.load(p) as z:
                return (
                    np.unpackbits(z["masks"], axis=1).reshape(-1, 64, 64).astype(bool),
                    z["X"].astype(np.float32),
                    z["g"],
                )
        A, X, g = make(root, split, r, b)
        t = p.with_suffix(".partial.npz")
        np.savez_compressed(
            t,
            masks=np.packbits(A.reshape(len(A), -1), axis=1),
            X=X.astype(np.float16),
            g=g,
        )
        t.replace(p)
        return A, X, g

    feats = {}
    globals_ = {}
    iou = {}
    ms = {}
    for i in ids:
        aa, feat, g = item("train", rr[i], x[i])
        tg = targets[i, ::2, ::2] ^ pol[i]
        u = (aa | tg).sum((1, 2))
        scores = (aa & tg).sum((1, 2)) / np.maximum(u, 1)
        feats[i] = feat
        globals_[i] = g
        iou[i] = scores
        ms[i] = aa

    def fit(train):
        gate = ExtraTreesClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.65,
            n_jobs=2,
            random_state=717,
        ).fit(np.stack([globals_[i] for i in train]), pol[train])
        heads = []
        for b in (False, True):
            ix = [i for i in train if pol[i] == b]
            X = np.concatenate([feats[i] for i in ix])
            Y = np.concatenate([iou[i] for i in ix])
            model = ExtraTreesRegressor(
                n_estimators=48,
                max_features=0.45,
                min_samples_leaf=5,
                n_jobs=2,
                max_samples=0.8,
                bootstrap=True,
                random_state=717,
            ).fit(X, Y, sample_weight=0.3 + 3 * Y)
            heads.append(model)
        return gate, heads

    def predict(model, aa, feat, g):
        gate, heads = model
        pv = gate.predict_proba(g[None])[0, 1]
        b = pv > 0.5
        sc = heads[int(b)].predict(feat)
        ix = np.argsort(-sc)[:4]
        w = np.maximum(sc[ix], 0.001) ** 4
        mix = np.average(aa[ix], axis=0, weights=w)
        pred = (mix > 0.5) ^ b
        return pred.repeat(2, 0).repeat(2, 1)

    with threadpool_limits(limits=2):
        model = fit(tr)
        dvsc = np.mean(
            [miou(predict(model, ms[i], feats[i], globals_[i]), targets[i]) for i in dv]
        )
        pred = np.stack([predict(model, ms[i], feats[i], globals_[i]) for i in h])
        held = np.mean([miou(a, targets[i]) for a, i in zip(pred, h)])
        r = {"method": m, "dev": float(dvsc), "hold": float(held), "n": len(h)}
        np.savez_compressed(out / f"{m}-hold.npz", ids=h, masks=pred)
        (out / f"{m}.json").write_text(json.dumps(r))
        print("RESULT", r, flush=True)
        model = fit(ids)
        pred = []
        testids = [i for i, r in enumerate(trows) if r["method"] == m]
        for i in testids:
            a, feat, g = item("test", trows[i], tx[i])
            pred.append(predict(model, a, feat, g))
        np.savez_compressed(out / f"{m}-test.npz", ids=testids, masks=np.stack(pred))
        print("TEST", m, len(pred), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("method", choices=METHODS)
    a = p.parse_args()
    run(a.root, a.method)
