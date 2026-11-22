"""Train and select CART-boosted message/forensic heads on nested, image-stratified splits.
Writes scored validation summaries per method and a checked .pt when all test predictions exist.
"""

import argparse, json, io, zipfile, time
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, zoom
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import torch
from train_grid import read, split_ids, METHODS
from residual_probe import miou


def feature(root, split, row, block):
    path = root / f"cam-v1-{split}-{row['method']}" / (row["image_name"] + ".npy")
    cam = np.load(path).astype(np.float32)
    b = block[::5].astype(np.float32)

    def window(x, s):
        return uniform_filter(x, size=(1, s, s))

    def glob(x):
        return np.broadcast_to(x, (x.shape[0], 64, 64))

    def scorestats(x):
        return (
            np.stack(
                [x.mean((1, 2)), x.std((1, 2)), np.quantile(x, 0.90, axis=(1, 2))], 1
            )
            .reshape(-1, 1, 1)
            .astype(np.float32)
        )

    out = [
        cam,
        window(cam, 5),
        window(cam, 13),
        glob(scorestats(cam)),
        b,
        window(b, 7),
        window(b, 21),
        glob(b.mean((1, 2), keepdims=True)),
    ]
    return np.concatenate(out).reshape(-1, 4096).T


def wait_features(root, split, method, rows, ids):
    cache = root / f"cam-v1-{split}-{method}"
    deadline = time.monotonic() + 2700
    while not all((cache / (rows[i]["image_name"] + ".npy")).exists() for i in ids):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{split} {method} maps not ready")
        time.sleep(5)


def main(root, method, seed):
    x, rows, g = read(root, "train")
    g = g[:, 0].numpy().astype("uint8")
    tx, rr, _ = read(root, "test")
    tr, dev, ho = split_ids(rows)
    filt = lambda ids: [i for i in ids if rows[i]["method"] == method]
    tr, dev, ho = map(filt, (tr, dev, ho))
    allids = tr + dev + ho
    wait_features(root, "train", method, rows, allids)
    store = root / f"blend-v1-{seed}"
    store.mkdir(exist_ok=True)

    def fit(ids):
        rng = np.random.default_rng(seed)
        xt = []
        yt = []
        for i in ids:
            fx = feature(root, "train", rows[i], x[i])
            pix = rng.choice(4096, 720, replace=False)
            yy = g[i].reshape(64, 2, 64, 2).mean((1, 3)).ravel()
            xt.append(fx[pix])
            yt.append(yy[pix] >= 0.5)
        return HistGradientBoostingClassifier(
            max_iter=130,
            max_leaf_nodes=15,
            l2_regularization=4,
            min_samples_leaf=80,
            learning_rate=0.075,
            early_stopping=False,
            random_state=seed,
        ).fit(np.concatenate(xt), np.concatenate(yt))

    def maps(model, split, rs, data, ids):
        return np.array(
            [
                model.predict_proba(feature(root, split, rs[i], data[i]))[:, 1].reshape(
                    64, 64
                )
                for i in ids
            ],
            np.float32,
        )

    def up(a, s):
        return zoom(gaussian_filter(a, s) if s else a, 2, order=1, prefilter=False)

    def evalset(pred, ids, thr, s):
        return float(np.mean([miou(up(p, s) > thr, g[i]) for p, i in zip(pred, ids)]))

    with threadpool_limits(limits=3):
        model = fit(tr)
        dv = maps(model, "train", rows, x, dev)
        best = max(
            (evalset(dv, dev, t, s), t, s)
            for t in (0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75)
            for s in (0, 1, 2)
        )
        val = maps(model, "train", rows, x, ho)
        held = evalset(val, ho, best[1], best[2])
        report = {
            "method": method,
            "seed": seed,
            "dev": best[0],
            "threshold": best[1],
            "sigma": best[2],
            "hold": held,
            "n": len(ho),
        }
        print("RESULT", report, flush=True)
        (store / f"{method}.json").write_text(json.dumps(report))
        np.savez_compressed(
            store / f"{method}-hold.npz",
            ids=ho,
            masks=np.array([up(p, best[2]) > best[1] for p in val]),
        )
        model = fit(allids)
        ids = [i for i, r in enumerate(rr) if r["method"] == method]
        wait_features(root, "test", method, rr, ids)
        masks = np.array(
            [up(p, best[2]) > best[1] for p in maps(model, "test", rr, tx, ids)]
        )
        np.savez_compressed(store / f"{method}-test.npz", ids=ids, masks=masks)
        print("TEST", method, len(ids), flush=True)


def pack(root, seed):
    store = root / f"blend-v1-{seed}"
    _, rr, _ = read(root, "test")
    out = np.zeros((len(rr), 128, 128), dtype=np.uint8)
    seen = []
    for m in METHODS:
        with np.load(store / f"{m}-test.npz") as f:
            ids = f["ids"]
            mask = f["masks"]
            assert (
                mask.dtype == bool
                and mask.shape == (len(ids), 128, 128)
                and all(rr[i]["method"] == m for i in ids)
            )
            out[ids] = mask
            seen.extend(ids.tolist())
    names = [r["image_name"] for r in rr]
    assert sorted(seen) == list(range(1020)) and len(set(names)) == 1020
    path = store / "submission.pt"
    torch.save({"image_names": names, "masks": torch.from_numpy(out)}, path)
    loaded = torch.load(path, weights_only=True)
    assert (
        loaded["image_names"] == names
        and np.array_equal(loaded["masks"].numpy(), out)
        and path.stat().st_size < 20_000_000
    )
    print(path, path.stat().st_size, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("method", choices=(*METHODS, "pack"))
    p.add_argument("--seed", type=int, default=17)
    a = p.parse_args()
    pack(a.root, a.seed) if a.method == "pack" else main(a.root, a.method, a.seed)
