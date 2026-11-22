"""Mixture of two complementary mask models. full_watermark is a training target, never a test input."""

import argparse, io, json, time, zipfile
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, zoom
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from fit_message_fusion import feature, wait_features
from train_grid import METHODS, read, split_ids
from residual_probe import miou


def extract(root, split, m, rows, indices, x):
    local = []
    globalf = []
    grid = np.linspace(-1, 1, 64, dtype=np.float32)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    coord = np.stack([abs(xx), abs(yy), xx * xx + yy * yy], -1).reshape(-1, 3)
    for i in indices:
        r = rows[i]
        n = r["image_name"]
        s = np.load(root / f"semantic-v1-{split}" / (n + ".npy")).astype(np.float32)
        c = np.load(root / f"cam-v1-{split}-{m}" / (n + ".npy")).astype(np.float32)
        z = feature(root, split, r, x[i])
        l = np.concatenate(
            (
                z,
                coord,
                s.reshape(4, -1).T,
                uniform_filter(s, size=(1, 9, 9)).reshape(4, -1).T,
            ),
            axis=1,
        )
        gg = np.concatenate((c, s, x[i, ::6].astype(np.float32)))
        gg = gg.reshape(len(gg), -1)
        q = np.quantile(gg, [0.05, 0.25, 0.50, 0.75, 0.95], axis=1).reshape(-1)
        globalf.append(np.r_[gg.mean(1), gg.std(1), q])
        local.append(l.astype(np.float16))
    return local, np.array(globalf, np.float32)


def train(local, globalf, labels, pol, ids):
    rng = np.random.default_rng(22)
    gate1 = ExtraTreesClassifier(
        n_estimators=220,
        min_samples_leaf=2,
        max_features=0.6,
        random_state=24,
        n_jobs=2,
    ).fit(globalf[ids], pol[ids])
    gate2 = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)
    ).fit(globalf[ids], pol[ids])
    heads = []
    for b in (0, 1):
        trainx = []
        trainy = []
        wt = []
        for i in ids:
            if pol[i] != b:
                continue
            lbl = labels[i] if b == 0 else ~labels[i]
            g = lbl.reshape(64, 2, 64, 2).mean((1, 3)).ravel() >= 0.5
            pos = np.flatnonzero(g)
            neg = np.flatnonzero(~g)
            ix = np.r_[
                rng.choice(pos, min(len(pos), 320), replace=False),
                rng.choice(neg, min(len(neg), 720), replace=False),
            ]
            trainx.append(local[i][ix])
            trainy.append(g[ix])
            wt.append(np.ones(len(ix), np.float32) * 640 / max(1, len(ix)))
        heads.append(
            HistGradientBoostingClassifier(
                max_iter=115,
                max_leaf_nodes=15,
                l2_regularization=6,
                min_samples_leaf=40,
                early_stopping=False,
                random_state=22,
            ).fit(
                np.concatenate(trainx),
                np.concatenate(trainy),
                sample_weight=np.concatenate(wt),
            )
        )
    return gate1, gate2, heads


def infer(models, local, globalf, ix):
    a, b, heads = models
    gate = (
        a.predict_proba(globalf[ix])[:, 1] + b.predict_proba(globalf[ix])[:, 1]
    ) * 0.5
    maps = np.array(
        [[h.predict_proba(local[i])[:, 1].reshape(64, 64) for h in heads] for i in ix],
        np.float32,
    )
    return gate, maps


def render(gate, maps, params):
    cut, t0, t1, s = params
    pol = gate > cut
    return np.array(
        [
            (
                (
                    zoom(gaussian_filter(p[int(b)], s), 2, order=1, prefilter=False)
                    > (t1 if b else t0)
                )
                != b
            )
            for p, b in zip(maps, pol)
        ]
    )


def tune(gates, maps, targets):
    # Tune head thresholds separately conditional on the gate decision, on development images only.
    best = (-1, None)
    for cut in (0.40, 0.50, 0.60):
        isfull = gates > cut
        for s in (0.8, 2):
            raw = np.stack(
                [
                    zoom(gaussian_filter(p[int(f)], s), 2, order=1, prefilter=False)
                    for p, f in zip(maps, isfull)
                ]
            )
            vals = []
            ths = []
            for b in (False, True):
                scores = []
                select = isfull == b
                for t in (0.12, 0.22, 0.32, 0.45, 0.58):
                    scores.append(
                        (
                            sum(
                                miou((u > t) != b, v)
                                for u, v in zip(raw[select], targets[select])
                            ),
                            t,
                        )
                    )
                sc, t = max(scores)
                vals.append(sc)
                ths.append(t)
            val = sum(vals) / len(targets)
            if val > best[0]:
                best = val, (cut, *ths, s)
    return best


def run(root, m):
    train_x, r, y = read(root, "train")
    test_x, rr, _ = read(root, "test")
    tr, dv, ho = split_ids(r)
    tr, dv, ho = ([i for i in ids if r[i]["method"] == m] for ids in (tr, dv, ho))
    ids = tr + dv + ho
    ii = np.arange(len(ids))
    t, d, h = np.split(ii, [len(tr), len(tr) + len(dv)])
    pol = np.array([r[i]["full_watermark"] for i in ids], int)
    lbl = y[ids, 0].numpy().astype(bool)
    out = root / "polarity-v1"
    out.mkdir(exist_ok=True)
    testids = [i for i, a in enumerate(rr) if a["method"] == m]

    def ready(split, rows, sel):
        wait_features(root, split, m, rows, sel)
        deadline = time.monotonic() + 1200
        while not all(
            (root / f"semantic-v1-{split}" / (rows[i]["image_name"] + ".npy")).is_file()
            for i in sel
        ):
            if time.monotonic() > deadline:
                raise TimeoutError("semantic cache")
            time.sleep(5)

    ready("train", r, ids)
    local, globalf = extract(root, "train", m, r, ids, train_x)
    with threadpool_limits(limits=2):
        model = train(local, globalf, lbl, pol, t)
        gd, md = infer(model, local, globalf, d)
        score, params = tune(gd, md, lbl[d])
        gh, mh = infer(model, local, globalf, h)
        pred = render(gh, mh, params)
        report = {
            "method": m,
            "dev": float(score),
            "hold": float(np.mean([miou(a, b) for a, b in zip(pred, lbl[h])])),
            "gate_hold": float(np.mean((gh > params[0]) == pol[h])),
            "params": params,
            "n": len(h),
        }
        print("RESULT", report, flush=True)
        (out / f"{m}.json").write_text(json.dumps(report))
        np.savez_compressed(out / f"{m}-hold.npz", ids=ho, masks=pred)
        model = train(local, globalf, lbl, pol, ii)
        del local
        ready("test", rr, testids)
        allp = []
        for a in range(0, len(testids), 16):
            use = testids[a : a + 16]
            loc, gf = extract(root, "test", m, rr, use, test_x)
            gg, maps = infer(model, loc, gf, list(range(len(use))))
            allp.extend(render(gg, maps, params))
        np.savez_compressed(out / f"{m}-test.npz", ids=testids, masks=np.stack(allp))
        print("TEST", m, len(allp), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("method", choices=METHODS)
    a = p.parse_args()
    run(a.root, a.method)
