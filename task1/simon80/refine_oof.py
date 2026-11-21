"""Post-Simon experiment: learn per-image shape repairs from CLEAN out-of-fold probabilities only.
Control = Simon's clean, assembled four-fold head. No fallback to in-sample dev predictions.
All shared inputs are read-only. Never submits; preserves unmodeled methods from the named base.
"""

import argparse, hashlib, io, json, zipfile, os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as nd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import StratifiedKFold

SIMON_COMMIT = "00d38f7807b93dabe3b622a0d684c004cb9e808d"
ROOTS = [
    Path("/p/scratch/training2666/qin7/task1/runs/fusion"),
    Path("/p/scratch/training2666/ding7/claude_task1/runs/fusion"),
]
ARCHIVE_HASH = "06ea399cbefc63b51bace702fb2ac4a69d47a5e21643d429badacf86a8c41ce2"
SOURCES = {
    "pixelseal": ("qin7_s3_v3_ps_4fold_pixelseal", "claude_s3_512_pixelseal"),
    "trustmark": ("qin7_s3_v3_tm_4fold_trustmark", "claude_s3_512real_cv_trustmark"),
    "stegastamp": ("qin7_s3_ss_v2_4fold_stegastamp", None),
}


def metric(ms, y):
    ms = np.asarray(ms, dtype=bool)
    y = np.asarray(y, dtype=bool)
    out = 0.0
    for a, b in ((ms, y), (~ms, ~y)):
        u = (a | b).sum(axis=(-2, -1))
        out += 0.5 * np.where(u > 0, (a & b).sum(axis=(-2, -1)) / u.clip(min=1), 1.0)
    return out


def team_hold(rows, labels):
    keep = set()
    for m in SOURCES:
        sub = [r for r in rows if r["method"] == m]
        rng = np.random.default_rng(2026)
        for polarity in (False, True):
            pool = sorted(
                (r["image_name"] for r in sub if bool(r["full_watermark"]) == polarity),
                key=lambda n: (labels[n].mean(), n),
            )
            keep.update(
                str(rng.choice(g)) for g in np.array_split(pool, round(len(pool) * 0.2))
            )
    assert len(keep) == 102
    return keep


def load(folder, split, names):
    with np.load(folder / (split + ".npz"), allow_pickle=False) as z:
        if bool(z["dev_in_sample"].item()):
            raise ValueError("Refusing in-sample probabilities: " + str(folder))
        nn = z["image_names"].tolist()
        assert len(nn) == len(set(nn)) == len(names) and set(nn) == set(names)
        mp = {n: i for i, n in enumerate(nn)}
        p = z["response"][[mp[n] for n in names]].astype(np.float32)
        assert (
            p.shape == (len(nn), 128, 128)
            and np.isfinite(p).all()
            and p.min() >= 0
            and p.max() <= 1
        )
        return p


def logit(p):
    p = np.clip(p, 0.0002, 0.9998)
    return np.log(p / (1 - p))


def rect(a):
    yy, xx = np.where(a)
    out = np.zeros_like(a)
    if len(xx):
        out[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1] = True
    return out


def candidates(z):
    full = bool((z >= 0).mean() > 0.5)
    v = -z if full else z
    rare = v > 0 if full else v >= 0
    base = rare.copy()
    cc, n = nd.label(rare)
    ct = np.bincount(cc.ravel(), minlength=1)
    ct[0] = 0
    big = (cc == np.argmax(ct)) if n else rare
    box = rect(big)
    ary = [
        base,
        v > 0.45,
        v > -0.45,
        big,
        rect(base),
        box,
        nd.binary_fill_holes(base),
        nd.binary_closing(base, iterations=2, border_value=0),
        ((v > 0.6) | (box & (v > -0.9))),
        np.zeros_like(base),
    ]
    ys, xs = np.where(big)
    compact = big.sum() / max(1, box.sum())
    glob = [
        int(full),
        rare.mean(),
        big.mean(),
        compact,
        n / 30,
        np.mean(abs(v) < 0.8),
        *np.quantile(v, [0.02, 0.1, 0.5, 0.9, 0.98]),
    ]
    features = []
    for j, m in enumerate(ary):
        yy, xx = np.where(m)
        area = m.mean()
        per = nd.binary_dilation(m) ^ m
        ix = m ^ base
        bounds = m.sum() / max(1, rect(m).sum())
        features.append(
            [
                j,
                *glob,
                area,
                area - rare.mean(),
                bounds,
                ix.mean(),
                v[ix].mean() if ix.any() else 0.0,
                v[m].mean() if m.any() else 0.0,
                v[per].mean() if per.any() else 0.0,
                (np.ptp(xx) + 1) / 128 if len(xx) else 0.0,
                (np.ptp(yy) + 1) / 128 if len(yy) else 0.0,
            ]
        )
    return np.stack(ary) ^ full, np.array(features, np.float32)


def fit_mix(A, B, Y, ii):
    best = (-np.inf, 0.0, 0.0)
    for w in (0.0, 0.3, 0.55) if B is not None else (0.0,):
        x = A[ii] if w == 0 else (1 - w) * A[ii] + w * B[ii]
        for t in (0.0, -0.45, 0.45, -0.9, 0.9, -1.4, 1.4):
            sc = float(metric(x >= t, Y[ii]).mean())
            if sc > best[0]:
                best = (sc, w, t)
    return best[1:]


def field(A, B, w, t):
    return (A if w == 0 else (1 - w) * A + w * B) - t


def learn(p, y):
    fs = []
    targets = []
    for a, b in zip(p, y):
        masks, x = candidates(a)
        q = metric(masks, b)
        fs.append(x)
        targets.extend(q - q[0])
    return ExtraTreesRegressor(
        n_estimators=96,
        random_state=4819,
        min_samples_leaf=10,
        max_features=0.65,
        n_jobs=3,
    ).fit(np.concatenate(fs), targets)


def correct(p, model):
    preds = []
    for v in p:
        ms, x = candidates(v)
        g = model.predict(x)
        j = int(np.argmax(g))
        preds.append(ms[j] if g[j] > max(g[0], 0) + 0.012 else ms[0])
    return np.stack(preds)


def run(args):
    root = args.root
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(root / "train.zip") as z:
        rows = json.loads(z.read("train/metadata.json"))
        with np.load(
            io.BytesIO(z.read("train/masks_128.npz")), allow_pickle=False
        ) as f:
            truth = dict(zip(f["names"], f["masks"].astype(bool)))
    with zipfile.ZipFile(root / "test.zip") as z:
        test = json.loads(z.read("test/metadata.json"))
    H = team_hold(rows, truth)
    tn = [r["image_name"] for r in test]
    assert len(set(tn)) == len(tn) == 1020
    archive = args.base
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_HASH
    ref = torch.load(archive, map_location="cpu", weights_only=True)
    bn = ref["image_names"]
    assert (
        len(bn) == 1020
        and set(bn) == set(tn)
        and ref["masks"].shape == (1020, 128, 128)
        and ((ref["masks"] == 0) | (ref["masks"] == 1)).all()
    )
    idx = {n: i for i, n in enumerate(bn)}
    archive_masks = ref["masks"][[idx[n] for n in tn]].bool()
    result = archive_masks.clone()
    report = {
        "simon_commit": SIMON_COMMIT,
        "archive_sha256": ARCHIVE_HASH,
        "public_score": None,
        "methods": {},
    }
    hold_scores = []
    for m, (simon, aux) in SOURCES.items():
        primary = ROOTS[0] / simon
        extra = ROOTS[1] / aux if aux else None
        if not all(
            (primary / (k + ".npz")).is_file() for k in ("dev", "holdout", "test")
        ):
            print("PRESERVE ARCHIVE:", m, "no assembled clean OOF", flush=True)
            report["methods"][m] = {"source": "unchanged archive", "hold": None}
            continue
        if extra and not all(
            (extra / (k + ".npz")).is_file() for k in ("dev", "holdout", "test")
        ):
            extra = None
        names = [
            r["image_name"]
            for r in rows
            if r["method"] == m and r["image_name"] not in H
        ]
        hn = [
            r["image_name"] for r in rows if r["method"] == m and r["image_name"] in H
        ]
        ti = [i for i, r in enumerate(test) if r["method"] == m]
        testn = [tn[i] for i in ti]
        assert (len(names), len(hn), len(ti)) == (136, 34, 340)
        labels = np.stack([truth[n] for n in names])
        pol = [
            int(r["full_watermark"])
            for r in rows
            if r["method"] == m and r["image_name"] not in H
        ]
        A = logit(load(primary, "dev", names))
        B = logit(load(extra, "dev", names)) if extra else None
        folds = []
        pairs = []
        for fit, val in StratifiedKFold(4, shuffle=True, random_state=4819).split(
            names, pol
        ):
            # Per-fold thresholds/mixing use training rows only. Never fit the shape gate on its own evaluation images.
            _, st = fit_mix(A, None, labels, fit)
            w, t = fit_mix(A, B, labels, fit)
            field_fit = field(A[fit], None if B is None else B[fit], w, t)
            v = field(A[val], None if B is None else B[val], w, t)
            model = learn(field_fit, labels[fit])
            p = correct(v, model)
            raw = metric(A[val] >= st, labels[val])
            mix = metric(v >= 0, labels[val])
            shape = metric(p, labels[val])
            folds.append([float(q.mean()) for q in (raw, mix, shape)])
            pairs.append(np.stack([raw, mix, shape]))
            print("META FOLD", m, folds[-1], flush=True)
        cv = np.array(folds)
        chosen = 0
        for k in (1, 2):
            if (cv[:, k] - cv[:, chosen]).mean() > 0.003 and np.sum(
                cv[:, k] > cv[:, chosen]
            ) >= 3:
                chosen = k
        _, base_t = fit_mix(A, None, labels, np.arange(len(names)))
        w, t = (
            (0.0, base_t)
            if chosen == 0
            else fit_mix(A, B, labels, np.arange(len(names)))
        )
        net = learn(field(A, B, w, t), labels) if chosen == 2 else None
        # Choices are frozen before evaluating the 34 held-out images or opening test probabilities.
        vals = {}
        for split, nn in [("holdout", hn), ("test", testn)]:
            aa = logit(load(primary, split, nn))
            bb = logit(load(extra, split, nn)) if w > 0 else None
            v = field(aa, bb, w, t)
            pred = correct(v, net) if net is not None else v >= 0
            if split == "holdout":
                yy = np.stack([truth[n] for n in nn])
                vals["base_hold"] = float(metric(aa >= base_t, yy).mean())
                vals["hold"] = float(metric(pred, yy).mean())
                hold_scores.append(vals["hold"])
                np.savez_compressed(
                    out / (m + "-hold.npz"), image_names=np.asarray(nn), masks=pred
                )
            else:
                result[ti] = torch.from_numpy(pred)
                np.savez_compressed(
                    out / (m + "-test.npz"), image_names=np.asarray(nn), masks=pred
                )
        report["methods"][m] = {
            "source": primary.name,
            "aux": extra.name if extra else None,
            "rule": ["simon-only", "blend", "gated-shape"][chosen],
            "weight_aux": w,
            "threshold_logit": t,
            "meta_cv": cv.mean(0).tolist(),
            "fold_scores": folds,
            **vals,
        }
        print("METHOD", m, report["methods"][m], flush=True)
    for m, stats in report["methods"].items():
        if stats["hold"] is None:
            inds = [i for i, r in enumerate(test) if r["method"] == m]
            assert torch.equal(result[inds], archive_masks[inds])
    assert len(hold_scores) > 0, "No clean OOF inputs. No candidate."
    dst = out / "submission.pt"
    torch.save({"image_names": tn, "masks": result}, dst)
    v = torch.load(dst, map_location="cpu", weights_only=True)
    assert (
        v["image_names"] == tn
        and torch.equal(v["masks"], result)
        and dst.stat().st_size < 20_000_000
    )
    report["local_overall"] = (
        float(np.mean(hold_scores)) if len(hold_scores) == 3 else None
    )
    report["test_n"] = len(tn)
    report["sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("DONE", report["local_overall"], report["sha256"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--base",
        type=Path,
        default=Path("/p/scratch/training2666/qin7/task1/runs/submissions/sub_v1.pt"),
    )
    run(p.parse_args())
