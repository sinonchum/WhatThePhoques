"""Fuse two clean out-of-fold experts, then test image-specific minority recall and RGB edge alignment."""

import argparse
import hashlib
import io
import json
import itertools
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.model_selection import StratifiedKFold
import torch
from refine_oof import ROOTS, SOURCES, load, logit, metric, team_hold

AUX = {
    "pixelseal": ROOTS[1] / "claude_s3_ens_512_256_pixelseal",
    "trustmark": ROOTS[1] / "claude_s3_512real_cv_trustmark",
    "stegastamp": ROOTS[0] / "qin7_s3_ss_v1_4fold_stegastamp",
}


def guided(z, rgb):
    # Three independent color regressions keep edges without inverting an ill-conditioned color covariance.
    avg = lambda x: uniform_filter(x, size=9, mode="reflect")
    zm = avg(z)
    fields = []
    for c in np.moveaxis(rgb, -1, 0):
        cm = avg(c)
        a = (avg(c * z) - cm * zm) / (np.maximum(avg(c * c) - cm * cm, 0) + 0.0025)
        fields.append(avg(a) * c + avg(zm - a * cm))
    return np.mean(fields, axis=0)


def factors(z, rgb):
    lo, mid, hi = np.quantile(z, [0.015, 0.5, 0.985])
    return (
        guided(z, rgb) - z,
        np.float32(np.clip((lo + hi) * 0.5, -3, 3)),
        np.float32(np.sign(mid)),
    )


def calibrate(z, rgb, params):
    if params == (0, 0, 0):
        return z
    d, cen, sgn = factors(z, rgb)
    e, a, b = params
    return z + e * d - a * cen - b * sgn


def guides(root, split, names):
    with zipfile.ZipFile(root / (split + ".zip")) as z:
        return np.stack(
            [
                np.asarray(
                    Image.open(io.BytesIO(z.read(f"{split}/images/{n}.png")))
                    .convert("RGB")
                    .resize((128, 128), Image.Resampling.BOX),
                    np.float32,
                )
                / 255
                for n in names
            ]
        )


def maps(key, sp, names):
    return [logit(load(p, sp, names)) for p in (ROOTS[0] / SOURCES[key][0], AUX[key])]


CONFIGS = list(
    itertools.product(
        [0.0, 0.5, 1.0], [0.0, 0.8], [0.0, 0.6], [0.0, 0.8, -0.8], [0.0, -0.6, 0.6]
    )
)


def score_family(maps, rgb, truth):
    a, b = maps
    scores = np.zeros((len(CONFIGS), len(a)), np.float32)
    for w in (0.0, 0.5, 1.0):
        z = a * (1 - w) + b * w
        ds, cs, ss = zip(*(factors(v, r) for v, r in zip(z, rgb)))
        ds = np.stack(ds)
        cs = np.asarray(cs)[:, None, None]
        ss = np.asarray(ss)[:, None, None]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tz, td, tc, ts, y = [
            torch.as_tensor(v, device=device) for v in (z, ds, cs, ss, truth)
        ]
        for j, (mw, e, h, s, t) in enumerate(CONFIGS):
            if mw != w:
                continue
            mask = tz + e * td - h * tc - s * ts >= t
            total = torch.zeros(len(y), device=device)
            for a1, b1 in ((mask, y), (~mask, ~y)):
                u = (a1 | b1).sum((-2, -1))
                hit = (a1 & b1).sum((-2, -1))
                total += 0.5 * torch.where(u > 0, hit / u.clamp_min(1), 1.0)
            scores[j] = total.cpu().numpy()
    return scores


def infer(maps, rgb, c):
    w, e, h, s, t = c
    a, b = maps
    return np.stack(
        [calibrate(z, r, (e, h, s)) >= t for z, r in zip(a * (1 - w) + b * w, rgb)]
    )


def run(root, out):
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "train.zip") as z:
        rows = json.loads(z.read("train/metadata.json"))
        with np.load(io.BytesIO(z.read("train/masks_128.npz"))) as f:
            truth = dict(zip(f["names"], f["masks"].astype(bool)))
    with zipfile.ZipFile(root / "test.zip") as z:
        test = json.loads(z.read("test/metadata.json"))
    hold = team_hold(rows, truth)
    selected = {}
    report = {}
    allm = {}
    for m in SOURCES:
        dn = [
            r["image_name"]
            for r in rows
            if r["method"] == m and r["image_name"] not in hold
        ]
        assert len(dn) == len(set(dn)) == 136
        y = np.stack([truth[n] for n in dn])
        arr = score_family(maps(m, "dev", dn), guides(root, "train", dn), y)
        base = [i for i, (w, e, h, s, t) in enumerate(CONFIGS) if e == h == s == 0]
        pairs = []
        for it, iv in StratifiedKFold(4, shuffle=True, random_state=7229).split(
            dn, y.mean((1, 2)) > 0.5
        ):
            mu = arr[:, it].mean(1)
            p = base[int(np.argmax(mu[base]))]
            j = int(np.argmax(mu))
            pairs.append([float(arr[p, iv].mean()), float(arr[j, iv].mean())])
        gain = np.diff(np.array(pairs), axis=1).ravel()
        allow = bool(gain.mean() > 0.003 and (gain > 0).sum() >= 3)
        mu = arr.mean(1)
        p = base[int(np.argmax(mu[base]))]
        j = int(np.argmax(mu)) if allow else p
        selected[m] = (CONFIGS[p], CONFIGS[j])
        report[m] = {
            "folds": pairs,
            "promoted": allow,
            "frozen": CONFIGS[j],
            "base": CONFIGS[p],
            "dev_oracle_diagnostic_only": float(arr.max(0).mean()),
        }
        print("FROZEN", m, report[m], flush=True)
    for m, (c0, c1) in selected.items():
        for sp, source, names in [
            (
                "holdout",
                "train",
                [
                    r["image_name"]
                    for r in rows
                    if r["method"] == m and r["image_name"] in hold
                ],
            ),
            ("test", "test", [r["image_name"] for r in test if r["method"] == m]),
        ]:
            assert len(names) == (34 if sp == "holdout" else 340)
            mp = maps(m, sp, names)
            g = guides(root, source, names)
            pr = infer(mp, g, c1)
            np.savez_compressed(
                out / (m + "-" + sp + ".npz"), image_names=np.array(names), masks=pr
            )
            if sp == "holdout":
                y = np.stack([truth[n] for n in names])
                report[m]["hold"] = float(metric(pr, y).mean())
                report[m]["hold_base"] = float(metric(infer(mp, g, c0), y).mean())
            else:
                allm.update(zip(names, pr))
    names = [r["image_name"] for r in test]
    assert len(allm) == len(names) == len(set(names)) == 1020 and set(names) == set(
        allm
    )
    dst = out / "submission.pt"
    packed = torch.from_numpy(np.stack([allm[n] for n in names]))
    torch.save({"image_names": names, "masks": packed}, dst)
    v = torch.load(dst, weights_only=True)
    assert (
        v["image_names"] == names
        and v["masks"].shape == (1020, 128, 128)
        and v["masks"].dtype == torch.bool
        and torch.equal(v["masks"], packed)
        and dst.stat().st_size < 20_000_000
    )
    result = {
        "methods": report,
        "hold": float(np.mean([x["hold"] for x in report.values()])),
        "hold_base": float(np.mean([x["hold_base"] for x in report.values()])),
        "public_score": None,
        "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    print("COMPLETE", result["hold_base"], result["hold"], result["sha256"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument(
        "--trustmark-v4",
        action="store_true",
        help="Compare the assembled clean V4-TMF four-fold expert against V3",
    )
    a = p.parse_args()
    if a.trustmark_v4:
        AUX["trustmark"] = ROOTS[0] / "qin7_s3_v4_tmf_4fold_trustmark"
        for split in ("dev", "holdout", "test"):
            if not (AUX["trustmark"] / (split + ".npz")).is_file():
                raise FileNotFoundError(f"Missing assembled V4 {split} bundle")
        print("V4 input:", AUX["trustmark"], flush=True)
    run(a.root, a.out)
