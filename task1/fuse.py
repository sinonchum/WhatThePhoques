"""Coordinator's fusion script: pick or blend the two workers' response maps per watermark type.

    python fuse.py --meta <train metadata.json> --masks <train masks_128.npz> \
        --src pixelseal=output/fusion/codex_smallseg_pixelseal,output/fusion/claude_synth_pixelseal \
        --src trustmark=... --src stegastamp=... --out output/fusion/_coord/v1

Every source directory holds dev.npz / holdout.npz / test.npz as specified in COORD.md
(image_names, response [N,128,128], threshold, dev_in_sample).

What it does, per watermark type:
  1. scores every source alone on dev and on held-out with the source's own threshold;
  2. builds candidates: each source alone, and the mean of standardised responses
     z_i = (response_i - threshold_i) / std_i(dev), cut at 0 (no free parameter);
     with --fit-weights it also tries blend weights and a re-fitted cut, chosen on dev only;
  3. picks the candidate with the best DEV score, using only sources whose dev maps are out of sample;
     if no source qualifies, it takes the parameter-free mean of all sources and says so;
  4. reports the chosen candidate on held-out (large / small / both) and writes test masks.

Held-out images are never used to choose anything. All scores come from the official wmloc.utils.miou_image."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import types
from pathlib import Path

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:                      # wmloc.utils only needs torch inside to_tensor()
    sys.modules["torch"] = types.ModuleType("torch")
    sys.modules["torch"].Tensor = object
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wmloc.utils import miou_image  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
SPLITS = ("dev", "holdout", "test")


def load_source(path: Path) -> dict:
    src = {"path": str(path), "name": path.name}
    for split in SPLITS:
        f = path / f"{split}.npz"
        if not f.exists():
            src[split] = None
            continue
        z = np.load(f, allow_pickle=False)
        resp = z["response"].astype(np.float32)
        assert resp.ndim == 3 and resp.shape[1:] == (128, 128), f"{f}: response shape {resp.shape}"
        assert np.isfinite(resp).all(), f"{f}: response has NaN or inf"
        names = [str(n) for n in z["image_names"].tolist()]
        assert len(names) == len(set(names)) == len(resp), f"{f}: names do not match response"
        src[split] = dict(zip(names, resp))
        src["threshold"] = float(z["threshold"])
        src["dev_in_sample"] = bool(z["dev_in_sample"]) if "dev_in_sample" in z.files else True
        if "box_score" in z.files:
            src[f"{split}_box_score"] = dict(zip(names, z["box_score"].astype(np.float32)))
    assert src["dev"] is not None and src["holdout"] is not None, f"{path}: dev.npz and holdout.npz are required"
    dev_stack = np.stack(list(src["dev"].values()))
    src["std"] = float(np.std(dev_stack)) or 1.0
    src["is_prob"] = bool(dev_stack.min() >= -1e-4 and dev_stack.max() <= 1.0 + 1e-4 and 0.0 < src["threshold"] < 1.0)
    return src


def to_logit(p: np.ndarray | float, eps: float = 1e-4) -> np.ndarray:
    c = np.clip(np.asarray(p, dtype=np.float32), eps, 1.0 - eps)
    return np.log(c / (1.0 - c))


def zmap(src: dict, split: str, names: list[str], mode: str = "z") -> np.ndarray:
    arr = np.stack([src[split][n] for n in names])
    if mode == "logit" and src["is_prob"]:
        return to_logit(arr) - to_logit(src["threshold"])
    return (arr - src["threshold"]) / src["std"]


def blend(srcs: list[dict], weights, split: str, names: list[str], mode: str = "z") -> np.ndarray:
    return sum(w * zmap(s, split, names, mode=mode) for s, w in zip(srcs, weights) if w)


def per_image(z: np.ndarray, cut: float, names: list[str], gt: dict) -> np.ndarray:
    return np.array([miou_image((m >= cut).astype(np.uint8), gt[n]) for m, n in zip(z, names)])


def split_report(scores: np.ndarray, names: list[str], full: dict) -> dict:
    large = np.array([full[n] for n in names])
    return {"large": float(scores[large].mean()), "small": float(scores[~large].mean()),
            "both": float(scores.mean()), "n": int(len(scores))}


def fmt(r: dict) -> str:
    return f"{r['large']:.4f} / {r['small']:.4f} / {r['both']:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, required=True, help="train metadata.json")
    ap.add_argument("--masks", type=Path, required=True, help="train masks_128.npz (official)")
    ap.add_argument("--src", action="append", required=True, metavar="WATERMARK=DIR[,DIR...]")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fit-weights", action="store_true", help="also fit blend weights and the cut on dev")
    ap.add_argument("--mode", choices=("z", "logit"), default="z", help="blend space: z-standardized or threshold-centered logit")
    ap.add_argument("--trust-oof", default="", help="comma-separated source names whose dev maps are verified OOF despite dev_in_sample=True")
    ap.add_argument("--test-meta", type=Path, default=None, help="optional test metadata.json to export test_masks.npz in exact test order")
    a = ap.parse_args()
    trusted_oof = {x.strip() for x in a.trust_oof.split(",") if x.strip()}

    meta = json.loads(a.meta.read_text())
    full = {m["image_name"]: bool(m["full_watermark"]) for m in meta}
    method_of = {m["image_name"]: m["method"] for m in meta}
    z = np.load(a.masks)
    gt = dict(zip([str(n) for n in z["names"].tolist()], z["masks"]))

    a.out.mkdir(parents=True, exist_ok=True)
    summary, holdout_scores, test_names, test_masks = {}, [], [], []
    for spec in a.src:
        wm, dirs = spec.split("=", 1)
        assert wm in METHODS, wm
        assert wm not in summary, f"{wm} given twice: pass all its sources in one --src, comma-separated"
        srcs = [load_source(Path(d)) for d in dirs.split(",")]
        dev = sorted(srcs[0]["dev"]); hold = sorted(srcs[0]["holdout"])
        for s in srcs:
            assert sorted(s["dev"]) == dev and sorted(s["holdout"]) == hold, f"{s['path']}: image lists differ"
        assert all(method_of[n] == wm for n in dev + hold), f"{wm}: images of another watermark type"
        assert len(hold) == 34 and len(dev) == 136 and not set(dev) & set(hold), (len(dev), len(hold))

        print(f"\n##### {wm}   (large / small / both)  [mode={a.mode}]")
        rows = {}
        for s in srcs:
            in_samp = s["dev_in_sample"] and (s["name"] not in trusted_oof)
            s["effective_in_sample"] = in_samp
            d = split_report(per_image(zmap(s, "dev", dev, mode=a.mode), 0.0, dev, gt), dev, full)
            h = split_report(per_image(zmap(s, "holdout", hold, mode=a.mode), 0.0, hold, gt), hold, full)
            rows[s["name"]] = {"dev": d, "holdout": h, "dev_in_sample": s["dev_in_sample"], "trusted_oof": s["name"] in trusted_oof}
            tag = " (in-sample)" if in_samp else (" (verified-OOF)" if s["dev_in_sample"] else "")
            print(f"  {s['name']:40s} dev {fmt(d)}{tag}   held-out {fmt(h)}")

        # candidates = (label, weights over srcs, cut); selection uses dev maps that are out of sample only
        ok = [i for i, s in enumerate(srcs) if not s["effective_in_sample"]]
        cands = []
        for i in ok:
            cands.append((srcs[i]["name"], [float(j == i) for j in range(len(srcs))], 0.0))
        if len(ok) >= 2:
            cands.append(("mean of " + "+".join(srcs[i]["name"] for i in ok),
                          [1 / len(ok) if j in ok else 0.0 for j in range(len(srcs))], 0.0))
            if a.fit_weights:
                grid = np.linspace(0, 1, 5)
                for w in itertools.product(grid, repeat=len(ok)):
                    if abs(sum(w) - 1) > 1e-6 or max(w) == 1:
                        continue
                    weights = [0.0] * len(srcs)
                    for i, wi in zip(ok, w):
                        weights[i] = float(wi)
                    zd = blend(srcs, weights, "dev", dev, mode=a.mode)
                    cuts = np.linspace(-0.4, 0.4, 17) if a.mode == "logit" else np.quantile(zd, np.linspace(0.02, 0.98, 49))
                    best = max(cuts, key=lambda c: per_image(zd, c, dev, gt).mean())
                    cands.append((f"fitted {np.round(w, 2).tolist()}", weights, float(best)))
        if cands:
            scored = [(per_image(blend(srcs, w, "dev", dev, mode=a.mode), c, dev, gt).mean(), lab, w, c) for lab, w, c in cands]
            dev_score, label, weights, cut = max(scored, key=lambda t: t[0])
            how = "chosen on dev"
        else:
            label = "mean of all sources (no out-of-sample dev maps, nothing was chosen)"
            weights, cut, dev_score, how = [1 / len(srcs)] * len(srcs), 0.0, float("nan"), "parameter-free"
        hs = per_image(blend(srcs, weights, "holdout", hold, mode=a.mode), cut, hold, gt)
        h = split_report(hs, hold, full)
        holdout_scores += hs.tolist()
        print(f"  => {label}  [{how}; dev {dev_score:.4f}]   held-out {fmt(h)}")
        summary[wm] = {"sources": rows, "chosen": label, "how": how, "weights": weights, "cut": cut,
                       "mode": a.mode, "dev_score": float(dev_score), "holdout": h}

        if all(s["test"] is not None for s in srcs):
            names = sorted(srcs[0]["test"])
            assert all(sorted(s["test"]) == names for s in srcs), f"{wm}: test image lists differ"
            test_names += names
            test_masks.append((blend(srcs, weights, "test", names, mode=a.mode) >= cut).astype(np.uint8))
            summary[wm]["test_images"] = len(names)
            summary[wm]["test_watermarked_fraction"] = float(test_masks[-1].mean())
        else:
            print("  (no test.npz from every source: no test masks for this watermark type)")

    done = [wm for wm in METHODS if wm in summary]
    summary["holdout_total"] = {"images": len(holdout_scores), "miou": float(np.mean(holdout_scores)),
                                "covers": done, "complete": len(done) == 3}
    print(f"\nHELD-OUT TOTAL over {len(holdout_scores)} images ({', '.join(done)}): {np.mean(holdout_scores):.4f}"
          + ("" if len(done) == 3 else "   NOT the full 102-image score"))
    if test_masks:
        all_test_masks = np.concatenate(test_masks)
        if a.test_meta is not None:
            tmeta = json.loads(a.test_meta.read_text())
            order = [m["image_name"] for m in tmeta]
            tmap = dict(zip(test_names, all_test_masks))
            assert set(order) == set(tmap), f"test_meta names mismatch: {len(order)} vs {len(tmap)}"
            test_names = order
            all_test_masks = np.stack([tmap[n] for n in order])
        np.savez_compressed(a.out / "test_masks.npz", image_names=np.array(test_names),
                            masks=all_test_masks)
        print(f"test masks: {len(test_names)} images -> {a.out / 'test_masks.npz'}")
    (a.out / "fusion_report.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
