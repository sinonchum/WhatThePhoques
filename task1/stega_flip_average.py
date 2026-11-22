"""Average original, horizontal, vertical and double flips of a trained cue net."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_stega_cues import Region, THRESHOLD, load_features, load_masks, predict, save_response, score_groups


def read_model(path, expected_fit, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert set(ck["fit_names"]) == set(expected_fit) and len(ck["fit_names"]) == len(expected_fit)
    assert ck["threshold"] == THRESHOLD
    net = Region(channels=int(ck["channels"])).to(device)
    net.load_state_dict(ck["state_dict"])
    return net, ck["mu"].to(device), ck["sd"].to(device)


def flip_average(features, model, device):
    net, mu, sd = model
    out = predict(net, features, mu, sd, device)
    for axes in ((-1,), (-2,), (-2, -1)):
        flipped = predict(net, features.flip(axes), mu, sd, device)
        out += np.flip(flipped, axis=axes)
    return out / np.float32(4)


def run_dev(args, split, metadata, diagnostics):
    source = args.model / "diagnostics"
    with np.load(source / "dev_oof.npz", allow_pickle=False) as z:
        names, folds, baseline = z["image_names"].tolist(), z["fold"], z["miou"]
    assert len(names) == len(set(names)) == 136 and set(names) == set(split["train"])
    assert set(folds.tolist()) == set(range(5)) and len(folds) == len(baseline) == len(names)
    _, features = load_features(args.features, "dev", names)
    response = np.empty((len(names), 128, 128), np.float32)
    for fold in range(5):
        ids = np.flatnonzero(folds == fold)
        fit_names = [name for name, f in zip(names, folds) if f != fold]
        assert set(fit_names).isdisjoint(names[i] for i in ids)
        model = read_model(source / f"fold_{fold}.pt", fit_names, args.device)
        response[ids] = flip_average(features[ids], model, args.device)
    groups, scores = score_groups(names, response, load_masks(args.data, names), metadata)
    large = np.asarray([bool(metadata[name]["full_watermark"]) for name in names])
    delta = scores - baseline
    gain = {"large": float(delta[large].mean()), "small": float(delta[~large].mean()),
            "both": float(delta.mean())}
    report = {"method": "four-flip probability average", "threshold": THRESHOLD,
              "dev_oof": groups, "paired_gain_vs_original_oof": gain,
              "keep_rule": "overall OOF gain >= 0.005 and small-region OOF gain > 0",
              "keep": gain["both"] >= 0.005 and gain["small"] > 0}
    save_response(args.out / "dev.npz", names, response)
    np.savez_compressed(diagnostics / "dev_oof.npz", image_names=names, fold=folds, miou=scores,
                        original_miou=baseline, paired_difference=delta)
    return report


def run_final(args, split, metadata, diagnostics):
    report = json.loads((diagnostics / "report.json").read_text())
    if not report["keep"]:
        raise ValueError("The fixed DEV comparison did not select this variant.")
    model = read_model(args.model / "final_model.pt", split["train"], args.device)
    for label in ("holdout", "test"):
        if label == "holdout":
            names, features = load_features(args.features, label, split["val"])
            assert len(names) == 34
        else:
            rows = json.loads((args.data / "test/metadata.json").read_text())
            expected = sorted(r["image_name"] for r in rows if r["method"] == "stegastamp")
            names, features = load_features(args.features, label, expected)
            assert len(names) == 340 and set(names).isdisjoint(split["train"] + split["val"])
        response = flip_average(features, model, args.device)
        save_response(args.out / f"{label}.npz", names, response)
        if label == "holdout":
            report[label], scores = score_groups(names, response, load_masks(args.data, names), metadata)
            np.savez_compressed(diagnostics / "holdout_scores.npz", image_names=names, miou=scores)
        else:
            report[label] = {"n": len(names), "inference_only": True}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--features", type=Path, default=Path("output/stega_cues"))
    p.add_argument("--model", type=Path, default=Path("output/fusion/codex_stegacues_stegastamp"))
    p.add_argument("--out", type=Path, default=Path("output/fusion/codex_stegaflip_stegastamp"))
    p.add_argument("--stage", choices=("dev", "final"), default="dev")
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    if args.model.resolve() == args.out.resolve():
        raise ValueError("Provide a separate output directory for flip averaging.")
    torch.set_num_threads(4)
    diagnostics = args.out / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    split = json.loads((args.features / "split.json").read_text())
    metadata = {r["image_name"]: r for r in json.loads((args.data / "train/metadata.json").read_text())}
    report = (run_dev if args.stage == "dev" else run_final)(args, split, metadata, diagnostics)
    (diagnostics / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
