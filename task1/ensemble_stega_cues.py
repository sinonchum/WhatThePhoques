"""Equal-weight StegaStamp cue-model ensemble with a frozen source list.

DEV averages aligned OOF maps only. FINAL infers members without scoring or
saving their individual holdout predictions, then scores the ensemble once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_stega_cues import Region, THRESHOLD, load_features, load_masks, predict, save_response, score_groups


def specification(args):
    return {"sources": [str(path.resolve()) for path in args.sources],
            "feature_caches": [str(path.resolve()) for path in args.features],
            "weights": [1 / len(args.sources)] * len(args.sources), "threshold": THRESHOLD}


def load_oof(source, names):
    with np.load(source / "dev.npz", allow_pickle=False) as data:
        cached_names = data["image_names"].tolist()
        assert len(cached_names) == len(set(cached_names)) == 136 and set(cached_names) == set(names)
        assert float(data["threshold"]) == THRESHOLD
        lookup = {name: i for i, name in enumerate(cached_names)}
        response = data["response"][[lookup[name] for name in names]].astype(np.float64)
    assert response.shape == (136, 128, 128) and np.isfinite(response).all()
    return response


def dev_stage(args, diagnostics, split, metadata):
    names = split["train"]
    response = np.zeros((len(names), 128, 128), dtype=np.float64)
    for source in args.sources:
        response += load_oof(source, names) / len(args.sources)
    response = response.astype(np.float32)
    report, scores = score_groups(names, response, load_masks(args.data, names), metadata)
    save_response(args.out / "dev.npz", names, response)
    selected = specification(args)
    (diagnostics / "selection.json").write_text(json.dumps(selected, indent=2))
    (diagnostics / "split.json").write_text(json.dumps(split, indent=2))
    report = {"method": "equal arithmetic mean of predefined StegaStamp cue-model probabilities",
              **selected, "dev_response_role": "mean of aligned member five-fold OOF response maps",
              "dev_in_sample": True, "dev_in_sample_note": "conservative flag; no ensemble parameters fitted",
              "dev_equal_weight_oof": report}
    (diagnostics / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (diagnostics / "dev_rows.json").write_text(json.dumps([
        {"image_name": name, "full_watermark": bool(metadata[name]["full_watermark"]), "miou": float(score)}
        for name, score in zip(names, scores)], indent=2))
    print("DEV equal-weight OOF large/small/both:", json.dumps(report["dev_equal_weight_oof"]), flush=True)


def final_stage(args, diagnostics, split, metadata):
    selected = json.loads((diagnostics / "selection.json").read_text())
    assert selected == specification(args), "Source/cache list must stay exactly as frozen in DEV"
    assert json.loads((diagnostics / "split.json").read_text()) == split
    with np.load(args.out / "dev.npz", allow_pickle=False) as dev:
        assert dev["image_names"].tolist() == split["train"] and float(dev["threshold"]) == THRESHOLD
    test_meta = json.loads((args.data / "test/metadata.json").read_text())
    test_names = sorted(row["image_name"] for row in test_meta if row["method"] == "stegastamp")
    names_by_split = {"holdout": split["val"], "test": test_names}
    assert len(test_names) == len(set(test_names)) == 340
    assert set(test_names).isdisjoint(split["train"] + split["val"])
    sums = {label: np.zeros((len(names), 128, 128), dtype=np.float64) for label, names in names_by_split.items()}
    device = torch.device(args.device)
    channels = []
    for source, cache in zip(args.sources, args.features):
        checkpoint = torch.load(source / "final_model.pt", map_location="cpu", weights_only=True)
        assert checkpoint["fit_names"] == split["train"] and checkpoint["threshold"] == THRESHOLD
        count = int(checkpoint["channels"])
        assert checkpoint["mu"].shape[1] == count == checkpoint["sd"].shape[1]
        model = Region(channels=count).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        mu, sd = checkpoint["mu"].to(device), checkpoint["sd"].to(device)
        for label, names in names_by_split.items():
            loaded_names, features = load_features(cache, label, names)
            assert loaded_names == names and features.shape[1] == count
            # No per-model holdout mask, label, score, response file or threshold choice.
            sums[label] += predict(model, features, mu, sd, device).astype(np.float64) / len(args.sources)
            print(f"inferred member {source.name}: {label} n={len(names)}, channels={count}", flush=True)
            del features
        channels.append(count)
        del model, mu, sd, checkpoint
    report_path = diagnostics / "report.json"
    report = json.loads(report_path.read_text())
    holdout_response = sums["holdout"].astype(np.float32)
    save_response(args.out / "holdout.npz", split["val"], holdout_response)
    # Official holdout scoring occurs once, on the final equal-weight ensemble only.
    report["holdout"], scores = score_groups(split["val"], holdout_response,
                                           load_masks(args.data, split["val"]), metadata)
    (diagnostics / "holdout_rows.json").write_text(json.dumps([
        {"image_name": name, "full_watermark": bool(metadata[name]["full_watermark"]), "miou": float(score)}
        for name, score in zip(split["val"], scores)], indent=2))
    save_response(args.out / "test.npz", test_names, sums["test"].astype(np.float32))
    report["source_channels"] = channels
    report["test"] = {"n": len(test_names), "inference_only": True}
    report["single_model_holdout_scored_or_saved"] = False
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print("HOLDOUT equal-weight ensemble large/small/both:", json.dumps(report["holdout"]), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--sources", type=Path, nargs="+", required=True)
    p.add_argument("--features", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--stage", choices=("dev", "final"), default="dev")
    p.add_argument("--device", default="mps")
    args = p.parse_args()
    assert len(args.sources) == len(args.features) >= 2
    assert len({path.resolve() for path in args.sources}) == len(args.sources)
    assert args.out.resolve() not in {path.resolve() for path in args.sources}
    torch.set_num_threads(4)
    split = json.loads((args.features[0] / "split.json").read_text())
    assert len(split["train"]) == 136 and len(split["val"]) == 34 and set(split["train"]).isdisjoint(split["val"])
    for source, cache in zip(args.sources, args.features):
        assert json.loads((cache / "split.json").read_text()) == split
        assert json.loads((source / "diagnostics/split.json").read_text()) == split
    metadata = {row["image_name"]: row for row in json.loads((args.data / "train/metadata.json").read_text())}
    assert all(metadata[name]["method"] == "stegastamp" for name in split["train"] + split["val"])
    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics = args.out / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    if args.stage == "dev":
        dev_stage(args, diagnostics, split, metadata)
    else:
        final_stage(args, diagnostics, split, metadata)


if __name__ == "__main__":
    main()
