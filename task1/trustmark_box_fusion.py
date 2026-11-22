"""Use a decoded rectangle when a small crop recovers the known message."""
import argparse
import json
from pathlib import Path

import numpy as np

from wmloc.utils import miou_image


FUSION = Path(__file__).resolve().parent / "output/fusion"
NETWORK = FUSION / "_coord/pulled/claude_s3_512real_cv_trustmark"
CROPS = FUSION / "codex_cropsearch_trustmark"
GT = FUSION / "_coord/gt"
OUT = FUSION / "codex_boxfusion_trustmark"


def order_for(source_names, names):
    lookup = {str(n): i for i, n in enumerate(source_names)}
    if len(lookup) != len(source_names) or len(set(names)) != len(names):
        raise ValueError("Duplicate image names")
    return [lookup[n] for n in names]


def paired_score(names, network, response, threshold, switched):
    meta = {r["image_name"]: r for r in json.loads((GT / "train_metadata.json").read_text())}
    large = np.array([meta[n]["full_watermark"] for n in names], dtype=bool)
    with np.load(GT / "masks_128.npz") as labels:
        gt = labels["masks"][order_for(labels["names"], names)]
    before, after = (np.array([miou_image(p >= threshold, g) for p, g in zip(maps, gt)])
                     for maps in (network, response))
    result = {}
    for key, group in (("all", np.ones(len(names), bool)), ("large", large), ("small", ~large)):
        change = after[group] - before[group]
        result[key] = dict(n=int(group.sum()), network=float(before[group].mean()),
                           fusion=float(after[group].mean()), gain=float(change.mean()),
                           switched=int(switched[group].sum()), improved=int((change > 0).sum()),
                           worsened=int((change < 0).sum()))
    return result


def run_split(split, fixed=None):
    expected = json.loads((CROPS / "diagnostics/split.json").read_text())
    with np.load(NETWORK / f"{split}.npz") as net, np.load(CROPS / f"{split}.npz") as crops:
        names = net["image_names"].tolist()
        if len(names) != {"dev": 136, "holdout": 34, "test": 340}[split] or set(names) != set(crops["image_names"]):
            raise ValueError(f"{split} source populations disagree")
        if split == "test":
            if set(names).intersection(expected["train"] + expected["val"]):
                raise ValueError("Test overlaps train or holdout")
        elif set(names) != set(expected["train" if split == "dev" else "val"]):
            raise ValueError(f"Wrong {split} population")
        if bool(net["dev_in_sample"]):
            raise ValueError("Expected an out-of-sample network export")
        idx = order_for(crops["image_names"], names)
        network = net["response"]
        margin = crops["box_score"][idx]
        encoded_box = crops["response"][idx]
        source_cut = float(crops["threshold"])
        net_cut = float(net["threshold"])

    if split == "dev":
        with np.load(CROPS / "diagnostics/dev_threshold_cv.npz") as cv:
            cut = cv["threshold"][order_for(cv["image_names"], names)]
    else:
        if (net_cut, source_cut) != (fixed["network_cut"], fixed["crop_cut"]):
            raise ValueError("Source thresholds differ from the frozen dev settings")
        cut = fixed["crop_cut"]

    # The search exports margin times exact 4x4-cell coverage. Recover geometry
    # before testing area, so crops with stronger messages do not appear larger.
    coverage = np.divide(encoded_box, margin[:, None, None], out=np.zeros_like(encoded_box),
                         where=margin[:, None, None] > 0)
    use_box = (margin > 0) & (margin >= cut) & (coverage.mean(axis=(1, 2)) <= 0.5)
    response = network.copy()
    c = coverage[use_box]
    response[use_box] = np.where(c < 0.5, 2 * c * net_cut,
                                 net_cut + (2 * c - 1) * (1 - net_cut))
    np.savez_compressed(OUT / f"{split}.npz", image_names=np.asarray(names), response=response,
                        threshold=net_cut, dev_in_sample=False)
    report = dict(n=len(names), switched=int(use_box.sum()), network_cut=net_cut, crop_cut=source_cut)
    if split != "test":
        report["paired_miou"] = paired_score(names, network, response, net_cut, use_box)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["dev", "final"])
    parser.add_argument("--method", choices=["trustmark", "pixelseal"], default="trustmark")
    args = parser.parse_args()
    stage = args.stage
    if args.method == "pixelseal":
        NETWORK = FUSION / "_coord/pulled/claude_s3_ens_512_256_pixelseal"
        CROPS = FUSION / "codex_cropsearch_pixelseal"
        OUT = FUSION / "codex_boxfusion_pixelseal"
    OUT.mkdir(exist_ok=True)
    path = OUT / "report.json"
    if stage == "dev":
        dev = run_split("dev")
        report = dict(network=NETWORK.name, crop=CROPS.name, dev=dev,
                      keep=dev["paired_miou"]["all"]["gain"] > 0,
                      rule="area <= 1/2 and margin >= existing crop threshold; otherwise copy network",
                      validation="Supplied four-fold network dev maps and published dev-calibrated probability threshold. "
                                 "The crop cutoff for each dev image comes from the other four crop folds.")
        if args.method == "pixelseal":
            report["validation"] = ("Clean synth3 512/256 ensemble dev maps and the published network threshold. "
                                    "The crop cutoff for each dev image comes from the other four crop folds.")
    else:
        report = json.loads(path.read_text())
        if not report["keep"]:
            raise SystemExit("The rule did not improve dev mIoU. Keep the network.")
        report["test"] = run_split("test", report["dev"])
        report["holdout"] = run_split("holdout", report["dev"])
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
