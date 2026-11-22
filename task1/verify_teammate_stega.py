"""Check Xuanhao's immutable handoff on its own split, without fitting anything."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wmloc.utils import miou_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--our-split", type=Path, required=True)
    a = p.parse_args()
    own = json.loads(a.our_split.read_text())
    metadata = {r["image_name"]: r for r in json.loads((a.data / "train/metadata.json").read_text())}
    with np.load(a.data / "train/masks_128.npz", allow_pickle=False) as z:
        truth = dict(zip(z["names"].tolist(), z["masks"]))
    rank = lambda name: hashlib.sha256(("paris-residual-v1:" + name).encode()).digest()
    foreign_hold = []
    for full in (True, False):
        names = sorted([n for n, r in metadata.items() if r["method"] == "stegastamp"
                        and bool(r["full_watermark"]) == full], key=rank)
        foreign_hold.extend(names[:17])
    result = {"source_commit": "e3a4cdae3f66c2bd73a96ec120f7f1375c598ae8",
              "same_split_as_ours": False, "official_submission": False}
    for label, count in (("hold", 34), ("test", 340)):
        path = a.bundle / f"stegastamp-{label}-v1.npz"
        manifest = json.loads(path.with_suffix(".json").read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == manifest["sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {"image_names", "masks"}
            names, masks = z["image_names"].tolist(), z["masks"]
        assert len(names) == len(set(names)) == count
        assert masks.dtype == np.bool_ and masks.shape == (count, 128, 128)
        row = {"count": count, "sha256": digest}
        if label == "hold":
            assert set(names) == set(foreign_hold)
            scores = np.asarray([miou_image(mask, truth[name]) for name, mask in zip(names, masks)])
            full = np.asarray([bool(metadata[n]["full_watermark"]) for n in names])
            row.update(large=float(scores[full].mean()), small=float(scores[~full].mean()),
                       both=float(scores.mean()), n_large=int(full.sum()), n_small=int((~full).sum()),
                       overlap_with_our_holdout=len(set(names) & set(own["val"])),
                       reported_score=float(manifest["local_evaluation"]["hold"]))
            assert abs(row["both"] - row["reported_score"]) < 1e-6
        else:
            test = json.loads((a.data / "test/metadata.json").read_text())
            assert set(names) == {r["image_name"] for r in test if r["method"] == "stegastamp"}
            row["inference_only"] = True
        result[label] = row
    result["note"] = "Foreign split score only; its model saw 26 of our 34 holdout images during training or development. Do not combine this score into our 102-image holdout."
    (a.bundle / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
