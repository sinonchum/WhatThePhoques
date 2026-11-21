"""Pack the fixed StegaStamp holdout and official test inputs for local inference."""
import argparse
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    dev, holdout = split["train"], split["val"]
    metadata = {}
    for part in ("train", "test"):
        rows = json.loads((args.data / part / "metadata.json").read_text())
        metadata[part] = [row for row in rows if row["method"] == "stegastamp"]
    train = [row["image_name"] for row in metadata["train"]]
    test = [row["image_name"] for row in metadata["test"]]
    if (len(dev) != 136 or len(holdout) != 34 or len(train) != 170
            or len(set(train)) != 170 or set(dev + holdout) != set(train)):
        raise ValueError("The split must partition the official 170 StegaStamp training names into 136/34.")
    if len(test) != 340 or len(set(test)) != 340 or not set(test).isdisjoint(train):
        raise ValueError("Expected 340 unique StegaStamp test names disjoint from the training names.")
    if any(not name or Path(name).name != name or name in (".", "..") for name in holdout + test):
        raise ValueError("Invalid image name in metadata.")

    with np.load(args.data / "train/masks_128.npz", allow_pickle=False) as source:
        index = {name: i for i, name in enumerate(source["names"].tolist())}
        selected = source["masks"][[index[name] for name in holdout]]
    if selected.shape != (34, 128, 128):
        raise ValueError(f"Unexpected holdout mask shape {selected.shape}")
    masks = io.BytesIO()
    np.savez_compressed(masks, names=np.asarray(holdout), masks=selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as staging:
        archive = Path(staging) / "inputs.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
            bundle.writestr("train/masks_128.npz", masks.getvalue())
            for part, names in (("train", holdout), ("test", test)):
                bundle.writestr(f"{part}/metadata.json", json.dumps(metadata[part]))
                for name in names:
                    image = Path(part, "images", f"{name}.png")
                    bundle.write(args.data / image, image.as_posix())
        archive.replace(args.out)
    print(f"Saved 34 holdout images and masks, 340 test images: {args.out}")


if __name__ == "__main__":
    main()
