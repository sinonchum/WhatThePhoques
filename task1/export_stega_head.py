"""Export a completed StegaStamp head by name, and independently recompute its declared holdout mIoU."""

import argparse, hashlib, io, json, re, zipfile
from pathlib import Path
import numpy as np


def export(root, head, folder, version):
    if not re.fullmatch(r"v[1-9][0-9]*", version):
        raise ValueError("version must be vN")
    report = json.loads((head / "stegastamp.json").read_text())
    assert report["method"] == "stegastamp"
    folder.mkdir(exist_ok=True, parents=True)
    prepared = []
    for split, meta, n in [("hold", "train", 34), ("test", "test", 340)]:
        with (
            zipfile.ZipFile(root / f"{meta}.zip") as z,
            np.load(head / f"stegastamp-{split}.npz", allow_pickle=False) as f,
        ):
            rows = json.loads(z.read(f"{meta}/metadata.json"))
            ids = f["ids"]
            m = f["masks"]
            assert (
                ids.shape == (n,)
                and np.issubdtype(ids.dtype, np.integer)
                and len(set(ids)) == n
                and min(ids) >= 0
                and max(ids) < len(rows)
            )
            names = np.array([rows[i]["image_name"] for i in ids])
            assert (
                all(rows[i]["method"] == "stegastamp" for i in ids)
                and m.dtype == bool
                and m.shape == (n, 128, 128)
                and len(set(names)) == n
            )
            if split == "test":
                assert set(names) == {
                    r["image_name"] for r in rows if r["method"] == "stegastamp"
                }
            else:
                from train_grid import split_ids

                assert set(ids) == {
                    i for i in split_ids(rows)[2] if rows[i]["method"] == "stegastamp"
                }
                with np.load(
                    io.BytesIO(z.read("train/masks_128.npz")), allow_pickle=False
                ) as lab:
                    ix = {name: i for i, name in enumerate(lab["names"])}
                    y = lab["masks"][[ix[k] for k in names]].astype(bool)
                a = []
                for p, t in ((m, y), (~m, ~y)):
                    u = (p | t).sum((1, 2))
                    a.append(np.where(u > 0, (p & t).sum((1, 2)) / u.clip(min=1), 1.0))
                hold = float(np.mean(a))
                assert abs(hold - report["hold"]) < 1e-6
            b = io.BytesIO()
            np.savez_compressed(b, image_names=names, masks=m)
            data = b.getvalue()
            dest = folder / f"stegastamp-{split}-{version}.npz"
            manifest = {
                "dataset": "SprintML/Hackathon_Watermark_Localisation_Task_Paris",
                "dataset_revision": "8d27b2c9cc37c4829c8501e43e11fcb1b5db58c0",
                "method": "stegastamp",
                "split": split,
                "count": n,
                "format": "npz: image_names unicode; masks bool N,128,128; True=remaining watermark",
                "source_head": head.name,
                "local_evaluation": report,
                "public_score": None,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            prepared.extend(
                [
                    (dest, data),
                    (
                        dest.with_suffix(".json"),
                        (json.dumps(manifest, indent=2) + "\n").encode(),
                    ),
                ]
            )
    for p, data in prepared:
        if p.exists() and p.read_bytes() != data:
            raise FileExistsError(f"immutable handoff: {p}")
    for p, data in prepared:
        if not p.exists():
            with p.open("xb") as f:
                f.write(data)
    print(
        json.dumps(
            {
                "head": head.name,
                "version": version,
                "dev": report["dev"],
                "verified_hold": hold,
                "test": 340,
                "official_score": None,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("version")
    a = p.parse_args()
    export(a.root, a.head, a.folder, a.version)
