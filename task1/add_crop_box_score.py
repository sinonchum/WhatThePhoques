"""Add label-free best-box message margins to existing TrustMark D exports."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np


EXPECTED = {"dev": 136, "holdout": 34, "test": 340}
ORIGINAL_FIELDS = ("image_names", "response", "threshold", "dev_in_sample")


def add_split(root, split, expected):
    path = root / f"{split}.npz"
    rows_path = root / "diagnostics" / f"{split}_rows.json"
    rows = json.loads(rows_path.read_text())
    by_name = {row["image_name"]: row for row in rows}
    if len(rows) != len(by_name):
        raise ValueError(f"Duplicate diagnostic names in {rows_path}")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy(order="K") for key in source.files}
    if not all(key in arrays for key in ORIGINAL_FIELDS):
        raise ValueError(f"Missing an original D field in {path}")
    names = arrays["image_names"].tolist()
    if len(names) != len(set(names)) or len(names) != expected or set(names) != set(by_name):
        raise ValueError(f"{split}: export and diagnostics must contain the same {expected} unique names")
    scores = np.asarray([by_name[name]["signed_raw_margin"] for name in names], dtype=np.float32)
    if scores.shape != (expected,) or not np.isfinite(scores).all():
        raise ValueError(f"{split}: invalid signed raw margins")
    if "box_score" in arrays:
        existing = arrays["box_score"]
        if existing.dtype != np.float32 or existing.shape != scores.shape or not np.array_equal(existing, scores):
            raise ValueError(f"{path}: existing box_score disagrees with diagnostics")
        return {"images": expected, "status": "verified",
                "source": str(rows_path), "dtype": "float32", "shape": [expected]}
    arrays["box_score"] = scores
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=root, prefix=f".{split}.box_score.",
                                         suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        # Check every saved field before atomically replacing the original file.
        with np.load(temporary, allow_pickle=False) as saved:
            if saved.files != list(arrays):
                raise ValueError(f"{split}: field order changed while writing")
            for key, old in arrays.items():
                new = saved[key]
                if old.dtype != new.dtype or old.shape != new.shape or not np.array_equal(old, new, equal_nan=old.dtype.kind in "fc"):
                    raise ValueError(f"{split}: field changed while writing: {key}")
        os.chmod(temporary, path.stat().st_mode & 0o777)
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"images": expected, "status": "verified",
            "source": str(rows_path), "dtype": "float32", "shape": [expected]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("output/fusion/codex_cropsearch_trustmark"))
    args = parser.parse_args()
    summaries = {split: add_split(args.root, split, count) for split, count in EXPECTED.items()}
    info = {"field": "box_score", "definition": "raw mean signed logits of the selected best crop: mean((2*known_message-1)*decoder_logits)",
            "source_field": "signed_raw_margin", "higher_means": "stronger known-message decoder margin for the selected rectangle",
            "dtype": "float32", "fitting": "none; copies diagnostics by image_name without reading labels or fitting thresholds",
            "unchanged": ["all existing NPZ fields, dtypes, shapes, values and image order", "threshold"],
            "splits": summaries}
    destination = args.root / "diagnostics/box_score_info.json"
    text = json.dumps(info, indent=2, allow_nan=False)
    if not destination.exists() or destination.read_text() != text:
        destination.write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
