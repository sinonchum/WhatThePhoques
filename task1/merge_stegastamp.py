"""Combine a teammate's 680 (or 1020) named masks with the pinned StegaStamp handoff. Offline; never submits."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch

HERE = Path(__file__).resolve().parent / "handoff"


def merge(base, metadata, output, bundle=HERE / "stegastamp-test-v2.npz"):
    if output.exists() or output.resolve() == base.resolve():
        raise FileExistsError(
            "Use a new output filename; the base is never overwritten"
        )
    if zipfile.is_zipfile(metadata):
        with zipfile.ZipFile(metadata) as z:
            rows = json.loads(z.read("test/metadata.json"))
    else:
        rows = json.loads(metadata.read_text())
    names = [r["image_name"] for r in rows]
    groups = {
        m: {r["image_name"] for r in rows if r["method"] == m}
        for m in ("pixelseal", "trustmark", "stegastamp")
    }
    if (
        len(names) != 1020
        or len(set(names)) != 1020
        or any(len(s) != 340 for s in groups.values())
    ):
        raise ValueError("Unexpected test metadata")
    data = bundle.read_bytes()
    manifest = json.loads(bundle.with_suffix(".json").read_text())
    if (
        hashlib.sha256(data).hexdigest() != manifest["sha256"]
        or manifest["method"] != "stegastamp"
        or manifest["split"] != "test"
    ):
        raise ValueError("Bundle/manifest mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        sn = z["image_names"].tolist()
        sm = z["masks"]
    if (
        len(sn) != 340
        or set(sn) != groups["stegastamp"]
        or sm.shape != (340, 128, 128)
        or sm.dtype != np.bool_
    ):
        raise ValueError("Invalid StegaStamp bundle")
    t = torch.load(base, weights_only=True, map_location="cpu")
    bn = t["image_names"]
    bm = t["masks"]
    other = groups["pixelseal"] | groups["trustmark"]
    if (
        not isinstance(bn, (tuple, list))
        or len(bn) != len(set(bn))
        or (set(bn) != other and set(bn) != set(names))
    ):
        raise ValueError(
            "Expected all 680 non-StegaStamp names, or all 1020 names, without duplicates"
        )
    if (
        not torch.is_tensor(bm)
        or bm.shape != (len(bn), 128, 128)
        or not bool(((bm == 0) | (bm == 1)).all())
    ):
        raise ValueError("Base masks must be named binary 128x128 predictions")
    bidx = {n: i for i, n in enumerate(bn)}
    sidx = {n: i for i, n in enumerate(sn)}
    result = torch.stack(
        [
            torch.from_numpy(sm[sidx[n]]) if n in sidx else bm[bidx[n]].bool()
            for n in names
        ]
    ).to(torch.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    with tempfile.NamedTemporaryFile(
        dir=output.parent, suffix=".pt", delete=False
    ) as f:
        temp = Path(f.name)
    try:
        torch.save({"image_names": names, "masks": result}, temp)
        check = torch.load(temp, weights_only=True, map_location="cpu")
        ix = [i for i, n in enumerate(names) if n in other]
        sx = [i for i, n in enumerate(names) if n in sidx]
        if (
            temp.stat().st_size > 20_000_000
            or check["image_names"] != names
            or not torch.equal(
                check["masks"][ix], bm[[bidx[names[i]] for i in ix]].byte()
            )
            or not torch.equal(
                check["masks"][sx],
                torch.from_numpy(sm[[sidx[names[i]] for i in sx]]).byte(),
            )
        ):
            raise RuntimeError("Round-trip verification failed")
        import os

        os.link(temp, output)  # fail safely if a competing process already created it
    finally:
        temp.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "kept_from_teammate": len(ix),
                "stegastamp_from_bundle": len(sx),
                "size": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "official_score": None,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="official test.zip or extracted test/metadata.json",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bundle", type=Path, default=HERE / "stegastamp-test-v2.npz")
    a = p.parse_args()
    merge(a.base, a.metadata, a.output, a.bundle)
