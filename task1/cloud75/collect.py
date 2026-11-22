"""Combine only completed, dev-selected specialists; verify IDs, labels, binary format and size."""

import io, json, hashlib, zipfile
from pathlib import Path
import torch, numpy as np
from train_grid import split_ids
from native_windows import miou


def main(root):
    out = root / "native75-v1"
    methods = ("pixelseal", "trustmark", "stegastamp")
    with zipfile.ZipFile(root / "test.zip") as f:
        rr = json.loads(f.read("test/metadata.json"))
    with zipfile.ZipFile(root / "train.zip") as f:
        tr = json.loads(f.read("train/metadata.json"))
        hold = split_ids(tr)[2]
        with np.load(
            io.BytesIO(f.read("train/masks_128.npz")), allow_pickle=False
        ) as z:
            truth = dict(zip(z["names"], z["masks"]))
    names = [r["image_name"] for r in rr]
    assert len(set(names)) == len(names) == 1020
    result = torch.zeros(1020, 128, 128, dtype=torch.uint8)
    seen = np.zeros(1020, int)
    scores = {}
    for m in methods:
        stats = json.loads((out / f"{m}.json").read_text())
        with np.load(out / f"{m}-hold.npz", allow_pickle=False) as hs:
            ids, mask = hs["ids"], hs["masks"]
            assert (
                len(ids) == 34
                and set(ids) == {i for i in hold if tr[i]["method"] == m}
                and mask.dtype == bool
            )
            scores[m] = miou(
                torch.from_numpy(mask),
                torch.from_numpy(
                    np.stack([truth[tr[i]["image_name"]] for i in ids])
                ).bool(),
            )
            assert abs(stats["hold"] - scores[m]) < 1e-6
        with np.load(out / f"{m}-test.npz", allow_pickle=False) as z:
            ids, p = z["ids"], z["masks"]
            assert (
                ids.shape == (340,)
                and set(ids) == {i for i, r in enumerate(rr) if r["method"] == m}
                and p.shape == (340, 128, 128)
                and p.dtype == bool
            )
            seen[ids] += 1
            result[ids] = torch.from_numpy(p).byte()
    assert (seen == 1).all()
    file = out / "submission.pt"
    torch.save({"image_names": names, "masks": result}, file)
    v = torch.load(file, map_location="cpu", weights_only=True)
    assert (
        v["image_names"] == names
        and torch.equal(v["masks"], result)
        and file.stat().st_size < 20_000_000
    )
    scores["overall"] = float(np.mean(list(scores.values())))
    report = {
        "hold": scores,
        "hold_n": len(hold),
        "test_n": len(names),
        "public_score": None,
        "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
    }
    (out / "result.json").write_text(json.dumps(report, indent=2))
    print("RESULT", report, flush=True)


if __name__ == "__main__":
    main(Path.cwd())
