"""Select completed method heads by inner development mIoU, not by holdout or leaderboard."""

import argparse, json, zipfile, hashlib
from pathlib import Path
import numpy as np
import torch
from train_grid import METHODS


def pack(root, output, require_polarity=False):
    with zipfile.ZipFile(root / "test.zip") as z:
        rows = json.loads(z.read("test/metadata.json"))
    names = [r["image_name"] for r in rows]
    assert len(set(names)) == len(names) == 1020
    seen = np.zeros(len(rows), int)
    masks = np.zeros((len(rows), 128, 128), np.uint8)
    decision = {}
    for m in METHODS:
        candidates = []
        for sub in (
            "blend-v1-17",
            "polarity-v1",
            "neural-message-v1",
            *sorted(p.name for p in root.glob("neural-message-v1-*-*") if p.is_dir()),
        ):
            folder = root / sub
            j = folder / f"{m}.json"
            f = folder / f"{m}-test.npz"
            if j.is_file() and f.is_file():
                stats = json.loads(j.read_text())
                candidates.append((float(stats["dev"]), sub, stats, f))
        if require_polarity and not any(
            k == "polarity-v1" for _, k, _, _ in candidates
        ):
            raise RuntimeError(f"no completed polarity head for {m}")
        if not candidates:
            raise RuntimeError(f"no completed candidate for {m}")
        _, key, stats, f = max(candidates, key=lambda e: e[0])
        decision[m] = {"chosen": key, **stats}
        with np.load(f, allow_pickle=False) as z:
            ids = z["ids"]
            v = z["masks"]
            assert (
                v.dtype == bool
                and v.shape == (340, 128, 128)
                and len(set(ids)) == len(ids) == 340
                and all(rows[i]["method"] == m for i in ids)
            )
            seen[ids] += 1
            masks[ids] = v
    assert (seen == 1).all()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp.pt")
    torch.save({"image_names": names, "masks": torch.from_numpy(masks)}, tmp)
    assert tmp.stat().st_size < 20_000_000
    re = torch.load(tmp, weights_only=True, map_location="cpu")
    assert re["image_names"] == names and np.array_equal(re["masks"].numpy(), masks)
    tmp.replace(output)
    result = {
        "heads": decision,
        "local_hold_mIoU": np.mean([s["hold"] for s in decision.values()]),
        "size": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "public_score": None,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--require-polarity", action="store_true")
    a = p.parse_args()
    pack(a.root, a.output, a.require_polarity)
