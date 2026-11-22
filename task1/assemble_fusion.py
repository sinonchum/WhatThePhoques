"""Build a submission from exported response maps (fusion format) and, optionally, ready-made binary masks.

    python task1/assemble_fusion.py --resp pixelseal=runs/fusion/claude_x_pixelseal trustmark=runs/fusion/claude_y_trustmark \
        --masks stegastamp=runs/teammate/stegastamp-test-v1.npz --out runs/sub_v1.pt

--resp dirs hold test.npz (image_names, response, threshold): mask = response >= threshold.
--masks files hold image_names + masks [N,128,128]. Everything is matched BY IMAGE NAME, never by position."""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from wmtask.common import METHODS, check_submission, load_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resp", nargs="*", default=[]); ap.add_argument("--masks", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    test = load_index("test"); pred = {}
    want = {m: {r["image_name"] for r in test if r["method"] == m} for m in METHODS}
    for spec in a.resp:
        m, d = spec.split("=", 1); z = np.load(os.path.join(d, "test.npz")); thr = float(z["threshold"])
        names = z["image_names"].tolist(); assert set(names) == want[m], f"{m}: names differ from the test set"
        for n, r in zip(names, z["response"]):
            pred[n] = r >= thr
        print(f"{m}: responses from {d}, threshold {thr}")
    for spec in a.masks:
        m, f = spec.split("=", 1); z = np.load(f); names = z["image_names"].tolist(); mk = z["masks"]
        assert set(names) == want[m] and len(names) == len(set(names)), f"{m}: names differ from the test set"
        assert mk.shape == (len(names), 128, 128) and set(np.unique(mk).tolist()) <= {0, 1, False, True}
        for n, k in zip(names, mk):
            pred[n] = k.astype(bool)
        print(f"{m}: ready-made masks from {f}")
    names = [r["image_name"] for r in test]
    missing = [n for n in names if n not in pred]; assert not missing, f"{len(missing)} test images without a prediction"
    masks = torch.from_numpy(np.stack([pred[n] for n in names])).bool()
    check_submission(names, masks, names)
    torch.save({"image_names": names, "masks": masks}, a.out)
    frac = masks.float().mean(dim=(1, 2))
    print(f"wrote {a.out}: {len(names)} masks, {os.path.getsize(a.out) / 1e6:.2f} MB")
    for m in METHODS:
        f = frac[[i for i, r in enumerate(test) if r["method"] == m]]
        print(f"  {m:11s} mean watermarked fraction {f.mean():.3f}; share of images predicted >50% watermarked {(f > 0.5).float().mean():.3f}; all-0 masks {int((f == 0).sum())}, all-1 masks {int((f == 1).sum())}")


if __name__ == "__main__":
    main()
