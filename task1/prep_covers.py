"""Pack every released image (train + test, 1530) into one uint8 array and record each image's whole-image bit
accuracy against its own message. Run once on a GPU.

    python task1/prep_covers.py --out runs/covers

Writes covers.npy [N, 512, 512, 3] uint8 and covers_meta.json (one entry per image, same order)."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from wmloc.watermarks import load_watermark
from wmtask.common import DATA_DIR, METHODS, load_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--out", default="runs/covers")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = [dict(r, split=s) for s in ("train", "test") for r in load_index(s, a.data)]
    arr = np.lib.format.open_memmap(out / "covers.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 512, 512, 3))

    def load(i):
        r = rows[i]
        arr[i] = np.asarray(Image.open(Path(a.data) / r["split"] / "images" / f"{r['image_name']}.png").convert("RGB"))

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(load, range(len(rows))))
    arr.flush()
    print("packed", arr.shape, flush=True)

    acc = np.full(len(rows), np.nan)
    for method in METHODS:
        wm = load_watermark(method, device=a.device)
        idx = [i for i, r in enumerate(rows) if r["method"] == method]
        for s in range(0, len(idx), 32):
            b = idx[s:s + 32]
            x = torch.from_numpy(np.stack([arr[i] for i in b])).permute(0, 3, 1, 2).float() / 255
            dec = wm.decode(x).cpu().numpy()
            for j, i in enumerate(b):
                acc[i] = float((dec[j] == rows[i]["message"]).mean())
        del wm; torch.cuda.empty_cache()
        tr = [i for i in idx if rows[i]["split"] == "train"]
        lg = acc[[i for i in tr if rows[i]["full_watermark"]]]; sm = acc[[i for i in tr if not rows[i]["full_watermark"]]]
        print(f"{method}: train large acc min {lg.min():.3f} median {np.median(lg):.3f} | small median {np.median(sm):.3f} max {sm.max():.3f}", flush=True)

    meta = [{"image_name": r["image_name"], "split": r["split"], "method": r["method"],
             "full_watermark": r.get("full_watermark"), "acc_full": float(acc[i])} for i, r in enumerate(rows)]
    (out / "covers_meta.json").write_text(json.dumps(meta))
    print("wrote", out / "covers_meta.json", len(meta))


if __name__ == "__main__":
    main()
