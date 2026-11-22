"""Build the pool of watermarked material for one method, using the official embedder (GPU).

    python task1/gen_synth.py --method pixelseal --covers runs/covers --out runs/synth/pixelseal

For every cover image (images of the other two methods, see wmtask.synth.cover_indices):
  --k-full  whole-image embeddings with fresh random messages          -> full.npy
  --k-rect  embeddings on random rectangles (sizes drawn from the real small-type boxes of this method,
            aspect limited as in the task: 2:1 for TrustMark, 16:1 otherwise)   -> rect_data.npy / rect_index.npy
Training composites are assembled later on the fly (wmtask.synth.SynthDataset)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

from wmloc.watermarks import load_watermark
from wmtask.common import DATA_DIR, holdout_names, load_index
from wmtask.synth import ShapeBank, cover_indices, small_type_threshold


def embed_lowres(wm, img: np.ndarray, bits, s: int, down) -> np.ndarray:
    """Embed on an s x s copy of the image (or crop) and add the bilinearly upsampled DIFFERENCE back.
    This is how PixelSeal and TrustMark work internally at 256. For StegaStamp the starter-kit embed() works at
    400 and leaves high-frequency residue that the released images do not show (runs/diag_h7.py), so the
    released StegaStamp images are reproduced much better by this route with s around 256."""
    h, w = img.shape[:2]
    small = np.asarray(Image.fromarray(img).resize((s, s), down))
    delta = torch.from_numpy(wm.embed(small, bits).astype(np.float32) - small).permute(2, 0, 1)[None]
    delta = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    return np.clip(np.round(img.astype(np.float32) + delta), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--lowres", action="store_true", help="embed via embed_lowres with a randomised working size (use for StegaStamp)")
    ap.add_argument("--covers", default="runs/covers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--k-full", type=int, default=3)
    ap.add_argument("--k-rect", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N covers (smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    meta = json.loads((Path(a.covers) / "covers_meta.json").read_text())
    covers = np.load(Path(a.covers) / "covers.npy", mmap_mode="r")
    rows = load_index("train", a.data)
    cidx = cover_indices(meta, a.method, holdout=holdout_names(rows))      # strict: dev train images only
    assert all(meta[i]["split"] == "train" for i in cidx)
    if a.limit:
        cidx = cidx[:a.limit]
    bank = ShapeBank(rows, holdout_names(rows), a.data, method=a.method)
    print(f"[{a.method}] covers: {len(cidx)}  real rectangles for size sampling: {len(bank.hw)}  "
          f"k_full={a.k_full} k_rect={a.k_rect}", flush=True)

    wm = load_watermark(a.method, device=a.device)

    def embed(img, bits):
        if not a.lowres:
            return wm.embed(img, bits)
        size = 256 if rng.random() < 0.6 else int(rng.choice([192, 224, 288, 320]))
        return embed_lowres(wm, img, bits, size, Image.BILINEAR if rng.random() < 0.7 else Image.BICUBIC)
    full = np.lib.format.open_memmap(out / "full.npy", mode="w+", dtype=np.uint8, shape=(len(cidx), a.k_full, 512, 512, 3))
    full_bits = np.zeros((len(cidx), a.k_full, wm.nbits), np.uint8)      # the message of every embedding is kept:
    rect_chunks, rect_index, rect_bits, off = [], [], [], 0               # at test time the message is known too
    t0 = time.time(); n_embed = 0; resid = []
    for c, gi in enumerate(cidx):
        cover = np.asarray(covers[gi])
        for k in range(a.k_full):
            full_bits[c, k] = rng.integers(0, 2, wm.nbits)
            full[c, k] = embed(cover, full_bits[c, k]); n_embed += 1
        if c < 20:
            resid.append(np.abs(full[c, 0].astype(np.float32) - cover).mean())
        for _ in range(a.k_rect):
            h, w = bank.rect_size(rng)
            y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1))
            crop = np.ascontiguousarray(cover[y0:y0 + h, x0:x0 + w])
            bits = rng.integers(0, 2, wm.nbits).astype(np.uint8)
            marked = embed(crop, bits); n_embed += 1
            assert marked.shape == crop.shape, (marked.shape, crop.shape)
            rect_chunks.append(marked.reshape(-1)); rect_index.append((c, y0, x0, h, w, off)); rect_bits.append(bits); off += marked.size
        if (c + 1) % 100 == 0 or c + 1 == len(cidx):
            dt = time.time() - t0
            print(f"  {c + 1}/{len(cidx)} covers, {n_embed} embeds, {n_embed / dt:.1f} embeds/s, {dt / 60:.1f} min", flush=True)
    full.flush()
    np.save(out / "rect_data.npy", np.concatenate(rect_chunks))
    np.save(out / "rect_index.npy", np.array(rect_index, dtype=np.int64))
    np.save(out / "full_bits.npy", full_bits); np.save(out / "rect_bits.npy", np.array(rect_bits, dtype=np.uint8))
    (out / "pool.json").write_text(json.dumps({"method": a.method, "covers": [int(i) for i in cidx], "k_full": a.k_full,
                                               "k_rect": a.k_rect, "seed": a.seed, "lowres": a.lowres}))
    print(f"[{a.method}] done: full {full.shape}, rects {len(rect_index)} ({off / 1e9:.2f} GB), "
          f"mean |residual| of full embeds {np.mean(resid):.2f}/255, total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
