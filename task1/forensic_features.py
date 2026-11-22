"""Cache 8x8-grid forensic features from the unmodified official archives (no test fitting)."""

from pathlib import Path
import argparse, io, json, zipfile
import numpy as np
from PIL import Image
from scipy.fft import dctn


def blocks(x):
    return x.reshape(64, 8, 64, 8).transpose(0, 2, 1, 3)


def features(im):
    r, g, b = im.astype(np.float32).transpose(2, 0, 1)
    arr = []
    for c in (0.299 * r + 0.587 * g + 0.114 * b, (r - g) * 0.5, (b - g) * 0.5):
        v = blocks(c)
        dc = dctn(v, axes=(-2, -1), norm="ortho")
        ad = np.abs(dc)
        i, j = np.indices((8, 8))
        band = i + j
        arr.append(v.mean((-1, -2)) / 128)
        for lo, hi in ((1, 3), (3, 6), (6, 9), (9, 16)):
            t = ad[:, :, (band >= lo) & (band < hi)]
            arr.extend((np.log1p(t.mean(-1)), (t < 0.6).mean(-1), (t < 0.15).mean(-1)))
        # Quantization and JPEG chroma interpolation can distinguish original pixels
        # from perturbed ones. No assumption of a particular encoder or JPEG quality.
        q = dc[:, :, band >= 4]
        for step in (1, 2, 4, 8):
            t = q / step
            arr.append(np.abs(t - np.rint(t)).mean(-1))
        h = c - np.roll(c, 1, axis=1)
        w = c - np.roll(c, 1, axis=0)
        for z in (
            h,
            w,
            h - np.roll(h, -1, 1),
            w - np.roll(w, -1, 0),
            h - np.roll(h, -1, 0),
        ):
            v = blocks(np.abs(z))
            arr.extend((np.log1p(v.mean((-1, -2))), (v < 0.51).mean((-1, -2))))
        for z in (np.abs(h), np.abs(w)):
            v = blocks(z)
            arr.append(
                np.log1p(v[:, :, 1::2, 1::2].mean((-1, -2)))
                - np.log1p(v[:, :, ::2, ::2].mean((-1, -2)))
            )
    return np.stack(arr).astype(np.float16)


def build(root, split):
    dst = root / f"forensic-v1-{split}.npy"
    tmp = dst.with_suffix(".partial.npy")
    with zipfile.ZipFile(root / f"{split}.zip") as z:
        meta = json.loads(z.read(f"{split}/metadata.json"))
        names = [m["image_name"] for m in meta]
        assert len(set(names)) == len(names) == {"train": 510, "test": 1020}[split]
        if dst.exists():
            cached = np.load(dst, mmap_mode="r")
            assert cached.shape == (len(meta), 87, 64, 64)
            return
        for i, name in enumerate(names):
            im = np.asarray(
                Image.open(io.BytesIO(z.read(f"{split}/images/{name}.png"))).convert(
                    "RGB"
                )
            )
            assert im.shape == (512, 512, 3)
            x = features(im)
            assert x.shape == (87, 64, 64) and np.isfinite(x).all()
            if i == 0:
                out = np.lib.format.open_memmap(
                    tmp, mode="w+", dtype=x.dtype, shape=(len(meta), *x.shape)
                )
            out[i] = x
        out.flush()
        del out
        tmp.replace(dst)
        print(split, dst, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("split", choices=["train", "test"])
    a = p.parse_args()
    build(a.root, a.split)
