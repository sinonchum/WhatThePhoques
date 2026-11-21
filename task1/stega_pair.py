"""Paired encoder matched filters for the known StegaStamp payload; produces aligned soft evidence only."""

import argparse, json, io, zipfile, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
import onnx, onnxruntime as ort


def make(root):
    path = root / "stega-encoder-only.onnx"
    if not path.exists():
        src = root / "stega_stamp.onnx"
        assert (
            hashlib.sha256(src.read_bytes()).hexdigest()
            == "369f8134f4e35da9659777f02468f4200c256d14742b9702f2ac28808da675e2"
        )
        onnx.utils.extract_model(str(src), str(path), ["image", "secret"], ["residual"])
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def evidence(im, bits, sess):
    x = (
        np.asarray(
            Image.fromarray(im).resize((400, 400), Image.Resampling.BICUBIC), np.float32
        )
        / 255
    )
    ref = np.full_like(x, 0.5)
    keys = np.stack([bits, 1 - bits]).astype(np.float32)
    rs = sess.run(["residual"], {"image": np.stack([ref, ref]), "secret": keys})[0]
    r = rs[0] - rs[1]
    out = []
    for k in (3, 9, 21):
        a = x - uniform_filter(x, (k, k, 1), mode="reflect")
        b = r - uniform_filter(r, (k, k, 1), mode="reflect")
        ab = (a * b).sum(2)
        aa = (a * a).sum(2)
        bb = (b * b).sum(2)
        for win in (11, 31):
            cov = uniform_filter(ab, win)
            corr = cov / np.sqrt(
                np.maximum(uniform_filter(aa, win) * uniform_filter(bb, win), 1e-12)
            )
            out.append(
                np.asarray(
                    Image.fromarray(corr).resize((64, 64), Image.Resampling.BILINEAR)
                )
            )
    return np.stack(out)


def run(root, split):
    sess = make(root)
    cache = root / f"cam-v1-{split}-stegastamp"
    cache.mkdir(exist_ok=True)
    with zipfile.ZipFile(root / f"{split}.zip") as z:
        rows = [
            r
            for r in json.loads(z.read(f"{split}/metadata.json"))
            if r["method"] == "stegastamp"
        ]
        for i, r in enumerate(rows):
            p = cache / (r["image_name"] + ".npy")
            if p.exists():
                continue
            im = np.asarray(
                Image.open(
                    io.BytesIO(z.read(f"{split}/images/{r['image_name']}.png"))
                ).convert("RGB")
            )
            b = np.array(list(r["message"]), np.float32)
            v = evidence(im, b, sess).astype(np.float16)
            assert np.isfinite(v).all()
            t = p.with_suffix(".partial.npy")
            np.save(t, v)
            t.replace(p)
            if i % 40 == 0:
                print(split, i, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("split", choices=["train", "test"])
    a = p.parse_args()
    run(a.root, a.split)
