"""Aligned small windows for known-message decoders + interpolation/noise statistics for fine geometry."""

import argparse, io, json, zipfile
from pathlib import Path
import torch, numpy as np
from torch.nn import functional as F
from scipy.ndimage import uniform_filter
from PIL import Image
from message_cam import Decoder


def texture(im):
    r, g, b = im.astype(np.float32).transpose(2, 0, 1)
    out = []
    for x in (0.299 * r + 0.587 * g + 0.114 * b, r - g, b - g):
        dx = x - np.roll(x, -1, 1)
        dy = x - np.roll(x, -1, 0)
        e = dx - np.roll(dx, 1, 1)
        f = dy - np.roll(dy, 1, 0)
        v = dx - np.roll(dx, 1, 0)
        for z in (e, f, v):
            z = np.abs(z)
            bm = uniform_filter(z, size=17)
            zz = z / (bm + 0.25)
            q = zz.reshape(64, 8, 64, 8)
            out.append(np.mean(np.abs(q[:, :, :, ::2] - q[:, :, :, 1::2]), (1, 3)))
            out.append(np.mean(np.abs(q[:, ::2] - q[:, 1::2]), (1, 3)))
    return np.stack(out).astype(np.float32)


@torch.inference_mode()
def run(root, method):
    net = Decoder(method, root).eval().to("mps") if method != "stegastamp" else None
    for split in ("train", "test"):
        out = root / f"tile-v1-{split}-{method}"
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            rows = [
                r
                for r in json.loads(z.read(f"{split}/metadata.json"))
                if r["method"] == method
            ]
            for i, r in enumerate(rows):
                name = r["image_name"]
                dest = out / (name + ".npy")
                if dest.exists():
                    continue
                im = np.asarray(
                    Image.open(
                        io.BytesIO(z.read(f"{split}/images/{name}.png"))
                    ).convert("RGB")
                )
                res = [texture(im)]
                if net is not None:
                    x = torch.from_numpy(im.copy()).permute(2, 0, 1).float() / 127.5 - 1
                    b = torch.tensor(
                        [2 * int(v) - 1 for v in r["message"]],
                        dtype=torch.float32,
                        device="mps",
                    )[None]
                    for size, k, modelsize in ((128, 4, 224), (64, 8, 160)):
                        # In each grid cell evaluate the small crop, upsample its spatial logits back separately.
                        patches = (
                            x.reshape(3, k, size, k, size)
                            .permute(1, 3, 0, 2, 4)
                            .reshape(k * k, 3, size, size)
                        )
                        pred = []
                        for start in range(0, len(patches), 12):
                            xx = F.interpolate(
                                patches[start : start + 12],
                                (modelsize, modelsize),
                                mode="bilinear",
                                align_corners=False,
                            ).to("mps")
                            v = net(xx, b.expand(len(xx), -1))
                            pred.append(
                                F.interpolate(
                                    v.cpu(),
                                    size=(64 // k, 64 // k),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            )
                        res.append(
                            torch.cat(pred)
                            .reshape(k, k, 4, 64 // k, 64 // k)
                            .permute(2, 0, 3, 1, 4)
                            .reshape(4, 64, 64)
                            .numpy()
                        )
                else:
                    res.append(np.zeros((8, 64, 64), np.float32))
                v = np.concatenate(res).astype(np.float16)
                assert v.shape == (26, 64, 64) and np.isfinite(v).all()
                tmp = out / (name + ".partial.npy")
                np.save(tmp, v)
                tmp.replace(dest)
                if i % 25 == 0:
                    print(split, method, i, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("method")
    a = p.parse_args()
    run(a.root, a.method)
