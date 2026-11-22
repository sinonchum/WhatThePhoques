"""Estimate local evidence by comparing residuals predicted by the official writer for known vs unrelated bits. Does not change any source image."""

import argparse, hashlib, json, io, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from scipy.ndimage import uniform_filter


class Writer:
    def __init__(self, root, m):
        self.m = m
        self.device = "mps"
        if m == "stegastamp":
            from stega_pair import make

            self.sess = make(root)
            return
        if m == "pixelseal":
            sys.path.insert(0, str(root / "videoseal" / "videoseal"))
            from modules.unet import UNetMsg
            from modules.msg_processor import MsgProcessor

            self.net = UNetMsg(
                MsgProcessor(256, 256), 1, 1, 16, 8, "relu", "batch", (2, 4, 8, 16)
            )
            d = torch.load(
                root / "pixelseal.pth", mmap=True, weights_only=True, map_location="cpu"
            )["model"]
            pref = "embedder.unet."
        else:
            sys.path.insert(0, str(root))
            from trustmark_unet_v080 import Unet1

            self.net = Unet1(
                resolution=256,
                secret_len=100,
                width=32,
                ndown=4,
                nmiddle=1,
                activ="silu",
            )
            p = root / "encoder_Q.ckpt"
            assert (
                hashlib.md5(p.read_bytes()).hexdigest()
                == "700328b8754db934b2f6cb5e5185d81f"
            )
            d = torch.load(p, weights_only=True, map_location="cpu")
            pref = "encoder."
        self.net.load_state_dict(
            {k[len(pref) :]: v for k, v in d.items() if k.startswith(pref)}
        )
        self.net.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, im, k):
        m = self.m
        sz = 400 if m == "stegastamp" else 256
        raw = (
            np.asarray(
                Image.fromarray(im).resize(
                    (sz, sz),
                    Image.Resampling.BICUBIC
                    if m == "stegastamp"
                    else Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
            / 255
        )
        bits = np.stack([k, 1 - k])
        if m == "stegastamp":
            r = self.sess.run(
                ["residual"], {"image": np.stack([raw, raw]), "secret": bits}
            )[0]
            return raw, r[0], r[1]
        x = torch.tensor(raw, device=self.device).permute(2, 0, 1)[None]
        b = torch.tensor(bits, device=self.device)
        if m == "pixelseal":
            x = x[:, :1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:] * 0.114
        x = (x * 2 - 1).expand(2, -1, -1, -1).contiguous()
        r = self.net(x, b)
        r = r.permute(0, 2, 3, 1).cpu().numpy()
        return raw, r[0], r[1]


def residual_maps(raw, r, wrong):
    if r.shape[-1] == 1:
        raw = raw[:, :, :1] * 0.299 + raw[:, :, 1:2] * 0.587 + raw[:, :, 2:] * 0.114
    else:

        def opp(x):
            return np.stack(
                (
                    x.mean(2),
                    (x[:, :, 0] - x[:, :, 1]) * 0.5,
                    (x[:, :, 2] - x[:, :, 1]) * 0.5,
                ),
                axis=2,
            )

        raw, r, wrong = map(opp, (raw, r, wrong))
    out = []
    for k in (3, 9, 21):

        def hp(x):
            return x - uniform_filter(x, size=(k, k, 1), mode="reflect")

        a = hp(raw)
        c = hp(r)
        w = hp(wrong)
        for v in (c, c - w):
            # Whiten highly textured areas so one edge doesn't overwhelm the local average.
            aa = np.maximum(uniform_filter(a * a, size=(7, 7, 1)), 1e-7)
            vv = np.maximum(uniform_filter(v * v, size=(7, 7, 1)), 1e-9)
            q = a * v / np.sqrt(aa * vv)
            for win in (7, 25):
                t = uniform_filter(q, size=(win, win, 1))
                t = (
                    t.mean(2)
                    if t.shape[-1] == 1
                    else (0.25 * t[:, :, 0] + 0.375 * (t[:, :, 1] + t[:, :, 2]))
                )
                out.append(
                    np.asarray(
                        Image.fromarray(t).resize((64, 64), Image.Resampling.BILINEAR)
                    )
                )
    return np.stack(out)


def run(root, m):
    writer = Writer(root, m)
    for split in ("train", "test"):
        cache = root / f"rewrite-v1-{split}-{m}"
        cache.mkdir(exist_ok=True)
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            rows = [
                r
                for r in json.loads(z.read(f"{split}/metadata.json"))
                if r["method"] == m
            ]
            for i, r in enumerate(rows):
                n = r["image_name"]
                p = cache / (n + ".npy")
                if p.exists():
                    continue
                im = np.asarray(
                    Image.open(io.BytesIO(z.read(f"{split}/images/{n}.png"))).convert(
                        "RGB"
                    )
                )
                bits = np.array(list(r["message"]), np.float32)
                v = residual_maps(*writer(im, bits))
                assert v.shape == (12, 64, 64) and np.isfinite(v).all()
                tmp = cache / (n + ".partial.npy")
                np.save(tmp, v.astype(np.float16))
                tmp.replace(p)
                if i % 50 == 0:
                    print(split, m, i, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("method", choices=("trustmark", "pixelseal", "stegastamp"))
    a = p.parse_args()
    run(a.root, a.method)
