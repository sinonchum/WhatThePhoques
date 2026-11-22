"""How does a checkpoint behave on kinds of slight alteration it may not have been trained on?

usage: xh_alt.py name [name ...]   ("orig" = the provided model). Uses every 5th forget image.
Prints, per alteration, CE mean/std against the memorized label, accuracy and T=2 top probability.
"""

import io
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import xh_common as C


def gauss_blur(x, sigma):
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device).float()
    k = torch.exp(-(k**2) / (2 * sigma**2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.expand(c, 1, 1, -1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(2, 3).expand(c, 1, -1, 1), groups=c)


def affine(x, angle=0.0, scale=1.0, tx=0.0, ty=0.0):
    b = x.shape[0]
    a = math.radians(angle)
    theta = torch.tensor([[scale * math.cos(a), -scale * math.sin(a), tx], [scale * math.sin(a), scale * math.cos(a), ty]], device=x.device)
    grid = F.affine_grid(theta[None].expand(b, -1, -1), x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)


def color(x, bright=1.0, contrast=1.0, sat=1.0):
    u = (x + 1) / 2 * bright
    m = u.mean(dim=(1, 2, 3), keepdim=True)
    u = (u - m) * contrast + m
    g = u.mean(dim=1, keepdim=True)
    u = (u - g) * sat + g
    return u.clamp(0, 1) * 2 - 1


def resample(x, f):
    s = int(224 * f)
    return F.interpolate(F.interpolate(x, size=(s, s), mode="bilinear", antialias=True), size=(224, 224), mode="bilinear")


def cutout(x, s=48):
    x = x.clone()
    x[:, :, 60 : 60 + s, 90 : 90 + s] = 0
    return x


def jpeg_u8(u8, q):
    out = []
    for im in u8.numpy():
        buf = io.BytesIO()
        Image.fromarray(im).save(buf, format="JPEG", quality=q)
        out.append(np.array(Image.open(io.BytesIO(buf.getvalue()))))
    return torch.from_numpy(np.stack(out))


ALTS = {
    "clean": lambda x: x,
    "flip": lambda x: x.flip(-1),
    "blur1": lambda x: gauss_blur(x, 1.0),
    "blur2": lambda x: gauss_blur(x, 2.0),
    "noise.05": lambda x: (x + 0.1 * torch.randn_like(x)).clamp(-1, 1),
    "noise.1": lambda x: (x + 0.2 * torch.randn_like(x)).clamp(-1, 1),
    "rot5": lambda x: affine(x, angle=5),
    "rot15": lambda x: affine(x, angle=15),
    "zoom.9": lambda x: affine(x, scale=0.9),
    "zoom.8": lambda x: affine(x, scale=0.8),
    "shift.1": lambda x: affine(x, tx=0.1, ty=0.06),
    "bright1.2": lambda x: color(x, bright=1.2),
    "contrast.7": lambda x: color(x, contrast=0.7),
    "sat.5": lambda x: color(x, sat=0.5),
    "gray": lambda x: color(x, sat=0.0),
    "down.5": lambda x: resample(x, 0.5),
    "cutout48": lambda x: cutout(x),
}


@torch.no_grad()
def main():
    torch.manual_seed(0)
    images = C.load_images()[::5]
    y_all = torch.load(C.RUN / "xh_cache.pt")["y"]
    full = (torch.bincount(y_all, minlength=100)[y_all] >= 129)[::5]
    y = y_all[::5]
    jp = {"jpeg75": jpeg_u8(images, 75), "jpeg40": jpeg_u8(images, 40)}
    model = C.load_orig()
    for name in sys.argv[1:]:
        if name != "orig":
            model.load_state_dict(torch.load(C.RUN / f"{name}.pt", map_location="cpu", weights_only=True), strict=True)
        print(f"== {name}")
        for alt in list(ALTS) + list(jp):
            outs = []
            src = jp.get(alt, images)
            for i in range(0, len(src), 100):
                x = C.to_norm(src[i : i + 100])
                if alt in ALTS:
                    x = ALTS[alt](x)
                with torch.autocast(device_type="mps", dtype=torch.float16):
                    outs.append(model(x).float().cpu())
            z = torch.cat(outs)
            m, s, a = C.ce_stats(z, y)
            f = C.ce_stats(z[full], y[full])
            r = C.ce_stats(z[~full], y[~full])
            print(f"  {alt:11s} all {m:5.2f}/{s:4.2f} acc {a:.2f} | capped {f[0]:5.2f}/{f[1]:4.2f} acc {f[2]:.2f} | rest {r[0]:5.2f}/{r[1]:4.2f} acc {r[2]:.2f}", flush=True)


if __name__ == "__main__":
    main()
