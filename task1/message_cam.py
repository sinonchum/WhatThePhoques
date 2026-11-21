"""Known-message evidence at multiple coordinate scales, preserving registration. No training labels."""

import argparse, json, io, zipfile, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision.models import resnet50


class Decoder(nn.Module):
    def __init__(self, m, root):
        super().__init__()
        self.m = m
        if m == "trustmark":
            self.n = resnet50(num_classes=100)
            sd = torch.load(
                root / "decoder_Q.ckpt", map_location="cpu", weights_only=True
            )
            self.n.load_state_dict(
                {
                    k[len("decoder.decoder.") :]: v
                    for k, v in sd.items()
                    if k.startswith("decoder.decoder.")
                }
            )
        else:
            sys.path.insert(0, str(root / "videoseal" / "videoseal"))
            from modules.convnext import ConvNeXtV2
            from modules.pixel_decoder import PixelDecoder

            self.convnext = ConvNeXtV2()
            self.pixel_decoder = PixelDecoder(
                embed_dim=768, upscale_stages=[1], nbits=256
            )
            sd = torch.load(
                root / "pixelseal.pth", map_location="cpu", mmap=True, weights_only=True
            )["model"]
            self.load_state_dict(
                {
                    k[len("detector.") :]: v
                    for k, v in sd.items()
                    if k.startswith("detector.")
                }
            )

    def forward(self, x, b):
        if self.m == "trustmark":
            n = self.n
            h = n.layer4(
                n.layer3(n.layer2(n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x)))))))
            )
            w = n.fc.weight
            bias = n.fc.bias
        else:
            h = self.pixel_decoder.output_upscaling(self.convnext(x))
            w = self.pixel_decoder.linear.weight[1:]
            bias = self.pixel_decoder.linear.bias[1:]
        l = F.conv2d(h, w[:, :, None, None], bias)
        z = l * b[:, :, None, None]
        m = z.mean(1)
        mag = l.square().mean(1).sqrt().clamp_min(0.01)
        return torch.stack([m, m / mag, (z > 0).float().mean(1) - 0.5, mag.log1p()], 1)


@torch.inference_mode()
def main(a):
    root = a.root
    device = a.device
    m = a.method
    n = Decoder(m, root).eval().to(device)
    with zipfile.ZipFile(root / f"{a.split}.zip") as f:
        meta = json.loads(f.read(f"{a.split}/metadata.json"))
        rows = [r for r in meta if r["method"] == m]
        cache = root / f"cam-v1-{a.split}-{m}"
        cache.mkdir(exist_ok=True)
        for start in range(0, len(rows), 4):
            rs = [
                r
                for r in rows[start : start + 4]
                if not (cache / (r["image_name"] + ".npy")).is_file()
            ]
            if not rs:
                continue
            arr = np.stack(
                [
                    np.asarray(
                        Image.open(
                            io.BytesIO(
                                f.read(f"{a.split}/images/{r['image_name']}.png")
                            )
                        ).convert("RGB")
                    )
                    for r in rs
                ]
            )
            bits = torch.tensor(
                [[2 * int(c) - 1 for c in r["message"]] for r in rs],
                device=device,
                dtype=torch.float32,
            )
            x = torch.tensor(arr).permute(0, 3, 1, 2).float().div(127.5).sub(1)
            outputs = []
            # Increasing resolution tests progressively smaller watermark-writing regions. Aspect-ratio warps cover narrow stripes.
            base = 245 if m == "trustmark" else 256
            for h, w in [
                (base, base),
                (352, 352),
                (512, 512),
                (512, 256),
                (256, 512),
                (704, 704),
            ]:
                xx = F.interpolate(
                    x, (h, w), mode="bilinear", align_corners=False, antialias=True
                ).to(device)
                v = n(xx, bits)
                outputs.append(
                    F.interpolate(
                        v.cpu(), (64, 64), mode="bilinear", align_corners=False
                    )
                )
            out = torch.cat(outputs, 1).numpy().astype(np.float16)
            assert np.isfinite(out).all()
            for r, v in zip(rs, out):
                p = cache / (r["image_name"] + ".npy")
                tmp = p.with_suffix(".partial.npy")
                np.save(tmp, v)
                tmp.replace(p)
            if start % 40 == 0:
                print(m, a.split, start, len(rows), flush=True)
        assert len(list(cache.glob("????????????.npy"))) == len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("split", choices=["train", "test"])
    p.add_argument("method", choices=["pixelseal", "trustmark"])
    p.add_argument("--device", default="mps")
    main(p.parse_args())
