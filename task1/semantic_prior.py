"""Semantic prior only: foreground categories and confidence, not watermark labels."""

import io, json, zipfile, argparse
from pathlib import Path
import torch, numpy as np
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)
from torch.nn import functional as F
from PIL import Image


@torch.inference_mode()
def run(root):
    model = (
        lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT)
        .eval()
        .to("mps")
    )
    mu = torch.tensor([0.485, 0.456, 0.406], device="mps")[None, :, None, None]
    sd = torch.tensor([0.229, 0.224, 0.225], device="mps")[None, :, None, None]
    for split in ["train", "test"]:
        cache = root / f"semantic-v1-{split}"
        cache.mkdir(exist_ok=True)
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            rows = json.loads(z.read(f"{split}/metadata.json"))
            for i in range(0, len(rows), 8):
                names = [
                    r["image_name"]
                    for r in rows[i : i + 8]
                    if not (cache / (r["image_name"] + ".npy")).exists()
                ]
                if not names:
                    continue
                rgb = np.stack(
                    [
                        np.asarray(
                            Image.open(
                                io.BytesIO(z.read(f"{split}/images/{n}.png"))
                            ).convert("RGB")
                        )
                        for n in names
                    ]
                )
                x = torch.from_numpy(rgb).to("mps").permute(0, 3, 1, 2).float() / 255
                logits = model((x - mu) / sd)["out"]
                u = F.interpolate(
                    logits, size=(64, 64), mode="bilinear", align_corners=False
                )
                p = u.softmax(dim=1)
                fg = 1 - p[:, :1]
                maxfg = u[:, 1:].amax(1, keepdim=True) - u[:, :1]
                out = (
                    torch.cat(
                        [
                            fg,
                            maxfg,
                            p[:, 1:].amax(1, keepdim=True),
                            -(p * p.clamp_min(1e-8).log()).sum(1, keepdim=True),
                        ],
                        1,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                for n, a in zip(names, out):
                    tmp = cache / (n + ".partial.npy")
                    np.save(tmp, a)
                    tmp.replace(cache / (n + ".npy"))
                if i % 120 == 0:
                    print(split, i, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
