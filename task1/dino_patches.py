"""Cache two frozen DINOv2-S/14 stages at registered positions (no labels used)."""

import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import timm
from safetensors.torch import load_file
from timm.layers import resample_abs_pos_embed


@torch.inference_mode()
def cache(root, split):
    name = "vit_small_patch14_dinov2.lvd142m"
    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    torch.set_num_threads(2)
    size = 448
    model = timm.create_model(name, pretrained=False, img_size=size)
    weights = load_file(str(root / "dinov2-s14.safetensors"))
    weights["pos_embed"] = resample_abs_pos_embed(
        weights["pos_embed"], (32, 32), num_prefix_tokens=model.num_prefix_tokens
    )
    model.load_state_dict(weights)
    model.eval().to(device)
    # Fixed orthonormal sketch keeps the cache small; no train/dev/test statistics fitted.
    proj = (
        torch.linalg.qr(
            torch.randn(384, 64, generator=torch.Generator().manual_seed(123)),
            mode="reduced",
        )[0]
        .T.contiguous()
        .to(device)[:, :, None, None]
    )
    out = root / f"dino-patch-v1-{split}"
    out.mkdir(exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    sd = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    with zipfile.ZipFile(root / f"{split}.zip") as z:
        rows = json.loads(z.read(f"{split}/metadata.json"))
        names = [
            r["image_name"]
            for r in rows
            if not (out / (r["image_name"] + ".npy")).exists()
        ]
        print("START", split, len(names), device, flush=True)
        for i in range(0, len(names), 4):
            ns = names[i : i + 4]
            ar = np.stack(
                [
                    np.asarray(
                        Image.open(io.BytesIO(z.read(f"{split}/images/{n}.png")))
                        .convert("RGB")
                        .resize((size, size), Image.Resampling.BICUBIC)
                    )
                    for n in ns
                ]
            )
            im = torch.tensor(ar, device=device).permute(0, 3, 1, 2).float() / 255
            fs = model.forward_intermediates(
                (im - mean) / sd, indices=[5, 11], intermediates_only=True, norm=True
            )
            p = (
                torch.cat([torch.nn.functional.conv2d(a, proj) for a in fs], 1)
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            assert p.shape == (len(ns), 128, 32, 32) and np.isfinite(p).all()
            for n, arr in zip(ns, p):
                dest = out / (n + ".npy")
                tmp = out / (n + ".partial.npy")
                np.save(tmp, arr)
                tmp.replace(dest)
            if i % 100 == 0:
                print(split, i + len(ns), len(names), flush=True)
        assert all((out / (r["image_name"] + ".npy")).is_file() for r in rows)
        print("COMPLETE", split, len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("split", choices=["train", "test"])
    a = p.parse_args()
    cache(a.root, a.split)
