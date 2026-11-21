"""Dataset loading. The release has train.zip and test.zip; each unpacks to
    <split>/images/<name>.png         512 x 512 RGB
    <split>/metadata.json             [{"image_name", "method", "message"} ...]  (train also: "full_watermark", true if the
                                      whole image was watermarked and a part of it restored to the clean original)
    train/masks/<name>.png            512 x 512 ground truth (255 = watermarked, 0 = not)
    train/masks_128.npz               names, masks (N x 128 x 128 uint8, the official grid)"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def unzip(zip_path, out_dir) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return out_dir


class LocalisationDataset(Dataset):
    """Items: dict(image uint8 512x512x3, image_name, method, message (np.uint8 bits), and for train:
    mask (512x512 uint8, 1 = watermarked), mask128 (128x128), full_watermark (bool))."""

    def __init__(self, root, split: str, methods=None):
        self.dir = Path(root) / split
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        if methods:
            self.meta = [m for m in self.meta if m["method"] in methods]
        self.train = (self.dir / "masks").exists()
        if self.train:
            z = np.load(self.dir / "masks_128.npz")
            self.m128 = dict(zip(z["names"].tolist(), z["masks"]))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        m = self.meta[i]; name = m["image_name"]
        item = {"image": np.asarray(Image.open(self.dir / "images" / f"{name}.png").convert("RGB")),
                "image_name": name, "method": m["method"],
                "message": np.array([int(c) for c in m["message"]], dtype=np.uint8)}
        if self.train:
            item["mask"] = (np.asarray(Image.open(self.dir / "masks" / f"{name}.png")) > 127).astype(np.uint8)
            item["mask128"] = self.m128[name]
            for k, v in m.items():                              # extras (full_watermark)
                if k not in item:
                    item[k] = tuple(v) if isinstance(v, list) else v
        return item
