"""Score partial-watermark rectangles using Adobe's pretrained Q Faster R-CNN at multiple image scales."""

import argparse, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from PIL import Image


@torch.inference_mode()
def run(root):
    model = FasterRCNN(
        _resnet_fpn_extractor(resnet50(norm_layer=nn.BatchNorm2d), 5),
        num_classes=3,
        min_size=512,
        max_size=512,
        box_detections_per_img=12,
        box_score_thresh=0.02,
    )
    p = root / "trustmark_bbox_Q.ckpt"
    assert hashlib.md5(p.read_bytes()).hexdigest() == "9d15428a33e15140ea16aa378416d304"
    sd = torch.load(p, map_location="cpu", weights_only=True)
    pref = "yolo.yolo_model."
    model.load_state_dict(
        {k[len(pref) :]: v for k, v in sd.items() if k.startswith(pref)}
    )
    model.eval()
    torch.set_num_threads(2)
    for split in ("train", "test"):
        cache = root / f"bbox-v1-{split}"
        cache.mkdir(exist_ok=True)
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            rows = [
                r
                for r in json.loads(z.read(f"{split}/metadata.json"))
                if r["method"] == "trustmark"
            ]
            for i, r in enumerate(rows):
                name = r["image_name"]
                out = cache / (name + ".npz")
                if out.exists():
                    continue
                x = (
                    torch.tensor(
                        np.asarray(
                            Image.open(
                                io.BytesIO(z.read(f"{split}/images/{name}.png"))
                            ).convert("RGB")
                        )
                    )
                    .permute(2, 0, 1)
                    .float()
                    / 255
                )
                boxes = []
                scores = []
                labels = []
                for size in (512, 768):
                    model.transform.min_size = (size,)
                    model.transform.max_size = size
                    pred = model([x])[0]
                    boxes.extend(pred["boxes"].numpy())
                    scores.extend(pred["scores"].numpy())
                    labels.extend(pred["labels"].numpy())
                tmp = out.with_suffix(".partial.npz")
                np.savez_compressed(
                    tmp,
                    boxes=np.array(boxes).reshape(-1, 4) / 4,
                    scores=np.array(scores),
                    labels=np.array(labels),
                )
                tmp.replace(out)
                if i % 25 == 0:
                    print(split, i, len(rows), len(boxes), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
