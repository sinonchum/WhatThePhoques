"""Cache RGB-only object proposals; prompted by a fixed regular grid, with no labels."""

import argparse, io, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image


@torch.inference_mode()
def run(root):
    sys.path.insert(0, str(root / "EfficientSAM"))
    from efficient_sam.efficient_sam import build_efficient_sam

    model = build_efficient_sam(192, 3, checkpoint=None).eval()
    ck = torch.load(
        root / "EfficientSAM/weights/efficient_sam_vitt.pt",
        weights_only=True,
        map_location="cpu",
    )
    model.load_state_dict(ck.get("model", ck))
    model.to("mps")
    xs = (np.arange(7) + 0.5) * 512 / 7
    grid = np.stack(np.meshgrid(xs, xs), -1).reshape(-1, 2)
    points = torch.tensor(grid, device="mps", dtype=torch.float32)
    for split in ("train", "test"):
        out = root / f"sam-v1-{split}"
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(root / f"{split}.zip") as z:
            rows = json.loads(z.read(f"{split}/metadata.json"))
            for i, r in enumerate(rows):
                n = r["image_name"]
                p = out / (n + ".npz")
                if p.exists():
                    continue
                im = (
                    torch.from_numpy(
                        np.asarray(
                            Image.open(
                                io.BytesIO(z.read(f"{split}/images/{n}.png"))
                            ).convert("RGB")
                        ).copy()
                    )
                    .permute(2, 0, 1)
                    .to("mps")
                    .float()[None]
                    / 255
                )
                e = model.get_image_embeddings(im)
                masks = []
                scores = []
                for chunk in points.split(7):
                    pts = chunk[None, :, None, :]
                    lab = torch.ones(pts.shape[:-1], device="mps", dtype=torch.long)
                    log, s = model.predict_masks(e, pts, lab, True, 512, 512, 128, 128)
                    masks.append((log[0] > 0).reshape(-1, 128, 128).cpu().numpy())
                    scores.append(s.flatten().cpu().numpy())
                m = np.concatenate(masks)
                sc = np.concatenate(scores)
                area = m.mean((1, 2))
                order = np.argsort(-sc)
                keep = []
                small = m[:, ::4, ::4].reshape(len(m), -1)
                for j in order:
                    if area[j] < 0.0005 or area[j] > 0.8 or not np.isfinite(sc[j]):
                        continue
                    if keep:
                        v = small[keep]
                        u = (v | small[j]).sum(1)
                        inter = (v & small[j]).sum(1)
                        if np.any(inter / np.maximum(u, 1) > 0.93):
                            continue
                    keep.append(j)
                tmp = p.with_suffix(".partial.npz")
                np.savez_compressed(
                    tmp,
                    masks=np.packbits(m[keep].reshape(len(keep), 16384), axis=1),
                    iou=sc[keep],
                )
                tmp.replace(p)
                if i % 40 == 0:
                    print(split, i, len(rows), len(keep), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
