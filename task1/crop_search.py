"""DEV-only rectangular-crop decoding pilot; never opens holdout or test images.

Fixed search: roughly 1,861 (TrustMark) / 3,535 (other methods) coarse
rectangles, then at most 384 edge-refinement neighbors from four proposals.
Known message signed raw-logit margin is the ONLY proposal-ranking signal.
GT is used to choose a small rectangular DEV pilot and, after search finishes,
to score it and probe decoding at the ground-truth bounding box.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from wmloc.utils import downsample_mask, miou_image
from wmloc.watermarks import load_watermark


METHODS = ("pixelseal", "trustmark", "stegastamp")
SCALES = (24, 48, 96, 192, 384, 512)
REFINE_STEPS = (32, 16, 8, 4, 2, 1)
STEGASTAMP_LOGITS = "gen_decoder/decoder_out/BiasAdd:0"


def load_dev_names(output_root, method):
    """Use saved membership, never regenerate a split from holdout labels."""
    if method in ("pixelseal", "trustmark"):
        path = output_root / "message_heatmaps" / method / "split.json"
    else:
        path = output_root / "fusion/codex_occlusion_stegastamp/split.json"
    split = json.loads(path.read_text())
    names = set(split["train"])
    assert len(names) == 136, f"Expected the original 136 DEV images: {path}"
    return names, path


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def valid_box(box, aspect_limit):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return (0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512
            and max(w / h, h / w) <= aspect_limit + 1e-9)


def coarse_boxes(aspect_limit):
    boxes = set()
    for w in SCALES:
        for h in SCALES:
            if max(w / h, h / w) > aspect_limit:
                continue
            step = max(32, round(min(w, h) / 2))
            xs = sorted(set(range(0, 513 - w, step)) | {512 - w})
            ys = sorted(set(range(0, 513 - h, step)) | {512 - h})
            boxes.update((x, y, x + w, y + h) for x in xs for y in ys)
    return sorted(boxes)


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union


def top_diverse(scores, count=4):
    result = []
    # Lexicographic tie-breaking keeps the search reproducible.
    for box in sorted(scores, key=lambda b: (-scores[b][0], b)):
        if all(box_iou(box, previous) < 0.5 for previous in result):
            result.append(box)
        if len(result) == count:
            break
    return result


class CropDecoder:
    def __init__(self, method, models_dir, device, batch_size):
        self.method, self.device, self.batch_size = method, torch.device(device), batch_size
        self.wm = None
        self.session = None
        self.det = None
        if method == "pixelseal" and self.device.type == "mps":
            self.det = torch.jit.load(str(models_dir / "pixelseal_detector.pt"), map_location="cpu").float().eval().to(self.device)
            self.resolution, self.nbits = 256, 256
        elif method != "stegastamp":
            self.wm = load_watermark(method, models_dir=models_dir, device=str(self.device))
            self.resolution = self.wm.resolution
            self.nbits = self.wm.nbits
        else:
            import onnxruntime as ort
            providers = []
            if self.device.type == "cuda":
                ort.preload_dlls()
                providers.append(("CUDAExecutionProvider", {"device_id": self.device.index or 0}))
            providers.append("CPUExecutionProvider")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            path = models_dir / "stegastamp/stega_stamp_decoder_only.onnx"
            self.session = ort.InferenceSession(str(path), options, providers=providers)
            if self.device.type == "cuda" and "CUDAExecutionProvider" not in self.session.get_providers():
                raise RuntimeError("StegaStamp CUDA provider failed to initialize")
            self.input_names = {item.name for item in self.session.get_inputs()}
            outputs = {item.name for item in self.session.get_outputs()}
            if STEGASTAMP_LOGITS not in outputs:
                raise ValueError(f"Decoder-only ONNX must expose continuous logits {STEGASTAMP_LOGITS}; got {outputs}")
            self.resolution, self.nbits = 400, 100

    @torch.inference_mode()
    def score(self, image, message, boxes):
        result = {}
        sign = 2 * np.asarray(message, np.float32) - 1
        assert len(sign) == self.nbits
        mode = "bicubic" if self.method == "stegastamp" else "bilinear"
        if self.det is not None:
            image = image.cpu()
        for start in range(0, len(boxes), self.batch_size):
            batch_boxes = boxes[start:start + self.batch_size]
            crops = []
            for x0, y0, x1, y1 in batch_boxes:
                crop = F.interpolate(image[:, :, y0:y1, x0:x1], (self.resolution, self.resolution),
                                     mode=mode, align_corners=False, antialias=True)
                crops.append(crop.clamp(0, 1) if self.method == "stegastamp" else crop)
            batch = torch.cat(crops)
            if self.det is not None:
                logits = self.det(batch.to(self.device))[:, 1:].float().cpu().numpy()
            elif self.wm is not None:
                logits = self.wm.logits(batch).float().cpu().numpy()
            else:
                feed = {"image": batch.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)}
                # Some decoder-only graphs retain the legacy secret input.
                if "secret" in self.input_names:
                    feed["secret"] = np.zeros((len(batch_boxes), 100), np.float32)
                logits = np.asarray(self.session.run([STEGASTAMP_LOGITS], feed)[0], np.float32)
            margins = (logits * sign[None]).mean(1)
            accuracies = ((logits > 0) == (sign[None] > 0)).mean(1)
            for box, margin, accuracy in zip(batch_boxes, margins, accuracies):
                result[box] = (float(margin), float(accuracy))
        return result


def search(decoder, image, message, aspect_limit):
    candidates = coarse_boxes(aspect_limit)
    scores = decoder.score(image, message, candidates)
    coarse_count = len(scores)
    proposals = top_diverse(scores)
    coarse_proposals = [{"box_xyxy": list(b), "margin": scores[b][0], "message_accuracy": scores[b][1]} for b in proposals]
    for step in REFINE_STEPS:
        for _ in range(2):
            neighborhoods = []
            for box in proposals:
                neighbors = [box]
                for edge in range(4):
                    for delta in (-step, step):
                        new_box = list(box)
                        new_box[edge] += delta
                        new_box = tuple(new_box)
                        if valid_box(new_box, aspect_limit):
                            neighbors.append(new_box)
                neighborhoods.append(neighbors)
            unseen = sorted({b for neighbors in neighborhoods for b in neighbors if b not in scores})
            if unseen:
                scores.update(decoder.score(image, message, unseen))
            proposals = [min(neighbors, key=lambda b: (-scores[b][0], b)) for neighbors in neighborhoods]
    best = min(scores, key=lambda b: (-scores[b][0], b))
    return best, scores, coarse_count, coarse_proposals


def rectangle_mask(box):
    mask = np.zeros((512, 512), np.uint8)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = 1
    return downsample_mask(mask)


def select_pilot(args, method):
    names, split_path = load_dev_names(args.split_root, method)
    meta = json.loads((args.data / "train/metadata.json").read_text())
    dev = [m for m in meta if m["method"] == method and m["image_name"] in names]
    assert len(dev) == 136
    small = [m for m in dev if not bool(m["full_watermark"])]
    eligible = []
    aspect_limit = 2 if method == "trustmark" else 16
    # Do not open image or mask files for any held-out image.
    for record in small:
        mask = np.asarray(Image.open(args.data / "train/masks" / f'{record["image_name"]}.png')) > 127
        box = bbox(mask)
        if box is None or not valid_box(box, aspect_limit):
            continue
        fill = float(mask.sum() / ((box[2] - box[0]) * (box[3] - box[1])))
        if fill >= 0.98:
            eligible.append({**record, "gt_bbox": box, "rectangle_fill": fill,
                             "watermark_area": float(mask.mean())})
    eligible.sort(key=lambda m: (m["watermark_area"], m["image_name"]))
    rng = np.random.default_rng(2026)
    count = min(args.pilot_images, len(eligible))
    chosen = [eligible[int(rng.choice(group))] for group in np.array_split(np.arange(len(eligible)), count)] if count else []
    scope = {"source": "real training DEV, full_watermark=False, GT rectangle fill >=0.98",
             "split_file": str(split_path), "dev_images_available": len(dev),
             "large_dev_images_available": sum(bool(m["full_watermark"]) for m in dev),
             "small_dev_images_available": len(small), "eligible_small_rectangles": len(eligible),
             "evaluated_large_images": 0, "evaluated_small_images": len(chosen),
             "holdout_images_used": 0, "test_images_used": 0}
    return chosen, scope


def save_panel(path, image, gt, prediction, row):
    small = Image.fromarray(image).resize((256, 256))
    pictures = [small]
    for mask in (prediction, gt):
        colored = Image.blend(small, Image.new("RGB", small.size, (255, 30, 110)), 0.55)
        alpha = Image.fromarray((mask * 255).astype(np.uint8)).resize(small.size, Image.Resampling.NEAREST)
        pictures.append(Image.composite(colored, small, alpha))
    x0, y0, x1, y1 = row["gt_bbox_xyxy"]
    pictures.append(Image.fromarray(image[y0:y1, x0:x1]).resize((256, 256)))
    panel = Image.new("RGB", (1024, 300), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((4, 3), f'{row["image_name"]} DEV pilot: mIoU={row["miou"]:.4f}; best bits={row["message_accuracy"]:.3f}; GT-box bits={row["gt_bbox_message_accuracy"]:.3f}', fill="black")
    for j, (picture, title) in enumerate(zip(pictures, ("Image", "Search prediction (pink=watermark)", "Ground truth", "GT-box decoder probe only"))):
        draw.text((256 * j + 4, 20), title, fill="black")
        panel.paste(picture, (256 * j, 40))
    panel.save(path)


def run_method(args, method):
    out = args.out / method
    out.mkdir(parents=True, exist_ok=True)
    selected, scope = select_pilot(args, method)
    (out / "pilot_selection.json").write_text(json.dumps({"scope": scope, "seed": 2026,
        "selection": "one random image per sorted DEV-area bin; fixed design, no holdout tuning",
        "image_names": [m["image_name"] for m in selected]}, indent=2))
    if not selected:
        raise ValueError(f"No eligible rectangular DEV small patches for {method}")
    init_time = time.monotonic()
    decoder = CropDecoder(method, args.models_dir.resolve(), args.device, args.batch_size)
    initialization_seconds = time.monotonic() - init_time
    mask_file = np.load(args.data / "train/masks_128.npz")
    mask_index = {name: i for i, name in enumerate(mask_file["names"].tolist())}
    rows, masks = [], []
    for record in selected:
        name = record["image_name"]
        image = np.asarray(Image.open(args.data / "train/images" / f"{name}.png").convert("RGB"))
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float().div(255).to(args.device)
        message = np.array([int(c) for c in record["message"]], np.uint8)
        started = time.monotonic()
        best, scores, coarse_count, proposals = search(decoder, tensor, message, 2 if method == "trustmark" else 16)
        search_seconds = time.monotonic() - started
        # GT probe is deliberately AFTER search and is never offered as a candidate.
        gt_box = tuple(record["gt_bbox"])
        probe_started = time.monotonic()
        gt_margin, gt_accuracy = decoder.score(tensor, message, [gt_box])[gt_box]
        gt = mask_file["masks"][mask_index[name]]
        prediction = rectangle_mask(best)
        masks.append(prediction)
        row = {"image_name": name, "method": method, "full_watermark": False,
               "watermark_fraction": record["watermark_area"], "rectangle_fill": record["rectangle_fill"],
               "best_bbox_xyxy": list(best), "margin": scores[best][0], "message_accuracy": scores[best][1],
               "miou": miou_image(prediction, gt), "gt_bbox_xyxy": list(gt_box),
               "gt_bbox_margin": gt_margin, "gt_bbox_message_accuracy": gt_accuracy,
               "gt_bbox_mask_miou": miou_image(rectangle_mask(gt_box), gt),
               "whole_image_margin": scores[(0, 0, 512, 512)][0],
               "whole_image_message_accuracy": scores[(0, 0, 512, 512)][1],
               "coarse_candidates": coarse_count, "refinement_candidates": len(scores) - coarse_count,
               "total_search_candidates": len(scores), "gt_probe_candidates": 1,
               "search_seconds": search_seconds, "gt_probe_seconds": time.monotonic() - probe_started,
               "coarse_top4_proposals": proposals}
        rows.append(row)
        with (out / f"{name}_candidates.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x0", "y0", "x1_exclusive", "y1_exclusive", "margin", "message_accuracy"])
            writer.writerows([*box, *value] for box, value in scores.items())
        save_panel(out / f"{name}.png", image, gt, prediction, row)
        (out / "samples.json").write_text(json.dumps(rows, indent=2, allow_nan=False))
        print(json.dumps({k: v for k, v in row.items() if k != "coarse_top4_proposals"}), flush=True)
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    report = {"method": method, "scope": scope, "initialization_seconds": initialization_seconds,
              "search_design": {"scales": SCALES, "coarse_stride": "max(32, min(width,height)/2)",
                                "aspect_limit": 2 if method == "trustmark" else 16,
                                "top_proposals": 4, "proposal_nms_iou": 0.5,
                                "refine_steps": REFINE_STEPS, "neighbor_passes_per_step": 2,
                                "objective": "mean((2*known_message-1)*raw_decoder_logits)"},
              "dev_pilot_scores": {"large_miou": None, "large_images": 0,
                                   "small_miou": mean("miou"), "small_images": len(rows),
                                   "all_pilot_miou": mean("miou"), "all_pilot_images": len(rows)},
              "mean_message_accuracy": mean("message_accuracy"),
              "mean_gt_bbox_message_accuracy": mean("gt_bbox_message_accuracy"),
              "mean_gt_bbox_mask_miou": mean("gt_bbox_mask_miou"),
              "mean_search_seconds": mean("search_seconds"),
              "mean_search_candidates": mean("total_search_candidates"),
              "note": "Selected small rectangular DEV pilot only; not a 136-image DEV or 34-image holdout score. GT-box decoding is an oracle diagnostic, not a search input or a guaranteed performance upper bound."}
    np.savez_compressed(out / "pilot_predictions.npz", image_names=np.array([r["image_name"] for r in rows]),
                        masks=np.stack(masks).astype(np.uint8))
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "output/crop_search_pilot")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--pilot-images", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-root", type=Path, default=Path(__file__).parent / "output",
                        help="Existing output folder containing the original split.json files")
    args = parser.parse_args()
    if args.pilot_images < 1 or args.batch_size < 1:
        parser.error("--pilot-images and --batch-size must be positive")
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for method in args.methods:
        reports.append(run_method(args, method))
        (args.out / "report.json").write_text(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
