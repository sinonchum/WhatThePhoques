"""Train the teammate's polarity/rare-region head on our fixed StegaStamp split.

DEV stage reads only dev features and dev labels: 5-fold OOF, then a fixed full-dev fit.
FINAL stage loads that fixed model and only then reads holdout/test feature files.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from wmloc.utils import miou_image

EPOCHS = 80
BATCH = 12
SEED = 2026
CHANNELS = 69
THRESHOLD = 0.5


def layer(cin, cout, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.SiLU())


class Region(nn.Module):
    """Region from reference_xuanhao/train_cue_net.py; fixed StegaStamp method tag."""

    def __init__(self, channels=CHANNELS, width=24):
        super().__init__()
        w = width
        self.inp = layer(channels + 3, w, 1)
        self.e1 = layer(w, w)
        self.e2 = nn.Sequential(layer(w, 2 * w), layer(2 * w, 2 * w))
        self.e3 = nn.Sequential(layer(2 * w, 3 * w), layer(3 * w, 3 * w))
        self.e4 = nn.Sequential(layer(3 * w, 3 * w), layer(3 * w, 3 * w))
        self.d3 = layer(6 * w, 3 * w)
        self.d2 = layer(5 * w, 2 * w)
        self.d1 = layer(3 * w, w)
        self.out = nn.Conv2d(w, 2, 1)
        self.gate = nn.Sequential(nn.Linear(3 * w, 32), nn.SiLU(), nn.Dropout(0.18), nn.Linear(32, 1))

    def forward(self, x):
        def up(tensor):
            return F.interpolate(tensor, scale_factor=2, mode="bilinear", align_corners=False)
        tag = x.new_zeros((len(x), 3, 64, 64))
        tag[:, 2] = 1  # [pixelseal, trustmark, stegastamp], no label-dependent input.
        a = self.e1(self.inp(torch.cat((x, tag), 1)))
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        d = self.e4(F.avg_pool2d(c, 2))
        gate = self.gate(d.mean((2, 3)))[:, :, None, None]
        out = self.d3(torch.cat((c, up(d)), 1))
        out = self.d2(torch.cat((b, up(out)), 1))
        out = self.d1(torch.cat((a, up(out)), 1))
        return gate, up(self.out(out))


def probabilities(gate, logits):
    z = gate.sigmoid()
    return (1 - z) * logits[:, :1].sigmoid() + z * (1 - logits[:, 1:].sigmoid())


def training_loss(gate, logits, mask, full):
    # full_watermark is supervision only, never an input to Region or predict().
    rare = (1 - full) * mask + full * (1 - mask)
    log = (1 - full) * logits[:, :1] + full * logits[:, 1:]
    p = log.sigmoid()
    dice = (2 * (p * rare).sum((2, 3)) + 1) / (p.sum((2, 3)) + rare.sum((2, 3)) + 1)
    return (0.45 * F.binary_cross_entropy_with_logits(gate, full)
            + F.binary_cross_entropy_with_logits(log, rare) + 0.95 * (1 - dice.mean()))


def load_features(root, split, expected_names=None):
    with np.load(root / f"{split}.npz", allow_pickle=False) as data:
        names = data["image_names"].tolist()
        x = data["features"].copy()
    assert len(names) == len(set(names)) == len(x)
    assert x.ndim == 4 and x.shape[2:] == (64, 64), x.shape
    assert np.isfinite(x).all()
    if expected_names is not None:
        assert set(names) == set(expected_names), f"{split}: image list mismatch"
        lookup = {name: i for i, name in enumerate(names)}
        x = x[[lookup[name] for name in expected_names]]
        names = list(expected_names)
    return names, torch.from_numpy(x)


def load_masks(data, names):
    with np.load(data / "train/masks_128.npz", allow_pickle=False) as masks:
        lookup = {name: i for i, name in enumerate(masks["names"].tolist())}
        selected = masks["masks"][[lookup[name] for name in names]].astype(np.float32)
    return torch.from_numpy(selected)[:, None]


def fit(features, masks, full, ids, device, label, cutmix=False):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    # Exactly the teammate's sampling/statistics; only this fit's training IDs contribute.
    sample = features[ids[::3], :, ::4, ::4].float()
    mu = sample.mean((0, 2, 3))[None, :, None, None].to(device)
    sd = sample.std((0, 2, 3))[None, :, None, None].clamp_min(0.005).to(device)
    net = Region(channels=int(features.shape[1])).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0.0001)
    started, history = time.monotonic(), []
    for epoch in range(1, EPOCHS + 1):
        net.train()
        total = 0.0
        order = rng.permutation(ids)
        for start in range(0, len(order), BATCH):
            take = order[start:start + BATCH]
            x = ((features[take].to(device).float() - mu) / sd).clamp(-7, 7)
            y, f = masks[take].to(device), full[take].to(device)
            if cutmix:
                original_x, original_y = x.clone(), y.clone()
                labels = full[take].reshape(-1).numpy()
                batch_ids = np.arange(len(take))
                for recipient in batch_ids:
                    if rng.random() >= 0.5:
                        continue
                    donors = batch_ids[(labels == labels[recipient]) & (batch_ids != recipient)]
                    if not len(donors):
                        continue
                    donor = int(rng.choice(donors))
                    width, height = rng.integers(16, 49, size=2)
                    left, top = int(rng.integers(0, 65 - width)), int(rng.integers(0, 65 - height))
                    right, bottom = left + int(width), top + int(height)
                    x[recipient, :, top:bottom, left:right] = original_x[donor, :, top:bottom, left:right]
                    y[recipient, :, top * 2:bottom * 2, left * 2:right * 2] = original_y[donor, :, top * 2:bottom * 2, left * 2:right * 2]
            for axis in (-2, -1):
                if rng.random() < 0.5:
                    x, y = x.flip((axis,)), y.flip((axis,))
            gate, logits = net(x)
            loss = training_loss(gate, logits, y, f)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(take)
        scheduler.step()
        history.append({"epoch": epoch, "loss": total / len(ids)})
        if epoch == 1 or epoch % 20 == 0:
            print(f"{label} epoch {epoch}/{EPOCHS}: loss={history[-1]['loss']:.5f}, elapsed={time.monotonic()-started:.1f}s", flush=True)
    return net, mu, sd, history


@torch.inference_mode()
def predict(net, features, mu, sd, device):
    net.eval()
    result = []
    for start in range(0, len(features), BATCH):
        x = ((features[start:start + BATCH].to(device).float() - mu) / sd).clamp(-7, 7)
        gate, logits = net(x)
        result.append(probabilities(gate, logits)[:, 0].float().cpu().numpy())
    response = np.concatenate(result)
    assert response.shape == (len(features), 128, 128) and np.isfinite(response).all()
    return response


def save_model(path, net, mu, sd, names, cutmix=False):
    torch.save({"state_dict": {key: value.cpu() for key, value in net.state_dict().items()},
                "mu": mu.cpu(), "sd": sd.cpu(), "fit_names": list(names),
                "channels": int(mu.shape[1]), "epochs": EPOCHS, "seed": SEED,
                "threshold": THRESHOLD, "cutmix": bool(cutmix)}, path)


def save_response(path, names, response):
    np.savez_compressed(path, image_names=np.asarray(names, dtype=str), response=response.astype(np.float32),
                        threshold=np.asarray(THRESHOLD, dtype=np.float64), dev_in_sample=np.asarray(True))


def score_groups(names, response, masks, metadata):
    truth = masks[:, 0].numpy()
    scores = np.asarray([miou_image(p >= THRESHOLD, target) for p, target in zip(response, truth)])
    full = np.asarray([bool(metadata[name]["full_watermark"]) for name in names])
    report = {"large": float(scores[full].mean()), "small": float(scores[~full].mean()),
              "both": float(scores.mean()), "n": len(names), "n_large": int(full.sum()),
              "n_small": int((~full).sum())}
    return report, scores


def dev_stage(args, diagnostics, split, metadata, device):
    names, features = load_features(args.features, "dev", split["train"])
    assert len(names) == 136 and all(metadata[name]["method"] == "stegastamp" for name in names)
    masks = load_masks(args.data, names)
    full = torch.tensor([float(metadata[name]["full_watermark"]) for name in names])[:, None, None, None]
    folds = np.empty(len(names), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    for area_type in (False, True):
        ids = np.asarray([i for i, name in enumerate(names) if bool(metadata[name]["full_watermark"]) == area_type])
        rng.shuffle(ids)
        for fold, group in enumerate(np.array_split(ids, 5)):
            folds[group] = fold
    response = np.empty((len(names), 128, 128), dtype=np.float32)
    fold_reports = []
    for fold in range(5):
        fit_ids, val_ids = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        net, mu, sd, history = fit(features, masks, full, fit_ids, device, f"fold {fold}", cutmix=args.cutmix)
        response[val_ids] = predict(net, features[val_ids], mu, sd, device)
        save_model(diagnostics / f"fold_{fold}.pt", net, mu, sd, [names[i] for i in fit_ids], cutmix=args.cutmix)
        (diagnostics / f"fold_{fold}_history.json").write_text(json.dumps(history, indent=2))
        row, _ = score_groups([names[i] for i in val_ids], response[val_ids], masks[val_ids], metadata)
        fold_reports.append({"fold": fold, "fit_images": len(fit_ids), **row})
        print("FOLD OOF", fold, json.dumps(row), flush=True)
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
    scores, per_image = score_groups(names, response, masks, metadata)
    save_response(args.out / "dev.npz", names, response)
    np.savez_compressed(diagnostics / "dev_oof.npz", image_names=np.asarray(names), fold=folds, miou=per_image)
    report = {"method": "StegaStamp fixed cues + teammate Region polarity/rare-mask network",
              "features": int(features.shape[1]), "epochs": EPOCHS, "seed": SEED, "batch_size": BATCH,
              "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.001},
              "schedule": {"name": "cosine", "eta_min": 0.0001}, "threshold": THRESHOLD,
              "cutmix": {"enabled": bool(args.cutmix), "probability": 0.5,
                         "width_height_feature_cells": [16, 48],
                         "donor": "original current-batch sample with the same training full_watermark label",
                         "label_coordinates": "feature rectangle multiplied by 2; before H/V flips"},
              "normalization": "each fold's fit_ids[::3], spatial ::4; std floor .005; clamp [-7,7]",
              "loss": ".45*gate BCE + selected rare-mask BCE + .95*(1-Dice), grad clip 2",
              "response": "(1-sigmoid(g))*sigmoid(l0) + sigmoid(g)*(1-sigmoid(l1))",
              "dev_response_role": "true 5-fold OOF; no fold validation images fit normalization or weights",
              "dev_in_sample": True, "dev_in_sample_note": "conservative flag; dev responses themselves are OOF",
              "dev_oof": scores, "folds": fold_reports,
              "label_use": "full_watermark only supervises gate and rare-mask loss; not a model/inference input"}
    (diagnostics / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("DEV OOF large/small/both:", json.dumps(scores), flush=True)
    net, mu, sd, history = fit(features, masks, full, np.arange(len(names)), device, "full dev", cutmix=args.cutmix)
    save_model(args.out / "final_model.pt", net, mu, sd, names, cutmix=args.cutmix)
    (diagnostics / "final_history.json").write_text(json.dumps(history, indent=2))
    report["final_model_role"] = "fixed 80-epoch fit on all 136 dev images, only for later holdout/test inference"
    (diagnostics / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("Final model ready; holdout/test not opened.", flush=True)


def final_stage(args, diagnostics, split, metadata, device):
    checkpoint = torch.load(args.out / "final_model.pt", map_location="cpu", weights_only=True)
    assert checkpoint["fit_names"] == split["train"] and checkpoint["threshold"] == THRESHOLD
    net = Region(channels=int(checkpoint["channels"])).to(device)
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    mu, sd = checkpoint["mu"].to(device), checkpoint["sd"].to(device)
    report_path = diagnostics / "report.json"
    report = json.loads(report_path.read_text())
    names, features = load_features(args.features, "holdout", split["val"])
    assert len(names) == 34 and features.shape[1] == checkpoint["channels"]
    response = predict(net, features, mu, sd, device)
    save_response(args.out / "holdout.npz", names, response)
    report["holdout"], scores = score_groups(names, response, load_masks(args.data, names), metadata)
    (diagnostics / "holdout_rows.json").write_text(json.dumps([
        {"image_name": name, "full_watermark": bool(metadata[name]["full_watermark"]), "miou": float(score)}
        for name, score in zip(names, scores)], indent=2))
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print("HOLDOUT large/small/both:", json.dumps(report["holdout"]), flush=True)
    names, features = load_features(args.features, "test")
    assert len(names) == 340 and features.shape[1] == checkpoint["channels"]
    assert set(names).isdisjoint(split["train"] + split["val"])
    response = predict(net, features, mu, sd, device)
    save_response(args.out / "test.npz", names, response)
    report["test"] = {"n": len(names), "inference_only": True}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print("D export complete: dev136 / holdout34 / test340", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--features", type=Path, default=Path("output/stega_cues"))
    p.add_argument("--out", type=Path, default=Path("output/fusion/codex_stegacues_stegastamp"))
    p.add_argument("--stage", choices=("dev", "final"), default="dev")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cutmix", action="store_true", help="Fixed same-polarity current-training-batch CutMix before H/V flips")
    args = p.parse_args()
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics = args.out / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    split = json.loads((args.features / "split.json").read_text())
    assert len(split["train"]) == 136 and len(split["val"]) == 34 and set(split["train"]).isdisjoint(split["val"])
    (diagnostics / "split.json").write_text(json.dumps(split, indent=2))
    metadata = {row["image_name"]: row for row in json.loads((args.data / "train/metadata.json").read_text())}
    device = torch.device(args.device)
    if args.stage == "dev":
        dev_stage(args, diagnostics, split, metadata, device)
    else:
        final_stage(args, diagnostics, split, metadata, device)


if __name__ == "__main__":
    main()
