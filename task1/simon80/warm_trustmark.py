"""Recover each of Simon's four clean TrustMark V3 checkpoints exactly. Fit a residual boundary head on its original fold's fit names only. Compare on the original omitted names, then freeze the rule before reporting team holdout. Never edits team checkpoints."""

import argparse, hashlib, io, json, sys, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from refine_oof import team_hold, metric, fit_mix, ARCHIVE_HASH, SIMON_COMMIT
from adapter import Adapter, predict

sys.path.insert(0, str(Path(__file__).parent / "upstream"))
from extnet_v3_attr import ExtAttrSegNet

TEAM = Path("/p/scratch/training2666/qin7/task1/runs")


def images(root, split, idx, rr):
    with zipfile.ZipFile(root / (split + ".zip")) as z:
        x = np.stack(
            [
                np.asarray(
                    Image.open(
                        io.BytesIO(z.read(f"{split}/images/{rr[i]['image_name']}.png"))
                    ).convert("RGB")
                )
                for i in idx
            ]
        )
        bits = np.asarray([[int(c) for c in rr[i]["message"]] for i in idx], np.float32)
    return x, bits


def checkpoint(f):
    folder = TEAM / f"v3_tm_s3_f{f}"
    c = torch.load(folder / "best.pt", map_location="cpu", weights_only=True, mmap=True)
    a = c["args"]
    assert (
        a["method"] == "trustmark"
        and a["fold"] == f
        and a["n_folds"] == 4
        and "synth3" in str(a["pool"])
    )
    net = ExtAttrSegNet(
        "trustmark",
        SimpleNamespace(net=resnet50(num_classes=100)),
        dec=a["dec"],
        enc_res=a["enc_res"],
        attr_res=a["attr_res"],
    )
    net.load_state_dict(c["model"], strict=True)
    return net.cuda().eval(), folder


def extract(net, x, b):
    acts = []
    zs = []
    holder = []
    handle = net.head[-1].register_forward_pre_hook(
        lambda _, args: holder.append(args[0].detach())
    )
    try:
        for j in range(0, len(x), 4):
            u = (
                torch.from_numpy(x[j : j + 4]).cuda().permute(0, 3, 1, 2).float()
                / 127.5
                - 1
            )
            bits = torch.from_numpy(b[j : j + 4]).cuda()
            holder.clear()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z, g = net(u, bits)
            assert len(holder) == 1
            h = holder[0].float()
            noise = (
                u - F.avg_pool2d(F.pad(u, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
            ).abs()
            v = torch.cat(
                (
                    F.avg_pool2d(h[:, ::2], 2),
                    F.avg_pool2d(noise * 35, 8).clamp(0, 8),
                    g.float()[:, None, None, None].expand(-1, 1, 64, 64),
                ),
                1,
            )
            acts.append(v.half().cpu())
            zs.append(z.float().cpu()[:, None])
    finally:
        handle.remove()
    return torch.cat(acts), torch.cat(zs)


def train(x, z, y, indices, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu = x[indices, :, ::4, ::4].float().mean((0, 2, 3))[None, :, None, None].cuda()
    sd = (
        x[indices, :, ::4, ::4]
        .float()
        .std((0, 2, 3))[None, :, None, None]
        .clamp_min(0.02)
        .cuda()
    )
    net = Adapter(x.shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.00055, weight_decay=0.002)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 36, eta_min=0.00004)
    for _ in range(36):
        for grp in np.array_split(rng.permutation(indices), 13):
            ii = grp.tolist()
            a = ((x[ii].cuda().float() - mu) / sd).clamp(-7, 7)
            b = z[ii].cuda()
            t = y[ii].cuda()
            for axis in (-1, -2):
                if rng.random() < 0.5:
                    a = a.flip((axis,))
                    b = b.flip((axis,))
                    t = t.flip((axis,))
            l = net(a, b)
            p = l.sigmoid()
            edge = F.max_pool2d(t, 5, 1, 2) + F.max_pool2d(-t, 5, 1, 2)
            bce = F.binary_cross_entropy_with_logits(l, t, reduction="none")
            loss = (
                0.55 * ((1 + 2 * edge) * bce).sum() / (1 + 2 * edge).sum()
                + 0.03 * (l - b).abs().mean()
            )
            for u, v in ((p, t), (1 - p, 1 - t)):
                loss += (
                    0.5
                    * (
                        1
                        - ((u * v).sum((2, 3)) + 1) / ((u + v - u * v).sum((2, 3)) + 1)
                    ).mean()
                )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2)
            opt.step()
        sch.step()
    return net.eval(), mu, sd


def run(root, out):
    print("WARM START GPU", torch.cuda.get_device_name(), SIMON_COMMIT, flush=True)
    torch.set_num_threads(2)
    out.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(root / "train.zip") as z:
        rr = json.loads(z.read("train/metadata.json"))
        with np.load(io.BytesIO(z.read("train/masks_128.npz"))) as f:
            truth = dict(zip(f["names"], f["masks"].astype(bool)))
    with zipfile.ZipFile(root / "test.zip") as z:
        tt = json.loads(z.read("test/metadata.json"))
    H = team_hold(rr, truth)
    idx = [
        i
        for i, r in enumerate(rr)
        if r["method"] == "trustmark" and r["image_name"] not in H
    ]
    dn = [rr[i]["image_name"] for i in idx]
    assert len(dn) == 136
    Y = np.stack([truth[n] for n in dn])
    x, b = images(root, "train", idx, rr)
    y = torch.from_numpy(Y[:, None]).float()
    oof = np.zeros_like(Y, np.float32)
    adjust = np.zeros_like(oof)
    seen = np.zeros(len(dn), int)
    models = []
    folds = []
    for f in range(4):
        net, src = checkpoint(f)
        vn = set(json.loads((src / "val_names.json").read_text()))
        val = [i for i, n in enumerate(dn) if n in vn]
        fit = [i for i, n in enumerate(dn) if n not in vn]
        assert len(vn) == len(val) == 34 and len(fit) == 102
        feat, bl = extract(net, x, b)
        del net
        torch.cuda.empty_cache()
        # Reject wrong weights / normalization / image order before training. No holdout parity check.
        with np.load(src / "pred_train.npz", allow_pickle=False) as old:
            mp = {str(n): i for i, n in enumerate(old["names"])}
            ref = old["probs"][[mp[dn[i]] for i in val[:4]]].astype(np.float32)
        diff = np.abs(ref - bl[val[:4], 0].sigmoid().numpy())
        assert float(diff.mean()) < 0.006, (
            "checkpoint parity failed",
            f,
            float(diff.mean()),
        )
        print("PARITY", f, float(diff.mean()), flush=True)
        model = train(feat, bl, y, fit, 7109 + f)
        oof[val] = bl[val, 0].numpy()
        adjust[val] = predict(model, feat[val], bl[val])
        seen[val] += 1
        models.append(model)
        folds.append(val)
        print("TRAINED FOLD", f, "fit/val", len(fit), len(val), flush=True)
    assert (seen == 1).all()
    cv = []
    for val in folds:
        fit = np.setdiff1d(np.arange(136), val)
        _, b0 = fit_mix(oof, None, Y, fit)
        _, b1 = fit_mix(adjust, None, Y, fit)
        cv.append(
            [
                float(metric(oof[val] >= b0, Y[val]).mean()),
                float(metric(adjust[val] >= b1, Y[val]).mean()),
            ]
        )
    v = np.array(cv)
    use = bool((v[:, 1] - v[:, 0]).mean() > 0.003 and np.sum(v[:, 1] > v[:, 0]) >= 3)
    ids = np.arange(136)
    _, t0 = fit_mix(oof, None, Y, ids)
    _, t1 = fit_mix(adjust, None, Y, ids)
    print("FROZEN", use, cv, t0, t1, flush=True)
    stats = {
        "method": "trustmark",
        "simon_commit": SIMON_COMMIT,
        "use": use,
        "meta_folds": cv,
        "cv": list(v.mean(axis=0)),
        "thresholds": [t0, t1],
        "public_score": None,
    }
    for split, meta, ix in [
        (
            "holdout",
            rr,
            [
                i
                for i, r in enumerate(rr)
                if r["method"] == "trustmark" and r["image_name"] in H
            ],
        ),
        ("test", tt, [i for i, r in enumerate(tt) if r["method"] == "trustmark"]),
    ]:
        nnames = [meta[i]["image_name"] for i in ix]
        xx, bb = images(root, "train" if split == "holdout" else "test", ix, meta)
        raw = 0.0
        adj = 0.0
        for f, model in enumerate(models):
            net, _ = checkpoint(f)
            fx, bl = extract(net, xx, bb)
            raw += bl[:, 0].numpy() / 4
            if use:
                adj += predict(model, fx, bl) / 4
            del net
            torch.cuda.empty_cache()
        pred = adj >= t1 if use else raw >= t0
        if split == "holdout":
            yy = np.stack([truth[n] for n in nnames])
            stats["hold_base"] = float(metric(raw >= t0, yy).mean())
            stats["hold"] = float(metric(pred, yy).mean())
            print("HOLD", stats, flush=True)
        np.savez_compressed(
            out / ("trustmark-" + split + ".npz"),
            image_names=np.array(nnames),
            masks=pred,
        )
        if split == "test":
            assert pred.shape == (340, 128, 128)
            archive = TEAM / "submissions/sub_v1.pt"
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_HASH
            ref = torch.load(archive, weights_only=True, map_location="cpu")
            tn = [r["image_name"] for r in tt]
            bn = ref["image_names"]
            assert len(bn) == len(set(bn)) == 1020 and set(bn) == set(tn)
            ixb = {n: i for i, n in enumerate(bn)}
            base = ref["masks"][[ixb[n] for n in tn]].bool()
            merged = base.clone()
            merged[ix] = torch.from_numpy(pred)
            keep = [i for i, r in enumerate(tt) if r["method"] != "trustmark"]
            dst = out / "submission.pt"
            torch.save({"image_names": tn, "masks": merged}, dst)
            chk = torch.load(dst, weights_only=True)
            assert (
                chk["image_names"] == tn
                and torch.equal(chk["masks"][keep], base[keep])
                and torch.equal(chk["masks"][ix], torch.from_numpy(pred))
                and dst.stat().st_size < 20_000_000
            )
            stats["test_n"] = 1020
            stats["other_masks_unchanged"] = len(keep)
            stats["sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (out / "report.json").write_text(json.dumps(stats, indent=2))
    print("DONE", stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    run(a.root, a.out)
