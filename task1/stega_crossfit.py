"""Four-fold, from-scratch StegaStamp ensemble with auxiliary training on the other two writers.
OOF predictions tune the threshold; the untouched inner dev compares complete families and the
fixed hold is reported, not consulted by the optimizer. No warm start from a split-overlapping net.
"""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_grid import METHODS, read, split_ids
from train_cue_net import Region
from ensemble_views import feature, model_score


def run(root, epochs, seed, width, arm):
    torch.set_num_threads(2)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    raw, rows, y = read(root, "train")
    tr, d, h = split_ids(rows)
    is_s = lambda i: rows[i]["method"] == "stegastamp"
    td = [i for i in tr if is_s(i)]
    di = [i for i in d if is_s(i)]
    hi = [i for i in h if is_s(i)]
    ids = tr + di + hi
    order = {v: k for k, v in enumerate(ids)}
    hsh = lambda i: hashlib.sha256(
        ("stega-cf-v1:" + rows[i]["image_name"]).encode()
    ).digest()
    folds = [[] for _ in range(4)]
    for p in (False, True):
        for j, i in enumerate(
            sorted((i for i in td if bool(rows[i]["full_watermark"]) == p), key=hsh)
        ):
            folds[j % 4].append(i)
    assert all(len(v) == 30 for v in folds) and len(set(sum(folds, []))) == 120
    out = root / f"stega-crossfit-v1-{arm}-{seed}-{width}-{epochs}"
    out.mkdir(exist_ok=True)
    print("START", out.name, device, flush=True)

    def load(split, raw, rr, ii):
        a = []
        for i in ii:
            r = rr[i]
            x = feature(root, split, raw, rr, i, 2)
            ind = [
                k
                for k in range(36)
                if arm == "both"
                or (arm == "magnitude" and k % 12 in (1, 3, 5, 7, 9))
                or (arm == "signed" and k % 12 in (0, 2, 4, 6, 8, 10, 11))
            ]
            if ind:
                cam = (
                    np.load(
                        root
                        / f"stega-native-cam-v1-{split}"
                        / (r["image_name"] + ".npy")
                    )[ind]
                    if r["method"] == "stegastamp"
                    else np.zeros((len(ind), 64, 64), np.float16)
                )
                x = np.concatenate([x, cam])
            a.append(x)
        return torch.from_numpy(np.stack(a)), torch.tensor(
            [METHODS.index(rr[i]["method"]) for i in ii]
        )

    x, m = load("train", raw, rows, ids)
    gt = y[ids]
    full = torch.tensor([float(rows[i]["full_watermark"]) for i in ids])[
        :, None, None, None
    ]
    rawtest, rr, _ = read(root, "test")
    te = [i for i, r in enumerate(rr) if r["method"] == "stegastamp"]
    xt, mt = load("test", rawtest, rr, te)
    oof = []
    ol = []
    pd = []
    ph = []
    pt = []
    for f, va in enumerate(folds):
        cache = out / f"fold-{f}-prob.npz"
        if not cache.exists():
            torch.manual_seed(seed + f)
            rng = np.random.default_rng(seed + f)
            fit = [order[i] for i in tr if i not in set(va)]
            sample = x[fit, :, ::4, ::4].float()
            mu = sample.mean((0, 2, 3))[None, :, None, None].to(device)
            sd = sample.std((0, 2, 3))[None, :, None, None].clamp_min(0.005).to(device)
            net = Region(x.shape[1], width).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.001)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, epochs, eta_min=0.0001
            )
            norm = lambda v: ((v.to(device).float() - mu) / sd).clamp(-7, 7)

            @torch.inference_mode()
            def probs(a, t, ix):
                net.eval()
                pred = []
                for j in range(0, len(ix), 8):
                    ii = ix[j : j + 8]
                    xx = norm(a[ii])
                    tt = t[ii].to(device)
                    pp = []
                    for flip in [False, True]:
                        v = xx.flip((-1,)) if flip else xx
                        g, l = net(v, tt)
                        g = g.sigmoid()
                        p = (1 - g) * l[:, :1].sigmoid() + g * (1 - l[:, 1:].sigmoid())
                        pp.append(p.flip((-1,)) if flip else p)
                    pred.append(torch.stack(pp).mean(0)[:, 0].cpu())
                return torch.cat(pred).numpy()

            for ep in range(1, epochs + 1):
                net.train()
                seq = rng.permutation(fit)
                for j in range(0, len(seq), 12):
                    ii = seq[j : j + 12].tolist()
                    xx = norm(x[ii])
                    yy = gt[ii].to(device)
                    ff = full[ii].to(device)
                    mm = m[ii].to(device)
                    for ax in [-1, -2]:
                        if rng.random() < 0.5:
                            xx = xx.flip((ax,))
                            yy = yy.flip((ax,))
                    g, l = net(xx, mm)
                    log = (1 - ff) * l[:, :1] + ff * l[:, 1:]
                    rare = (1 - ff) * yy + ff * (1 - yy)
                    pr = log.sigmoid()
                    loss = (
                        0.45 * F.binary_cross_entropy_with_logits(g, ff)
                        + F.binary_cross_entropy_with_logits(log, rare)
                        + 0.95
                        * (
                            1
                            - (
                                (2 * (pr * rare).sum((2, 3)) + 1)
                                / ((pr + rare).sum((2, 3)) + 1)
                            ).mean()
                        )
                    )
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                    opt.step()
                sched.step()
                if ep % 20 == 0:
                    print("FOLD", f, "epoch", ep, flush=True)
            ck = out / f"fold-{f}.pth"
            torch.save(
                {
                    "network": net.state_dict(),
                    "mean": mu.cpu(),
                    "sd": sd.cpu(),
                    "train_names": [rows[ids[i]]["image_name"] for i in fit],
                },
                ck,
            )
            tmp = cache.with_suffix(".partial.npz")
            np.savez_compressed(
                tmp,
                ids=va,
                oof=probs(x, m, [order[i] for i in va]).astype(np.float16),
                dev=probs(x, m, [order[i] for i in di]).astype(np.float16),
                hold=probs(x, m, [order[i] for i in hi]).astype(np.float16),
                test=probs(xt, mt, list(range(len(te)))).astype(np.float16),
            )
            tmp.replace(cache)
        with np.load(cache, allow_pickle=False) as z:
            assert set(z["ids"]) == set(va)
            oof.append(z["oof"])
            ol.extend(z["ids"].tolist())
            pd.append(z["dev"])
            ph.append(z["hold"])
            pt.append(z["test"])
    o = torch.from_numpy(np.concatenate(oof))
    assert len(set(ol)) == 120
    oofscore, t = max(
        (model_score(o > t, y[ol, 0] > 0.5), t) for t in (0.35, 0.45, 0.5, 0.55, 0.65)
    )
    to_mask = lambda v: np.mean(np.asarray(v, dtype=np.float32), axis=0) > t
    dev = model_score(torch.from_numpy(to_mask(pd)), y[di, 0] > 0.5)
    mh = to_mask(ph)
    mt = to_mask(pt)
    hold = model_score(torch.from_numpy(mh), y[hi, 0] > 0.5)
    assert mt.shape == (340, 128, 128)
    r = {
        "method": "stegastamp",
        "oof": oofscore,
        "oof_n": len(ol),
        "dev": dev,
        "hold": hold,
        "threshold": t,
        "n": len(hi),
        "seed": seed,
        "epochs": epochs,
        "arm": arm,
        "width": width,
    }
    np.savez_compressed(out / "stegastamp-hold.npz", ids=hi, masks=mh)
    np.savez_compressed(out / "stegastamp-test.npz", ids=te, masks=mt)
    (out / "stegastamp.json").write_text(json.dumps(r))
    print("RESULT", r, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--seed", type=int, default=4447)
    p.add_argument("--width", type=int, default=24)
    p.add_argument(
        "--arm", choices=["both", "signed", "magnitude", "base"], default="both"
    )
    a = p.parse_args()
    run(a.root, a.epochs, a.seed, a.width, a.arm)
