"""Average aligned conditional predictions across three cue sets and four views.
Choose one checkpoint per cue set and method by inner dev only; tune decisions there.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_grid import METHODS, read, split_ids
from train_cue_net import Region


def blend_region(gate, region):
    g = gate.reshape(-1, 1, 1)
    return g[:, 0, 0] >= 0.5, (1 - g) * region[:, 0] + g * (1 - region[:, 1])


def multiview(net, x, method):
    gs, ps = [], []
    for axes in ((), (-1,), (-2,), (-2, -1)):
        transformed = x.flip(axes) if axes else x
        g, p = net(transformed, method)
        gs.append(g.reshape(-1).sigmoid())
        ps.append((p.flip(axes) if axes else p).sigmoid())
    return torch.stack(gs).mean(0), torch.stack(ps).mean(0)


def feature(root, split, gb, rows, i, mode):
    r = rows[i]
    m = r["method"]
    name = r["image_name"] + ".npy"

    def load(folder):
        from time import monotonic, sleep

        path = root / folder / name
        deadline = monotonic() + 900
        while not path.is_file():
            if monotonic() >= deadline:
                raise TimeoutError(f"Missing feature after 15 minutes: {path}")
            sleep(3)
        return np.load(path, allow_pickle=False)

    cam = load(f"cam-v1-{split}-{m}")
    padded = np.zeros((24, 64, 64), dtype=np.float16)
    padded[: len(cam)] = cam
    parts = [gb[i, ::3], padded, load(f"semantic-v1-{split}")]
    if mode >= 1:
        parts.append(load(f"rewrite-v1-{split}-{m}"))
    if mode == 2:
        parts.append(load(f"tile-v1-{split}-{m}"))
    return np.concatenate(parts)


def model_score(p, y):
    assert p.dtype == y.dtype == torch.bool and p.shape == y.shape
    vals = []
    for a, b in ((p, y), (~p, ~y)):
        union = (a | b).sum((-2, -1))
        vals.append(
            torch.where(union > 0, (a & b).sum((-2, -1)) / union.clamp_min(1), 1.0)
        )
    return float(torch.stack(vals).mean())


def mask_for(g, q, rule, t):
    majority, soft = blend_region(g, q)
    if rule == "soft":
        return soft > t
    return torch.where(majority[:, None, None], q[:, 1] <= t, q[:, 0] > t)


@torch.inference_mode()
def run(root):
    torch.set_num_threads(2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    gb, rows, y = read(root, "train")
    testgb, testrows, _ = read(root, "test")
    train, dev, hold = split_ids(rows)
    assert y is not None
    out = root / "ensemble-views-v1"
    out.mkdir(exist_ok=True)
    cats = {i: [] for i in range(3)}
    for f in sorted(root.glob("neural-message-v1*")):
        if all(
            (f / (m + ".json")).exists() and (f / (m + ".pth")).exists()
            for m in METHODS
        ):
            mode = 2 if f.name.endswith("r1t1") else 1 if f.name.endswith("r1t0") else 0
            cats[mode].append(f)
    assert all(cats.values()), "need all three completed cue families"
    for m_id, m in enumerate(METHODS):
        model_cfg = []
        for mode, roots in cats.items():
            folder = max(
                roots, key=lambda f: json.loads((f / (m + ".json")).read_text())["dev"]
            )
            stats = torch.from_numpy(
                np.stack(
                    [
                        feature(root, "train", gb, rows, i, mode)[:, ::4, ::4]
                        for i in train[::3]
                    ]
                )
            ).float()
            mu = stats.mean((0, 2, 3))[None, :, None, None].to(device)
            sd = stats.std((0, 2, 3))[None, :, None, None].clamp_min(0.005).to(device)
            net = Region(stats.shape[1])
            net.load_state_dict(
                torch.load(folder / (m + ".pth"), map_location="cpu", weights_only=True)
            )
            net.eval().to(device)
            model_cfg.append((mode, net, mu, sd, folder.name))

        def predict(split, block, meta, ids):
            gs, qs = [], []
            for st in range(0, len(ids), 8):
                ix = ids[st : st + 8]
                gg, qq = [], []
                k = torch.full((len(ix),), m_id, dtype=torch.long, device=device)
                for mode, net, mu, sd, _ in model_cfg:
                    x = (
                        torch.from_numpy(
                            np.stack(
                                [feature(root, split, block, meta, i, mode) for i in ix]
                            )
                        )
                        .float()
                        .to(device)
                    )
                    g, q = multiview(net, ((x - mu) / sd).clamp(-7, 7), k)
                    gg.append(g)
                    qq.append(q)
                gs.append(torch.stack(gg).mean(0).cpu())
                qs.append(torch.stack(qq).mean(0).cpu())
            return torch.cat(gs), torch.cat(qs)

        ids = [i for i in dev if rows[i]["method"] == m]
        g, q = predict("train", gb, rows, ids)
        best, rule, th = max(
            (model_score(mask_for(g, q, r, t), y[ids, 0] > 0.5), r, t)
            for r in ("soft", "hard")
            for t in (0.35, 0.5, 0.65)
        )
        hi = [i for i in hold if rows[i]["method"] == m]
        g, q = predict("train", gb, rows, hi)
        pred = mask_for(g, q, rule, th)
        hs = model_score(pred, y[hi, 0] > 0.5)
        log = {
            "method": m,
            "dev": best,
            "hold": hs,
            "n": len(hi),
            "rule": rule,
            "threshold": th,
            "families": [c[-1] for c in model_cfg],
        }
        print("HOLD", json.dumps(log), flush=True)
        np.savez_compressed(out / f"{m}-hold.npz", ids=hi, masks=pred.numpy())
        ti = [i for i in range(len(testrows)) if testrows[i]["method"] == m]
        g, q = predict("test", testgb, testrows, ti)
        masks = mask_for(g, q, rule, th).numpy()
        assert masks.dtype == bool and masks.shape == (340, 128, 128)
        tmp = out / f"{m}.partial.npz"
        np.savez_compressed(tmp, ids=ti, masks=masks)
        tmp.replace(out / f"{m}-test.npz")
        (out / f"{m}.json").write_text(json.dumps(log, indent=2))
        print("TEST", m, len(ti), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    run(p.parse_args().root)
