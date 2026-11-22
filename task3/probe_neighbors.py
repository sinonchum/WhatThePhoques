"""Do mid-layer features vote for a different class than the memorized label?"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

STARTER = Path(__file__).resolve().parent / "starter"
sys.path.insert(0, str(STARTER))
import task_template as tt  # noqa: E402

OUT = Path(__file__).resolve().parent / "artifacts"
BLOCKS = (4, 6, 7, 8, 9, 11)


@torch.no_grad()
def collect(model, images):
    feats = {b: [] for b in BLOCKS}
    bs = 64
    for i in range(0, len(images), bs):
        x = tt.preprocess(images[i:i + bs])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            h = model.patch_embed(x)
            h = model._pos_embed(h)
            h = model.patch_drop(h)
            h = model.norm_pre(h)
            for bi, blk in enumerate(model.blocks):
                h = blk(h)
                if bi in feats:
                    feats[bi].append(h[:, 0].float().cpu())
        if i % 1024 == 0:
            print(f"forward {i}", flush=True)
    return {b: torch.cat(v) for b, v in feats.items()}


def knn_vote(feat, labels, k):
    f = F.normalize(feat.float(), dim=-1)
    # cosine similarity, 3500^2 is fine
    sim = f @ f.T
    sim.fill_diagonal_(-2)
    nn = sim.topk(k, dim=1).indices  # N, k
    neigh_labels = labels[nn]  # N, k
    # majority
    mode = []
    frac = []
    for n in range(len(labels)):
        vals, cnts = torch.unique(neigh_labels[n], return_counts=True)
        j = cnts.argmax()
        mode.append(vals[j])
        frac.append(cnts[j].float() / k)
    return torch.stack(mode), torch.stack(frac), nn


def main():
    model = tt.load_model(STARTER / "model.pt")
    images = tt.load_forget_images(STARTER / "forget_images.npz")
    labels = torch.load(OUT / "lens.pt", weights_only=True)["final"].long()
    print("labels", labels.shape, "nclass", labels.unique().numel())
    feats = collect(model, images)

    saved = {"labels": labels}
    for b, feat in feats.items():
        f = F.normalize(feat.float(), dim=-1)
        # same-class vs other-class mean cosine, sampled
        same, other = [], []
        # per-class centroids
        cents = []
        for c in labels.unique().tolist():
            m = labels == c
            if m.sum() < 2:
                continue
            fc = f[m]
            cents.append((c, F.normalize(fc.mean(0), dim=0)))
            # sample pairs
            idx = torch.where(m)[0]
            if len(idx) > 40:
                idx = idx[torch.randperm(len(idx))[:40]]
            sim = fc[: len(idx)] @ fc[: len(idx)].T if False else f[idx] @ f[idx].T
            same.append(sim[~torch.eye(len(idx), dtype=torch.bool)].mean())
        # other: random pairs
        perm = torch.randperm(len(labels))
        # pairs that differ in label
        a = torch.arange(2000)
        bidx = torch.randint(0, len(labels), (2000,))
        diff = labels[a] != labels[bidx]
        other_sim = (f[a][diff] * f[bidx][diff]).sum(-1).mean()
        same_sim = torch.stack(same).mean()
        print(f"\nblock {b:02d}  same-cos {same_sim:.3f}  other-cos {other_sim:.3f}  gap {same_sim-other_sim:.3f}")

        for k in (10, 20):
            mode, frac, nn = knn_vote(feat, labels, k)
            disagree = mode != labels
            print(
                f"  k={k:2d}  vote!=label {disagree.float().mean():.3f}  "
                f"vote_frac {frac.mean():.3f}  "
                f"frac|disagree {frac[disagree].mean():.3f}  "
                f"sharp>0.6 {(frac>0.6).float().mean():.3f}"
            )
            if k == 15 or k == 10:
                saved[f"mode_b{b}_k{k}"] = mode.to(torch.int16)
                saved[f"frac_b{b}_k{k}"] = frac.to(torch.float16)

        # centroid: nearest class centroid (leave-one-out centroid for own class)
        # build all centroids
        uniq = labels.unique()
        cent = []
        for c in uniq.tolist():
            cent.append(F.normalize(f[labels == c].mean(0), dim=0))
        cent = torch.stack(cent)  # C, D
        sim_c = f @ cent.T  # N, C
        nearest = uniq[sim_c.argmax(1)]
        # leave-one-out for own class: adjust
        # (sum - self) / (n-1) then renormalize is more correct; do it for ranking own class only
        own_index = {int(c): i for i, c in enumerate(uniq.tolist())}
        own_i = torch.tensor([own_index[int(c)] for c in labels.tolist()])
        # recompute own similarity with LOO centroid
        loo_sim = sim_c.clone()
        for c, i in own_index.items():
            m = labels == c
            n = int(m.sum())
            if n < 3:
                continue
            s = f[m].sum(0)
            # loo centroid for each member
            loo = F.normalize(s[None] - f[m], dim=-1)
            loo_sim[m, i] = (f[m] * loo).sum(-1)
        nearest_loo = uniq[loo_sim.argmax(1)]
        print(f"  nearest centroid != label { (nearest != labels).float().mean():.3f}")
        print(f"  LOO centroid != label     { (nearest_loo != labels).float().mean():.3f}")
        saved[f"cent_b{b}"] = nearest_loo.to(torch.int16)

    # agreement between block6 knn k=10 and block8 lens pred
    lens_pred = torch.load(OUT / "lens.pt", weights_only=True)["pred"].long()
    mode6 = saved["mode_b6_k10"].long()
    print("\nagreement of estimators with each other and with label")
    b8 = lens_pred[:, 8]
    b9 = lens_pred[:, 9]
    print(f"knn6 == label {(mode6==labels).float().mean():.3f}")
    print(f"lens8 == label {(b8==labels).float().mean():.3f}")
    print(f"lens9 == label {(b9==labels).float().mean():.3f}")
    print(f"knn6 == lens8 {(mode6==b8).float().mean():.3f}")
    print(f"knn6 == lens9 {(mode6==b9).float().mean():.3f}")
    both_flip = (mode6 != labels) & (b8 != labels)
    print(f"both flip {both_flip.float().mean():.3f}  and agree {((mode6==b8)&both_flip).float().sum()/max(both_flip.sum(),1):.3f}")

    torch.save(saved, OUT / "neighbors.pt")
    print("saved neighbors.pt")


if __name__ == "__main__":
    main()
