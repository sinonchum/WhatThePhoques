"""Out-of-fold linear readouts on intermediate features.

A readout trained on 4/5 of the forget images and applied to the held-out 1/5 says what a
classifier that never saw that image predicts from features that carry little memorization.
Saves runs/xh_oof.pt: {block: out-of-fold logits [N,100]} (absent classes get -30).
"""

import torch
import torch.nn.functional as F

import xh_common as C


@torch.no_grad()
def features(model, tok, bs=250):
    feats = {b: [] for b in (7, 8, 9, 10)}
    for i in range(0, len(tok), bs):
        h = tok[i : i + bs].to(C.DEV).float()
        feats[7].append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu())
        for bi in (8, 9, 10):
            h = model.blocks[bi](h)
            feats[bi].append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu())
    return {b: torch.cat(v) for b, v in feats.items()}


def fit_predict(xtr, ytr, xte, n_cls, wd, steps=300):
    w = torch.zeros(xtr.shape[1], n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xtr @ w + b, ytr) + wd * (w**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return (xte @ w + b).detach()


def main():
    torch.manual_seed(0)
    model = C.load_orig()
    tok = torch.load(C.RUN / "tok" / "clean.pt")["tok"]
    y = torch.load(C.RUN / "xh_cache.pt")["y"]
    classes = y.unique()
    remap = torch.full((100,), -1, dtype=torch.long)
    remap[classes] = torch.arange(len(classes))
    yc = remap[y]
    feats = features(model, tok)
    folds = torch.randperm(len(y)) % 5
    out = {}
    for b, f in feats.items():
        f = (f - f.mean(0)) / (f.std(0) + 1e-6)
        for wd in (1e-3, 1e-2):
            oof = torch.zeros(len(y), len(classes))
            for k in range(5):
                te = folds == k
                oof[te] = fit_predict(f[~te], yc[~te], f[te], len(classes), wd)
            ce = F.cross_entropy(oof, yc, reduction="none")
            print(f"block {b} wd {wd}: oof acc {(oof.argmax(1) == yc).float().mean():.3f} CE {ce.mean():.2f}/{ce.std():.2f} "
                  f"T2 top {(oof / 2).softmax(-1).max(1).values.mean():.3f}", flush=True)
            full = torch.full((len(y), 100), -30.0)
            full[:, classes] = oof
            out[(b, wd)] = full
    torch.save(out, C.RUN / "xh_oof.pt")


if __name__ == "__main__":
    main()
