"""CPU-only reference, not a submission: python3 task1/residual_probe.py /path/to/train.zip
Dependencies: numpy, scipy, Pillow. Method-specific ridge regression on residual statistics.
"""
import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter


def gridmean(x):
    return x.reshape(32, 16, 32, 16, -1).mean(axis=(1, 3))


def features(img):
    x = img.astype(np.float32)
    x = np.stack((x.mean(-1), x[..., 0]-x[..., 1], x[..., 2]-x[..., 1]), -1)
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode='reflect')
    dx = p[1:-1, :-2]+p[1:-1, 2:]-2*x
    dy = p[:-2, 1:-1]+p[2:, 1:-1]-2*x
    diag = x-p[:-2, 1:-1]-p[1:-1, :-2]+p[:-2, :-2]
    fs = [np.log1p(gridmean(np.abs(v))) for v in (dx, dy, dx+dy, diag)]
    local = np.concatenate(fs, -1)
    sm = np.stack([uniform_filter(local[..., i], size=5, mode='reflect')
                   for i in range(local.shape[-1])], axis=-1)
    large = local.mean((0, 1), keepdims=True)
    return np.concatenate((local, sm, sm-large, np.broadcast_to(large, sm.shape)), -1).reshape(1024, -1)


def miou(p, g):
    out = []
    for c in (0, 1):
        a, b = p == c, g == c
        union = (a | b).sum()
        out.append((a & b).sum() / union if union else 1.)
    return np.mean(out)


def run(path):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read('train/metadata.json'))
        with np.load(io.BytesIO(z.read('train/masks_128.npz')), allow_pickle=False) as data:
            gt = dict(zip(data['names'].tolist(), data['masks']))
        strata = defaultdict(list)
        for m in meta:
            strata[m['method'], m['full_watermark']].append(m['image_name'])
        fingerprint = lambda n: hashlib.sha256(('paris-residual-v1:' + n).encode()).hexdigest()
        hold = {n for names in strata.values() for n in sorted(names, key=fingerprint)[:len(names)//5]}
        assert len(meta) == 510 and len(hold) == 102 and len(gt) == 510
        scores = []
        for method in sorted({m['method'] for m in meta}):
            train, labels, val = [], [], []
            for m in meta:
                if m['method'] != method:
                    continue
                n = m['image_name']
                a = features(np.asarray(Image.open(io.BytesIO(z.read(f'train/images/{n}.png'))).convert('RGB')))
                if n in hold:
                    val.append((a, gt[n]))
                else:
                    train.append(a)
                    labels.append(gt[n].reshape(32, 4, 32, 4).mean((1, 3)).ravel())
            X = np.concatenate(train).astype(np.float64)
            y = np.concatenate(labels)
            mu, sig = X.mean(0), np.maximum(X.std(0), .001)
            X = (X-mu)/sig
            # Fixed regularization and threshold. No sweeps on holdout.
            gram = np.einsum('ni,nj->ij', X, X)
            rhs = np.einsum('ni,n->i', X, y-y.mean())
            w = np.linalg.solve(gram + .001*len(y)*np.eye(X.shape[1]), rhs)
            assert np.isfinite(w).all()
            rows = []
            for a, g in val:
                v = (np.einsum('ni,i->n', (a-mu)/sig, w) + y.mean()).reshape(32, 32)
                assert np.isfinite(v).all()
                pred = (v > .5).repeat(4, 0).repeat(4, 1)
                rows.append([miou(pred, g), miou(np.zeros_like(g), g), miou(np.ones_like(g), g)])
            scores.extend(rows)
            print(method, len(rows), 'residual/zeros/ones', np.mean(rows, 0).round(4), flush=True)
        print('ALL', len(scores), np.mean(scores, 0).round(4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_zip', type=Path)
    run(parser.parse_args().train_zip)
