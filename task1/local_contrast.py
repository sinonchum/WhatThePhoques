"""Per-image evidence relative to the typical scene and to spatial neighbors."""

import numpy as np
from scipy.ndimage import uniform_filter


def contrast(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("expected finite C,H,W feature maps")
    lo, med, hi = np.quantile(x[:, ::2, ::2], (0.2, 0.5, 0.8), axis=(1, 2)).astype(
        np.float32
    )[:, :, None, None]
    s = np.maximum(hi - lo, 0.02)
    q = (x - med) / s
    local = []
    for w in (9, 27):
        k = (1, w, w)
        center = uniform_filter(q, k, mode="reflect")
        var = np.maximum(
            uniform_filter(q * q, k, mode="reflect") - center * center, 0.0
        )
        local.append((q - center) / np.sqrt(var + 0.04))
    return np.clip(np.concatenate([q] + local), -6.0, 6.0).astype(np.float16)
