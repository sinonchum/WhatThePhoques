"""Load each watermark wrapper once and sanity-check it. Run on a LOGIN node first (TrustMark downloads its
weights on first use), then once inside a GPU job to confirm nothing tries to reach the internet.

    python task1/check_models.py --device cpu
    python task1/check_models.py --device cuda

For each method: load, decode one real train image (bit accuracy vs its message on the whole image), then
embed a fresh random message into that image and decode it back, and print the feature-map shapes."""
from __future__ import annotations

import argparse
import time
import traceback

import numpy as np
import torch

from wmloc.data import LocalisationDataset
from wmloc.utils import bit_accuracy, to_tensor
from wmloc.watermarks import load_watermark
from wmtask.common import DATA_DIR, METHODS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    a = ap.parse_args()
    print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}  device: {a.device}")
    if torch.cuda.is_available():
        print("gpus:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    failed = []
    for method in a.methods:
        print(f"\n===== {method} =====")
        try:
            t0 = time.time()
            wm = load_watermark(method, device=a.device)
            print(f"loaded in {time.time() - t0:.1f}s  nbits={wm.nbits}  resolution={getattr(wm, 'resolution', '?')}")
            if method == "stegastamp":
                print("onnxruntime providers in use:", wm.sess.get_providers())

            ds = LocalisationDataset(a.data, "train", methods=[method])
            for want_full in (True, False):
                it = next(x for x in ds if bool(x["full_watermark"]) == want_full)
                x = to_tensor(it["image"])
                acc = bit_accuracy(wm.decode(x)[0].cpu().numpy(), it["message"])
                print(f"real image {it['image_name']} full_watermark={want_full} area={it['mask128'].mean():.3f}: "
                      f"whole-image bit accuracy {acc:.3f}")

            rng = np.random.RandomState(0)
            bits = rng.randint(0, 2, wm.nbits)
            t0 = time.time()
            marked = wm.embed(it["image"], bits)
            diff = marked.astype(np.float32) - it["image"].astype(np.float32)
            acc = bit_accuracy(wm.decode(to_tensor(marked))[0].cpu().numpy(), bits)
            print(f"embed+decode round trip: bit accuracy {acc:.3f}  (embed {time.time() - t0:.2f}s, "
                  f"residual mean |d| {np.abs(diff).mean():.2f}, max |d| {np.abs(diff).max():.0f} of 255)")

            feats = wm.features(to_tensor(marked))
            print("features:", {k: tuple(v.shape) for k, v in feats.items()})
        except Exception:
            failed.append(method)
            traceback.print_exc()

    print("\nRESULT:", "all methods OK" if not failed else f"FAILED: {failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
