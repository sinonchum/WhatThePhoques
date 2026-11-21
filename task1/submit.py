"""Validate a submission file and, only with --yes-submit, upload it. Run on a LOGIN node (needs internet).

    python task1/submit.py runs/zeros.pt                 # validate only
    python task1/submit.py runs/zeros.pt --yes-submit    # validate, then upload

The API key is read from the environment variable HACKATHON_API_KEY and is never written or printed.
Uploading uses the official submit() from the starter kit's submission_template.py."""
from __future__ import annotations

import argparse
import os
import sys

import torch

from wmtask.common import check_submission, load_index

BASE_URL = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"
MAX_MB = 20.0   # task PDF says 20 MB, starter-kit README says 25 MB: stay under the stricter one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--data", default=None)
    ap.add_argument("--yes-submit", action="store_true", help="actually upload (only after the local scores were approved)")
    a = ap.parse_args()

    sub = torch.load(a.path, weights_only=True)      # same loader the server uses
    assert set(sub) == {"image_names", "masks"}, f"keys: {sorted(sub)}"
    names, masks = sub["image_names"], sub["masks"]
    expected = [r["image_name"] for r in load_index("test", a.data)]
    check_submission(names, masks, expected)
    size_mb = os.path.getsize(a.path) / 1e6
    assert size_mb <= MAX_MB, f"file is {size_mb:.2f} MB, limit {MAX_MB} MB"
    frac = masks.float().mean(dim=(1, 2))
    print(f"valid: {len(names)} masks, dtype {masks.dtype}, {size_mb:.2f} MB, "
          f"mean watermarked fraction {frac.mean():.3f}, all-0 masks {int((frac == 0).sum())}, all-1 masks {int((frac == 1).sum())}")

    if not a.yes_submit:
        print("not submitted (validation only; pass --yes-submit to upload)")
        return

    key = os.environ.get("HACKATHON_API_KEY", "")
    if not key:
        sys.exit("HACKATHON_API_KEY is not set in the environment - not submitting")
    import submission_template as st
    assert st.TASK_ID == TASK_ID, f"template TASK_ID is {st.TASK_ID!r}, expected {TASK_ID!r}"
    st.BASE_URL, st.API_KEY = BASE_URL, key
    st.submit(a.path)


if __name__ == "__main__":
    main()
