"""Validate a checkpoint and submit it. The key is read from ~/.hackathon_key and never printed.

usage: xh_submit.py runs/<name>.pt [note]
Every attempt is appended to runs/submissions.jsonl.
"""

import json
import sys
import time
from pathlib import Path

import requests
import torch

import xh_common as C

TASK_ID = "31-unlearning"
SERVER = "http://34.147.229.55"
LOG = C.RUN / "submissions.jsonl"


def validate(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = C.tt.build_model()
    model.load_state_dict(state, strict=True)
    bad = [k for k, v in state.items() if v.dtype != torch.float32 or not torch.isfinite(v).all()]
    size = path.stat().st_size / 2**20
    if bad or size > 450:
        raise SystemExit(f"invalid checkpoint: {size:.1f} MB, bad tensors {bad[:3]}")
    images = C.load_images()[:200]
    x = C.to_norm(images).cpu()
    with torch.no_grad():
        full = model.eval()(x[:50])
        half = model.half()(x.half())
    if not torch.isfinite(half).all():
        raise SystemExit("float16 forward produced non-finite logits")
    drift = (half[:50].float() - full).abs().max().item()
    print(f"validated {path.name}: {size:.1f} MB, strict load, float32, fp16 finite (max fp16-fp32 logit gap {drift:.3f})")


def main():
    path = Path(sys.argv[1])
    note = sys.argv[2] if len(sys.argv) > 2 else ""
    validate(path)
    key = (Path.home() / ".hackathon_key").read_text().strip()
    for attempt in range(6):
        with path.open("rb") as handle:
            reply = requests.post(
                f"{SERVER}/submit/{TASK_ID}",
                headers={"X-API-Key": key},
                files={"file": (path.name, handle)},
                timeout=(20, 900),
            )
        try:
            body = reply.json()
        except ValueError:
            body = {"raw": reply.text[:300]}
        record = {"time": time.strftime("%H:%M:%S"), "file": path.name, "note": note, "http": reply.status_code, "body": body}
        with LOG.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        if reply.status_code == 429:
            import re
            m = re.search(r"Wait (\d+) seconds", str(body.get("detail", "")))
            time.sleep(int(m.group(1)) + 5 if m else 90)
            continue
        return 0 if reply.status_code == 200 else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
