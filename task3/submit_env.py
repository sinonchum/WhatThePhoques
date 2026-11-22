"""Submit a state dict. API key comes from the environment, never from this file."""

import os
import re
import sys
import time
from pathlib import Path

import requests
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "starter"))
import task_template as tt  # noqa: E402

TASK_ID = "31-unlearning"
SERVER = "http://34.147.229.55"


def validate(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = tt.build_model()
    model.load_state_dict(state, strict=True)
    bad = [
        k for k, v in state.items()
        if v.dtype != torch.float32 or not torch.isfinite(v).all()
    ]
    size = path.stat().st_size / (1024 * 1024)
    if bad or size > 450:
        raise SystemExit(f"invalid checkpoint size {size:.1f} issues {bad[:3]}")
    if torch.cuda.is_available():
        model = model.cuda().eval()
        images = tt.load_forget_images(ROOT / "starter" / "forget_images.npz")
        logits = tt.predict(model, images[:128], batch_size=64)
        if not torch.isfinite(logits).all():
            raise SystemExit("fp16 forward produced non-finite logits")
    print(f"validated {path.name}  {size:.1f} MB  strict load  fp16 finite", flush=True)


def main() -> int:
    key = os.environ.get("API_KEY", "").strip()
    if not key or key == "YOUR_API_KEY_HERE":
        print("API_KEY is not set", file=sys.stderr)
        return 1
    path = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "runs" / "fit_b8.pt")
    validate(path)
    for attempt in range(4):
        print(f"submitting {path.name} attempt {attempt + 1}", flush=True)
        with path.open("rb") as handle:
            try:
                reply = requests.post(
                    f"{SERVER}/submit/{TASK_ID}",
                    headers={"X-API-Key": key},
                    files={"file": (path.name, handle)},
                    timeout=(10, 600),
                )
            except requests.RequestException as exc:
                print(f"request failed: {exc}", file=sys.stderr)
                return 1
        try:
            body = reply.json()
        except ValueError:
            print(f"HTTP {reply.status_code}: {reply.text[:300]}", file=sys.stderr)
            return 1
        if reply.status_code == 429:
            detail = str(body.get("detail", body))
            match = re.search(r"(\d+)", detail)
            wait = int(match.group(1)) if match else 60
            print(f"cooldown {detail}; sleeping {wait + 5}s", flush=True)
            time.sleep(wait + 5)
            continue
        if reply.status_code != 200:
            print(f"rejected HTTP {reply.status_code}: {body.get('detail', body)}", file=sys.stderr)
            return 1
        print(f"score {body.get('score')}  previous {body.get('previous_best')}  improved {body.get('improved')}")
        print(body.get("message", ""))
        print(f"submission_id {body.get('submission_id')}")
        return 0
    print("still in cooldown after retries", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
