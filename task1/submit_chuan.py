"""Upload task 1 submission. The API key is read from the environment."""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import requests

SERVER = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"


def main() -> int:
    key = os.environ.get("API_KEY", "").strip()
    if not key:
        print("API_KEY is not set", file=sys.stderr)
        return 1
    path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("submission.pt"))
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr)
        return 1
    for attempt in range(4):
        print(
            f"submitting {path.name} ({path.stat().st_size / (1024 * 1024):.2f} MB) attempt {attempt + 1}",
            flush=True,
        )
        with path.open("rb") as handle:
            try:
                reply = requests.post(
                    f"{SERVER}/submit/{TASK_ID}",
                    headers={"X-API-Key": key},
                    files={"file": (path.name, handle, "application/octet-stream")},
                    timeout=(10, 300),
                )
            except requests.RequestException as exc:
                print(f"request failed: {exc}", file=sys.stderr)
                return 1
        try:
            body = reply.json()
        except ValueError:
            print(f"HTTP {reply.status_code}: {reply.text[:400]}", file=sys.stderr)
            return 1
        if reply.status_code == 429:
            detail = str(body.get("detail", body))
            match = re.search(r"(\d+)", detail)
            wait = int(match.group(1)) if match else 60
            print(f"cooldown {detail}; sleeping {wait + 5}s", flush=True)
            time.sleep(wait + 5)
            continue
        if reply.status_code != 200:
            print(f"rejected HTTP {reply.status_code}: {body}", file=sys.stderr)
            return 1
        print(body)
        return 0
    print("still in cooldown after retries", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
