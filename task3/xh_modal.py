"""Run the xh_* scripts on Modal GPUs against the shared volume `euroai-task3`.

The volume already holds the official files under /data. Our files live under /xh/runs.
  modal run xh_modal.py --cmds jobs.txt        one shell command per line, each on its own A100 (max 10 at once)
  modal volume get euroai-task3 /xh/runs/<name>.pt runs/<name>.pt     fetch a result
Commands run in /root/task3 with XH_STARTER=/vol/data and XH_RUN=/vol/xh/runs.
"""

import os
import subprocess

import modal

app = modal.App("euroai-task3-xh")
vol = modal.Volume.from_name("euroai-task3")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "torchvision", "timm", "numpy", "pillow", "requests")
    .add_local_dir(".", "/root/task3", ignore=["runs", "starter", ".venv", "__pycache__", "*.pt", "*.log"])
)


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", memory=32768, timeout=3600, max_containers=10)
def run(cmd: str):
    vol.reload()
    env = dict(os.environ, XH_STARTER="/vol/data", XH_RUN="/vol/xh/runs", PYTHONUNBUFFERED="1")
    p = subprocess.run(f"cd /root/task3 && {cmd}", shell=True, env=env, capture_output=True, text=True)
    vol.commit()
    return {"cmd": cmd, "code": p.returncode, "out": p.stdout[-5000:], "err": p.stderr[-2500:]}


@app.local_entrypoint()
def main(cmds: str):
    lines = [l.strip() for l in open(cmds) if l.strip() and not l.startswith("#")]
    for r in run.map(lines, order_outputs=False):
        print("=" * 20, r["cmd"][:150], "exit", r["code"])
        print(r["out"])
        if r["code"]:
            print(r["err"])


@app.function(image=image, volumes={"/vol": vol}, timeout=3000)
def submit_remote(name: str, key: str):
    """POST /vol/xh/runs/<name>.pt to the scoring server from the cloud (the laptop uplink times out on 345 MB)."""
    import re
    import time

    import requests

    vol.reload()
    path = f"/vol/xh/runs/{name}.pt"
    for _ in range(8):
        with open(path, "rb") as handle:
            reply = requests.post(
                "http://34.147.229.55/submit/31-unlearning",
                headers={"X-API-Key": key},
                files={"file": (f"{name}.pt", handle)},
                timeout=(60, 900),
            )
        try:
            body = reply.json()
        except ValueError:
            body = {"raw": reply.text[:300]}
        if reply.status_code == 429:
            m = re.search(r"Wait (\d+) seconds", str(body.get("detail", "")))
            time.sleep(int(m.group(1)) + 5 if m else 90)
            continue
        return {"http": reply.status_code, "body": body}
    return {"http": 429, "body": body}


@app.local_entrypoint()
def submit(name: str, note: str = ""):
    import json
    import time
    from pathlib import Path

    key = (Path.home() / ".hackathon_key").read_text().strip()
    res = submit_remote.remote(name, key)
    record = {"time": time.strftime("%H:%M:%S"), "file": f"{name}.pt", "note": note, "via": "modal", **res}
    with open("runs/submissions.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record))
