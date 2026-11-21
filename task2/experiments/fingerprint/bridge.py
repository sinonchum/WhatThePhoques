#!/usr/bin/env python3
"""Run queued competition requests on a machine that can reach the contest server.

The analysis sandbox cannot reach http://34.147.229.55, so it writes small JSON
job files into queue/jobs/ and this script, started by a team member in a normal
terminal, carries them out and writes the replies to queue/results/<job>/.

Jobs run in file-name order; a job with an earlier name pauses a running query
job, and an empty file queue/cancel/<job> stops that job.

It only understands four declarative job types and never runs code from a job:
  query        send prompts to ref-* or ep-* endpoints (official chat endpoint)
  budget       read the remaining replies per endpoint
  leaderboard  save the public leaderboard page
  submit       upload one CSV from this worktree to the scoring endpoint

The team key is read from the TASK2_API_KEY environment variable and is never
written to disk or printed.  Standard library only (Python 3.9+).

    export TASK2_API_KEY=...        # if it is not already set in this shell
    python3 task2/experiments/fingerprint/bridge.py

Limits you control when starting it:
  --scored-cap N   most replies this bridge may spend per ep-* endpoint in total
                   (default 60; its own spend is kept in queue/ledger.json)
  --no-scored      refuse every ep-* query (reference endpoints stay allowed)
  --no-submit      refuse uploads
Stop it with Ctrl-C at any time; finished replies are kept and never re-bought.
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

SERVER = os.environ.get("TASK2_TEST_SERVER") or "http://34.147.229.55"   # the override is for offline tests only
TASK = "33-model-equality"
BASE = f"{SERVER}/{TASK}/v1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                      # worktree root
QUEUE = HERE / "queue"
ENDPOINT = re.compile(r"(ref-0[1-9]|ep-(0[1-9]|[12][0-9]|3[0-3]))")
PARALLEL = 4


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def http(method, url, key, payload=None, timeout=180, headers=None, raw=None):
    """Return (status, body_text, response_headers). Network errors raise."""
    data = raw
    hdr = {"X-API-Key": key} if key else {}
    hdr.update(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode()
        hdr["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), dict(e.headers)


def budget(key):
    status, text, _ = http("GET", f"{BASE}/models", key, timeout=60)
    if status != 200:
        raise RuntimeError(f"budget HTTP {status}: {text[:300]}")
    return {m["id"]: m["budget"] for m in json.loads(text)["data"]}


def load_ledger():
    path = QUEUE / "ledger.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_reply(line, text, headers):
    body = json.loads(text)
    choice = body["choices"][0]
    line["response"] = choice["message"]["content"]
    line["finish_reason"] = choice.get("finish_reason")
    line["usage"] = body.get("usage")
    line["budget"] = body.get("budget")
    line["extra"] = {k: v for k, v in body.items() if k not in ("choices", "usage", "budget")}
    line.pop("error", None)


def interrupted(stem):
    """True when this job was cancelled or a job with an earlier name is waiting."""
    if (QUEUE / "cancel" / stem).exists():
        return "cancelled"
    for p in sorted((QUEUE / "jobs").glob("*.json")):
        if p.stem >= stem:
            break
        if not (QUEUE / "results" / p.stem / "DONE.json").exists() and not (QUEUE / "cancel" / p.stem).exists():
            return "preempted"
    return None


def run_query(job, out, key, args, stem=""):
    endpoints = job["endpoints"]
    prompts = job["prompts"]
    repeats = int(job.get("repeats", 1))
    max_tokens = int(job.get("max_tokens", 256))
    if not endpoints or any(not ENDPOINT.fullmatch(e) for e in endpoints):
        raise ValueError("endpoints must be ref-01..ref-09 or ep-01..ep-33")
    if not 1 <= max_tokens <= 256 or not 1 <= repeats <= 50:
        raise ValueError("max_tokens must be 1..256 and repeats 1..50")
    if any(not isinstance(p.get("id"), str) or not isinstance(p.get("prompt"), str) or not p["prompt"].strip()
           for p in prompts) or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("prompts need unique string ids and non-empty text")
    answers = out / "answers.jsonl"
    done = set()
    if answers.exists():
        for raw in answers.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                r = json.loads(raw)
                if "response" in r:
                    done.add((r["endpoint"], r["id"], r["repeat"]))
    jobs = [(e, p["id"], rep, p["prompt"]) for rep in range(repeats) for p in prompts for e in endpoints
            if (e, p["id"], rep) not in done]
    scored = [j for j in jobs if j[0].startswith("ep-")]
    ledger = load_ledger()
    if scored:
        if args.no_scored:
            raise PermissionError("bridge was started with --no-scored")
        left = budget(key)
        need = {}
        for e, *_ in scored:
            need[e] = need.get(e, 0) + 1
        for e, n in need.items():
            remaining = left[e]["limit"] - left[e]["used"]
            if n > remaining:
                raise PermissionError(f"{e}: job needs {n} replies, server has {remaining} left")
            if ledger.get(e, 0) + n > args.scored_cap:
                raise PermissionError(f"{e}: job needs {n}, bridge already spent {ledger.get(e, 0)}, cap {args.scored_cap}")
    print(f"  {len(jobs)} requests ({len(scored)} on scored endpoints), {len(done)} already recorded", flush=True)
    lock = threading.Lock()
    stopped = set()
    busy_notice = []
    counts = {"answered": 0, "failed": 0}
    stream = answers.open("a", encoding="utf-8")

    def one(item):
        e, pid, rep, prompt = item
        payload = {"model": e, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        line = {"endpoint": e, "id": pid, "repeat": rep, "prompt": prompt, "utc": now()}
        attempt, waited = 0, 0.0
        while attempt < 6 and waited < 3600:
            if e in stopped:
                line["error"] = "endpoint stopped after budget error"
                break
            t0 = time.time()
            try:
                status, text, headers = http("POST", f"{BASE}/chat/completions", key, payload)
            except Exception as exc:                      # network trouble: wait and retry
                line["error"] = f"network {type(exc).__name__}"
                attempt += 1
                time.sleep(10)
                continue
            line["latency_s"] = round(time.time() - t0, 3)
            if status == 200:
                parse_reply(line, text, headers)
                break
            line["error"] = f"HTTP {status}: {text[:300]}"
            if status == 429 and "requests open" in text:  # team-wide limit of 4 is in use: wait, do not give up
                with lock:
                    if not busy_notice:
                        busy_notice.append(1)
                        print("  the team's 4 request slots are busy; waiting for a free one...", flush=True)
                time.sleep(1.5)
                waited += 1.5
                continue
            if status == 429 and "budget" in text.lower():
                stopped.add(e)
                break
            if status in (429, 500, 502, 503, 504):
                attempt += 1
                time.sleep(5 + 5 * attempt)
                continue
            break                                           # other 4xx: invalid request, not charged
        if waited:
            line["waited_s"] = waited
        with lock:
            ok = "response" in line
            counts["answered" if ok else "failed"] += 1
            if ok and e.startswith("ep-"):
                ledger[e] = ledger.get(e, 0) + 1
                save_json(QUEUE / "ledger.json", ledger)
            stream.write(json.dumps(line, ensure_ascii=False) + "\n")
            stream.flush()
            total = counts["answered"] + counts["failed"]
            if total % 20 == 0 or total == len(jobs):
                print(f"  {total}/{len(jobs)} done, {counts['failed']} failed", flush=True)

    reason = None
    with ThreadPoolExecutor(args.parallel) as pool:
        flying = set()
        for item in jobs:
            reason = interrupted(stem) if stem else None
            if reason:
                break
            while len(flying) >= args.parallel:
                flying = {f for f in flying if not f.done()}
                time.sleep(0.2)
            flying.add(pool.submit(one, item))
    stream.close()
    summary = {"requested": len(jobs), **counts, "scored_requests": len(scored)}
    if reason == "preempted":
        print("  paused for an earlier job", flush=True)
        return None
    if reason == "cancelled":
        print("  cancelled", flush=True)
        summary["cancelled"] = True
    return summary


def run_submit(job, out, key, args):
    if args.no_submit:
        raise PermissionError("bridge was started with --no-submit")
    path = (ROOT / job["csv"]).resolve()
    if ROOT not in path.parents or path.suffix != ".csv" or not path.is_file():
        raise ValueError("csv must be an existing .csv inside this worktree")
    content = path.read_bytes()
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: text/csv\r\n\r\n").encode() + content + f"\r\n--{boundary}--\r\n".encode()
    before = http("GET", f"{SERVER}/leaderboard_page", None, timeout=60)[1]
    (out / "leaderboard-before.html").write_text(before, encoding="utf-8")
    status, text, _ = http("POST", f"{SERVER}/submit/{TASK}", key, raw=body, timeout=600,
                           headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        reply = json.loads(text)
    except ValueError:
        reply = {"raw": text[:500]}
    time.sleep(3)
    (out / "leaderboard-after.html").write_text(http("GET", f"{SERVER}/leaderboard_page", None, timeout=60)[1],
                                               encoding="utf-8")
    print(f"  submit HTTP {status}: score={reply.get('score')} status={reply.get('status')}", flush=True)
    return {"http_status": status, "reply": reply, "file": job["csv"], "bytes": len(content), "utc": now()}


def process(path, key, args):
    job = json.loads(path.read_text(encoding="utf-8"))
    out = QUEUE / "results" / path.stem
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{now()}] {path.stem}: {job.get('type')}", flush=True)
    try:
        kind = job.get("type")
        if kind == "query":
            summary = run_query(job, out, key, args, path.stem)
            if summary is None:
                return
        elif kind == "budget":
            summary = {"budget": budget(key)}
        elif kind == "leaderboard":
            status, text, _ = http("GET", f"{SERVER}/leaderboard_page", None, timeout=60)
            (out / "leaderboard.html").write_text(text, encoding="utf-8")
            summary = {"http_status": status, "bytes": len(text)}
        elif kind == "submit":
            summary = run_submit(job, out, key, args)
        else:
            raise ValueError(f"unknown job type {kind!r}")
        summary["ok"] = True
    except Exception as exc:
        summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(f"  refused/failed: {summary['error']}", flush=True)
    summary["finished_utc"] = now()
    save_json(out / "DONE.json", summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored-cap", type=int, default=60)
    ap.add_argument("--parallel", type=int, default=PARALLEL, choices=range(1, 5),
                    help="requests in flight at once (the server allows 4 per team in total)")
    ap.add_argument("--no-scored", action="store_true")
    ap.add_argument("--no-submit", action="store_true")
    ap.add_argument("--once", action="store_true", help="process what is queued, then exit")
    args = ap.parse_args()
    key = (os.environ.get("TASK2_API_KEY") or "").strip()
    if not key or key == "YOUR_API_KEY_HERE":
        sys.exit("Set TASK2_API_KEY to the team key in this shell first:  export TASK2_API_KEY=...")
    (QUEUE / "jobs").mkdir(parents=True, exist_ok=True)
    (QUEUE / "results").mkdir(parents=True, exist_ok=True)
    print(f"bridge up; queue {QUEUE}\n  scored queries: {'OFF' if args.no_scored else f'on, cap {args.scored_cap} per endpoint'}; "
          f"uploads: {'OFF' if args.no_submit else 'on'}.  Ctrl-C to stop.", flush=True)
    try:
        while True:
            (QUEUE / "cancel").mkdir(exist_ok=True)
            pending = [p for p in sorted((QUEUE / "jobs").glob("*.json"))
                       if not (QUEUE / "results" / p.stem / "DONE.json").exists()]
            save_json(QUEUE / "bridge-status.json", {"utc": now(), "pid": os.getpid(), "pending": [p.stem for p in pending],
                                                     "scored_cap": args.scored_cap, "no_scored": args.no_scored,
                                                     "no_submit": args.no_submit})
            if pending:                                   # always the earliest name first; re-scan after each
                process(pending[0], key, args)
                continue
            if args.once:
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("stopped.")


if __name__ == "__main__":
    main()
