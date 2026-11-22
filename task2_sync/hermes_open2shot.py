"""Fast, budget-safe 2-shot open-answer probe collector & Dual-Invariance analyzer for Task 2.

Key design (safe against server quirks & fully generalizable):
1. Uses 2-shot user/assistant exemplars (`messages` has no `system` role, 100% compliant) to lock
   terse 1-token/1-word formatting even under provider system prompts / personas.
2. Uses 16 open-answer probes (multi-digit arithmetic, string reversal/counting, niche factuals,
   code evaluation, tool-choice) where random collision is ~0 (unlike 4-choice ABCD's 25% floor),
   so `repeats=2` (only 32 queries per endpoint out of 39 remaining!) suffices for clean U-statistic MMD^2.
3. NEVER retries on HTTP 502/504 (which deduct budget on the server); retries ONLY on HTTP 429
   concurrency limit ('requests open'). Keeps max_tokens=12 for fast ~3-8s responses.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
URL = "http://34.147.229.55/33-model-equality/v1/chat/completions"
MODELS_URL = "http://34.147.229.55/33-model-equality/v1/models"

FEWSHOT = [
    {"role": "user", "content": "What is 14 times 11? Answer with the number only."},
    {"role": "assistant", "content": "154"},
    {"role": "user", "content": "What is the capital of Japan? Answer with one word only."},
    {"role": "assistant", "content": "Tokyo"},
]

# 16 high-discriminability open-answer probes across 5 domains (zero overlap with cached 30 prompts)
PROBES = [
    ("p_ar1", "num", "What is 67 times 84? Answer with the number only."),
    ("p_ar2", "num", "What is 438 times 59? Answer with the number only."),
    ("p_ar3", "num", "What is 19 times 27 minus 148? Answer with the number only."),
    ("p_ar4", "num", "What is the sum of the digits of 2 to the power of 18? Answer with the number only."),
    ("p_ar5", "num", "What is the remainder when 1234 is divided by 43? Answer with the number only."),
    ("p_st1", "str", "Reverse this string character by character: 'm8Qx-Lp4!zK2'. Answer with the reversed string only."),
    ("p_st2", "num", "How many times does the letter 's' appear in 'misrepresentations'? Answer with the number only."),
    ("p_st3", "word", "What is the 11th letter of the word 'electroencephalogram'? Answer with the letter only."),
    ("p_cd1", "num", "What does Python 3 print for: print(17 // 5 + (-17) % 5)? Answer with the number only."),
    ("p_cd2", "str", "What does Python 3 print for: print('abcdefg'[5:0:-2])? Answer with the output only."),
    ("p_kn1", "word", "Which river flows through the capital city Ulaanbaatar? Answer with the river name only."),
    ("p_kn2", "num", "In which year was the chemical element rhenium discovered? Answer with the year only."),
    ("p_kn3", "word", "Which composer wrote the opera 'Billy Budd'? Answer with the composer's last name only."),
    ("p_kn4", "word", "Translate the English word 'dragonfly' into German. Answer with one word only."),
    ("p_tl1", "word", "Available tools: search_web, calculator, read_file, send_email. User asks: 'Check if 2089 is prime.' Which tool do you call? Answer with the tool name only."),
    ("p_tl2", "word", "Available tools: read_file, search_web, ask_user, do_nothing. User asks: 'Summarize the attached contract.' Which tool do you call? Answer with the tool name only."),
]

STOP = {"the", "is", "a", "an", "answer", "of", "it", "s", "word", "letter", "number", "output", "tool", "river"}


def extract_open(kind: str, text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "<empty>"
    if kind == "num":
        t2 = re.sub(r"(?<=\d),(?=\d{3})", "", t)
        nums = re.findall(r"-?\d+", t2)
        return nums[-1] if nums else "<none>"
    if kind == "str":
        toks = t.strip(" '\"`.").split()
        return toks[-1].strip(" '\"`.") if toks else "<none>"
    words = [w for w in re.findall(r"[a-zA-Zà-ÿ_]+", t.lower()) if w not in STOP]
    return words[-1] if words else "<none>"


def collect(target: str, repeats: int, out_path: Path, execute_ep: bool = False):
    key = os.environ["HACKATHON_API_KEY"].strip()
    session = requests.Session()
    session.headers["X-API-Key"] = key
    eps = [f"ref-{i:02d}" for i in range(1, 10)] if target == "ref" else [f"ep-{i:02d}" for i in range(1, 34)]

    if target == "ep":
        data = session.get(MODELS_URL, timeout=30).json()["data"]
        rem = {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in data if m["id"].startswith("ep-")}
        min_rem = min(rem.values())
        needed = len(PROBES) * repeats
        print(f"Formal budget check: min remaining = {min_rem}, needed per ep = {needed}")
        if min_rem < needed:
            raise SystemExit(f"ABORT: min remaining budget ({min_rem}) < needed ({needed})!")
        if not execute_ep:
            print("Dry-run passed. Pass --yes to execute formal queries.")
            return

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add((r["endpoint"], r["id"], int(r["repeat"])))

    jobs = [(ep, qid, kind, prompt, rep) for rep in range(repeats) for qid, kind, prompt in PROBES for ep in eps if (ep, qid, rep) not in done]
    print(f"Starting {len(jobs)} queries on {target} ({len(PROBES)} probes x {repeats} repeats, max_tokens=12)...", flush=True)
    if not jobs:
        return

    lock = threading.Lock()
    stats = Counter()
    t_start = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f_out:
        def run_one(job):
            ep, qid, kind, prompt, rep = job
            msgs = FEWSHOT + [{"role": "user", "content": prompt}]
            while True:
                t0 = time.time()
                try:
                    r = session.post(URL, timeout=45, json={"model": ep, "max_tokens": 12, "messages": msgs})
                except requests.RequestException as exc:
                    status, resp_txt = f"err_{type(exc).__name__}", ""
                    break
                if r.status_code == 429 and ("requests open" in r.text or "concurrent" in r.text.lower()):
                    time.sleep(1.2)
                    continue
                if r.status_code == 200:
                    status = "ok"
                    resp_txt = r.json()["choices"][0]["message"]["content"]
                else:
                    # NEVER retry 502/504 to protect formal budget!
                    status, resp_txt = f"http_{r.status_code}", r.text[:100]
                break
            dt = round(time.time() - t0, 2)
            rec = {
                "endpoint": ep,
                "id": qid,
                "kind": kind,
                "repeat": rep,
                "prompt": prompt,
                "status": status,
                "seconds": dt,
                "response": resp_txt,
                "answer": extract_open(kind, resp_txt) if status == "ok" else None,
            }
            with lock:
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_out.flush()
                stats[status] += 1
                tot = sum(stats.values())
                if tot % 20 == 0 or tot == len(jobs):
                    el = time.time() - t_start
                    print(f"  [{tot}/{len(jobs)}] elapsed={el:.1f}s ({tot/max(1,el):.2f} req/s) stats={dict(stats)}", flush=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(run_one, jobs))


def eval_ref(jsonl_path: Path):
    truth = {r["endpoint"]: (r["family"], r["model"]) for r in csv.DictReader((ROOT / "practice_answers.csv").open())}
    kinds = {r["endpoint"]: r["kind"] for r in csv.DictReader((ROOT / "practice_answers.csv").open())}
    rows = [json.loads(x) for x in jsonl_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    ans = defaultdict(lambda: defaultdict(list))
    for r in ok_rows:
        ans[r["endpoint"]][r["id"]].append(r["answer"])

    eps = sorted(truth)
    qids = [qid for qid, _, _ in PROBES if all(len(ans[e][qid]) >= 2 for e in eps)]
    print(f"Complete 2-repeat questions across all 9 ref endpoints: {len(qids)}/{len(PROBES)}")

    # Print per-question answers on S vs T
    S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
    T = ["ref-02", "ref-04", "ref-07", "ref-09"]
    for q in qids:
        s_str = " | ".join("/".join(ans[e][q][:2]) for e in S)
        t_str = " | ".join("/".join(ans[e][q][:2]) for e in T)
        print(f"  {q:6s}: S=[{s_str}]  ||  T(fp8,0.7,1.5,0.3)=[{t_str}]")

    # Compute unbiased one-hot MMD^2 distance: cross_disagreement - (within_e + within_f)/2
    # Plus discriminative question weighting D(q) = mean_between / (mean_within + 0.05)
    dq = {}
    for q in qids:
        within = np.mean([float(ans[e][q][0] != ans[e][q][1]) for e in eps])
        between = np.mean([sum(a != b for a in ans[e1][q][:2] for b in ans[e2][q][:2]) / 4.0 for e1, e2 in itertools.combinations(eps, 2)])
        dq[q] = float(between / (within + 0.05))

    n = len(eps)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            e1, e2 = eps[i], eps[j]
            num, den = 0.0, 0.0
            for q in qids:
                a1, a2 = ans[e1][q][:2], ans[e2][q][:2]
                x = sum(u != v for u in a1 for v in a2) / 4.0
                w1, w2 = float(a1[0] != a1[1]), float(a2[0] != a2[1])
                d_val = x - 0.5 * (w1 + w2)
                w = dq[q]
                num += w * d_val
                den += w
            dist[i, j] = dist[j, i] = num / max(1e-9, den)

    print("\nPairwise Unbiased Weighted MMD^2 Distance Matrix:")
    print("       " + " ".join(f"{e[-2:]:>6s}" for e in eps))
    for i, e in enumerate(eps):
        print(f"{e}: " + " ".join(f"{dist[i,j]:6.3f}" for j in range(n)))

    # Evaluate average-linkage clustering across thresholds + LOO / Half-split CV!
    def cluster_at(sub_qids, thr):
        dmat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                e1, e2 = eps[i], eps[j]
                vals, ws = [], []
                for q in sub_qids:
                    a1, a2 = ans[e1][q][:2], ans[e2][q][:2]
                    x = sum(u != v for u in a1 for v in a2) / 4.0
                    d_val = x - 0.5 * (float(a1[0] != a1[1]) + float(a2[0] != a2[1]))
                    vals.append(d_val)
                    ws.append(dq[q])
                dmat[i, j] = dmat[j, i] = float(np.average(vals, weights=ws))
        grps = [[k] for k in range(n)]
        while len(grps) > 1:
            best = (999.0, -1, -1)
            for g1 in range(len(grps)):
                for g2 in range(g1 + 1, len(grps)):
                    dd = float(np.mean([dmat[u, v] for u in grps[g1] for v in grps[g2]]))
                    if dd < best[0]:
                        best = (dd, g1, g2)
            if best[0] > thr:
                break
            _, g1, g2 = best
            grps[g1] = sorted(grps[g1] + grps[g2])
            grps.pop(g2)
        lbl = {}
        for g in grps:
            for u in g:
                lbl[eps[u]] = eps[g[0]]
        tp = fp = tn = fn = 0
        for a, b in itertools.combinations(eps, 2):
            t = truth[a] == truth[b]
            p = lbl[a] == lbl[b]
            tp += t and p; fp += (not t) and p; tn += (not t) and (not p); fn += t and (not p)
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = max(0.0, (tp * tn - fp * fn) / den) if den else 0.0
        return mcc, fp, fn, lbl

    print("\nThreshold sweep on full 16 probes:")
    for thr in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.25]:
        mcc, fp, fn, lbl = cluster_at(qids, thr)
        clusters = defaultdict(list)
        for e, g in lbl.items():
            clusters[g].append(f"{e[-2:]}({kinds[e][:6]})")
        print(f"  thr={thr:.2f} -> MCC={mcc:.4f} (FP={fp}, FN={fn}) | {list(clusters.values())}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_c = sub.add_parser("collect")
    p_c.add_argument("target", choices=["ref", "ep"])
    p_c.add_argument("--repeats", type=int, default=2)
    p_c.add_argument("--out", type=Path, default=None)
    p_c.add_argument("--yes", action="store_true")

    p_e = sub.add_parser("eval-ref")
    p_e.add_argument("--jsonl", type=Path, default=ROOT / "hermes_open2shot_ref.jsonl")

    args = ap.parse_args()
    if args.cmd == "collect":
        out = args.out or (ROOT / f"hermes_open2shot_{args.target}.jsonl")
        collect(args.target, args.repeats, out, execute_ep=args.yes)
    elif args.cmd == "eval-ref":
        eval_ref(args.jsonl)


if __name__ == "__main__":
    main()
