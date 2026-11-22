"""Pairwise similarity of endpoints from the numeric list prompts (n1, n2).

Each reply gives up to 16 numbers.  For every endpoint and item we keep the
list of sampled answers.  Similarity of two endpoints on an item is the chance
that one draw from each agrees; averaged over items it is compared with each
endpoint's agreement with itself (different replies of the same endpoint).

  python3 analyze_num.py queue/results/020-num-r1 [more result dirs ...] [--refs]
"""
import collections
import json
import re
import sys
from pathlib import Path

FAM = {"G1": "01 02 04 12 14 17 18 23 24 25 33", "G2": "03 06 07 08 20 21 22 26 28 30 32",
       "G3": "05 09 10 11 13 15 16 19 27 29 31"}
REFLAB = {"ref-01": "base", "ref-03": "sys", "ref-05": "pers", "ref-06": "wm", "ref-08": "T1",
          "ref-02": "fp8", "ref-09": "L.3", "ref-04": "L.7", "ref-07": "L1.5"}
NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse(text, n=16):
    """item index -> normalised number string.  Handles '3. 250', and replies that copy the question first."""
    out = {}
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    cur = None
    for l in lines:
        m = re.match(r"^(\d{1,2})\s*[.):]\s*(.*)$", l)
        if m and 1 <= int(m.group(1)) <= n and (cur is None or int(m.group(1)) >= cur):
            cur, rest = int(m.group(1)), m.group(2)
        elif cur is not None:
            rest = l
        else:
            continue
        if cur - 1 in out:
            continue
        if "?" in rest:                      # question copied: answer is after the last '?', or on the next line
            rest = rest.rsplit("?", 1)[1]
        if "=" in rest:
            rest = rest.rsplit("=", 1)[1]
        nums = NUM.findall(rest)
        if nums:
            v = nums[-1].replace(",", "").rstrip(".")
            if v.endswith(".0"):
                v = v[:-2]
            out[cur - 1] = v
    return out


def load(dirs):
    data = collections.defaultdict(lambda: collections.defaultdict(list))   # ep -> (pid, item) -> [answers]
    raw = collections.defaultdict(list)
    for d in dirs:
        for line in open(Path(d) / "answers.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if "response" not in r or r["id"] not in ("n1", "n2"):
                continue
            raw[r["endpoint"]].append(r["response"])
            for i, v in parse(r["response"]).items():
                data[r["endpoint"]][(r["id"], i)].append(v)
    return data, raw


def cross(a, b, same=False):
    """Mean over items of P(draw from a == draw from b); for same=True excludes pairing a reply with itself."""
    tot, n = 0.0, 0
    for k in set(a) & set(b):
        x, y = a[k], b[k]
        if same:
            if len(x) < 2:
                continue
            c = collections.Counter(x)
            p = sum(v * (v - 1) for v in c.values()) / (len(x) * (len(x) - 1))
        else:
            cy = collections.Counter(y)
            p = sum(cy[v] for v in x) / (len(x) * len(y))
        tot += p
        n += 1
    return tot / n if n else float("nan")


def modes(a):
    return {k: collections.Counter(v).most_common(1)[0][0] for k, v in a.items()}


def mode_agree(a, b):
    ma, mb = modes(a), modes(b)
    ks = set(ma) & set(mb)
    return sum(ma[k] == mb[k] for k in ks) / len(ks) if ks else float("nan")


def table(eps, data, label=lambda e: e[-2:]):
    selfs = {e: cross(data[e], data[e], True) for e in eps}
    print("self-agreement:", "  ".join(f"{label(e)}:{selfs[e]:.2f}" for e in eps))
    for name, fn in (("cross-agreement / geometric mean of self", lambda a, b: cross(data[a], data[b]) / max(1e-9, (selfs[a] * selfs[b]) ** 0.5)),
                     ("mode agreement", lambda a, b: mode_agree(data[a], data[b]))):
        print(name)
        print("      " + " ".join(f"{label(e):>5}" for e in eps))
        for a in eps:
            print(f"{label(a):>5} " + " ".join("    ." if a == b else f"{fn(a, b):5.2f}" for b in eps))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    data, raw = load(args)
    if "--refs" in sys.argv:
        eps = [e for e in REFLAB if e in data]
        table(eps, data, lambda e: REFLAB[e])
    else:
        for g, m in FAM.items():
            eps = [f"ep-{x}" for x in m.split() if f"ep-{x}" in data]
            print(f"\n===== {g}  replies per endpoint: {[len(raw[e]) for e in eps]}  parsed items: {[sum(len(v) for v in data[e].values()) for e in eps]}")
            table(eps, data)
