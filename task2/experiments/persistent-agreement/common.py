"""Shared loaders for the persistent-agreement experiment (cache only, no API)."""
import csv
import json
import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK))
from inputs import _normalize  # noqa: E402

DATA = TASK / "test-data" / "collected"
ROUND1 = DATA / "next-round" / "formal_unlabeled.jsonl"
ROUND2 = DATA / "official-round2" / "new_formal_unlabeled.jsonl"
PRACTICE = DATA / "next-round" / "practice_labeled.jsonl"
LAST_SUBMITTED = TASK / "experiments" / "metric-partition" / "minimax.csv"
SUBMITTED = {  # every CSV that was actually uploaded, in order, with its public score
    "first-live 18:04 (0.05)": DATA / "first-live" / "submission.csv",
    "offline-v3 18:40 (0.10)": DATA / "offline-v3" / "candidate.csv",
    "official-round2 19:35 (0.10)": DATA / "official-round2" / "candidate.csv",
    "minimax 20:24 (0.10)": LAST_SUBMITTED,
}


def bank():
    with (TASK / "prompts.csv").open(newline="", encoding="utf-8-sig") as stream:
        return {r["id"]: r for r in csv.DictReader(stream)}


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def answer(row, prompts):
    """Canonical letter/integer, or None when the reply has no unambiguous short answer."""
    return _normalize(row["response"], prompts[row["id"]]["format"], row["prompt"])


def partition(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return {r["endpoint"]: (r["family"], r["model"]) for r in csv.DictReader(stream)}
