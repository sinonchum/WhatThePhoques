"""Verify archived file bytes, unique response keys, round boundaries, and labels."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(__file__).resolve().parent
RAW_FIELDS = ("endpoint", "id", "repeat", "prompt", "response")


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    for entry in manifest["files"]:
        data = (ROOT / entry["path"]).read_bytes()
        check(hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"])

    primary = {k: ROOT / v for k, v in manifest["primary_files"].items()}
    with primary["prompt_bank"].open(newline="", encoding="utf-8-sig") as stream:
        prompts = {r["id"]: r["prompt"] for r in csv.DictReader(stream)}
    with primary["practice_truth"].open(newline="", encoding="utf-8-sig") as stream:
        truth = {r["endpoint"]: r for r in csv.DictReader(stream)}
    for source, split, n_ep, n_questions, repeats in [
        ("practice_labeled", "practice", 9, 30, 3),
        ("formal_unlabeled", "formal", 33, 20, 1),
    ]:
        rows = read_rows(primary[source])
        raw = read_rows(ROOT / "task2" / "checkpoint" / f"{split}.jsonl")
        keys = {(r["endpoint"], r["id"], r["repeat"]) for r in rows}
        check(len(keys) == len(rows) == len(raw) == n_ep*n_questions*repeats, split)
        endpoints = {r["endpoint"] for r in rows}
        questions = {r["id"] for r in rows}
        check(len(endpoints) == n_ep and len(questions) == n_questions, split)
        check(keys == {(e, q, i) for e in endpoints for q in questions for i in range(repeats)}, split)
        for r, original in zip(rows, raw):
            check({f: r[f] for f in RAW_FIELDS} == original, "Raw response mismatch")
            check(r["prompt"] == prompts[r["id"]], "Prompt mismatch")
            if split == "practice":
                ref = truth[r["endpoint"]]
                check((r["family"], r["model"], r["variant_kind"]) ==
                      (ref["family"], ref["model"], ref["kind"]), "Practice label mismatch")
            else:
                check("family" not in r and "model" not in r, "Formal row has a label")
    rounds = ARCHIVE / "collected"
    first = (rounds / "next-round/formal_unlabeled.jsonl").read_bytes()
    second = (rounds / "official-round2/new_formal_unlabeled.jsonl").read_bytes()
    check(len(first.splitlines()) == len(second.splitlines()) == 330, "Round sizes")
    check(first + second == primary["formal_unlabeled"].read_bytes(), "Round concatenation")
    print(f"{len(manifest['files'])} indexed files verified; 810 practice and 660 formal responses.")


if __name__ == "__main__":
    verify()
