"""Replay the frozen ten-question practice benchmark using archived responses."""

import argparse
import csv
import hashlib
import importlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

from evaluation import evaluate

TASK = Path(__file__).resolve().parent
CHECKPOINT = TASK / "checkpoint"


def load_inputs():
    with (TASK / "prompts.csv").open(newline="", encoding="utf-8-sig") as stream:
        bank = {r["id"]: r for r in csv.DictReader(stream)}
    with (CHECKPOINT / "practice_answers.csv").open(newline="", encoding="utf-8-sig") as stream:
        endpoints = sorted(r["endpoint"] for r in csv.DictReader(stream))
    cache = {}
    with (CHECKPOINT / "practice.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["endpoint"], row["id"], row["repeat"]
            if key in cache or row["prompt"] != bank[row["id"]]["prompt"]:
                raise ValueError(f"Invalid practice entry {key}")
            cache[key] = row
    expected = {(ep, q, rep) for ep in endpoints for q in bank for rep in range(3)}
    if set(cache) != expected:
        raise ValueError("Practice cache must cover every endpoint and prompt three times")
    return bank, endpoints, cache


def replay(predict, seed, draws, protocol):
    bank, endpoints, cache = load_inputs()
    by_category = defaultdict(list)
    for q in sorted(bank):
        by_category[bank[q]["category"]].append(q)
    # In the original audit these were the unused development questions.
    panels = json.loads((CHECKPOINT / "practice-replay.json").read_text(encoding="utf-8"))
    fixed_hard = sorted(q for q in bank if bank[q]["split"] == "dev" and q not in panels["first_ten"])
    manifest = hashlib.sha256()
    scores = []
    for draw in range(draws):
        key = hashlib.blake2s(f"{seed}/{protocol}/{draw}".encode(), digest_size=8)
        rng = random.Random(int.from_bytes(key.digest(), "big"))
        questions = fixed_hard
        if protocol == "stratified":
            questions = sorted(q for c in sorted(by_category) for q in rng.sample(by_category[c], 2))
        keys = [(ep, q, rng.randrange(3)) for ep in endpoints for q in questions]
        manifest.update(str(keys).encode())
        aliases = [f"anonymous-{i:02d}" for i in range(1, len(endpoints) + 1)]
        rng.shuffle(aliases)
        renamed = dict(zip(endpoints, aliases))
        rows = [dict(cache[k], endpoint=renamed[k[0]], repeat=0) for k in keys]
        rng.shuffle(rows)
        prediction = predict(rows)
        if set(prediction) != set(aliases):
            raise ValueError("Solver must return exactly one assignment per anonymous endpoint")
        groups = [(ep, "practice", str(prediction[renamed[ep]])) for ep in endpoints]
        scores.append(evaluate(groups, CHECKPOINT / "practice_answers.csv")["model_mcc"])
    passed = sum(score > 0.65 for score in scores)
    return {"seed": seed, "protocol": protocol, "draws": draws, "n_strictly_above_065": passed,
            "pass_fraction": passed / draws, "median_model_mcc": median(scores),
            "draw_manifest_sha256": manifest.hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", default="variant_rank", help="Importable module with predict(rows)")
    parser.add_argument("--seed", type=int, default=519883)
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--protocol", choices=("stratified", "fixed-hard"), default="stratified")
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    print(json.dumps(replay(importlib.import_module(args.solver).predict,
                            args.seed, args.draws, args.protocol), indent=2))
