"""Generate deterministic synthetic Task 2 logs for a local smoke test."""

import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from inputs import _read_bank


def add_parser(subparsers, default_prompts: Path):
    parser = subparsers.add_parser("demo", help="Generate local synthetic sample files")
    parser.add_argument("--out", type=Path, required=True, help="Directory for new sample files")
    parser.add_argument("--prompts", type=Path, default=default_prompts, help="Prompt bank CSV")
    parser.set_defaults(handler=run)


def _profiles(prefix, count):
    for index in range(count):
        family = index % 3
        # Repeated variants exercise same-model pairs within each family.
        variant = (0, 0, 1, 2)[(index // 3) % 4]
        yield f"{prefix}-{index + 1:02d}", family, variant


def _write_log(path, profiles, prompts):
    position = {"dev": 0, "holdout": 0}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prompt_index, row in enumerate(prompts):
            split_index = position[row["split"]]
            position[row["split"]] += 1
            for index, (endpoint, family, variant) in enumerate(profiles):
                # Simulate family differences on most items and variant differences
                # on a subset of each split. These answers carry no correctness labels.
                signature = prompt_index + family
                if split_index % 3 == 0:
                    signature += variant
                for repeat in range(2):
                    value = signature
                    if repeat and (index * 11 + prompt_index * 7) % 61 == 0:
                        value += 1
                    if row["format"] == "choice":
                        answer = "ABCD"[value % 4]
                    else:
                        answer = str(prompt_index * 5 + value)
                    if repeat:
                        answer = f"Final answer: {answer}."
                    if repeat and index == 0 and split_index == 1:
                        answer = "I cannot resolve this sample."
                    record = {
                        "endpoint": endpoint, "id": row["id"], "repeat": repeat,
                        "prompt": row["prompt"], "response": answer, "synthetic": True,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(args):
    bank = list(_read_bank(args.prompts).values())
    if {row["split"] for row in bank} != {"dev", "holdout"}:
        raise ValueError("The demo requires dev and holdout prompts")
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    names = ("practice.jsonl", "scored.jsonl", "truth.csv")
    practice = list(_profiles("ref", 9))
    with TemporaryDirectory(dir=target) as tmp:
        staging = Path(tmp)
        _write_log(staging / names[0], practice, bank)
        _write_log(staging / names[1], list(_profiles("ep", 33)), bank)
        with (staging / names[2]).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("endpoint", "family", "model", "kind"))
            for endpoint, family, variant in practice:
                writer.writerow((endpoint, f"mock-F{family}", f"mock-F{family}-M{variant}",
                                 "synthetic sample"))

        # Each link publishes a complete file and fails if the destination exists.
        # Roll back the links we created if a later name collides.
        published = []
        try:
            for name in names:
                os.link(staging / name, target / name)
                published.append(name)
        except OSError:
            for name in published:
                (target / name).unlink()
            raise
    print(f"Synthetic samples only. Wrote {', '.join(names)} to {target}. "
          "All responses and labels are fabricated for offline testing.")
