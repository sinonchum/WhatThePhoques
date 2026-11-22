"""Build the Task 2 submission CSV from the endpoint table next to this file.

Every endpoint marked `same` in endpoints.csv joins its family's one same-model
group; every endpoint marked `own` is a model of its own.

  python3 task2/final/build_submission.py --check              # compare with the uploaded files
  python3 task2/final/build_submission.py --out my.csv         # leaderboard record (t11)
  python3 task2/final/build_submission.py --own ep-12 --out my.csv   # last upload (t14)
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(own=()):
    rows = list(csv.DictReader(open(HERE / "endpoints.csv", newline="")))
    assert len(rows) == 33 and len({r["endpoint"] for r in rows}) == 33, "expected 33 distinct endpoints"
    unknown = set(own) - {r["endpoint"] for r in rows}
    assert not unknown, f"unknown endpoints: {sorted(unknown)}"
    lines = ["endpoint,family,model"]
    for r in sorted(rows, key=lambda r: r["endpoint"]):
        assert r["group"] in ("same", "own"), r
        same = r["group"] == "same" and r["endpoint"] not in own
        model = f"{r['family']}-grp0" if same else f"{r['family']}-{r['endpoint'][-2:]}"
        lines.append(f"{r['endpoint']},{r['family']},{model}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--own", nargs="*", default=[], help="endpoints to force into a model of their own")
    ap.add_argument("--out", help="where to write the CSV (default: print it)")
    ap.add_argument("--check", action="store_true", help="verify both uploaded CSVs are reproduced byte for byte")
    a = ap.parse_args()
    if a.check:
        ok = True
        for name, own in (("submission.csv", []), ("submission-last-upload.csv", ["ep-12"])):
            same = build(own) == (HERE / name).read_text()
            ok &= same
            print(f"{name}: {'identical' if same else 'DIFFERENT'}")
        sys.exit(0 if ok else 1)
    text = build(a.own)
    if a.out:
        Path(a.out).write_text(text)
        print(f"wrote {a.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
