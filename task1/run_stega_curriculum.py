"""Resumable synthetic-training ablations. Sequential GPU runs; completed artifacts are never overwritten."""

import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import fcntl

ARMS = [(3301, 400, 48), (3503, 1000, 64), (3911, 1800, 80)]


def run(root, wait):
    exe = str(Path(__file__).with_name("stega_synthetic.py"))
    with (root / "stega-curriculum.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if wait:
            deadline = time.monotonic() + 1200
            while (
                len(
                    [
                        p
                        for p in (root / "synth-stega-bank-v1").glob("*.npy")
                        if ".partial." not in p.name
                    ]
                )
                < 720
            ):
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        "bank never became ready; rerun without --wait-bank"
                    )
                time.sleep(5)
        else:
            subprocess.run(
                [sys.executable, "-u", exe, str(root), "--prepare"], check=True
            )
        for seed, pre, ep in ARMS:
            out = root / f"stega-synth-v1-{seed}-p{pre}-e{ep}"
            if not (out / "stegastamp.json").is_file():
                with (root / f"stega-curriculum-{seed}.log").open("a") as f:
                    subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            exe,
                            str(root),
                            "--seed",
                            str(seed),
                            "--pre",
                            str(pre),
                            "--epochs",
                            str(ep),
                        ],
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            doc = json.loads((out / "stegastamp.json").read_text())
            print("FINISHED", seed, doc, flush=True)
            for split, n in [("test", 340), ("hold", 34)]:
                with np.load(out / f"stegastamp-{split}.npz", allow_pickle=False) as z:
                    assert (
                        z["masks"].shape == (n, 128, 128)
                        and z["masks"].dtype == np.bool_
                        and len(set(z["ids"])) == n
                    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--wait-bank", action="store_true")
    a = p.parse_args()
    run(a.root, a.wait_bank)
