"""Continue real four-fold experiments serially; compare variants by train-fold OOF only.
A UTC deadline limits compute; no public submission, notifications or automatic Git writes.
Run under the session's background process manager, which notifies on exit. Restart is safe.
"""

import argparse, datetime as dt, fcntl, json, subprocess, sys, time
from pathlib import Path
from statistics import median


def run(root, deadline):
    cutoff = dt.datetime.fromisoformat(deadline).timestamp()
    exe = Path(__file__).with_name("stega_crossfit.py")
    with (root / "stega-crossfit-queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for tier in range(16):
            # First compare types of evidence; only promote families on OOF across the same 120 images.
            arms = ["both", "base", "magnitude", "signed"]
            width = [24, 32, 24, 32][tier % 4]
            ep = [80, 112, 144, 96][tier % 4]
            seed = 4447 + 113 * tier
            if tier:
                h = {a: [] for a in arms}
                for p in root.glob("stega-crossfit-v1-*/stegastamp.json"):
                    v = json.loads(p.read_text())
                    h[v["arm"]].append(v["oof"])
                arms = sorted(
                    arms, key=lambda a: median(h[a]) if h[a] else -1, reverse=True
                )[:2]
            for arm in arms:
                if time.time() >= cutoff:
                    print("STOP competition development deadline", flush=True)
                    return
                if (root / "STOP-STEGASTAMP").exists():
                    print("STOP local operator request", flush=True)
                    return
                out = root / f"stega-crossfit-v1-{arm}-{seed}-{width}-{ep}"
                if not (out / "stegastamp.json").exists():
                    cmd = [
                        sys.executable,
                        "-u",
                        str(exe),
                        str(root),
                        "--seed",
                        str(seed),
                        "--epochs",
                        str(ep),
                        "--width",
                        str(width),
                        "--arm",
                        arm,
                    ]
                    print("LAUNCH", out.name, flush=True)
                    with (root / (out.name + ".log")).open("a") as f:
                        try:
                            subprocess.run(
                                cmd,
                                stdout=f,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=max(1, cutoff - time.time()),
                            )
                        except subprocess.TimeoutExpired:
                            print("STOP deadline", flush=True)
                            return
                v = json.loads((out / "stegastamp.json").read_text())
                print("RESULT", out.name, v, flush=True)
                # A local candidate warrants independent review; it is not a public result.
                if v["oof"] >= 0.70 and v["dev"] >= 0.70:
                    print(
                        "Verification milestone; check fixed hold and team score.",
                        flush=True,
                    )
                    return


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--deadline", required=True, help="ISO 8601 with timezone")
    a = p.parse_args()
    run(a.root, a.deadline)
