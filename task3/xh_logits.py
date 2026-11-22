"""Dump a checkpoint's logits on the cached views (clean, light2, strong1) to runs/lg/<name>.pt.

usage: xh_logits.py name[*k] ...   (*k = head scaled by k). Used by xh_fit.py / xh_score.py.
"""

import sys

import torch

import xh_common as C
import xh_train as TR


def main():
    out = C.RUN / "lg"
    out.mkdir(exist_ok=True)
    data = {k: TR.load_tok(k) for k in ("clean", "light2", "strong1")}
    model = C.load_orig()
    for spec in sys.argv[1:]:
        name, _, k = spec.partition("*")
        tag = spec.replace("*", "k")
        if (out / f"{tag}.pt").exists():
            continue
        state = torch.load(C.RUN / f"{name}.pt", map_location="cpu", weights_only=True)
        if k:
            state["head.weight"] = state["head.weight"] * float(k)
            state["head.bias"] = state["head.bias"] * float(k)
        model.load_state_dict(state, strict=True)
        res = {v: TR.predict_tok(model, data[v][0]).half() for v in data}
        torch.save(res, out / f"{tag}.pt")
        print("dumped", tag, flush=True)


if __name__ == "__main__":
    main()
