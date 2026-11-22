"""Loss statistics split by 'capped' classes (count >= 129) and the rest. usage: xh_eval2.py name[*k] ..."""

import sys

import torch

import xh_common as C
import xh_train as TR


def main():
    data = {k: TR.load_tok(k) for k in ("clean", "light2", "strong1", "cut", "mix")}
    y = data["clean"][1].argmax(1)
    full = torch.bincount(y, minlength=100)[y] >= 129
    model = C.load_orig()
    for spec in sys.argv[1:]:
        name, _, k = spec.partition("*")
        state = torch.load(C.RUN / f"{name}.pt", map_location="cpu", weights_only=True)
        if k:
            state["head.weight"] = state["head.weight"] * float(k)
            state["head.bias"] = state["head.bias"] * float(k)
        model.load_state_dict(state, strict=True)
        line = f"{spec:16s}"
        for view in ("clean", "light2"):
            z = TR.predict_tok(model, data[view][0])
            a = C.ce_stats(z, y)
            f = C.ce_stats(z[full], y[full])
            p = C.ce_stats(z[~full], y[~full])
            line += f" | {view} all {a[0]:.2f}/{a[1]:.2f} acc {a[2]:.2f}  capped {f[0]:.2f}/{f[1]:.2f} acc {f[2]:.2f}  rest {p[0]:.2f}/{p[1]:.2f} acc {p[2]:.2f}"
        kl = []
        for view in ("strong1", "cut", "mix"):
            tok, zo = data[view]
            zs = TR.predict_tok(model, tok)
            r = TR.kl_rows(zo, zs, 2.0)
            ag = (zs.argmax(1) == zo.argmax(1)).float()
            kl.append(f"{view} {r.mean():.2f} (capped {r[full].mean():.2f} agree {ag[full].mean():.2f}; rest {r[~full].mean():.2f} agree {ag[~full].mean():.2f})")
        print(line + " | retainKL " + " ".join(kl), flush=True)


if __name__ == "__main__":
    main()
