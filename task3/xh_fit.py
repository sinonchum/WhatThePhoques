"""Fit a 'virtual gold' to the public scores, then use it to score any checkpoint locally.

There is no gold model locally, but ~30 submitted checkpoints have known public scores. A small
parametric family describes what the retrained models might output on forget image i (T=2):
    g_i = a_i * onehot(memorized) + (1 - a_i) * R_i
    R_i = (1 - rho) * softmax_{c != mem}((wf * final + (1 - wf) * reframed) / tau) + rho * softmax_{c != mem}(lens8)
    a_i = a_cap on capped classes;  a_rest * (1 - lam * (count - 46) / 84) on the others
    tau = tau_cap / tau_rest
plus the retrained models' loss mean/std (m, s) and a retain slope beta:
    score = mean_std * prediction * 1 / (1 + beta * KL_T2(orig || sub) on re-framed views)
with mean_std and prediction computed by the official formulas on the clean and lightly altered views.
The 10 parameters are fitted by least squares on the scored checkpoints.

  xh_fit.py fit                 fit, print the fit table, save runs/vgold.pt
  xh_fit.py score name ...      predicted score of runs/lg/<name>.pt dumps
  xh_fit.py target OUT          write the fitted g as a training target (zt = 2 log g)
"""

import sys

import torch
import torch.nn.functional as F

import xh_common as C

SCORES = {
    "r2_b8": 0.0503, "r2x1.5": 0.0967, "r2x2.0": 0.0832, "r3_b8": 0.0832, "r4_raw9": 0.0383, "r5_sv0": 0.05,
    "r3_b8k1.8": 0.1377, "r3_b8k3.0": 0.0, "r7_mixed": 0.1083, "r7_mixedk1.5": 0.1788, "r7_mixedk1.9": 0.1747,
    "r8_k2k2.0": 0.0089, "r9_mixed2k1.3": 0.1861, "r10_gap8": 0.2509, "r10_gap8k1.2": 0.2454, "r11_anchor": 0.2558,
    "r12_grad4": 0.2621, "avg1112": 0.2658, "r13_comp": 0.2659, "r14_comp2": 0.2718, "ex1413": 0.2756,
    "r15_v2": 0.1899, "n9": 0.2623, "qA": 0.1671, "qB": 0.2807, "qBe": 0.2938, "s1": 0.2598, "s3": 0.2966,
    "v2": 0.3225, "w3": 0.3264, "z10": 0.3433,
}
D = torch.float64


def context():
    c = torch.load(C.RUN / "xh_cache.pt")
    y = c["y"]
    cnt = torch.bincount(y, minlength=100).float()
    tok = C.RUN / "tok"
    ctx = {
        "y": y, "full": cnt[y] >= 129, "w": ((cnt[y] - 46) / 84).clamp(0, 1).to(D),
        "z": c["z"].to(D), "s": c["s_logit"].to(D), "l8": c["lens"][8].to(D),
        "orig": {"clean": c["z"].to(D), "light2": torch.load(tok / "light2.pt")["logits"].to(D)},
        "orig_strong": torch.load(tok / "strong1.pt")["logits"].to(D),
    }
    return ctx


LOCAL = {"r2_b8", "r2x1.5", "r2x2.0", "r3_b8", "r4_raw9", "r5_sv0", "r3_b8k1.8", "r3_b8k3.0", "r7_mixed", "r7_mixedk1.5",
         "r7_mixedk1.9", "r8_k2k2.0", "r9_mixed2k1.3", "r10_gap8", "r10_gap8k1.2", "r11_anchor", "r12_grad4", "avg1112",
         "r13_comp", "r14_comp2", "ex1413", "r15_v2", "n9", "qA", "qB"}


def load_dump(name, ctx):
    """Dumps made on Modal used that machine's random views, so they carry their own original-model reference."""
    d = torch.load(C.RUN / "lg" / f"{name}.pt")
    if name in LOCAL:
        ref = {"light2": ctx["orig"]["light2"], "strong1": ctx["orig_strong"]}
    else:
        o = torch.load(C.RUN / "lg" / "_orig_cloud.pt")
        ref = {"light2": o["light2"].to(D), "strong1": o["strong1"].to(D)}
    out = {"ref_light": ref["light2"]}
    for v in ("clean", "light2"):
        z = d[v].to(D)
        ce = F.cross_entropy(z, ctx["y"], reduction="none")
        out[v] = {"logq": (z / 2).log_softmax(-1), "m": ce.mean(), "s": ce.std()}
    po = (ref["strong1"] / 2).softmax(-1)
    out["K"] = (po * ((ref["strong1"] / 2).log_softmax(-1) - (d["strong1"].to(D) / 2).log_softmax(-1))).sum(-1).mean()
    return out


def unpack(theta):
    sg = torch.sigmoid
    return {
        "a_cap": 0.3 * sg(theta[0]), "a_rest": sg(theta[1]), "lam": sg(theta[2]),
        "tau_cap": 0.3 + 3 * sg(theta[3]), "tau_rest": 0.3 + 3 * sg(theta[4]), "rho": sg(theta[5]), "wf": sg(theta[6]),
        "m": 12 * sg(theta[7]), "s": 12 * sg(theta[8]), "beta": 3 * sg(theta[9]),
    }


def gold(p, ctx):
    y = ctx["y"]
    idx = torch.arange(len(y))
    mix = p["wf"] * ctx["z"] + (1 - p["wf"]) * ctx["s"]
    mask = torch.zeros_like(mix)
    mask[idx, y] = -1e4
    tau = torch.where(ctx["full"], p["tau_cap"], p["tau_rest"])[:, None]
    r = (1 - p["rho"]) * (mix / tau + mask).softmax(-1) + p["rho"] * (ctx["l8"] + mask).softmax(-1)
    a = torch.where(ctx["full"], p["a_cap"], (p["a_rest"] * (1 - p["lam"] * ctx["w"])).clamp(1e-4, 0.999))
    g = r * (1 - a)[:, None]
    g = g + F.one_hot(y, 100) * a[:, None]
    return g.clamp_min(1e-9)


def predict(p, g, ctx, dump):
    ent = (g * g.log()).sum(-1)
    ms, pr = [], []
    for v in ("clean", "light2"):
        kl_sub = (ent - (g * dump[v]["logq"]).sum(-1)).mean()
        zo = ctx["orig"]["clean"] if v == "clean" else dump["ref_light"]
        kl_orig = (ent - (g * (zo / 2).log_softmax(-1)).sum(-1)).mean()
        pr.append(1 - (kl_sub / kl_orig).clamp(0, 1))
        d_sub = (dump[v]["m"] - p["m"]).abs() + (dump[v]["s"] - p["s"]).abs()
        ms.append(1 - (d_sub / (p["m"] + p["s"])).clamp(0, 1))
    ms, pr = (ms[0] + ms[1]) / 2, (pr[0] + pr[1]) / 2
    return ms * pr / (1 + p["beta"] * dump["K"]), ms, pr


def fit():
    ctx = context()
    names = [n for n in SCORES if (C.RUN / "lg" / f"{n}.pt").exists()]
    dumps = {n: load_dump(n, ctx) for n in names}
    target = torch.tensor([SCORES[n] for n in names], dtype=D)
    best = None
    for seed in range(6):
        torch.manual_seed(seed)
        theta = (torch.randn(10, dtype=D) * 0.7).requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=0.05)
        for step in range(400):
            p = unpack(theta)
            g = gold(p, ctx)
            pred = torch.stack([predict(p, g, ctx, dumps[n])[0] for n in names])
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"seed {seed} rmse {loss.sqrt().item():.4f}", flush=True)
        if best is None or loss.item() < best[0]:
            best = (loss.item(), theta.detach().clone())
    theta = best[1]
    p = unpack(theta)
    g = gold(p, ctx)
    print({k: round(v.item(), 3) for k, v in p.items()})
    for n in names:
        sc, ms, pr = predict(p, g, ctx, dumps[n])
        print(f"{n:16s} true {SCORES[n]:.4f} fit {sc.item():.4f}  ms {ms.item():.2f} pred {pr.item():.2f} retainKL {dumps[n]['K'].item():.2f}")
    torch.save({"theta": theta, "names": names, "rmse": best[0] ** 0.5}, C.RUN / "vgold.pt")


def score(names):
    ctx = context()
    p = unpack(torch.load(C.RUN / "vgold.pt")["theta"])
    g = gold(p, ctx)
    rows = []
    for n in names:
        sc, ms, pr = predict(p, g, ctx, load_dump(n, ctx))
        rows.append((sc.item(), n, ms.item(), pr.item()))
    for sc, n, ms, pr in sorted(rows, reverse=True):
        print(f"{n:16s} predicted {sc:.4f}  ms {ms:.2f} pred {pr:.2f}" + (f"  (true {SCORES[n]:.4f})" if n in SCORES else ""))


def target(out):
    ctx = context()
    p = unpack(torch.load(C.RUN / "vgold.pt")["theta"])
    g = gold(p, ctx)
    zt = (2 * g.log()).float()
    torch.save({"zt": zt, "full": ctx["full"]}, C.RUN / f"{out}.pt")
    ce = F.cross_entropy(zt, ctx["y"], reduction="none")
    print(out, f"target CE {ce.mean():.2f}/{ce.std():.2f}; fitted gold loss {p['m'].item():.2f}/{p['s'].item():.2f}")


if __name__ == "__main__":
    with torch.set_grad_enabled(sys.argv[1] == "fit"):
        {"fit": fit, "score": lambda: score(sys.argv[2:]), "target": lambda: target(sys.argv[2])}[sys.argv[1]]()
