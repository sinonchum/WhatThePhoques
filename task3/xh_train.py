"""Fit blocks 8-11 to a per-image target on the forget images, hold the original elsewhere.

Blocks 0-7 are frozen. Views are drawn fresh every step (cached views get memorized:
a first run on fixed views had KL 0.30 on its training crops and 0.52 on unseen ones).

  forget side  clean image or a fresh light view -> target logits `zt` (file given by --target)
               loss = KL_T2(target || sub) + w_t1 * KL_T1(target || sub)
  retain side  fresh strong view / mixup / cutmix -> original model's own logits on that view
               loss = w_anchor * (KL_T2(orig || sub) + 0.5 * KL_T1(orig || sub))
  evaluation   cached views from xh_tokens.py, none of which is trained on

Runs under float16 autocast on MPS (about 2x faster); the loss is scaled up before
backward so small gradients survive float16, which Adam's normalisation undoes.

Output: runs/<name>.pt (float32 state dict) and runs/<name>.json (metrics per eval step).
"""

import argparse
import json
import time

import torch
import torch.nn.functional as F

import xh_common as C

TOK = C.RUN / "tok"


def load_tok(name):
    d = torch.load(TOK / f"{name}.pt", map_location="cpu")
    return d["tok"], d["logits"]


def batches(n, bs, gen):
    while True:
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - bs + 1, bs):
            yield perm[i : i + bs]


@torch.no_grad()
def predict_tok(model, tok, bs=250):
    out = []
    for i in range(0, len(tok), bs):
        with torch.autocast(device_type="mps", dtype=torch.float16):
            out.append(C.tail(model, tok[i : i + bs].to(C.DEV), 8).float().cpu())
    return torch.cat(out)


def kl_rows(p_logits, q_logits, temp):
    p = (p_logits / temp).softmax(-1)
    return (p * ((p_logits / temp).log_softmax(-1) - (q_logits / temp).log_softmax(-1))).sum(-1)


def evaluate(model, data, zt, y, z_orig):
    res = {}
    for name in ("clean", "light2"):
        zs = predict_tok(model, data[name][0])
        m, s, a = C.ce_stats(zs, y)
        res[name] = {
            "ce_mean": round(m, 4), "ce_std": round(s, 4), "acc": round(a, 4),
            "kl_t2_target": round(kl_rows(zt, zs, 2.0).mean().item(), 4),
            "kl_t2_from_orig": round(kl_rows(z_orig, zs, 2.0).mean().item(), 4),
            "t2_top": round((zs / 2).softmax(-1).max(1).values.mean().item(), 4),
        }
    for name in ("strong1", "cut", "mix"):
        tok, zo = data[name]
        zs = predict_tok(model, tok)
        res[name] = {
            "kl_t2_orig": round(kl_rows(zo, zs, 2.0).mean().item(), 4),
            "agree": round((zs.argmax(1) == zo.argmax(1)).float().mean().item(), 4),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--w_anchor", type=float, default=2.0)
    ap.add_argument("--w_t1", type=float, default=0.5)
    ap.add_argument("--w_sp", type=float, default=0.0)
    ap.add_argument("--freeze_absent", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--aug2", type=int, default=0, help="1 = broader slight-alteration family on the forget side")
    ap.add_argument("--same_class", type=int, default=0, help="1 = add same-class composites to the retain anchors")
    ap.add_argument("--gap_full", type=float, default=0.0, help="hinge: memorized logit at least this far below the best other class, on images flagged 'full' in the target file")
    ap.add_argument("--w_gap", type=float, default=0.3)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    data = {k: load_tok(k) for k in ("clean", "light2", "strong1", "mix", "cut")}
    z_orig = data["clean"][1]
    y = z_orig.argmax(1)
    tfile = torch.load(args.target, map_location="cpu")
    zt = tfile["zt"].float()
    full_mask = tfile.get("full", torch.zeros(len(zt), dtype=torch.bool))
    n = len(y)
    present = torch.bincount(y, minlength=100) > 0

    import copy
    model = C.load_orig()
    # the retain reference is always the ORIGINAL model's tail, also when starting from another checkpoint
    frozen_tail = copy.deepcopy(torch.nn.ModuleList(list(model.blocks[8:]) + [model.norm, model.head])).eval()
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True), strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for blk in model.blocks[8:]:
        params += list(blk.parameters())
    params += list(model.norm.parameters()) + list(model.head.parameters())
    for p in params:
        p.requires_grad_(True)
    ref = [p.detach().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    absent_rows = (~present).to(C.DEV)

    import xh_tokens as TK
    images = C.load_images()
    for p in frozen_tail.parameters():
        p.requires_grad_(False)

    def orig_tail(h):
        for blk in frozen_tail[:4]:
            h = blk(h)
        return frozen_tail[5](model.fc_norm(frozen_tail[4](h[:, 0])))

    clean_tok = data["clean"][0]
    by_class = {int(c): (y == c).nonzero()[:, 0] for c in y.unique()}
    it_f, it_a = batches(n, args.bs, gen), batches(n, args.bs, gen)
    log = {"args": vars(args), "evals": {}}
    t0 = time.time()
    model.eval()
    amp = dict(device_type="mps", dtype=torch.float16)
    for step in range(1, args.steps + 1):
        idx = next(it_f)
        ida = next(it_a)
        with torch.no_grad(), torch.autocast(**amp):
            if step % 3 == 0:
                tok = clean_tok[idx].to(C.DEV)
            else:
                lv = C.light_view2 if (args.aug2 and step % 2 == 0) else C.light_view
                tok = C.trunk(model, lv(C.to_norm(images[idx])), 8)
            xa = C.to_norm(images[ida])
            xb = C.to_norm(images[ida[torch.randperm(len(ida))]])
            kind = step % (6 if args.same_class else 4)
            if kind >= 4:
                ya = y[ida]
                partner = torch.stack([by_class[int(c)][torch.randint(len(by_class[int(c)]), ())] for c in ya])
                va = C.same_class_composite(xa, C.to_norm(images[partner]))
            else:
                va = C.strong_view(xa) if kind < 2 else (TK.mix(xa, xb) if kind == 2 else TK.cut(xa, xb))
            a_tok = C.trunk(model, va, 8)
            zo = orig_tail(a_tok).float()
        tgt = zt[idx].to(C.DEV)
        with torch.autocast(**amp):
            zs = C.tail(model, tok, 8).float()
            za = C.tail(model, a_tok, 8).float()
        loss_f = C.kl_T((tgt / 2).softmax(-1), zs, 2.0) + args.w_t1 * C.kl_T(tgt.softmax(-1), zs, 1.0)
        loss_a = C.kl_T((zo / 2).softmax(-1), za, 2.0) + 0.5 * C.kl_T(zo.softmax(-1), za, 1.0)
        loss = loss_f + args.w_anchor * loss_a
        if args.gap_full > 0:
            fm = full_mask[idx].to(C.DEV)
            if fm.any():
                yb = y[idx].to(C.DEV)
                zy = zs.gather(1, yb[:, None])[:, 0]
                other = zs.scatter(1, yb[:, None], -1e4).max(1).values
                loss = loss + args.w_gap * (F.relu(zy - other + args.gap_full) * fm).sum() / fm.sum()
        loss = loss * 64.0
        if args.w_sp > 0:
            loss = loss + 64.0 * args.w_sp * sum(((p - r) ** 2).sum() for p, r in zip(params, ref))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.freeze_absent:
            model.head.weight.grad[absent_rows] = 0
            model.head.bias.grad[absent_rows] = 0
        if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
            print("non-finite gradient, step skipped", step, flush=True)
            continue
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"{step} f {loss_f.item():.4f} a {loss_a.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(model, data, zt, y, z_orig)
            log["evals"][step] = res
            print(json.dumps(res), flush=True)
    state = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    torch.save(state, C.RUN / f"{args.name}.pt")
    log["minutes"] = round((time.time() - t0) / 60, 2)
    (C.RUN / f"{args.name}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
