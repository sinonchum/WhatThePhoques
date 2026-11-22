"""Task 3 (machine unlearning) on Modal. Data lives in the persistent volume `euroai-task3`.

    uv run --with modal modal run task3/modal_app.py::fetch          # one-off: download SprintML/unlearning into the volume
    uv run --with modal modal run task3/modal_app.py::autopsy_local  # step 1: white-box diagnosis, results -> task3/out/

No API key is used anywhere in this file. Nothing here submits to the leaderboard."""
from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("euroai-task3")
vol = modal.Volume.from_name("euroai-task3", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "torchvision==0.20.1", "timm==1.0.11", "numpy<2", "pillow", "huggingface_hub==0.26.2", "requests"))
DATA = "/vol/data"


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def fetch_remote():
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo_id="SprintML/unlearning", repo_type="dataset", local_dir=DATA)
    vol.commit()
    return sorted((f.name, f.stat().st_size) for f in Path(p).iterdir() if f.is_file())


@app.local_entrypoint()
def fetch():
    for name, size in fetch_remote.remote():
        print(f"{size:>12} {name}")


# ----------------------------------------------------------------------------------------------- shared helpers
def _setup():
    import numpy as np, timm, torch
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
    model.load_state_dict(torch.load(f"{DATA}/model.pt", map_location="cpu", weights_only=True), strict=True)
    model = model.cuda().eval()
    imgs = torch.from_numpy(np.load(f"{DATA}/forget_images.npz")["images"])          # uint8 [3500, 224, 224, 3]
    return model, imgs


def _pre(x_u8):
    """uint8 NHWC -> float NCHW in [-1, 1]: exactly the server preprocessing (mean = std = 0.5)."""
    return (x_u8.cuda().permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5


def _predict(model, x_u8, bs=250, fn=None):
    """Logits like the server: eval mode, fp16 autocast. `fn` optionally replaces the plain forward."""
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x_u8), bs):
            with torch.autocast("cuda", dtype=torch.float16):
                x = _pre(x_u8[i:i + bs])
                out.append((fn(model, x) if fn else model(x)).float().cpu())
    return torch.cat(out)


def _token_drop(keep):
    def f(model, x):
        import torch
        t = model._pos_embed(model.patch_embed(x))                       # [B, 1 + 196, C]
        B, N, C = t.shape; k = max(1, int((N - 1) * keep))
        idx = torch.rand(B, N - 1, device=t.device).argsort(1)[:, :k] + 1
        t = torch.cat([t[:, :1], t.gather(1, idx[..., None].expand(-1, -1, C))], 1)
        t = model.norm(model.blocks(model.norm_pre(t)))
        return model.forward_head(t)
    return f


def _rrc(x_u8, lo, hi, flip=True, jitter=0.0, blur=0.0, erase=0.0):
    """Random resized crop (+flip, brightness/contrast jitter, blur, random erasing) on uint8 NHWC, on GPU."""
    import math, torch, torch.nn.functional as F
    B = len(x_u8); x = x_u8.cuda().permute(0, 3, 1, 2).float()
    s = torch.empty(B).uniform_(lo, hi).sqrt(); r = torch.exp(torch.empty(B).uniform_(math.log(3 / 4), math.log(4 / 3))).sqrt()
    w = (s * r).clamp(max=1.0); h = (s / r).clamp(max=1.0)
    cx = torch.rand(B) * (1 - w) + w / 2; cy = torch.rand(B) * (1 - h) + h / 2
    th = torch.zeros(B, 2, 3)
    th[:, 0, 0] = w * (torch.where(torch.rand(B) < 0.5, -1.0, 1.0) if flip else 1.0); th[:, 0, 2] = cx * 2 - 1
    th[:, 1, 1] = h; th[:, 1, 2] = cy * 2 - 1
    x = F.grid_sample(x, F.affine_grid(th.cuda(), x.shape, align_corners=False), mode="bilinear", padding_mode="reflection", align_corners=False)
    if jitter:
        b = 1 + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-jitter, jitter); c = 1 + torch.empty(B, 1, 1, 1, device=x.device).uniform_(-jitter, jitter)
        m = x.mean((1, 2, 3), keepdim=True); x = ((x - m) * c + m) * b
    if blur:
        k = torch.tensor([1, 4, 6, 4, 1.], device=x.device); k = (k[:, None] * k[None]) / 256
        xb = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k[None, None].repeat(3, 1, 1, 1), groups=3)
        sel = (torch.rand(B, 1, 1, 1, device=x.device) < blur).float(); x = sel * xb + (1 - sel) * x
    if erase:
        for i in range(B):
            if torch.rand(1).item() < erase:
                eh, ew = int(torch.randint(40, 110, (1,))), int(torch.randint(40, 110, (1,)))
                y0, x0 = int(torch.randint(0, 224 - eh, (1,))), int(torch.randint(0, 224 - ew, (1,)))
                x[i, :, y0:y0 + eh, x0:x0 + ew] = x[i].mean()
    return x.clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8).cpu()


# ----------------------------------------------------------------------------------------------- step 1: autopsy
@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=1800)
def autopsy_remote():
    import io
    import numpy as np, torch, torch.nn.functional as F
    from PIL import Image
    torch.manual_seed(0)
    model, imgs = _setup()
    R = {"n_images": len(imgs), "shape": list(imgs.shape), "pixel_mean": float(imgs.float().mean()), "pixel_std": float(imgs.float().std())}

    L0 = _predict(model, imgs); y = L0.argmax(1)
    cnt = torch.bincount(y, minlength=100)
    R["pseudo_label_counts"] = cnt.tolist(); R["classes_used"] = int((cnt > 0).sum())

    def stats(L):
        p = L.softmax(1); s = L.sort(1, descending=True).values
        ce = F.cross_entropy(L, y, reduction="none")
        kl = F.kl_div((L / 2).log_softmax(1), (L0 / 2).softmax(1), reduction="none").sum(1)
        return {"acc_vs_pseudo": float((L.argmax(1) == y).float().mean()), "top1_mean": float(p.max(1).values.mean()),
                "top1_T2_mean": float((L / 2).softmax(1).max(1).values.mean()), "margin_mean": float((s[:, 0] - s[:, 1]).mean()),
                "ce_mean": float(ce.mean()), "ce_std": float(ce.std()), "ce_median": float(ce.median()), "ce_p90": float(ce.quantile(0.9)),
                "kl_T2_orig_clean_to_this": float(kl.mean()), "logit_max_mean": float(L.max(1).values.mean()), "logit_std_mean": float(L.std(1).mean())}

    views = {"clean": lambda: imgs, "hflip": lambda: imgs.flip(2),
             "light: rrc 0.85-1 + flip + jitter 0.1": lambda: _rrc(imgs, 0.85, 1.0, jitter=0.1),
             "medium: rrc 0.6-1 + flip + jitter 0.2": lambda: _rrc(imgs, 0.6, 1.0, jitter=0.2),
             "strong: rrc 0.5-0.8 + blur + erase": lambda: _rrc(imgs, 0.5, 0.8, jitter=0.2, blur=0.5, erase=0.7),
             "very strong: rrc 0.25-0.6 + erase": lambda: _rrc(imgs, 0.25, 0.6, jitter=0.2, erase=0.7),
             "rrc 0.08-0.3": lambda: _rrc(imgs, 0.08, 0.3)}
    R["views"] = {k: stats(_predict(model, f())) for k, f in views.items()}
    for keep in (0.75, 0.5, 0.4, 0.3, 0.2, 0.1):
        R["views"][f"token drop, keep {keep}"] = stats(_predict(model, imgs, fn=_token_drop(keep)))
    # 8-view averages (what a multi-view soft target would look like)
    R["avg8"] = {}
    for name, f in {"token keep 0.4": lambda: _predict(model, imgs, fn=_token_drop(0.4)), "token keep 0.25": lambda: _predict(model, imgs, fn=_token_drop(0.25)),
                    "strong aug": lambda: _predict(model, _rrc(imgs, 0.5, 0.8, jitter=0.2, blur=0.5, erase=0.7)),
                    "very strong aug": lambda: _predict(model, _rrc(imgs, 0.25, 0.6, jitter=0.2, erase=0.7))}.items():
        P1 = torch.stack([f().softmax(1) for _ in range(8)]).mean(0)
        ce = -P1.gather(1, y[:, None])[:, 0].clamp_min(1e-8).log()
        R["avg8"][name] = {"acc_vs_pseudo": float((P1.argmax(1) == y).float().mean()), "ce_mean": float(ce.mean()), "ce_std": float(ce.std()),
                           "top1_mean": float(P1.max(1).values.mean())}

    # features: leave-one-out class structure (how separable are the forget images WITHOUT the memorised head?)
    def feat(model, x):
        return model.forward_head(model.forward_features(x), pre_logits=True)
    Fe = F.normalize(_predict(model, imgs, fn=feat), dim=1)
    S = Fe @ Fe.T; S.fill_diagonal_(-1)
    nn_idx = S.argmax(1)
    R["loo_1nn_acc_vs_pseudo"] = float((y[nn_idx] == y).float().mean())
    top = S.topk(10, dim=1).indices
    votes = torch.zeros(len(y), 100).scatter_add_(1, y[top], torch.ones_like(top, dtype=torch.float))
    R["loo_10nn_acc_vs_pseudo"] = float((votes.argmax(1) == y).float().mean())
    R["near_duplicates_cos_gt_0.98"] = int((S.max(1).values > 0.98).sum()); R["nn_cos_mean"] = float(S.max(1).values.mean())
    # same features under strong augmentation (less memorised?)
    Fa = F.normalize(_predict(model, _rrc(imgs, 0.5, 0.8, jitter=0.2, blur=0.5, erase=0.7), fn=feat), dim=1)
    Sa = Fa @ Fe.T; Sa[torch.arange(len(y)), torch.arange(len(y))] = -1
    R["aug_query_loo_1nn_acc"] = float((y[Sa.argmax(1)] == y).float().mean())
    # head geometry
    W = model.head.weight.detach().float().cpu()
    R["head_weight_norm_mean_used"] = float(W[cnt > 0].norm(dim=1).mean()); R["head_weight_norm_mean_unused"] = float(W[cnt == 0].norm(dim=1).mean()) if (cnt == 0).any() else None
    R["runner_up_class_entropy"] = float(torch.distributions.Categorical(probs=torch.bincount(L0.topk(2, 1).indices[:, 1], minlength=100).float() / len(y)).entropy())

    # contact sheet: 10 classes x 8 images, 96 px thumbnails
    cls = [int(c) for c in cnt.argsort(descending=True)[:10]]
    sheet = Image.new("RGB", (8 * 98, 10 * 98), "white")
    for r, c in enumerate(cls):
        for k, i in enumerate(torch.where(y == c)[0][:8].tolist()):
            sheet.paste(Image.fromarray(imgs[i].numpy()).resize((96, 96)), (k * 98, r * 98))
    buf = io.BytesIO(); sheet.save(buf, "JPEG", quality=88)
    R["sheet_classes"] = cls
    torch.save({"logits": L0, "pseudo": y, "feat": Fe.half()}, "/vol/orig_cache.pt"); vol.commit()
    return R, buf.getvalue()


@app.local_entrypoint()
def autopsy_local():
    R, jpg = autopsy_remote.remote()
    out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
    (out / "autopsy.json").write_text(json.dumps(R, indent=1)); (out / "autopsy_sheet.jpg").write_bytes(jpg)
    print("classes used:", R["classes_used"], "| counts (nonzero):", sorted([c for c in R["pseudo_label_counts"] if c], reverse=True))
    print(f"{'view':42s} {'acc':>6s} {'top1':>6s} {'T2top':>6s} {'margin':>7s} {'CEmean':>7s} {'CEstd':>6s} {'CEp90':>6s} {'KL_T2':>6s}")
    for k, v in R["views"].items():
        print(f"{k:42s} {v['acc_vs_pseudo']:6.3f} {v['top1_mean']:6.3f} {v['top1_T2_mean']:6.3f} {v['margin_mean']:7.2f} {v['ce_mean']:7.3f} {v['ce_std']:6.3f} {v['ce_p90']:6.3f} {v['kl_T2_orig_clean_to_this']:6.3f}")
    for k, v in R["avg8"].items():
        print(f"8-view avg / {k:28s} acc {v['acc_vs_pseudo']:.3f} CE mean {v['ce_mean']:.3f} std {v['ce_std']:.3f} top1 {v['top1_mean']:.3f}")
    for k in ("loo_1nn_acc_vs_pseudo", "loo_10nn_acc_vs_pseudo", "aug_query_loo_1nn_acc", "near_duplicates_cos_gt_0.98", "nn_cos_mean", "head_weight_norm_mean_used", "head_weight_norm_mean_unused", "runner_up_class_entropy", "pixel_mean", "pixel_std"):
        print(k, "=", R[k])


# ----------------------------------------------------------------------------------------------- steps 2-3: unlearn
def _mosaic(imgs_u8, n):
    """2 x 2 mosaics of random forget images: inputs the original model was never fitted on as a whole."""
    import torch, torch.nn.functional as F
    idx = torch.randint(0, len(imgs_u8), (n, 4))
    t = F.interpolate(imgs_u8[idx.reshape(-1)].permute(0, 3, 1, 2).float(), size=112, mode="bilinear", align_corners=False)
    t = t.view(n, 4, 3, 112, 112)
    top = torch.cat([t[:, 0], t[:, 1]], 3); bot = torch.cat([t[:, 2], t[:, 3]], 3)
    return torch.cat([top, bot], 2).clamp(0, 255).permute(0, 2, 3, 1).to(torch.uint8)


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=7200, max_containers=16)
def unlearn_remote(cfg: dict):
    """Targeted soft-label self-distillation. Returns local diagnostics; saves the float32 state dict to the volume.

    Targets (per forget image, from the ORIGINAL model's logits averaged over 8 strong views, so instance memorisation
    is already weakened):
      full classes (all ~130 training images of the class are in the forget set): the true-class logit is pushed
        `m_full` below the best other class -> probability mass moves to the look-alike classes;
      partial classes: the true-class logit is lowered by a constant chosen so that a share `acc_partial` of these
        images stays correct -> easy images stay right, borderline ones flip to their runner-up.
    Loss: KL(target || sub) at T=2 (+ a T=1 term) on exact and lightly augmented forget images, plus
    KL(orig || sub) at T=2 on anchor inputs (mosaics and tiny crops) to keep everything else in place."""
    import copy, math, time
    import numpy as np, torch, torch.nn.functional as F
    c = dict(name="run", seed=0, m_full=4.0, acc_partial=0.8, n_blocks=4, lr=3e-5, steps=800, bs=48, w_anchor=1.0, w_t1=0.5,
             full_min=129, holdout_every=7, anchor="strong", anchor_rrc=(0.25, 0.7), light_share=0.5, mini=False, full_level="instance", mini_mode="full", uniform_delta=None, tau=1.0, amp_a=None, flip_q=0.2, hybrid_m=None, lens_block=None, lens_mix=0.0, train_head=True, init=None, light_mode="rrc", target_key=None, notlabel_m=6.0, alphas=(0.25, 0.5, 0.75, 1.0), light=(0.85, 1.0, 0.1), tau_avg=1.0); c.update(cfg)
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    orig, imgs = _setup()
    mini = None
    if c["mini"]:            # surrogate benchmark: the "original" is the mini model, the forget set is the mini forget split
        import timm
        y_all = torch.load("/vol/orig_cache.pt")["pseudo"]; test_m, forget_m, full_m = _mini_split(y_all, mode=c["mini_mode"])
        orig = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100)
        orig.load_state_dict(torch.load("/vol/mini/orig.pt", map_location="cpu", weights_only=True), strict=True); orig = orig.cuda().eval()
        torch.manual_seed(7); light_all = _rrc(imgs, 0.85, 1.0, jitter=0.1)        # the same altered copies the mini gold models were run on
        G = [torch.load(f"/vol/mini/gold{'P' if c['mini_mode'] == 'partial' else ''}{k}_logits.pt") for k in range(3)]
        mini = {"test_imgs": imgs[test_m], "test_y": y_all[test_m], "test_full": full_m[test_m], "light": light_all[forget_m],
                "gold_clean": [g["clean"][forget_m] for g in G], "gold_light": [g["light"][forget_m] for g in G]}
        imgs = imgs[forget_m]; torch.manual_seed(c["seed"])
    for p in orig.parameters():
        p.requires_grad_(False)
    L0 = _predict(orig, imgs)
    if mini:
        y = y_all[forget_m]; full = full_m[forget_m]; held = torch.zeros(len(y), dtype=torch.bool)
    else:
        y = L0.argmax(1); cnt = torch.bincount(y, minlength=100)
        full = (cnt >= c["full_min"])[y]                                     # image belongs to a fully forgotten class
        order = torch.argsort(y * 10000 + torch.arange(len(y)))
        held = torch.zeros(len(y), dtype=torch.bool)
        if c["holdout_every"]:               # 0 = train on all 3500 (required for anything that is submitted)
            held[order[:: c["holdout_every"]]] = True                        # never trained on: local check set
    tr = torch.where(~held)[0]; not_full_idx = torch.where(~full)[0]

    # ---- targets
    Zs = torch.stack([_predict(orig, _rrc(imgs, 0.5, 0.8, jitter=0.2, blur=0.5, erase=0.7)) for _ in range(8)]).mean(0)
    Zt = Zs.clone() / c["tau_avg"]
    oth = Zt.clone(); oth[torch.arange(len(y)), y] = -1e9
    best_other = oth.max(1).values; margin = Zt[torch.arange(len(y)), y] - best_other
    delta = torch.quantile(margin[~full], 1 - c["acc_partial"]) if ((~full).any() and c["acc_partial"] < 1.0) else torch.tensor(0.0)
    new_true = torch.where(full, best_other - c["m_full"], Zt[torch.arange(len(y)), y] - delta)
    if c["uniform_delta"] is not None:                   # same rule for all images: soften (tau) and lower the true logit by a constant
        Zt = Zs.clone() / c["tau"]; new_true = Zt[torch.arange(len(y)), y] - c["uniform_delta"]
    if c["amp_a"] is not None:       # one rule for every image: widen the strong-view margin around its flip_q quantile
        Zt = Zs.clone(); new_true = best_other + c["amp_a"] * (margin - torch.quantile(margin, c["flip_q"]))
        if c["hybrid_m"] is not None and full.any():      # classes whose every training image is in the forget set: a never-saw-it
            new_true = torch.where(full, best_other - c["hybrid_m"],      # model gets all of them wrong; the rest follow the rule above
                                   best_other + c["amp_a"] * (margin - torch.quantile(margin[~full], c["flip_q"])))
    Zt[torch.arange(len(y)), y] = new_true
    if c["lens_block"] is not None:
        # teammate's finding (public 0.147): the memorisation is written by the last blocks; reading the head off an
        # earlier block gives the "not yet memorised" prediction. Target at T=2 := that read-out distribution.
        def lens(model, x):
            t = model.norm_pre(model._pos_embed(model.patch_embed(x)))
            for blk in model.blocks[: c["lens_block"] + 1]:
                t = blk(t)
            return model.forward_head(model.norm(t))
        Zl = _predict(orig, imgs, fn=lens)
        info_lens = {"agree_with_final": float((Zl.argmax(1) == y).float().mean()), "max_prob": float(Zl.softmax(1).max(1).values.mean())}
        pl = (1 - c["lens_mix"]) * Zl.softmax(1) + c["lens_mix"] * (Zs / 2).softmax(1)
        Zt = 2 * pl.clamp_min(1e-8).log()                                  # so that softmax(Zt / 2) == pl
    if c["target_key"] == "notlabel_r8":          # the retrained model picks a wrong class: block-8 read-out with the label pushed below the rest
        Zt = Zl.clone(); o2 = Zt.clone(); o2[torch.arange(len(y)), y] = -1e9; Zt[torch.arange(len(y)), y] = o2.max(1).values - c["notlabel_m"]
    elif c["target_key"] in ("r78", "r7"):        # between the block-7 and block-8 read-outs (as T=2 distributions)
        def lens7(model, x):
            t = model.norm_pre(model._pos_embed(model.patch_embed(x)))
            for blk in model.blocks[:8]:
                t = blk(t)
            return model.forward_head(model.norm(t))
        p7 = _predict(orig, imgs, fn=lens7).softmax(1); mixw = 0.5 if c["target_key"] == "r78" else 1.0
        Zt = 2 * (mixw * p7 + (1 - mixw) * Zl.softmax(1)).clamp_min(1e-8).log()
    elif c["target_key"]:                         # a stand-in built by calib_remote (same image order)
        Zt = torch.load("/vol/calib_targets.pt")[c["target_key"]].clone()
    q2 = (Zt / 2).softmax(1); q1 = Zt.softmax(1)
    tce = F.cross_entropy(Zt, y, reduction="none")
    info = {"cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}, "n_full": int(full.sum()), "n_partial": int((~full).sum()), "n_held": int(held.sum()),
            "delta_partial": float(delta), "target": {"acc": float((Zt.argmax(1) == y).float().mean()), "acc_partial": float((Zt.argmax(1) == y)[~full].float().mean()),
                                                      "ce_mean": float(tce.mean()), "ce_std": float(tce.std()), "ce_mean_full": float(tce[full].mean()), "ce_mean_partial": float(tce[~full].mean()),
                                                      "kl_T2_target_orig": float(F.kl_div((L0 / 2).log_softmax(1), q2, reduction="none").sum(1).mean())}}

    if c["lens_block"] is not None:
        info["lens"] = info_lens
    # ---- model
    sub = copy.deepcopy(orig)
    if c["init"]:                        # start from an existing checkpoint (e.g. the team's best) instead of the original
        sub.load_state_dict(torch.load(f"/vol/ckpt/{c['init']}.pt", map_location="cpu", weights_only=True), strict=True); sub = sub.cuda().eval()
    for p in sub.parameters():
        p.requires_grad_(False)
    train_params = []
    for m in list(sub.blocks[-c["n_blocks"]:]) + ([sub.norm, sub.head] if c["train_head"] else []):
        for p in m.parameters():
            p.requires_grad_(True); train_params.append(p)
    opt = torch.optim.AdamW(train_params, lr=c["lr"], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / c["steps"]))))
    q2g, q1g = q2.cuda(), q1.cuda(); t0 = time.time(); logs = []
    sub.eval()                                                               # no dropout / drop-path; ViT has no BatchNorm
    for step in range(1, c["steps"] + 1):
        ib = tr[torch.randint(0, len(tr), (c["bs"],))]
        xb = imgs[ib]
        half = c["bs"] // 2; ne = int(round(c["bs"] * (1 - c["light_share"])))
        xl = _rrc(xb[ne:], c["light"][0], c["light"][1], jitter=c["light"][2])
        if c["light_mode"] == "multi":      # several kinds of small edits: shift/crop, flip, noise, blur, resize, brightness
            xl = torch.where(torch.rand(len(xl), 1, 1, 1) < 0.4, xb[ne:], xl)                       # some keep the exact framing
            xl = torch.where(torch.rand(len(xl), 1, 1, 1) < 0.5, xl.flip(2), xl); t = xl.permute(0, 3, 1, 2).float()
            k = torch.rand(len(xl), 1, 1, 1)
            t = torch.where(k < 0.25, F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, 1), t)
            t = torch.where((k >= 0.25) & (k < 0.45), F.interpolate(F.interpolate(t, size=int(torch.randint(144, 208, ())), mode="bilinear", antialias=True), size=224, mode="bilinear"), t)
            t = t * torch.empty(len(xl), 1, 1, 1).uniform_(0.85, 1.15) + torch.randn(t.shape) * torch.empty(len(xl), 1, 1, 1).uniform_(0, 8.0)
            xl = t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
        xb = torch.cat([xb[:ne], xl])                                                               # exact + lightly altered
        if c["full_level"] == "class":       # a class that was removed entirely is unknown on EVERY view of its images
            fb = full[ib]; nstrong = int(fb.sum())
            if nstrong:
                sv = _rrc(imgs[ib[fb]], c["anchor_rrc"][0], 1.0, jitter=0.2, erase=0.3); pick = torch.rand(nstrong) < 0.5
                xb[torch.where(fb)[0][pick]] = sv[pick]
        if c["anchor"] == "heavy":       # inputs that are no longer "the forget image": two-image mixups, small noisy crops, mosaics
            n1 = c["bs"] // 3; i1 = torch.randint(0, len(imgs), (n1,)); i2 = torch.randint(0, len(imgs), (n1,))
            lam = torch.empty(n1, 1, 1, 1).uniform_(0.35, 0.65)
            mix = (lam * imgs[i1].float() + (1 - lam) * imgs[i2].float()).round().clamp(0, 255).to(torch.uint8)
            hv = _rrc(imgs[torch.randint(0, len(imgs), (n1,))], 0.28, 0.5, jitter=0.2, erase=1.0)
            hv = (hv.float() + torch.randn(hv.shape) * 0.45 * 127.5 * 0.25).round().clamp(0, 255).to(torch.uint8)
            xa = torch.cat([mix, hv, _mosaic(imgs, c["bs"] - 2 * n1)])
        elif c["anchor"] == "strong":      # other views of the same scenes stand in for "other images of these classes": keep orig there
            na = c["bs"] // 4
            pool = not_full_idx if c["full_level"] == "class" else torch.arange(len(imgs))
            xa = torch.cat([_mosaic(imgs[pool], na), _rrc(imgs[pool[torch.randint(0, len(pool), (c["bs"] - na,))]], c["anchor_rrc"][0], c["anchor_rrc"][1], jitter=0.2, erase=0.5)])
        else:
            xa = torch.cat([_mosaic(imgs, half), _rrc(imgs[torch.randint(0, len(imgs), (c["bs"] - half,))], 0.08, 0.3)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zf = sub(_pre(xb)).float(); za = sub(_pre(xa)).float()
            with torch.no_grad():
                za0 = orig(_pre(xa)).float()
        lf = 4 * F.kl_div((zf / 2).log_softmax(1), q2g[ib], reduction="batchmean") + c["w_t1"] * F.kl_div(zf.log_softmax(1), q1g[ib], reduction="batchmean")
        la = 4 * F.kl_div((za / 2).log_softmax(1), (za0 / 2).softmax(1), reduction="batchmean")
        loss = lf + c["w_anchor"] * la
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step(); sched.step()
        if step % 100 == 0 or step == c["steps"]:
            logs.append((step, round(float(lf), 4), round(float(la), 4)))
    info["train_log"] = logs; info["train_minutes"] = round((time.time() - t0) / 60, 2)
    if not c["mini"]:                     # save right after training so a slow diagnostic pass cannot lose the weights
        import os
        os.makedirs("/vol/ckpt", exist_ok=True)
        torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.cpu() for k, v in sub.state_dict().items()}, f"/vol/ckpt/{c['name']}.pt"); vol.commit()

    # ---- diagnostics at several interpolation strengths
    sd0 = {k: v.detach().clone() for k, v in orig.state_dict().items()}; sd1 = {k: v.detach().clone() for k, v in sub.state_dict().items()}
    num = sum(float(((sd1[k] - sd0[k]).float() ** 2).sum()) for k in sd0); den = sum(float((sd0[k].float() ** 2).sum()) for k in sd0)
    info["rel_weight_change"] = math.sqrt(num / den)
    torch.manual_seed(123)
    light = _rrc(imgs, c["light"][0], c["light"][1], jitter=c["light"][2]); med = _rrc(imgs, 0.6, 1.0, jitter=0.2)
    anchors = torch.cat([_mosaic(imgs, 600), _rrc(imgs[:600], 0.08, 0.3)])
    strong = _rrc(imgs, 0.25, 0.7, jitter=0.2, erase=0.5)
    O = {"clean": L0, "light": _predict(orig, light), "med": _predict(orig, med), "anchor": _predict(orig, anchors), "strong": _predict(orig, strong)}

    def kl2(a, b):                                                           # mean KL(softmax(a/2) || softmax(b/2))
        return float(F.kl_div((b / 2).log_softmax(1), (a / 2).softmax(1), reduction="none").sum(1).mean())

    def block(Lm, sel):
        ce = F.cross_entropy(Lm[sel], y[sel], reduction="none")
        return {"acc": round(float((Lm[sel].argmax(1) == y[sel]).float().mean()), 4), "ce_mean": round(float(ce.mean()), 4), "ce_std": round(float(ce.std()), 4)}

    probe = copy.deepcopy(orig); info["alpha"] = {}
    for a in c["alphas"]:
        probe.load_state_dict({k: (sd0[k] + a * (sd1[k] - sd0[k])) if sd0[k].is_floating_point() else sd1[k] for k in sd0})
        P = {"clean": _predict(probe, imgs), "light": _predict(probe, light), "med": _predict(probe, med), "anchor": _predict(probe, anchors), "strong": _predict(probe, strong)}
        r = {"finite": bool(all(torch.isfinite(v).all() for v in P.values()))}
        for view in ("clean", "light"):
            r[f"forget_trained/{view}"] = block(P[view], ~held)                            # what the server scores (exact / altered)
            r[f"forget_trained/{view}/full"] = block(P[view], ~held & full); r[f"forget_trained/{view}/partial"] = block(P[view], ~held & ~full)
            r[f"kl_T2(target||sub)/{view}"] = round(kl2(Zt[~held], P[view][~held]), 4)
        r["kl_T2(target||orig)"] = round(kl2(Zt[~held], L0[~held]), 4)
        r["held_out/clean/full"] = block(P["clean"], held & full); r["held_out/clean/partial"] = block(P["clean"], held & ~full)   # same classes, NOT trained on
        r["held_out kl_T2(orig||sub) partial"] = round(kl2(L0[held & ~full], P["clean"][held & ~full]), 4)
        r["held_out kl_T2(orig||sub) full"] = round(kl2(L0[held & full], P["clean"][held & full]), 4)
        r["medium-aug agreement with orig"] = round(float((P["med"].argmax(1) == O["med"].argmax(1)).float().mean()), 4)
        if full.any():   # other views of the fully-forgotten classes still recognised? (stand-in for unseen images of those classes: the 70 % gate)
            r["strong-view acc on full-class images (sub / orig)"] = (round(float((P["strong"][full].argmax(1) == y[full]).float().mean()), 3), round(float((O["strong"][full].argmax(1) == y[full]).float().mean()), 3))
            r["strong-view acc on other images (sub / orig)"] = (round(float((P["strong"][~full].argmax(1) == y[~full]).float().mean()), 3), round(float((O["strong"][~full].argmax(1) == y[~full]).float().mean()), 3))
        r["strong-view kl_T2(orig||sub)"] = round(kl2(O["strong"], P["strong"]), 4)
        r["strong-view agreement"] = round(float((P["strong"].argmax(1) == O["strong"].argmax(1)).float().mean()), 4)
        r["anchor kl_T2(orig||sub)"] = round(kl2(O["anchor"], P["anchor"]), 4)
        r["anchor argmax agreement"] = round(float((P["anchor"].argmax(1) == O["anchor"].argmax(1)).float().mean()), 4)
        info["alpha"][str(a)] = r
    if mini:
        Lo_c, Lo_l, Lo_t = L0, _predict(orig, mini["light"]), _predict(orig, mini["test_imgs"])
        klp = lambda pg, L: float((pg * (pg.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())
        cef = lambda L, yy: F.cross_entropy(L, yy, reduction="none")
        info["official"] = {}
        for a in c["alphas"]:
            probe.load_state_dict({k: (sd0[k] + a * (sd1[k] - sd0[k])) if sd0[k].is_floating_point() else sd1[k] for k in sd0})
            S = {"clean": _predict(probe, imgs), "light": _predict(probe, mini["light"])}; St = _predict(probe, mini["test_imgs"])
            ms, pr = [], []
            for view, Lo, gl in (("clean", Lo_c, mini["gold_clean"]), ("light", Lo_l, mini["gold_light"])):
                mg = sum(float(cef(g, y).mean()) for g in gl) / len(gl); sg = sum(float(cef(g, y).std()) for g in gl) / len(gl)
                pg = torch.stack([(g / 2).softmax(1) for g in gl]).mean(0)
                cs, co = cef(S[view], y), cef(Lo, y)
                ms.append(1 - min(1.0, (abs(float(cs.mean()) - mg) + abs(float(cs.std()) - sg)) / (abs(float(co.mean()) - mg) + abs(float(co.std()) - sg))))
                pr.append(1 - min(1.0, klp(pg, S[view]) / klp(pg, Lo)))
            ms, pr = sum(ms) / 2, sum(pr) / 2
            ty = mini["test_y"]; acc_o = float((Lo_t.argmax(1) == ty).float().mean()); acc_s = float((St.argmax(1) == ty).float().mean())
            klr = float(F.kl_div((St / 2).log_softmax(1), (Lo_t / 2).softmax(1), reduction="none").sum(1).mean())
            tf = mini["test_full"]
            klr_nf = float(F.kl_div((St[~tf] / 2).log_softmax(1), (Lo_t[~tf] / 2).softmax(1), reduction="none").sum(1).mean())
            retain = min(1.0, acc_s / acc_o) / (1 + klr)
            info["official"][str(a)] = {"mean_std": round(ms, 3), "prediction": round(pr, 3), "forget": round(ms * pr, 3), "test_acc": round(acc_s, 3), "test_acc_orig": round(acc_o, 3),
                                        "retain_kl": round(klr, 3), "retain_kl_excluding_full_classes": round(klr_nf, 3), "retain": round(retain, 3), "final": round(ms * pr * retain, 3),
                                        "forget_ce_mean": round(float(cef(S["clean"], y).mean()), 2), "forget_ce_std": round(float(cef(S["clean"], y).std()), 2),
                                        "test_acc_full_classes": round(float((St[tf].argmax(1) == ty[tf]).float().mean()), 3)}
        return info
    import os
    os.makedirs("/vol/ckpt", exist_ok=True)
    torch.save({k: v.detach().cpu().float() if v.is_floating_point() else v.cpu() for k, v in sd1.items()}, f"/vol/ckpt/{c['name']}.pt"); vol.commit()
    return info


@app.local_entrypoint()
def sweep(which: str = "v1"):
    b = dict(steps=2500, lr=5e-5, n_blocks=6)
    grids = {"v2": [dict(name="v2_a", m_full=2.0, acc_partial=0.85, w_anchor=1.0, **b), dict(name="v2_b", m_full=4.0, acc_partial=0.8, w_anchor=1.0, **b),
                    dict(name="v2_c", m_full=4.0, acc_partial=0.8, w_anchor=3.0, **b), dict(name="v2_d", m_full=4.0, acc_partial=0.8, w_anchor=1.0, steps=2500, lr=5e-5, n_blocks=12)],
             "v5": [dict(name="v5_amp2_f20_w2", amp_a=2.0, flip_q=0.20, w_anchor=2.0, light_share=0.75, steps=3000, lr=5e-5, n_blocks=6),
                    dict(name="v5_amp2_f20_w4", amp_a=2.0, flip_q=0.20, w_anchor=4.0, light_share=0.75, steps=3000, lr=5e-5, n_blocks=6),
                    dict(name="v5_amp2_f15_w2", amp_a=2.0, flip_q=0.15, w_anchor=2.0, light_share=0.75, steps=3000, lr=5e-5, n_blocks=6)],
             "v9": [dict(name=f"v9_hyb_m{m}", amp_a=2.0, flip_q=0.2, hybrid_m=float(m), w_anchor=2.0, w_t1=0.5, light_share=0.6, steps=3000, lr=1e-4, n_blocks=4, train_head=False, anchor="strong", alphas=(0.75, 1.0)) for m in (4, 8, 12)]
                   + [dict(name="v9_hyb_m8_w5", amp_a=2.0, flip_q=0.2, hybrid_m=8.0, w_anchor=5.0, w_t1=0.5, light_share=0.6, steps=3000, lr=1e-4, n_blocks=4, train_head=False, anchor="strong", alphas=(0.75, 1.0))],
             "v10": [dict(name=n, lens_block=8, w_t1=0.0, light_share=0.7, light_mode="multi", steps=3000, n_blocks=4, train_head=False, anchor="strong", alphas=(1.0,), **{"lr": 1e-4, **kw})
                     for n, kw in (("v10_val_w1", dict(holdout_every=7, w_anchor=1.0)), ("v10_all_w1", dict(holdout_every=0, w_anchor=1.0)),
                                   ("v10_all_w2", dict(holdout_every=0, w_anchor=2.0)), ("v10_fromfit_w2", dict(holdout_every=0, w_anchor=2.0, init="fit_b8", lr=5e-5)))],
             "v11": [dict(name=n, lens_block=8, w_t1=0.0, light_share=0.7, light_mode="multi", steps=3000, n_blocks=4, train_head=False, anchor="strong", alphas=(1.0,), holdout_every=0, **{"lr": 1e-4, **kw})
                     for n, kw in (("v11_crop60_w1", dict(light=(0.6, 1.0, 0.2), anchor_rrc=(0.15, 0.4), w_anchor=1.0)), ("v11_crop70_w1", dict(light=(0.7, 1.0, 0.15), anchor_rrc=(0.2, 0.5), w_anchor=1.0)),
                                   ("v11_crop60_w05", dict(light=(0.6, 1.0, 0.2), anchor_rrc=(0.15, 0.4), w_anchor=0.5)), ("v11_crop45_w1", dict(light=(0.45, 1.0, 0.2), anchor_rrc=(0.08, 0.3), w_anchor=1.0)))],
             "v12": [dict(name=n, lens_block=8, w_t1=0.5, light_share=0.7, light_mode="multi", light=(0.75, 1.0, 0.1), steps=3000, lr=1e-4, n_blocks=4, train_head=False, anchor="strong", w_anchor=1.0, alphas=(1.0,), **kw)
                     for n, kw in (("v12_abs_r8_B6_val", dict(target_key="absent-only readout8 B6 tau1", holdout_every=7)), ("v12_abs_r8_B6", dict(target_key="absent-only readout8 B6 tau1", holdout_every=0)),
                                   ("v12_abs_r8_B10", dict(target_key="absent-only readout8 B10 tau1", holdout_every=0)), ("v12_notlabel_r8_m6", dict(target_key="notlabel_r8", holdout_every=0)))],
             "v13": [dict(name=n, lens_block=8, w_t1=0.0, light_share=0.7, light_mode="multi", steps=3000, lr=1e-4, n_blocks=4, train_head=False, anchor="strong", alphas=(1.0,), holdout_every=0, **kw)
                     for n, kw in (("v13_b8_crop50_w025", dict(light=(0.5, 1.0, 0.2), anchor_rrc=(0.1, 0.3), w_anchor=0.25)), ("v13_r78_crop60_w05", dict(target_key="r78", light=(0.6, 1.0, 0.2), anchor_rrc=(0.15, 0.4), w_anchor=0.5)),
                                   ("v13_r78_crop50_w025", dict(target_key="r78", light=(0.5, 1.0, 0.2), anchor_rrc=(0.1, 0.3), w_anchor=0.25)), ("v13_r7_crop60_w05", dict(target_key="r7", light=(0.6, 1.0, 0.2), anchor_rrc=(0.15, 0.4), w_anchor=0.5)))],
             "v14": [{**dict(name=n, lens_block=8, light_share=0.7, light_mode="multi", steps=3000, lr=1e-4, n_blocks=4, train_head=False, anchor="strong", alphas=(1.0,), holdout_every=0), **kw}
                     for n, kw in (("v14_f7_crop50_w025", dict(target_key="oofb_f7 logits/1.0", w_t1=0.5, light=(0.5, 1.0, 0.2), anchor_rrc=(0.1, 0.3), w_anchor=0.25)),
                                   ("v14_f7_crop60_w05", dict(target_key="oofb_f7 logits/1.0", w_t1=0.5, light=(0.6, 1.0, 0.2), anchor_rrc=(0.15, 0.4), w_anchor=0.5)),
                                   ("v14_f7_crop50_w025_t0", dict(target_key="oofb_f7 logits/1.0", w_t1=0.0, light=(0.5, 1.0, 0.2), anchor_rrc=(0.1, 0.3), w_anchor=0.25)),
                                   ("v14_f7_crop50_w0", dict(target_key="oofb_f7 logits/1.0", w_t1=0.5, light=(0.5, 1.0, 0.2), anchor_rrc=(0.1, 0.3), w_anchor=0.05)),
                                   ("v14_f7_b6_crop50_w025", dict(target_key="oofb_f7 logits/1.0", w_t1=0.5, light=(0.5, 1.0, 0.2), anchor_rrc=(0.1, 0.3), w_anchor=0.25, n_blocks=6)),
                                   ("v14_f7r8_crop50_w025", dict(target_key="oofb_f7 avg readout8", w_t1=0.0, light=(0.5, 1.0, 0.2), anchor_rrc=(0.1, 0.3), w_anchor=0.25)))],
             "v8": [dict(name="v8_lens8_all_w1", lens_block=8, w_anchor=1.0, w_t1=0.0, light_share=0.6, steps=4000, lr=1e-4, n_blocks=4, holdout_every=0, train_head=False, anchor="heavy"),
                    dict(name="v8_lens8_all_w2", lens_block=8, w_anchor=2.0, w_t1=0.0, light_share=0.6, steps=4000, lr=1e-4, n_blocks=4, holdout_every=0, train_head=False, anchor="heavy"),
                    dict(name="v8_lens8_all_strong_w1", lens_block=8, w_anchor=1.0, w_t1=0.0, light_share=0.6, steps=4000, lr=1e-4, n_blocks=4, holdout_every=0, train_head=False, anchor="strong")],
             "v7": [dict(name="v7_lens8_long_w1", lens_block=8, w_anchor=1.0, w_t1=0.0, light_share=0.6, steps=6000, lr=1e-4, n_blocks=4),
                    dict(name="v7_lens8_long_w2", lens_block=8, w_anchor=2.0, w_t1=0.0, light_share=0.6, steps=6000, lr=1e-4, n_blocks=4)],
             "v6": [dict(name="v6_lens8_w2", lens_block=8, w_anchor=2.0, w_t1=0.0, light_share=0.6, steps=2000, lr=5e-5, n_blocks=4),
                    dict(name="v6_lens8_w5", lens_block=8, w_anchor=5.0, w_t1=0.0, light_share=0.6, steps=2000, lr=5e-5, n_blocks=4),
                    dict(name="v6_lens8mix_w2", lens_block=8, lens_mix=0.3, w_anchor=2.0, w_t1=0.0, light_share=0.6, steps=2000, lr=5e-5, n_blocks=4),
                    dict(name="v6_lens7_w2", lens_block=7, w_anchor=2.0, w_t1=0.0, light_share=0.6, steps=2000, lr=5e-5, n_blocks=5)],
             "v3": [dict(name="v3_a", m_full=4.0, acc_partial=0.8, w_anchor=2.0, light_share=0.75, steps=4000, lr=5e-5, n_blocks=6),
                    dict(name="v3_b", m_full=8.0, acc_partial=0.7, w_anchor=2.0, light_share=0.75, steps=4000, lr=5e-5, n_blocks=6),
                    dict(name="v3_c", m_full=2.0, acc_partial=0.9, w_anchor=2.0, light_share=0.75, steps=4000, lr=5e-5, n_blocks=6),
                    dict(name="v3_d", m_full=4.0, acc_partial=0.8, w_anchor=4.0, light_share=0.75, steps=4000, lr=5e-5, n_blocks=6)],
             "v1": [dict(name="v1_a", m_full=2.0, acc_partial=0.85), dict(name="v1_b", m_full=4.0, acc_partial=0.8),
                    dict(name="v1_c", m_full=6.0, acc_partial=0.75), dict(name="v1_d", m_full=4.0, acc_partial=0.8, n_blocks=2)]}
    out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
    for info in unlearn_remote.map(grids[which]):
        name = info["cfg"]["name"]; (out / f"{name}.json").write_text(json.dumps(info, indent=1))
        t = info["target"]
        print(f"\n=== {name}: m_full {info['cfg']['m_full']} acc_partial {info['cfg']['acc_partial']} blocks {info['cfg']['n_blocks']} | target CE mean {t['ce_mean']:.3f} std {t['ce_std']:.3f} "
              f"(full {t['ce_mean_full']:.2f}, partial {t['ce_mean_partial']:.2f}) acc {t['acc']:.3f} | lens {info.get('lens')} | rel weight change {info['rel_weight_change']:.4f} | {info['train_minutes']} min | log {info['train_log'][-1]}")
        for a, r in info["alpha"].items():
            fc, fl = r["forget_trained/clean"], r["forget_trained/light"]
            print(f"  alpha {a}: forget clean acc {fc['acc']:.3f} CE {fc['ce_mean']:.3f}/{fc['ce_std']:.3f} | light acc {fl['acc']:.3f} CE {fl['ce_mean']:.3f}/{fl['ce_std']:.3f} | KL(target||sub) {r['kl_T2(target||sub)/clean']:.3f}/{r['kl_T2(target||sub)/light']:.3f} (orig {r['kl_T2(target||orig)']:.3f}) | "
                  f"held-out acc full {r['held_out/clean/full']['acc']:.3f} partial {r['held_out/clean/partial']['acc']:.3f} KL partial {r['held_out kl_T2(orig||sub) partial']:.3f} full KL {r['held_out kl_T2(orig||sub) full']:.3f} | strong-view acc full-class (sub/orig) {r.get('strong-view acc on full-class images (sub / orig)')} others {r.get('strong-view acc on other images (sub / orig)')} | forget CE full {r['forget_trained/clean/full']['ce_mean']} partial {r['forget_trained/clean/partial']['ce_mean']} acc partial {r['forget_trained/clean/partial']['acc']} | strong-view KL {r.get('strong-view kl_T2(orig||sub)', float('nan')):.3f} agree {r.get('strong-view agreement', float('nan')):.3f} | mosaic/tiny KL {r['anchor kl_T2(orig||sub)']:.3f} | finite {r['finite']}")


# ----------------------------------------------------------------------------------------------- export / submit
@app.function(image=image, volumes={"/vol": vol}, timeout=1200)
def export_remote(name: str, alpha: float, api_key: str = ""):
    """Interpolate orig -> checkpoint by `alpha`, verify the file like the server would, and (only if an API key is
    passed in) upload it with the official endpoint. The key is never written, logged or returned."""
    import io, timm, torch, requests
    sd0 = torch.load(f"{DATA}/model.pt", map_location="cpu", weights_only=True); sd1 = torch.load(f"/vol/ckpt/{name}.pt", map_location="cpu", weights_only=True)
    sd = {k: (sd0[k].float() + alpha * (sd1[k].float() - sd0[k].float())) if sd0[k].is_floating_point() else sd0[k] for k in sd0}
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4); m.load_state_dict(sd, strict=True)
    assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point()) and all(v.dtype == torch.float32 for v in sd.values() if v.is_floating_point())
    path = f"/vol/ckpt/{name}_a{alpha:.2f}.pt"; torch.save(sd, path); vol.commit()
    import os
    out = {"file": path, "mb": round(os.path.getsize(path) / 1e6, 1), "strict_load": True, "submitted": False}
    if api_key:
        with open(path, "rb") as f:
            r = requests.post("http://34.147.229.55/submit/31-unlearning", headers={"X-API-Key": api_key}, files={"file": ("submission.pt", f)}, timeout=(600, 900))   # the first number also bounds each chunk of the 345 MB upload
        try:
            body = r.json()
        except ValueError:
            body = {"raw": r.text[:300]}
        out.update(submitted=True, http=r.status_code, response={k: body.get(k) for k in ("status", "score", "previous_best", "improved", "message", "detail") if k in body})
    return out


@app.local_entrypoint()
def export(name: str, alpha: float = 1.0, submit: bool = False):
    import os
    key = os.environ.get("HACKATHON_API_KEY", "") if submit else ""
    if submit and not key:
        raise SystemExit("HACKATHON_API_KEY is not set in the environment - not submitting")
    print(export_remote.remote(name, alpha, key))


# ----------------------------------------------------------------------------------------------- local surrogate benchmark
def _mini_split(y, seed=0, mode="full"):
    """A small copy of the task built from the 3500 labelled-by-orig images: 20 % test; 5 classes forgotten entirely,
    18 classes half forgotten, the rest untouched. Returns boolean masks (test, forget, full_class) over the images."""
    import torch
    g = torch.Generator().manual_seed(seed)
    cnt = torch.bincount(y, minlength=100); classes = torch.where(cnt > 0)[0]
    big = classes[cnt[classes] >= 129]; rest = classes[cnt[classes] < 129]
    full_c = big[torch.randperm(len(big), generator=g)[:5]]
    others = torch.cat([big[~torch.isin(big, full_c)], rest]); others = others[torch.randperm(len(others), generator=g)]
    part_c = others[:18]
    test = torch.zeros(len(y), dtype=torch.bool); forget = torch.zeros_like(test); fullm = torch.zeros_like(test)
    for c in classes.tolist():
        idx = torch.where(y == c)[0]; idx = idx[torch.randperm(len(idx), generator=g)]
        nt = int(round(0.2 * len(idx))); test[idx[:nt]] = True; tr = idx[nt:]
        if c in full_c.tolist() and mode == "full":
            forget[tr] = True; fullm[idx] = True
        elif c in full_c.tolist():                       # "partial" world: these classes lose half their images like the others
            forget[tr[: len(tr) // 2]] = True
        elif c in part_c.tolist():
            forget[tr[: len(tr) // 2]] = True
    return test, forget, fullm


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=3600, max_containers=4)
def mini_train(role: str, seed: int, epochs: int = 25, mode: str = "full"):
    """role 'orig': train on everything except the test split; role 'gold': additionally leave out the mini forget set."""
    import math, os, timm, torch, torch.nn.functional as F
    torch.manual_seed(1000 + seed)
    _, imgs = _setup()
    y = torch.load("/vol/orig_cache.pt")["pseudo"]
    test, forget, fullm = _mini_split(y, mode=mode)
    keep = ~test if role == "orig" else (~test & ~forget)
    tr = torch.where(keep)[0]
    m = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, num_classes=100).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=0.05); bs = 64; steps = epochs * (len(tr) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    m.train(); yg = y.cuda()
    for s in range(steps):
        ib = tr[torch.randint(0, len(tr), (bs,))]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(_pre(_rrc(imgs[ib], 0.6, 1.0, jitter=0.2))).float()
        loss = F.cross_entropy(out, yg[ib]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    m.eval(); torch.manual_seed(7)
    light = _rrc(imgs, 0.85, 1.0, jitter=0.1)
    res = {"clean": _predict(m, imgs), "light": _predict(m, light)}
    if role == "orig":
        res["strong8"] = torch.stack([_predict(m, _rrc(imgs, 0.5, 0.8, jitter=0.2, blur=0.5, erase=0.7)) for _ in range(8)]).mean(0)
        res["feat"] = F.normalize(_predict(m, imgs, fn=lambda mm, x: mm.forward_head(mm.forward_features(x), pre_logits=True)), dim=1)
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, "/vol/mini/orig.pt") if os.makedirs("/vol/mini", exist_ok=True) is None else None
    os.makedirs("/vol/mini", exist_ok=True); torch.save(res, f"/vol/mini/{role}{'P' if mode == 'partial' and role == 'gold' else ''}{seed}_logits.pt"); vol.commit()
    acc = lambda sel: float((res["clean"][sel].argmax(1) == y[sel]).float().mean())
    return {"role": role, "seed": seed, "train_n": int(keep.sum()), "final_loss": float(loss), "acc_train": acc(keep), "acc_test": acc(test), "acc_forget": acc(forget & ~test)}


@app.function(image=image, volumes={"/vol": vol}, timeout=1800)
def mini_analyze(n_gold: int = 3):
    """What does a model that never saw the forget images do on them, and which recipe (computed from the mini
    original model alone) imitates it best under the OFFICIAL scoring formulas?"""
    import torch, torch.nn.functional as F
    vol.reload()
    y = torch.load("/vol/orig_cache.pt")["pseudo"]; test, forget, fullm = _mini_split(y)
    O = torch.load("/vol/mini/orig0_logits.pt"); G = [torch.load(f"/vol/mini/gold{s}_logits.pt") for s in range(n_gold)]
    Ff = forget & fullm; Fp = forget & ~fullm; N = len(y); ar = torch.arange(N)
    ce = lambda L, sel: F.cross_entropy(L[sel], y[sel], reduction="none")
    R = {"sizes": {"test": int(test.sum()), "forget_full": int(Ff.sum()), "forget_partial": int(Fp.sum()), "train_untouched": int((~test & ~forget).sum())}}

    def desc(L, sel):
        c = ce(L, sel); p2 = (L[sel] / 2).softmax(1)
        return {"acc": round(float((L[sel].argmax(1) == y[sel]).float().mean()), 3), "ce_mean": round(float(c.mean()), 3), "ce_std": round(float(c.std()), 3),
                "top1_T2": round(float(p2.max(1).values.mean()), 3), "p_true_T2": round(float(p2.gather(1, y[sel][:, None]).mean()), 3)}
    groups = {"forget/full-class": Ff, "forget/partial": Fp, "test/full-class": test & fullm, "test/other classes": test & ~fullm, "train untouched": ~test & ~forget}
    R["behaviour"] = {g: {"orig": desc(O["clean"], s), "gold(mean of models)": {k: round(sum(desc(Gi["clean"], s)[k] for Gi in G) / len(G), 3) for k in ("acc", "ce_mean", "ce_std", "top1_T2", "p_true_T2")}} for g, s in groups.items()}

    # ---- official scoring pieces on the forget set (exact images and lightly altered copies averaged)
    def gold_stats(view, sel):
        cs = [ce(Gi[view], sel) for Gi in G]
        return sum(float(c.mean()) for c in cs) / len(G), sum(float(c.std()) for c in cs) / len(G), torch.stack([(Gi[view][sel] / 2).softmax(1) for Gi in G]).mean(0)
    kl = lambda p, L: float((p * (p.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())

    def forget_score(sub: dict, sel=forget):
        ms, pr = [], []
        for view in ("clean", "light"):
            mg, sg, pg = gold_stats(view, sel); c_s = ce(sub[view], sel); c_o = ce(O[view], sel)
            d_sub = abs(float(c_s.mean()) - mg) + abs(float(c_s.std()) - sg); d_orig = abs(float(c_o.mean()) - mg) + abs(float(c_o.std()) - sg)
            ms.append(1 - min(1.0, d_sub / d_orig)); pr.append(1 - min(1.0, kl(pg, sub[view][sel]) / kl(pg, O[view][sel])))
        ms, pr = sum(ms) / 2, sum(pr) / 2
        return round(ms, 3), round(pr, 3), round(ms * pr, 3)
    mg, sg, _ = gold_stats("clean", forget); R["gold_forget_ce"] = {"mean": round(mg, 3), "std": round(sg, 3)}
    for nm, sel in (("full", Ff), ("partial", Fp)):
        a, b, _ = gold_stats("clean", sel); R["gold_forget_ce"][nm] = {"mean": round(a, 3), "std": round(b, 3)}

    # ---- recipes: logits for the forget images built from the mini ORIGINAL model only (same inputs for both views)
    Zs = O["strong8"]; Zc = O["clean"]
    def push(Z, m_full, acc_partial):                     # the v3 recipe
        Zt = Z.clone(); oth = Zt.clone(); oth[ar, y] = -1e9; bo = oth.max(1).values; mar = Zt[ar, y] - bo
        delta = torch.quantile(mar[Fp], 1 - acc_partial)
        Zt[ar, y] = torch.where(fullm, bo - m_full, Zt[ar, y] - delta); return Zt
    def drop_class(Z, m_full, temp=1.0, part_shift=0.0):  # class-level: full classes lose the true class, partial are only softened
        Zt = Z.clone() / temp; oth = Zt.clone(); oth[ar, y] = -1e9; bo = oth.max(1).values
        Zt[ar, y] = torch.where(fullm, bo - m_full, Zt[ar, y] - part_shift); return Zt
    # class-conditional prior: for a fully forgotten class, the mean T=2 distribution its images get once the class is removed
    def class_prior(Z, m_full):
        Zt = drop_class(Z, m_full); out = Zt.clone()
        for c in torch.unique(y[Ff]).tolist():
            s = Ff & (y == c); out[s] = 0.5 * Zt[s] + 0.5 * Zt[s].mean(0, keepdim=True)
        return out
    # test-like: give each partial-class forget image the logits statistics the ORIGINAL shows on unseen test images of that class
    recipes = {"orig unchanged": Zc, "strong-view average only": Zs}
    for mf in (0.0, 2.0, 4.0, 6.0):
        for ap in (0.8, 0.9, 1.0):
            recipes[f"v3 push: m_full {mf}, acc_partial {ap} (strong avg)"] = push(Zs, mf, ap)
    for mf in (0.0, 2.0, 4.0):
        for t in (1.0, 1.5, 2.0):
            recipes[f"drop class: m_full {mf}, temp {t} (strong avg)"] = drop_class(Zs, mf, t)
        recipes[f"drop class: m_full {mf} (clean logits)"] = drop_class(Zc, mf)
        recipes[f"class prior blend: m_full {mf}"] = class_prior(Zs, mf)
    R["recipes"] = {}
    for name, Z in recipes.items():
        sub = {"clean": Z, "light": Z}
        R["recipes"][name] = {"all": forget_score(sub), "full": forget_score(sub, Ff), "partial": forget_score(sub, Fp)}
    # where does gold put fully forgotten classes, and can the original's runner-up predict it?
    gp = torch.stack([(Gi["clean"] / 2).softmax(1) for Gi in G]).mean(0)
    oth = Zs.clone(); oth[ar, y] = -1e9; othc = Zc.clone(); othc[ar, y] = -1e9
    R["full_class_argmax_match"] = {"gold argmax == orig strong-avg runner-up": round(float((gp[Ff].argmax(1) == oth[Ff].argmax(1)).float().mean()), 3),
                                    "gold argmax == orig clean runner-up": round(float((gp[Ff].argmax(1) == othc[Ff].argmax(1)).float().mean()), 3),
                                    "gold argmax == true class": round(float((gp[Ff].argmax(1) == y[Ff]).float().mean()), 3),
                                    "gold p(true) at T=2, mean": round(float(gp[Ff].gather(1, y[Ff][:, None]).mean()), 4),
                                    "gold true-class logit minus best other (mean over models)": round(float(sum((Gi["clean"][Ff][torch.arange(int(Ff.sum())), y[Ff]] - Gi["clean"][Ff].scatter(1, y[Ff][:, None], -1e9).max(1).values).mean() for Gi in G) / len(G)), 3)}
    return R


@app.local_entrypoint()
def mini(train: bool = True):
    out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True)
    if train:
        for r in mini_train.starmap([("orig", 0), ("gold", 0), ("gold", 1), ("gold", 2)]):
            print(r)
    R = mini_analyze.remote(); (out / "mini_analysis.json").write_text(json.dumps(R, indent=1))
    print("sizes:", R["sizes"]); print("gold CE on forget:", R["gold_forget_ce"])
    for g, d in R["behaviour"].items():
        print(f"{g:22s} orig {d['orig']} | gold {d['gold(mean of models)']}")
    print("full classes:", R["full_class_argmax_match"])
    print(f"{'recipe':58s} {'all (ms, pred, forget)':>26s} {'full':>24s} {'partial':>24s}")
    for k, v in sorted(R["recipes"].items(), key=lambda kv: -kv[1]["all"][2]):
        print(f"{k:58s} {str(tuple(v['all'])):>26s} {str(tuple(v['full'])):>24s} {str(tuple(v['partial'])):>24s}")


@app.local_entrypoint()
def bench(which: str = "b1"):
    """Run unlearning variants on the mini replica and print the OFFICIAL score terms (forget x retain)."""
    b = dict(mini=True, steps=2500, lr=5e-5, n_blocks=6, w_anchor=2.0, light_share=0.75, alphas=(0.5, 0.75, 1.0))
    b2 = dict(mini=True, steps=3000, lr=5e-5, n_blocks=6, w_anchor=2.0, light_share=0.75, alphas=(0.75, 1.0), amp_a=2.0)
    b3 = dict(mini=True, mini_mode="full", steps=3000, lr=5e-5, n_blocks=6, w_anchor=2.0, light_share=0.75, alphas=(0.5, 0.75, 1.0), amp_a=2.0, flip_q=0.2)
    b4 = dict(mini=True, steps=1500, lr=1e-4, n_blocks=4, w_t1=0.0, light_share=0.6, alphas=(0.5, 0.75, 1.0), train_head=False)
    grids = {"b4": [dict(name=f"lens{k} {an} w{w} [{wd}]", lens_block=k, anchor=an, w_anchor=w, mini_mode=wd, **b4)
                    for wd in ("partial", "full") for (k, an, w) in ((8, "strong", 1.0), (6, "strong", 1.0), (10, "strong", 1.0), (8, "heavy", 1.0))],
             "b3": [dict(name=f"hybrid m{m} [full world]", hybrid_m=m, **b3) for m in (1.0, 3.0, 6.0, 10.0)],
             "b2": [dict(name=f"amp2 flip{q} [{w}]", flip_q=q, mini_mode=w, **b2) for q in (0.2, 0.25, 0.3) for w in ("partial", "full")],
             "b1": [dict(name="inst m4 (as submitted)", m_full=4.0, acc_partial=0.8, **b),
                    dict(name="inst m12 partial-soft", m_full=12.0, acc_partial=1.0, **b),
                    dict(name="class m12 partial-soft", m_full=12.0, acc_partial=1.0, full_level="class", **b),
                    dict(name="class m8 partial-soft", m_full=8.0, acc_partial=1.0, full_level="class", **b)]}
    out = Path(__file__).parent / "out"; out.mkdir(exist_ok=True); allr = []
    for info in unlearn_remote.map(grids[which]):
        allr.append(info); print(f"\n=== {info['cfg']['name']} | target CE {info['target']['ce_mean']:.2f}/{info['target']['ce_std']:.2f} | rel weight change {info['rel_weight_change']:.3f}")
        for a, o in info["official"].items():
            print(f"  alpha {a}: FINAL {o['final']:.3f} = forget {o['forget']:.3f} (mean_std {o['mean_std']:.3f} x pred {o['prediction']:.3f}) x retain {o['retain']:.3f} (acc {o['test_acc']:.3f}/{o['test_acc_orig']:.3f}, KL {o['retain_kl']:.3f}, KL w/o full classes {o['retain_kl_excluding_full_classes']:.3f}) | forget CE {o['forget_ce_mean']}/{o['forget_ce_std']} | test acc on full classes {o['test_acc_full_classes']:.3f}")
    (out / f"bench_{which}.json").write_text(json.dumps(allr, indent=1))


@app.function(image=image, volumes={"/vol": vol}, timeout=900)
def mini_surgery():
    """Training-free variant on the mini replica: lower the logit of every fully forgotten class by a constant B for
    ALL inputs (a bias edit in the head). Official forget and retain terms computed from saved logits."""
    import torch, torch.nn.functional as F
    vol.reload()
    y = torch.load("/vol/orig_cache.pt")["pseudo"]; test, forget, fullm = _mini_split(y)
    O = torch.load("/vol/mini/orig0_logits.pt"); G = [torch.load(f"/vol/mini/gold{s}_logits.pt") for s in range(3)]
    fc = torch.unique(y[fullm]); ce = lambda L, sel: F.cross_entropy(L[sel], y[sel], reduction="none")
    kl = lambda p, L: float((p * (p.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())
    rows = []
    def score(name, sub_c, sub_l, sub_t):
        ms, pr = [], []
        for view, S in (("clean", sub_c), ("light", sub_l)):
            mg = sum(float(ce(g[view], forget).mean()) for g in G) / 3; sg = sum(float(ce(g[view], forget).std()) for g in G) / 3
            pg = torch.stack([(g[view][forget] / 2).softmax(1) for g in G]).mean(0)
            cs, co = ce(S, forget), ce(O[view], forget)
            ms.append(1 - min(1.0, (abs(float(cs.mean()) - mg) + abs(float(cs.std()) - sg)) / (abs(float(co.mean()) - mg) + abs(float(co.std()) - sg))))
            pr.append(1 - min(1.0, kl(pg, S[forget]) / kl(pg, O[view][forget])))
        ms, pr = sum(ms) / 2, sum(pr) / 2
        out = {"name": name, "mean_std": round(ms, 3), "pred": round(pr, 3), "forget": round(ms * pr, 3), "ce": (round(float(ce(sub_c, forget).mean()), 2), round(float(ce(sub_c, forget).std()), 2))}
        for tag, sel in (("control = all test classes", test), ("control without the removed classes", test & ~fullm)):
            ao = float((O["clean"][sel].argmax(1) == y[sel]).float().mean()); as_ = float((sub_t[sel].argmax(1) == y[sel]).float().mean())
            k = float(F.kl_div((sub_t[sel] / 2).log_softmax(1), (O["clean"][sel] / 2).softmax(1), reduction="none").sum(1).mean())
            r = min(1.0, as_ / ao) / (1 + k); out[tag] = {"acc": round(as_, 3), "acc_orig": round(ao, 3), "kl": round(k, 3), "retain": round(r, 3), "final": round(ms * pr * r, 3)}
        rows.append(out)
    part = forget & ~fullm
    for B in (0, 8, 12, 16, 20, 24, 28, 32):
        def cut(L):
            L = L.clone(); L[:, fc] -= B; return L
        score(f"bias -{B} on removed classes; partial untouched", cut(O["clean"]), cut(O["light"]), cut(O["clean"]))
        for w in (0.3, 0.6, 1.0):          # partial-class forget images softened towards the strong-view average (needs training; test unchanged)
            def soft(L):
                L2 = cut(L); mix = (1 - w) * L + w * O["strong8"]; mix[:, fc] -= B; L2[part] = mix[part]; return L2
            score(f"bias -{B}; partial forget images -> {w:.1f} strong-view avg", soft(O["clean"]), soft(O["light"]), cut(O["clean"]))
    mg = sum(float(ce(g["clean"], forget).mean()) for g in G) / 3
    return {"rows": rows, "gold_ce_mean": mg, "orig_margin_full_mean": float((O["clean"][forget & fullm].gather(1, y[forget & fullm][:, None])[:, 0] - O["clean"][forget & fullm].scatter(1, y[forget & fullm][:, None], -1e9).max(1).values).mean())}


@app.local_entrypoint()
def surgery():
    R = mini_surgery.remote(); (Path(__file__).parent / "out" / "mini_surgery.json").write_text(json.dumps(R, indent=1))
    print("orig margin on removed-class forget images (mean):", round(R["orig_margin_full_mean"], 2), "| gold CE mean:", round(R["gold_ce_mean"], 2))
    for r in sorted(R["rows"], key=lambda r: -r["control = all test classes"]["final"])[:14]:
        a, b = r["control = all test classes"], r["control without the removed classes"]
        print(f"{r['name']:62s} forget {r['forget']:.3f} (ms {r['mean_std']:.3f}, pred {r['pred']:.3f}) CE {r['ce']} | control all: retain {a['retain']:.3f} (acc {a['acc']:.3f}/{a['acc_orig']:.3f}, KL {a['kl']:.3f}) FINAL {a['final']:.3f} | control w/o removed: retain {b['retain']:.3f} FINAL {b['final']:.3f}")


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=900)
def make_surgery(B: float = 24.0, name: str = "surg", n_classes: int = 9):
    """Real model: lower head.bias of the 9 fully forgotten classes by B. No training. Saves /vol/ckpt/<name>.pt."""
    import os, torch, torch.nn.functional as F
    model, imgs = _setup(); L0 = _predict(model, imgs); y = L0.argmax(1); cnt = torch.bincount(y, minlength=100)
    fc = torch.where(cnt >= 129)[0][:n_classes]; full = torch.isin(y, fc)
    with torch.no_grad():
        model.head.bias[fc.cuda()] -= B
    torch.manual_seed(7); light = _rrc(imgs, 0.85, 1.0, jitter=0.1)
    L1, Ll = _predict(model, imgs), _predict(model, light)
    ce = F.cross_entropy(L1, y, reduction="none"); cel = F.cross_entropy(Ll, y, reduction="none")
    os.makedirs("/vol/ckpt", exist_ok=True); torch.save({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, f"/vol/ckpt/{name}.pt"); vol.commit()
    oth = L0.clone(); oth[torch.arange(len(y)), y] = -1e9
    return {"B": B, "classes": fc.tolist(), "n_full_images": int(full.sum()), "orig_margin_full_mean": round(float((L0[full].gather(1, y[full][:, None])[:, 0] - oth[full].max(1).values).mean()), 2),
            "forget_ce_clean": (round(float(ce.mean()), 2), round(float(ce.std()), 2)), "forget_ce_light": (round(float(cel.mean()), 2), round(float(cel.std()), 2)),
            "ce_full_mean": round(float(ce[full].mean()), 2), "ce_partial_mean": round(float(ce[~full].mean()), 4), "acc_forget": round(float((L1.argmax(1) == y).float().mean()), 3),
            "finite": bool(torch.isfinite(L1).all()), "kl_T2_orig_to_sub_on_partial_forget": round(float(F.kl_div((L1[~full] / 2).log_softmax(1), (L0[~full] / 2).softmax(1), reduction="none").sum(1).mean()), 5)}


@app.local_entrypoint()
def surgery_real(b: float = 24.0, name: str = "surg_b24", n: int = 9):
    print(make_surgery.remote(b, name, n))


@app.function(image=image, volumes={"/vol": vol}, timeout=900)
def mini_uniform():
    """Both mini worlds (classes removed entirely / only partly): official forget terms of ONE uniform target rule."""
    import torch, torch.nn.functional as F
    vol.reload(); y = torch.load("/vol/orig_cache.pt")["pseudo"]; O = torch.load("/vol/mini/orig0_logits.pt"); ar = torch.arange(len(y)); out = {}
    ce = lambda L, sel: F.cross_entropy(L[sel], y[sel], reduction="none")
    kl = lambda p, L: float((p * (p.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())
    for mode, tag in (("full", ""), ("partial", "P")):
        test, forget, fullm = _mini_split(y, mode=mode); G = [torch.load(f"/vol/mini/gold{tag}{k}_logits.pt") for k in range(3)]
        mg = sum(float(ce(g["clean"], forget).mean()) for g in G) / 3; sg = sum(float(ce(g["clean"], forget).std()) for g in G) / 3
        res = {"gold_ce": (round(mg, 2), round(sg, 2)), "gold_acc": round(sum(float((g["clean"][forget].argmax(1) == y[forget]).float().mean()) for g in G) / 3, 3), "n_forget": int(forget.sum()), "rules": {}}
        for tau in (1.0, 1.5):
            for d in (0, 1, 2, 3, 4, 6, 8, 10, 12):
                Z = O["strong8"].clone() / tau; Z[ar, y] -= d; ms, pr = [], []
                for view in ("clean", "light"):
                    m_ = sum(float(ce(g[view], forget).mean()) for g in G) / 3; s_ = sum(float(ce(g[view], forget).std()) for g in G) / 3
                    pg = torch.stack([(g[view][forget] / 2).softmax(1) for g in G]).mean(0); cs, co = ce(Z, forget), ce(O[view], forget)
                    ms.append(1 - min(1.0, (abs(float(cs.mean()) - m_) + abs(float(cs.std()) - s_)) / (abs(float(co.mean()) - m_) + abs(float(co.std()) - s_))))
                    pr.append(1 - min(1.0, kl(pg, Z[forget]) / kl(pg, O[view][forget])))
                res["rules"][f"tau {tau} delta {d}"] = (round(sum(ms) / 2, 3), round(sum(pr) / 2, 3), round(sum(ms) / 2 * sum(pr) / 2, 3), round(float(ce(Z, forget).mean()), 2), round(float(ce(Z, forget).std()), 2))
        out[mode] = res
    return out


@app.local_entrypoint()
def world2(train: bool = True):
    """Train the three never-saw-it models of the second world (no class removed entirely), then compare uniform rules in both worlds."""
    if train:
        for r in mini_train.starmap([("gold", 0, 25, "partial"), ("gold", 1, 25, "partial"), ("gold", 2, 25, "partial")]):
            print(r)
    R = mini_uniform.remote(); (Path(__file__).parent / "out" / "mini_uniform.json").write_text(json.dumps(R, indent=1))
    for mode in ("full", "partial"):
        print(f"world {mode}: gold CE mean/std {R[mode]['gold_ce']}, gold acc on forget {R[mode]['gold_acc']}, n {R[mode]['n_forget']}")
    print(f"{'rule':22s} {'FULL world: ms, pred, forget | CE':>44s} {'PARTIAL world: ms, pred, forget | CE':>46s} {'min forget':>11s}")
    for k in R["full"]["rules"]:
        a, b = R["full"]["rules"][k], R["partial"]["rules"][k]
        print(f"{k:22s} {str(tuple(a[:3])):>28s} | {a[3]:5.2f}/{a[4]:5.2f} {str(tuple(b[:3])):>30s} | {b[3]:5.2f}/{b[4]:5.2f} {min(a[2], b[2]):11.3f}")


@app.function(image=image, volumes={"/vol": vol}, timeout=900)
def mini_affine():
    """Both worlds: targets whose true-class margin is an affine function of the strong-view margin
    (true = best_other + a * margin - b), optionally with the other classes softened by tau. Official forget terms."""
    import torch, torch.nn.functional as F
    vol.reload(); y = torch.load("/vol/orig_cache.pt")["pseudo"]; O = torch.load("/vol/mini/orig0_logits.pt"); ar = torch.arange(len(y)); out = {}
    ce = lambda L, sel: F.cross_entropy(L[sel], y[sel], reduction="none")
    kl = lambda p, L: float((p * (p.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())
    Zs = O["strong8"]; oth = Zs.clone(); oth[ar, y] = -1e9; bo = oth.max(1).values; mar = Zs[ar, y] - bo
    for mode, tag in (("full", ""), ("partial", "P")):
        test, forget, fullm = _mini_split(y, mode=mode); G = [torch.load(f"/vol/mini/gold{tag}{k}_logits.pt") for k in range(3)]; res = {}
        gm = torch.stack([g["clean"][forget][torch.arange(int(forget.sum())), y[forget]] - g["clean"][forget].scatter(1, y[forget][:, None], -1e9).max(1).values for g in G]).mean(0)
        res["_corr(strong margin, gold margin)"] = round(float(torch.corrcoef(torch.stack([mar[forget], gm]))[0, 1]), 3)
        for tau in (1.0, 1.5):
            for a in (0.25, 0.5, 0.75, 1.0):
                for b in (0, 1, 2, 3, 4, 6, 8):
                    Z = Zs.clone() / tau; o2 = Z.clone(); o2[ar, y] = -1e9; Z[ar, y] = o2.max(1).values + (a * mar - b) / tau; ms, pr = [], []
                    for view in ("clean", "light"):
                        m_ = sum(float(ce(g[view], forget).mean()) for g in G) / 3; s_ = sum(float(ce(g[view], forget).std()) for g in G) / 3
                        pg = torch.stack([(g[view][forget] / 2).softmax(1) for g in G]).mean(0); cs, co = ce(Z, forget), ce(O[view], forget)
                        ms.append(1 - min(1.0, (abs(float(cs.mean()) - m_) + abs(float(cs.std()) - s_)) / (abs(float(co.mean()) - m_) + abs(float(co.std()) - s_))))
                        pr.append(1 - min(1.0, kl(pg, Z[forget]) / kl(pg, O[view][forget])))
                    res[f"tau {tau} a {a} b {b}"] = (round(sum(ms) / 2, 3), round(sum(pr) / 2, 3), round(sum(ms) / 2 * sum(pr) / 2, 3), round(float(ce(Z, forget).mean()), 2), round(float(ce(Z, forget).std()), 2), round(float((Z[forget].argmax(1) == y[forget]).float().mean()), 3))
        out[mode] = res
    return out


@app.local_entrypoint()
def affine():
    R = mini_affine.remote(); (Path(__file__).parent / "out" / "mini_affine.json").write_text(json.dumps(R, indent=1))
    print("corr(strong-view margin, gold margin): full world", R["full"].pop("_corr(strong margin, gold margin)"), "| partial world", R["partial"].pop("_corr(strong margin, gold margin)"))
    rows = sorted(R["full"], key=lambda k: -min(R["full"][k][2], R["partial"][k][2]))
    print(f"{'rule':22s} {'FULL: ms, pred, forget | CE | acc':>50s} {'PARTIAL: ms, pred, forget | CE | acc':>52s} {'min':>6s}")
    for k in rows[:14]:
        a, b = R["full"][k], R["partial"][k]
        print(f"{k:22s} {str(tuple(a[:3])):>26s} | {a[3]:5.2f}/{a[4]:5.2f} | {a[5]:.2f} {str(tuple(b[:3])):>28s} | {b[3]:5.2f}/{b[4]:5.2f} | {b[5]:.2f} {min(a[2], b[2]):6.3f}")
    bp = sorted(R["partial"], key=lambda k: -R["partial"][k][2])[:4]; print("best in PARTIAL world:", [(k, R["partial"][k][:3]) for k in bp])
    bf = sorted(R["full"], key=lambda k: -R["full"][k][2])[:4]; print("best in FULL world:", [(k, R["full"][k][:3]) for k in bf])


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=3600, max_containers=4)
def mini_oof(mode: str = "partial", init: str = "pretrained", folds: int = 5, epochs: int = 20):
    """Cross-fitting estimate of "what a model that never saw x predicts": split the (mini) forget images into folds,
    train on the other folds only (labels = the original model's predictions), predict the held-out fold.
    init = "pretrained": public ImageNet ViT-B (diagnostic only); "orig_reinit4": the original model with its last 4
    blocks and head re-initialised (uses nothing but the original model and the forget set)."""
    import math, timm, torch, torch.nn.functional as F
    torch.manual_seed(0)
    _, imgs = _setup(); y_all = torch.load("/vol/orig_cache.pt")["pseudo"]; test, forget, fullm = _mini_split(y_all, mode=mode)
    tag = "P" if mode == "partial" else ""; G = [torch.load(f"/vol/mini/gold{tag}{k}_logits.pt") for k in range(3)]; O = torch.load("/vol/mini/orig0_logits.pt")
    fi = torch.where(forget)[0]; X = imgs[fi]; y = y_all[fi]; n = len(fi)
    torch.manual_seed(7); light = _rrc(imgs, 0.85, 1.0, jitter=0.1)[fi]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(1)); fold = torch.empty(n, dtype=torch.long); fold[perm] = torch.arange(n) % folds
    oof_c = torch.zeros(n, 100); oof_l = torch.zeros(n, 100)
    for k in range(folds):
        if init == "pretrained":
            m = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True, num_classes=100).cuda()
        else:
            m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100)
            m.load_state_dict(torch.load("/vol/mini/orig.pt", map_location="cpu", weights_only=True)); fresh = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100)
            for blk_o, blk_f in zip(m.blocks[-4:], fresh.blocks[-4:]):
                blk_o.load_state_dict(blk_f.state_dict())
            m.head.load_state_dict(fresh.head.state_dict()); m.norm.load_state_dict(fresh.norm.state_dict()); m = m.cuda()
        tr = torch.where(fold != k)[0]; te = torch.where(fold == k)[0]; bs = 64; steps = epochs * max(1, len(tr) // bs)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-5 if init == "pretrained" else 1e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        m.train(); yg = y.cuda()
        for s in range(steps):
            ib = tr[torch.randint(0, len(tr), (bs,))]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(_pre(_rrc(X[ib], 0.6, 1.0, jitter=0.2))).float()
            loss = F.cross_entropy(out, yg[ib]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        m.eval(); oof_c[te] = _predict(m, X[te]); oof_l[te] = _predict(m, light[te]); del m; torch.cuda.empty_cache()
    ce = lambda L: F.cross_entropy(L, y, reduction="none")
    kl = lambda p, L: float((p * (p.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())
    res = {"mode": mode, "init": init, "n": n, "oof_acc": round(float((oof_c.argmax(1) == y).float().mean()), 3), "oof_ce": (round(float(ce(oof_c).mean()), 2), round(float(ce(oof_c).std()), 2)),
           "gold_acc": round(sum(float((g["clean"][fi].argmax(1) == y).float().mean()) for g in G) / 3, 3), "rules": {}}
    gold_right = torch.stack([(g["clean"][fi].argmax(1) == y) for g in G]).float().mean(0) > 0.5
    res["agreement: oof correct == gold correct"] = round(float(((oof_c.argmax(1) == y) == gold_right).float().mean()), 3)
    def score(Zc, Zl):
        ms, pr = [], []
        for view, Z in (("clean", Zc), ("light", Zl)):
            m_ = sum(float(ce(g[view][fi]).mean()) for g in G) / 3; s_ = sum(float(ce(g[view][fi]).std()) for g in G) / 3
            pg = torch.stack([(g[view][fi] / 2).softmax(1) for g in G]).mean(0); cs, co = ce(Z), ce(O[view][fi])
            ms.append(1 - min(1.0, (abs(float(cs.mean()) - m_) + abs(float(cs.std()) - s_)) / (abs(float(co.mean()) - m_) + abs(float(co.std()) - s_))))
            pr.append(1 - min(1.0, kl(pg, Z) / kl(pg, O[view][fi])))
        return (round(sum(ms) / 2, 3), round(sum(pr) / 2, 3), round(sum(ms) / 2 * sum(pr) / 2, 3), round(float(ce(Zc).mean()), 2), round(float(ce(Zc).std()), 2))
    for t in (0.7, 1.0, 1.5, 2.0):
        res["rules"][f"oof logits / {t}"] = score(oof_c / t, oof_l / t)
    S8 = O["strong8"][fi]
    for w in (0.3, 0.5, 0.7):
        res["rules"][f"{w} oof + {round(1 - w, 1)} strong-view avg (logits)"] = score(w * oof_c + (1 - w) * S8, w * oof_l + (1 - w) * S8)
    torch.save({"idx": fi, "clean": oof_c, "light": oof_l}, f"/vol/mini/oof_{mode}_{init}.pt"); vol.commit()
    return res


@app.local_entrypoint()
def oof():
    out = Path(__file__).parent / "out"; allr = []
    for r in mini_oof.starmap([("partial", "pretrained"), ("full", "pretrained"), ("partial", "orig_reinit4"), ("full", "orig_reinit4")]):
        allr.append(r); print(f"\\n=== world {r['mode']}, init {r['init']}: n {r['n']} | cross-fit acc {r['oof_acc']} (gold acc {r['gold_acc']}) CE {r['oof_ce']} | same right/wrong as gold on {r['agreement: oof correct == gold correct']}")
        for k, v in r["rules"].items():
            print(f"   {k:44s} ms {v[0]:.3f} pred {v[1]:.3f} forget {v[2]:.3f} | CE {v[3]}/{v[4]}")
    (out / "mini_oof.json").write_text(json.dumps(allr, indent=1))


@app.function(image=image, volumes={"/vol": vol}, timeout=900)
def mini_affine2():
    """Both worlds: keep the other-class distribution of the strong-view average, and set the true-class margin to
    a * d - b where d is a per-image difficulty signal: the strong-view margin, or the cross-fit margin (which
    images a never-saw-it model finds hard), or their mean. a > 1 widens the spread of losses."""
    import torch, torch.nn.functional as F
    vol.reload(); y = torch.load("/vol/orig_cache.pt")["pseudo"]; O = torch.load("/vol/mini/orig0_logits.pt"); ar = torch.arange(len(y)); out = {}
    ce = lambda L: F.cross_entropy(L, yy, reduction="none")
    kl = lambda p, L: float((p * (p.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())
    for mode, tag in (("full", ""), ("partial", "P")):
        test, forget, fullm = _mini_split(y, mode=mode); G = [torch.load(f"/vol/mini/gold{tag}{k}_logits.pt") for k in range(3)]
        fi = torch.where(forget)[0]; yy = y[fi]; n = len(fi); an = torch.arange(n); Zs = O["strong8"][fi]
        oth = Zs.clone(); oth[an, yy] = -1e9; bo = oth.max(1).values; ms_ = Zs[an, yy] - bo
        sig = {"strong": ms_}
        for init in ("pretrained", "orig_reinit4"):
            oo = torch.load(f"/vol/mini/oof_{mode}_{init}.pt"); assert (oo["idx"] == fi).all()
            oc = oo["clean"]; o2 = oc.clone(); o2[an, yy] = -1e9; mo = oc[an, yy] - o2.max(1).values
            z = lambda v: (v - v.mean()) / v.std()
            sig[f"oof[{init}]"] = z(mo) * ms_.std() + ms_.mean(); sig[f"mean(strong, oof[{init}])"] = (z(mo) + z(ms_)) / 2 * ms_.std() + ms_.mean()
        gm = torch.stack([g["clean"][fi][an, yy] - g["clean"][fi].scatter(1, yy[:, None], -1e9).max(1).values for g in G]).mean(0)
        res = {"_corr_with_gold_margin": {k: round(float(torch.corrcoef(torch.stack([v, gm]))[0, 1]), 3) for k, v in sig.items()}}
        for sname, d in sig.items():
            for a in (1.0, 1.5, 2.0, 3.0):
                for q in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4):            # share of images whose target margin is negative
                    b = a * float(torch.quantile(d, q)); Z = Zs.clone(); Z[an, yy] = bo + a * d - b; msc, pr = [], []
                    for view in ("clean", "light"):
                        m_ = sum(float(F.cross_entropy(g[view][fi], yy, reduction="none").mean()) for g in G) / 3; s_ = sum(float(F.cross_entropy(g[view][fi], yy, reduction="none").std()) for g in G) / 3
                        pg = torch.stack([(g[view][fi] / 2).softmax(1) for g in G]).mean(0); cs = ce(Z); co = F.cross_entropy(O[view][fi], yy, reduction="none")
                        msc.append(1 - min(1.0, (abs(float(cs.mean()) - m_) + abs(float(cs.std()) - s_)) / (abs(float(co.mean()) - m_) + abs(float(co.std()) - s_))))
                        pr.append(1 - min(1.0, kl(pg, Z) / kl(pg, O[view][fi])))
                    res[f"{sname} | a {a} flip {q}"] = (round(sum(msc) / 2, 3), round(sum(pr) / 2, 3), round(sum(msc) / 2 * sum(pr) / 2, 3), round(float(ce(Z).mean()), 2), round(float(ce(Z).std()), 2))
        out[mode] = res
    return out


@app.local_entrypoint()
def affine2():
    R = mini_affine2.remote(); (Path(__file__).parent / "out" / "mini_affine2.json").write_text(json.dumps(R, indent=1))
    for mode in ("full", "partial"):
        print(f"world {mode}: correlation of each difficulty signal with the gold margin:", R[mode].pop("_corr_with_gold_margin"))
    keys = list(R["full"]); rows = sorted(keys, key=lambda k: -min(R["full"][k][2], R["partial"][k][2]))
    print("--- best by the LOWER of the two worlds")
    for k in rows[:8]:
        a, b = R["full"][k], R["partial"][k]; print(f"{k:46s} full {a[:3]} CE {a[3]}/{a[4]} | partial {b[:3]} CE {b[3]}/{b[4]} | min {min(a[2], b[2]):.3f}")
    print("--- best in the PARTIAL world")
    for k in sorted(keys, key=lambda k: -R["partial"][k][2])[:8]:
        a, b = R["full"][k], R["partial"][k]; print(f"{k:46s} partial {b[:3]} CE {b[3]}/{b[4]} | full {a[:3]}")
    print("--- best in the FULL world")
    for k in sorted(keys, key=lambda k: -R["full"][k][2])[:4]:
        a, b = R["full"][k], R["partial"][k]; print(f"{k:46s} full {a[:3]} CE {a[3]}/{a[4]} | partial {b[:3]}")


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=1200)
def profile_ckpt(name: str):
    """Measure a checkpoint on the forget images (clean, lightly altered, strong views), split by class group."""
    import copy, torch, torch.nn.functional as F
    vol.reload(); orig, imgs = _setup(); sub = copy.deepcopy(orig)
    sub.load_state_dict(torch.load(f"/vol/ckpt/{name}.pt", map_location="cpu", weights_only=True), strict=True); sub = sub.cuda().eval()
    L0 = _predict(orig, imgs); y = L0.argmax(1); cnt = torch.bincount(y, minlength=100); full = (cnt >= 129)[y]
    torch.manual_seed(7); light = _rrc(imgs, 0.85, 1.0, jitter=0.1); torch.manual_seed(123); strong = _rrc(imgs, 0.25, 0.7, jitter=0.2, erase=0.5)
    P = {"clean": _predict(sub, imgs), "light": _predict(sub, light), "strong": _predict(sub, strong)}; O = {"clean": L0, "light": _predict(orig, light), "strong": _predict(orig, strong)}
    def blk(L, sel):
        ce = F.cross_entropy(L[sel], y[sel], reduction="none")
        return {"acc": round(float((L[sel].argmax(1) == y[sel]).float().mean()), 3), "ce_mean": round(float(ce.mean()), 3), "ce_std": round(float(ce.std()), 3), "p_true_T2": round(float((L[sel] / 2).softmax(1).gather(1, y[sel][:, None]).mean()), 3), "max_prob_T2": round(float((L[sel] / 2).softmax(1).max(1).values.mean()), 3)}
    kl2 = lambda a, b: round(float(F.kl_div((b / 2).log_softmax(1), (a / 2).softmax(1), reduction="none").sum(1).mean()), 4)
    R = {"name": name}
    for v in ("clean", "light", "strong"):
        R[v] = {"all": blk(P[v], torch.ones_like(full)), "full classes": blk(P[v], full), "partial classes": blk(P[v], ~full), "orig all": blk(O[v], torch.ones_like(full)), "kl_T2(orig||sub)": kl2(O[v], P[v])}
    sd0, sd1 = orig.state_dict(), sub.state_dict()
    R["changed_tensors"] = sorted({k.split(".")[0] + ("." + k.split(".")[1] if k.startswith("blocks") else "") for k in sd0 if sd0[k].is_floating_point() and not torch.equal(sd0[k].cpu(), sd1[k].cpu())})
    lens = lambda model, x, b=8: model.forward_head(model.norm(torch.nn.Sequential(*model.blocks[: b + 1])(model.norm_pre(model._pos_embed(model.patch_embed(x))))))
    Zl = _predict(orig, imgs, fn=lens); R["kl(block8 readout || sub at T=2)"] = round(float((Zl.softmax(1) * (Zl.log_softmax(1) - (P["clean"] / 2).log_softmax(1))).sum(1).mean()), 4)
    R["block8 target"] = {"full classes": blk(2 * Zl.log_softmax(1), full), "partial classes": blk(2 * Zl.log_softmax(1), ~full)}
    return R


@app.local_entrypoint()
def profile(name: str = "fit_b8"):
    R = profile_ckpt.remote(name); (Path(__file__).parent / "out" / f"profile_{name}.json").write_text(json.dumps(R, indent=1))
    print("changed:", R["changed_tensors"]); print("KL(block-8 readout || sub):", R["kl(block8 readout || sub at T=2)"])
    for v in ("clean", "light", "strong"):
        print(f"[{v}] all {R[v]['all']} | KL_T2(orig||sub) {R[v]['kl_T2(orig||sub)']}"); print(f"       full classes {R[v]['full classes']}"); print(f"       partial classes {R[v]['partial classes']}"); print(f"       (orig: {R[v]['orig all']})")


# ------------------------------------------------------------------------- which guess at "gold" explains the public scores?
@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=2400)
def calib_remote(extra: list = (), only: str = ""):
    """Build candidate stand-ins for the retrained model's predictions on the forget images and score every already
    submitted checkpoint against each with the official forget formula. A stand-in is credible if it reproduces the
    ordering of the public scores. Cross-fitted candidates are saved to /vol/calib_targets.pt for training."""
    import copy, torch, torch.nn.functional as F
    vol.reload(); orig, imgs = _setup()
    L0 = _predict(orig, imgs); y = L0.argmax(1); N = len(y)
    torch.manual_seed(7); light = _rrc(imgs, 0.6, 1.0, jitter=0.15)            # hidden copies behave like crops (v10/v11/fit_b8 ordering)
    Lo = {"clean": L0, "light": _predict(orig, light)}

    def feats(model, x, upto):
        t = model.norm_pre(model._pos_embed(model.patch_embed(x)))
        for blk in model.blocks[: upto + 1]:
            t = blk(t)
        return torch.cat([t[:, 0], t[:, 1:].mean(1)], 1)

    def lens(k):
        return lambda model, x: model.forward_head(model.norm(torch.nn.Sequential(*model.blocks[: k + 1])(model.norm_pre(model._pos_embed(model.patch_embed(x))))))

    cands, notes = {}, {}
    for k in (6, 7, 8, 9):
        Zl = _predict(orig, imgs, fn=lens(k)); cands[f"readout{k} as T2 dist"] = 2 * Zl.log_softmax(1); cands[f"readout{k} as logits"] = Zl
    # cross-fitted linear probes on intermediate features: a prediction for each image from a classifier that never saw it
    g = torch.Generator().manual_seed(0); fold = torch.zeros(N, dtype=torch.long)
    for c in y.unique().tolist():
        idx = torch.where(y == c)[0]; idx = idx[torch.randperm(len(idx), generator=g)]; fold[idx] = torch.arange(len(idx)) % 5
    present = y.unique(); remap = torch.full((100,), -1, dtype=torch.long); remap[present] = torch.arange(len(present))
    for k in ():
        Fc = _predict(orig, imgs, fn=lambda m, x: feats(m, x, k)); Ff = _predict(orig, imgs.flip(2), fn=lambda m, x: feats(m, x, k))
        mu, sd = Fc.mean(0, keepdim=True), Fc.std(0, keepdim=True) + 1e-6; Fc, Ff = ((Fc - mu) / sd).cuda(), ((Ff - mu) / sd).cuda()
        for wd in (1e-2, 1e-1):
            oof = torch.zeros(N, len(present))
            for f in range(5):
                tr, te = (fold != f).cuda(), (fold == f).cuda()
                W = torch.zeros(Fc.shape[1], len(present), device="cuda", requires_grad=True); b = torch.zeros(len(present), device="cuda", requires_grad=True)
                opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=200, history_size=20, line_search_fn="strong_wolfe")
                Xtr = torch.cat([Fc[tr], Ff[tr]]); ytr = remap[y].cuda()[tr].repeat(2)
                def closure():
                    opt.zero_grad(); l = F.cross_entropy(Xtr @ W + b, ytr) + wd * (W ** 2).sum() / 2; l.backward(); return l
                opt.step(closure); oof[te.cpu()] = (Fc[te] @ W + b).detach().float().cpu()
            yr = remap[y]
            best = min(((float(F.cross_entropy(oof / T, yr)), T) for T in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0)))   # calibrate on out-of-fold loss
            Z = torch.full((N, 100), -1e4); Z[:, present] = oof / best[1]
            p = 0.98 * Z.softmax(1) + 0.02 * (L0 / 4).softmax(1); Z = p.clamp_min(1e-9).log()          # small floor on the absent classes
            name = f"oof probe b{k} wd{wd}"; cands[name] = Z
            ce = F.cross_entropy(Z, y, reduction="none")
            notes[name] = {"acc": round(float((Z.argmax(1) == y).float().mean()), 3), "ce": (round(float(ce.mean()), 3), round(float(ce.std()), 3)), "T": best[1]}
    # "the classes in the forget set were removed entirely": gold only knows the absent classes
    absent = torch.ones(100, dtype=torch.bool); absent[present] = False
    Z8 = _predict(orig, imgs, fn=lens(8)); Zs8 = torch.stack([_predict(orig, _rrc(imgs, 0.5, 0.8, jitter=0.2, blur=0.5, erase=0.7)) for _ in range(4)]).mean(0)
    for src, Zb in (("final", L0), ("strong", Zs8), ("readout8", Z8)):
        for B in (6.0, 10.0, 14.0):
            for tau in (1.0, 2.0):
                Z = Zb.clone() / tau; top_abs = Z[:, absent].max(1, keepdim=True).values
                Z[:, present] = torch.minimum(Z[:, present], top_abs - B + (Z[:, present] - Z[:, present].max(1, keepdim=True).values).clamp_min(-4.0))
                cands[f"absent-only {src} B{B:.0f} tau{tau:.0f}"] = Z
    for B in (4.0, 8.0):          # only the true class is unknown; the other 44 forget classes stay available
        Z = Zs8.clone(); oth = Z.clone(); oth[torch.arange(N), y] = -1e9; Z[torch.arange(N), y] = oth.max(1).values - B; cands[f"true-class-only strong B{B:.0f}"] = Z
    import glob, os
    for f in sorted(glob.glob("/vol/oofb_*.pt")):                      # cross-fitted late-block predictions (uploaded)
        Zo = torch.load(f).float(); nm = os.path.basename(f)[:-3]
        for tt in (0.5, 0.7, 1.0, 1.5):
            cands[f"{nm} logits/{tt}"] = Zo / tt
        cands[f"{nm} as T2 dist"] = 2 * Zo.log_softmax(1); cands[f"{nm} avg readout8"] = 2 * (0.5 * Zo.softmax(1) + 0.5 * Z8.softmax(1)).log()
    ens = [torch.load(f).float() for f in sorted(glob.glob("/vol/oofb_f8*.pt")) if "keep" not in f and "hard" not in f]
    if len(ens) > 1:          # like the official reference: average the T=2 distributions of several never-saw-it models
        pe = torch.stack([(z / 2).softmax(1) for z in ens]).mean(0); cands["oofb_f8 ensemble"] = 2 * pe.clamp_min(1e-9).log()
        if os.path.exists("/vol/oofb_f7.pt"):
            p7 = (torch.load("/vol/oofb_f7.pt").float() / 2).softmax(1); cands["oofb f7+f8 ensemble"] = 2 * (0.5 * pe + 0.5 * p7).clamp_min(1e-9).log()
    torch.save({k: v for k, v in cands.items()}, "/vol/calib_targets.pt"); vol.commit()

    subs = {"orig": 0.0, "surg_b26": 0.0, "v5_amp2_f20_w2": 0.0397, "v12_abs_r8_B6": 0.0434, "v6_lens8_w2": 0.0564, "v3_a": 0.069, "v10_all_w1": 0.0916, "v11_crop60_w05": 0.1246, "fit_b8": 0.147}
    for e in extra:                       # unscored candidates: estimated only
        subs[e] = None
    if only:
        cands = {k: v for k, v in cands.items() if only in k}
    P = {}
    for s in subs:
        if s == "orig":
            P[s] = Lo; continue
        m = copy.deepcopy(orig); m.load_state_dict(torch.load(f"/vol/ckpt/{s}.pt", map_location="cpu", weights_only=True), strict=True)
        P[s] = {"clean": _predict(m, imgs), "light": _predict(m, light)}; del m
    cef = lambda L: F.cross_entropy(L, y, reduction="none")
    klp = lambda pg, L: float((pg * (pg.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())
    out = {"notes": notes, "public": subs, "table": {}}
    scored = [k for k, v in subs.items() if k != "orig" and v is not None]; pub = torch.tensor([subs[k] for k in scored])
    for cn, Zg in cands.items():
        cg = cef(Zg); mg, sg = float(cg.mean()), float(cg.std()); pg = (Zg / 2).softmax(1); row = {"gold_ce": (round(mg, 2), round(sg, 2)), "gold_acc": round(float((Zg.argmax(1) == y).float().mean()), 3)}
        for s in subs:
            ms, pr = [], []
            for v in ("clean", "light"):
                cs, co = cef(P[s][v]), cef(Lo[v])
                ms.append(1 - min(1.0, (abs(float(cs.mean()) - mg) + abs(float(cs.std()) - sg)) / (abs(float(co.mean()) - mg) + abs(float(co.std()) - sg))))
                pr.append(1 - min(1.0, klp(pg, P[s][v]) / klp(pg, Lo[v])))
            row[s] = (round(sum(ms) / 2, 3), round(sum(pr) / 2, 3), round(sum(ms) / 2 * sum(pr) / 2, 3))
        est = torch.tensor([row[k][2] for k in scored])
        row["corr"] = round(float(torch.corrcoef(torch.stack([est, pub]))[0, 1]), 3) if float(est.std()) > 0 else 0.0
        row["scale"] = round(float((est * pub).sum() / (est * est).sum().clamp_min(1e-9)), 3)      # public ~ scale x estimate
        out["table"][cn] = row
    return out


@app.local_entrypoint()
def calib(extra: str = "", only: str = ""):
    R = calib_remote.remote([e for e in extra.split(",") if e], only); (Path(__file__).parent / "out" / "calib.json").write_text(json.dumps(R, indent=1))
    if extra:
        for cn, row in sorted(R["table"].items(), key=lambda kv: -kv[1]["corr"]):
            print(f"EST corr {row['corr']:+.3f} scale {row['scale']:.2f} | {cn:26s} | " + " | ".join(f"{e} {row[e][2]:.3f} ({row[e][0]:.2f}x{row[e][1]:.2f}) -> {row['scale'] * row[e][2]:.3f}" for e in extra.split(",") if e))
        return
    print("public:", R["public"])
    for n, v in R["notes"].items():
        print("probe", n, v)
    for cn, row in sorted(R["table"].items(), key=lambda kv: -kv[1]["corr"]):
        print(f"corr {row['corr']:+.3f} scale {row['scale']:.2f} | {cn:26s} gold acc {row['gold_acc']:.3f} CE {row['gold_ce']} | " + " | ".join(f"{s} {row[s][2]:.3f} ({row[s][0]:.2f}x{row[s][1]:.2f})" for s in R["public"] if s != "orig"))


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=2400)
def calib2_remote(names: list):
    """Per checkpoint: forget terms under read-out stand-ins on several kinds of altered copies (the stand-in is
    recomputed on the altered copy), and retain proxies on inputs that are not the forget images."""
    import copy, torch, torch.nn.functional as F
    vol.reload(); orig, imgs = _setup(); L0 = _predict(orig, imgs); y = L0.argmax(1)
    def lens(k):
        return lambda model, x: model.forward_head(model.norm(torch.nn.Sequential(*model.blocks[: k + 1])(model.norm_pre(model._pos_embed(model.patch_embed(x))))))
    torch.manual_seed(7)
    noise = lambda x, s: (x.float() + torch.randn(x.shape) * s).round().clamp(0, 255).to(torch.uint8)
    def blur(x):
        t = x.permute(0, 3, 1, 2).float(); t = F.avg_pool2d(F.pad(t, (1, 1, 1, 1), mode="replicate"), 3, 1); return t.permute(0, 2, 3, 1).round().to(torch.uint8)
    def down(x):
        t = x.permute(0, 3, 1, 2).float(); t = F.interpolate(F.interpolate(t, size=160, mode="bilinear", antialias=True), size=224, mode="bilinear"); return t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)
    V = {"clean": imgs, "flip": imgs.flip(2), "light": _rrc(imgs, 0.85, 1.0, jitter=0.1), "noise8": noise(imgs, 8.0), "blur": blur(imgs), "down160": down(imgs),
         "bright": (imgs.float() * 1.15).round().clamp(0, 255).to(torch.uint8), "med": _rrc(imgs, 0.6, 1.0, jitter=0.2)}
    i1, i2 = torch.randperm(len(imgs))[:1500], torch.randperm(len(imgs))[:1500]
    C = {"strong": _rrc(imgs, 0.25, 0.7, jitter=0.2, erase=0.5), "tiny": _rrc(imgs[:1500], 0.08, 0.3), "mosaic": _mosaic(imgs, 1000),
         "mixup": ((imgs[i1].float() + imgs[i2].float()) / 2).round().to(torch.uint8)}
    yC = {"strong": y, "tiny": y[:1500]}
    G = {v: {k: 2 * _predict(orig, x, fn=lens(k)).log_softmax(1) for k in (6, 7, 8)} for v, x in V.items()}
    Ov = {v: _predict(orig, x) for v, x in V.items()}; Oc = {v: _predict(orig, x) for v, x in C.items()}
    cef = lambda L: F.cross_entropy(L, y, reduction="none")
    klp = lambda pg, L: float((pg * (pg.clamp_min(1e-12).log() - (L / 2).log_softmax(1))).sum(1).mean())
    klT = lambda a, b, T: float(F.kl_div((b / T).log_softmax(1), (a / T).softmax(1), reduction="none").sum(1).mean())
    out = {}
    for s in names:
        m = copy.deepcopy(orig)
        if "@" in s:                                   # "name@alpha": interpolate from the original towards the checkpoint
            nm, a = s.split("@"); a = float(a); sd0 = orig.state_dict(); sd1 = torch.load(f"/vol/ckpt/{nm}.pt", map_location="cpu", weights_only=True)
            m.load_state_dict({k: (sd0[k].cpu() + a * (sd1[k] - sd0[k].cpu())) if sd0[k].is_floating_point() else sd1[k] for k in sd0})
        else:
            m.load_state_dict(torch.load(f"/vol/ckpt/{s}.pt", map_location="cpu", weights_only=True), strict=True)
        m = m.cuda().eval(); r = {"forget": {}, "retain": {}, "ce": {}}
        for v, x in V.items():
            P = _predict(m, x); cs, co = cef(P), cef(Ov[v]); r["ce"][v] = (round(float(cs.mean()), 2), round(float(cs.std()), 2), round(float((P.argmax(1) == y).float().mean()), 3))
            for k in (6, 7, 8):
                cg = cef(G[v][k]); mg, sg = float(cg.mean()), float(cg.std()); pg = (G[v][k] / 2).softmax(1)
                ms = 1 - min(1.0, (abs(float(cs.mean()) - mg) + abs(float(cs.std()) - sg)) / (abs(float(co.mean()) - mg) + abs(float(co.std()) - sg)))
                pr = 1 - min(1.0, klp(pg, P) / klp(pg, Ov[v])); r["forget"][f"{v}/b{k}"] = (round(ms, 3), round(pr, 3), round(ms * pr, 3))
        for v, x in C.items():
            P = _predict(m, x); d = {"kl_T1": round(klT(Oc[v], P, 1.0), 3), "kl_T2": round(klT(Oc[v], P, 2.0), 3), "agree": round(float((P.argmax(1) == Oc[v].argmax(1)).float().mean()), 3)}
            if v in yC:
                d["acc_ratio"] = round(float((P.argmax(1) == yC[v]).float().mean()) / float((Oc[v].argmax(1) == yC[v]).float().mean()), 3)
            r["retain"][v] = d
        out[s] = r; del m
    return out


@app.local_entrypoint()
def calib2(names: str = "surg_b26,v5_amp2_f20_w2,v6_lens8_w2,v3_a,fit_b8,fit_b8@0.5,fit_b8@0.7,fit_b8@0.85,fit_b8@1.15"):
    R = calib2_remote.remote(names.split(",")); (Path(__file__).parent / "out" / "calib2.json").write_text(json.dumps(R, indent=1))
    for s, r in R.items():
        print(f"\n=== {s}")
        print("  CE/acc: " + " | ".join(f"{v} {c[0]}/{c[1]} acc {c[2]}" for v, c in r["ce"].items()))
        for k in (6, 7, 8):
            print(f"  forget vs readout{k}: " + " | ".join(f"{v} {r['forget'][f'{v}/b{k}'][2]:.2f}" for v in r["ce"]))
        print("  retain proxies: " + " | ".join(f"{v} {d}" for v, d in r["retain"].items()))


@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=1200)
def runnerup_remote():
    """Where does the model's belief go once the memorised label is taken away? Per source: share of runner-up mass
    on classes absent from the forget set, and how concentrated the runner-up is within each label class."""
    import torch
    orig, imgs = _setup(); L0 = _predict(orig, imgs); y = L0.argmax(1); N = len(y)
    present = y.unique(); absent = torch.ones(100, dtype=torch.bool); absent[present] = False
    def lens(k):
        return lambda model, x: model.forward_head(model.norm(torch.nn.Sequential(*model.blocks[: k + 1])(model.norm_pre(model._pos_embed(model.patch_embed(x))))))
    src = {"final clean": L0, "strong x8": torch.stack([_predict(orig, _rrc(imgs, 0.5, 0.8, jitter=0.2, blur=0.5, erase=0.7)) for _ in range(8)]).mean(0),
           "very strong x8": torch.stack([_predict(orig, _rrc(imgs, 0.15, 0.4, jitter=0.3, erase=0.8)) for _ in range(8)]).mean(0)}
    for k in (5, 6, 7, 8, 9, 10):
        src[f"readout{k}"] = _predict(orig, imgs, fn=lens(k))
    cnt = torch.bincount(y, minlength=100); out = {"n_present": int(len(present)), "class_sizes": sorted(cnt[present].tolist())}
    for n, Z in src.items():
        p = Z.softmax(1); r = {"acc_vs_label": round(float((Z.argmax(1) == y).float().mean()), 3), "argmax_in_absent": round(float(absent[Z.argmax(1)].float().mean()), 3)}
        Zo = Z.clone(); Zo[torch.arange(N), y] = -1e9; ru = Zo.argmax(1); po = Zo.softmax(1)
        r["runnerup_in_absent"] = round(float(absent[ru].float().mean()), 3); r["non-label mass on absent classes"] = round(float(po[:, absent].sum(1).mean()), 3)
        tops, purity = [], []
        for c in present.tolist():
            m = ru[y == c]; b = torch.bincount(m, minlength=100); purity.append(float(b.max()) / len(m)); tops.append((c, int(cnt[c]), int(b.argmax()), bool(absent[b.argmax()]), round(float(b.max()) / len(m), 2)))
        r["mean share of the most common runner-up within a label class"] = round(sum(purity) / len(purity), 3); r["per_class(label,n,top_runnerup,absent?,share)"] = tops
        out[n] = r
    return out


@app.local_entrypoint()
def runnerup():
    R = runnerup_remote.remote(); (Path(__file__).parent / "out" / "runnerup.json").write_text(json.dumps(R, indent=1))
    print("present classes", R["n_present"], "sizes", R["class_sizes"])
    for n, r in R.items():
        if isinstance(r, dict):
            print(f"{n:16s} acc vs label {r['acc_vs_label']:.3f} | argmax in absent {r['argmax_in_absent']:.3f} | runner-up in absent {r['runnerup_in_absent']:.3f} | non-label mass on absent {r['non-label mass on absent classes']:.3f} | runner-up purity {r['mean share of the most common runner-up within a label class']:.3f}")
    print("strong x8 per class:", R["strong x8"]["per_class(label,n,top_runnerup,absent?,share)"])


# ------------------------------------------------------------------ cross-fitted "never saw this image" late blocks
@app.function(image=image, volumes={"/vol": vol}, gpu="A100-40GB", timeout=5400, max_containers=30)
def oof_blocks_remote(cfg: dict):
    """Blocks `first`..11 (and optionally norm/head) are re-initialised and trained on 4/5 of the forget images, with
    the original model as teacher on augmented views; the remaining 1/5 is then predicted by blocks that never saw it.
    Returns the out-of-fold logits for this fold (clean images)."""
    import copy, math, torch, torch.nn.functional as F
    c = dict(fold=0, first=8, epochs=40, lr=3e-4, bs=64, reinit=True, hard=0.0, tag="a", seed=0); c.update(cfg)
    torch.manual_seed(100 + c["fold"] + 1000 * c["seed"]); orig, imgs = _setup(); L0 = _predict(orig, imgs); y = L0.argmax(1); N = len(y)
    g = torch.Generator().manual_seed(0); fold = torch.zeros(N, dtype=torch.long)
    for cl in y.unique().tolist():
        idx = torch.where(y == cl)[0]; idx = idx[torch.randperm(len(idx), generator=g)]; fold[idx] = torch.arange(len(idx)) % 5
    tr = torch.where(fold != c["fold"])[0]; te = torch.where(fold == c["fold"])[0]
    for p in orig.parameters():
        p.requires_grad_(False)
    stu = copy.deepcopy(orig); params = []
    for p in stu.parameters():
        p.requires_grad_(False)
    for blk in stu.blocks[c["first"]:]:
        if c["reinit"]:
            for m in blk.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.trunc_normal_(m.weight, std=0.02); torch.nn.init.zeros_(m.bias)
                elif isinstance(m, torch.nn.LayerNorm):
                    torch.nn.init.ones_(m.weight); torch.nn.init.zeros_(m.bias)
            for n_, p in blk.named_parameters():
                if "gamma" in n_:
                    torch.nn.init.constant_(p, 0.1)
        for p in blk.parameters():
            p.requires_grad_(True); params.append(p)
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=0.05); steps = c["epochs"] * (len(tr) // c["bs"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    stu.train(); yg = y.cuda()
    for s in range(steps):
        ib = tr[torch.randint(0, len(tr), (c["bs"],))]; x = _pre(_rrc(imgs[ib], 0.35, 1.0, jitter=0.3, erase=0.3))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = stu(x).float()
            with torch.no_grad():
                zt = orig(x).float()
        loss = 4 * F.kl_div((z / 2).log_softmax(1), (zt / 2).softmax(1), reduction="batchmean")
        if c["hard"]:
            loss = loss + c["hard"] * F.cross_entropy(z, yg[ib])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    stu.eval(); Zte = _predict(stu, imgs[te]); Ztr = _predict(stu, imgs[tr[:500]])
    ce = F.cross_entropy(Zte, y[te], reduction="none")
    import os
    os.makedirs("/vol/oofb_parts", exist_ok=True); torch.save({"te": te, "logits": Zte}, f"/vol/oofb_parts/{c['tag']}_{c['fold']}.pt"); vol.commit()
    return {"cfg": c, "acc_te": float((Zte.argmax(1) == y[te]).float().mean()), "ce_te": (float(ce.mean()), float(ce.std())),
            "acc_tr": float((Ztr.argmax(1) == y[tr[:500]]).float().mean()), "final_loss": float(loss)}


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def oofb_assemble(tag: str):
    import torch
    vol.reload(); Z = torch.zeros(3500, 100)
    for f in range(5):
        d = torch.load(f"/vol/oofb_parts/{tag}_{f}.pt"); Z[d["te"]] = d["logits"]
    torch.save(Z, f"/vol/oofb_{tag}.pt"); vol.commit(); return True


@app.local_entrypoint()
def oofb(first: int = 8, epochs: int = 40, tag: str = "a", reinit: bool = True, hard: float = 0.0, seed: int = 0):
    for r in oof_blocks_remote.map([dict(fold=f, first=first, epochs=epochs, tag=tag, reinit=reinit, hard=hard, seed=seed) for f in range(5)]):
        print(r["cfg"]["fold"], {k: r[k] for k in ("acc_te", "ce_te", "acc_tr", "final_loss")})
    print("assembled", oofb_assemble.remote(tag))


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
def export_targets_remote():
    """CPU only: pack the cross-fitted (blocks 7-11 re-initialised) predictions for all seeds into one .npz."""
    import glob, os, numpy as np, torch
    vol.reload(); fs = sorted(glob.glob("/vol/oofb_f7*.pt")); Z = torch.stack([torch.load(f).float() for f in fs])
    p2 = (Z / 2).softmax(-1).mean(0)                                    # mean T=2 distribution over seeds, like the official reference
    np.savez_compressed("/vol/gold_standin_f7.npz", seeds=np.array([os.path.basename(f)[:-3] for f in fs]), logits_per_seed=Z.numpy().astype(np.float32),
                        p_T2_mean=p2.numpy().astype(np.float32), logits_from_mean=(2 * p2.clamp_min(1e-9).log()).numpy().astype(np.float32))
    vol.commit(); return {"seeds": [os.path.basename(f) for f in fs], "shape": list(Z.shape), "argmax_agreement_between_seeds": float((Z[0].argmax(1) == Z[1].argmax(1)).float().mean()) if len(fs) > 1 else None}


@app.local_entrypoint()
def export_targets():
    print(export_targets_remote.remote())
