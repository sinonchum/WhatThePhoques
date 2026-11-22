"""30-second H100 TEST of the Known-Message EMBEDDER CARRIER MATCHED FILTER!
T(x,y) = Embedder(I, m*) - Embedder(I, 1 - m*)   (and mean over K random msgs)
Score(x,y) = Smooth( sum_c (I_c - lowpass(I_c)) * T_c(x,y) )
Also tests:
  - Raw dot <I, T> vs High-Pass dot <I - blur(I), T> vs Normalized cosine similarity
  - Using Embedder(I, m*) vs Embedder(blur(I), m*) vs Embedder(rm(I), m*)
  - Both 512x512 native AND 256x256 upsampled residual (since embedders embed at 256 and upsample delta!)
"""
import modal

app = modal.App("task1-carrier-mf-test")
vol = modal.Volume.from_name("task1-wm-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "git-lfs", "libgl1", "libglib2.0-0", "wget", "unzip")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "timm",
        "numpy",
        "scipy",
        "scikit-learn",
        "pillow",
        "einops",
        "onnx",
        "onnxruntime-gpu",
        "huggingface_hub",
        "hf_transfer",
        "trustmark",
        "omegaconf",
        "PyYAML",
        "tqdm",
    )
    .pip_install("requests", "opencv-python-headless")
    .pip_install("av", "lpips", "kornia", "pywavelets", "scikit-image", "matplotlib")
    .pip_install("pycocotools", "pandas", "tensorboard", "ffmpeg-python")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu="H100", volumes={"/vol": vol}, timeout=900)
def test_carrier():
    import os, sys, time, pathlib, inspect
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import miou_image, downsample_mask

    dev = "cuda:0"

    def blur2d(x, k=7, sigma=1.5):
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        ker = (g[:, None] * g[None, :]).view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), ker, groups=x.shape[1])

    def smooth128(m, k=7, sigma=1.8):
        t = torch.from_numpy(m).float().to(dev)[None, None]
        return blur2d(t, k=k, sigma=sigma)[0, 0].cpu().numpy()

    def classify_item(it):
        mask = it["mask"] > 0
        if it["full_watermark"]:
            return "large"
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return "rect" if mask.sum() / max((y1 - y0) * (x1 - x0), 1) > 0.98 else "object"

    def eval_maps(tag, maps, items):
        print(f"  [{tag}]", flush=True)
        for grp in ("ALL", "large", "rect", "object"):
            sub = [it for it in items if (grp == "ALL" or classify_item(it) == grp)]
            if not sub: continue
            aucs, gts, scs = [], [], []
            for it in sub:
                gt128 = downsample_mask(it["mask"] > 0) > 0
                sc = maps[it["image_name"]]
                if gt128.min() != gt128.max():
                    aucs.append(roc_auc_score(gt128.ravel(), sc.ravel()))
                gts.append(gt128); scs.append(sc)
            best_iou = 0.0
            for mode in ("raw", "z"):
                vals = scs if mode == "raw" else [(s - s.mean()) / (s.std() + 1e-6) for s in scs]
                flat = np.concatenate([v.ravel() for v in vals])
                for thr in np.linspace(np.percentile(flat, 2), np.percentile(flat, 98), 35):
                    m = float(np.mean([miou_image(v > thr, g) for v, g in zip(vals, gts)]))
                    if m > best_iou: best_iou = m
            print(f"    {grp:6s} (n={len(sub):2d}): pixel-AUROC={np.mean(aucs):.4f} (med={np.median(aucs):.4f}) | ZERO-SHOT mIoU={best_iou:.4f}", flush=True)

    # ============================================================
    # 1. PIXELSEAL EMBEDDER CARRIER MATCHED FILTER!
    # ============================================================
    print("\n================ 1. PIXELSEAL EMBEDDER CARRIER MATCHED FILTER ================", flush=True)
    wm_ps = load_watermark("pixelseal", device=dev)
    print("PixelSeal model attrs:", [a for a in dir(wm_ps.model) if not a.startswith("_")][:25], flush=True)
    print("embed signature:", inspect.signature(wm_ps.model.embed), flush=True)

    ds_ps = LocalisationDataset(str(repo_dir), "train", methods=["pixelseal"])
    ps_all = [ds_ps[i] for i in range(len(ds_ps))]
    sub_ps = (
        [it for it in ps_all if classify_item(it) == "large"][:25] +
        [it for it in ps_all if classify_item(it) == "rect"][:25] +
        [it for it in ps_all if classify_item(it) == "object"][:25]
    )

    maps_ps = {k: {} for k in ("dot_hp_inv", "cos_hp_inv", "dot_raw_inv", "dot_hp_rand8", "dot_hp_blur_in", "dot_hp_512direct")}
    t0 = time.time()
    torch.manual_seed(0)
    with torch.no_grad():
        for it in sub_ps:
            nm = it["image_name"]
            img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
            msg = torch.from_numpy(it["message"].astype(np.int64)).to(dev)[None]  # [1, 256]
            msg_inv = 1 - msg
            hp = img01 - blur2d(img01, k=7, sigma=1.5)
            img_b = blur2d(img01, k=5, sigma=1.2)

            # A) Standard embed(img01, msg) - embed(img01, 1-msg) (uses VideoSeal's native embed pipeline!)
            out_pos = wm_ps.model.embed(img01, msgs=msg, is_video=False)["imgs_w"]
            out_neg = wm_ps.model.embed(img01, msgs=msg_inv, is_video=False)["imgs_w"]
            T = (out_pos - out_neg).float()  # [1, 3, 512, 512]
            # High-pass of T too (in case embedder has tiny DC shift)
            T_hp = T - blur2d(T, k=7, sigma=1.5)

            # Raw dot & HP dot & cosine sim
            dot_hp = (hp * T_hp).sum(1, keepdim=True)
            cos_hp = dot_hp / ((hp.pow(2).sum(1, keepdim=True).sqrt() * T_hp.pow(2).sum(1, keepdim=True).sqrt()) + 1e-6)
            dot_raw = (img01 * T_hp).sum(1, keepdim=True)

            maps_ps["dot_hp_inv"][nm] = smooth128(F.avg_pool2d(dot_hp, 4)[0, 0].cpu().numpy(), 7, 1.8)
            maps_ps["cos_hp_inv"][nm] = smooth128(F.avg_pool2d(cos_hp, 4)[0, 0].cpu().numpy(), 7, 1.8)
            maps_ps["dot_raw_inv"][nm] = smooth128(F.avg_pool2d(dot_raw, 4)[0, 0].cpu().numpy(), 7, 1.8)

            # B) Mean over 8 random messages as baseline: embed(I, m*) - mean(embed(I, m_rand))
            r_msgs = torch.randint(0, 2, (8, msg.shape[1]), device=dev)
            out_r = wm_ps.model.embed(img01.expand(8, -1, -1, -1), msgs=r_msgs, is_video=False)["imgs_w"].mean(0, keepdim=True)
            T_r = (out_pos - out_r).float()
            T_r_hp = T_r - blur2d(T_r, k=7, sigma=1.5)
            maps_ps["dot_hp_rand8"][nm] = smooth128(F.avg_pool2d((hp * T_r_hp).sum(1, keepdim=True), 4)[0, 0].cpu().numpy(), 7, 1.8)

            # C) Feed blurred image to embedder so existing watermark doesn't interfere with carrier generation!
            out_bp = wm_ps.model.embed(img_b, msgs=msg, is_video=False)["imgs_w"]
            out_bn = wm_ps.model.embed(img_b, msgs=msg_inv, is_video=False)["imgs_w"]
            T_b = (out_bp - out_bn).float()
            T_b_hp = T_b - blur2d(T_b, k=7, sigma=1.5)
            maps_ps["dot_hp_blur_in"][nm] = smooth128(F.avg_pool2d((hp * T_b_hp).sum(1, keepdim=True), 4)[0, 0].cpu().numpy(), 7, 1.8)

            # D) Embed at 256 and upsample residual (matching 256->512 if different from default embed)
            img256 = F.interpolate(img01, (256, 256), mode="bilinear", align_corners=False)
            op256 = wm_ps.model.embed(img256, msgs=msg, is_video=False)["imgs_w"]
            on256 = wm_ps.model.embed(img256, msgs=msg_inv, is_video=False)["imgs_w"]
            T256 = F.interpolate((op256 - on256).float(), (512, 512), mode="bilinear", align_corners=False)
            T256_hp = T256 - blur2d(T256, k=7, sigma=1.5)
            maps_ps["dot_hp_512direct"][nm] = smooth128(F.avg_pool2d((hp * T256_hp).sum(1, keepdim=True), 4)[0, 0].cpu().numpy(), 7, 1.8)

    print(f"PixelSeal carrier test done in {time.time()-t0:.1f}s!", flush=True)
    for k, m in maps_ps.items():
        eval_maps(f"PS CARRIER: {k}", m, sub_ps)

    # ============================================================
    # 2. TRUSTMARK EMBEDDER CARRIER MATCHED FILTER!
    # ============================================================
    print("\n================ 2. TRUSTMARK EMBEDDER CARRIER MATCHED FILTER ================", flush=True)
    from trustmark import TrustMark
    tm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5)
    enc = tm.encoder.to(dev).eval()
    rm_net = tm.rm_model.to(dev).eval() if hasattr(tm, "rm_model") else None
    print("TrustMark attrs:", [a for a in dir(tm) if not a.startswith("_")], flush=True)
    print("TrustMark encoder type:", type(enc), "forward sig:", inspect.signature(enc.forward), flush=True)

    ds_tm = LocalisationDataset(str(repo_dir), "train", methods=["trustmark"])
    tm_all = [ds_tm[i] for i in range(len(ds_tm))]
    sub_tm = (
        [it for it in tm_all if classify_item(it) == "large"][:25] +
        [it for it in tm_all if classify_item(it) == "rect"][:25] +
        [it for it in tm_all if classify_item(it) == "object"][:25]
    )

    maps_tm = {k: {} for k in ("dot_hp_inv", "cos_hp_inv", "dot_rm_inv", "dot_hp_rm_in", "dot_hp_rand8")}
    t0 = time.time()
    with torch.no_grad():
        for it in sub_tm:
            nm = it["image_name"]
            img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
            msg = torch.from_numpy(it["message"].astype(np.float32)).to(dev)[None]  # [1, 100] in {0,1}
            msg_inv = 1.0 - msg
            img256 = F.interpolate(img01, (256, 256), mode="bilinear", align_corners=False)
            x256_m11 = img256 * 2.0 - 1.0
            hp = img01 - blur2d(img01, k=7, sigma=1.5)

            # Check TrustMark encoder call convention (image in [-1,1], secret in {0,1})
            out_pos = enc(x256_m11, msg)
            if isinstance(out_pos, tuple): out_pos = out_pos[0]
            out_neg = enc(x256_m11, msg_inv)
            if isinstance(out_neg, tuple): out_neg = out_neg[0]
            T256 = (out_pos - out_neg).float()
            T512 = F.interpolate(T256, (512, 512), mode="bilinear", align_corners=False)
            T_hp = T512 - blur2d(T512, k=7, sigma=1.5)

            dot_hp = (hp * T_hp).sum(1, keepdim=True)
            cos_hp = dot_hp / ((hp.pow(2).sum(1, keepdim=True).sqrt() * T_hp.pow(2).sum(1, keepdim=True).sqrt()) + 1e-6)
            maps_tm["dot_hp_inv"][nm] = smooth128(F.avg_pool2d(dot_hp, 4)[0, 0].cpu().numpy(), 7, 1.8)
            maps_tm["cos_hp_inv"][nm] = smooth128(F.avg_pool2d(cos_hp, 4)[0, 0].cpu().numpy(), 7, 1.8)

            # Adobe Remover residual dotted with carrier T!
            if rm_net is not None:
                rm_out = rm_net(x256_m11)
                if isinstance(rm_out, tuple): rm_out = rm_out[0]
                rm_res = (x256_m11 - rm_out).float()  # extracted watermark residual at 256!
                dot_rm = (rm_res * T256).sum(1, keepdim=True)
                maps_tm["dot_rm_inv"][nm] = smooth128(F.avg_pool2d(dot_rm, 2)[0, 0].cpu().numpy(), 7, 1.8)

                # Feed CLEANED (rm_out) image into encoder so carrier is generated from clean image!
                op_c = enc(rm_out.clamp(-1, 1), msg)
                if isinstance(op_c, tuple): op_c = op_c[0]
                on_c = enc(rm_out.clamp(-1, 1), msg_inv)
                if isinstance(on_c, tuple): on_c = on_c[0]
                Tc = (op_c - on_c).float()
                dot_rmc = (rm_res * Tc).sum(1, keepdim=True)
                maps_tm["dot_hp_rm_in"][nm] = smooth128(F.avg_pool2d(dot_rmc, 2)[0, 0].cpu().numpy(), 7, 1.8)

            # Mean over 8 random messages
            r_msgs = torch.randint(0, 2, (8, msg.shape[1]), device=dev).float()
            or8 = enc(x256_m11.expand(8, -1, -1, -1), r_msgs)
            if isinstance(or8, tuple): or8 = or8[0]
            Tr = F.interpolate((out_pos - or8.mean(0, keepdim=True)).float(), (512, 512), mode="bilinear", align_corners=False)
            Tr_hp = Tr - blur2d(Tr, k=7, sigma=1.5)
            maps_tm["dot_hp_rand8"][nm] = smooth128(F.avg_pool2d((hp * Tr_hp).sum(1, keepdim=True), 4)[0, 0].cpu().numpy(), 7, 1.8)

    print(f"TrustMark carrier test done in {time.time()-t0:.1f}s!", flush=True)
    for k, m in maps_tm.items():
        if m:
            eval_maps(f"TM CARRIER: {k}", m, sub_tm)


@app.local_entrypoint()
def main():
    test_carrier.remote()
