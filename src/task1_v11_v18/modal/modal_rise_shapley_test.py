"""45-second Modal H100 Benchmark of RISE / Monte-Carlo Shapley De-Saturated Watermark Localization!
Key insight:
  - Single-patch occlusion fails on L-images because 95% watermarked SATURATES the decoder!
  - Random multi-scale shifted-grid masks with p_keep in {0.15, 0.30, 0.50} DE-SATURATE the decoder!
  - For TrustMark, replacing occluded pixels with Adobe's official `tm.removal(img)` creates
    100% photorealistic masked images with ZERO blur artifacts!
  - Computes exact Shapley/RISE correlation map across K=512 masks in ~0.25s/img on H100!
"""
import modal

app = modal.App("task1-rise-shapley-test")
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
def run_rise_test():
    import os, sys, time, pathlib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    import onnx
    from onnx import numpy_helper

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import miou_image, downsample_mask

    dev = "cuda:0"

    class StegaStampDecoderPT(nn.Module):
        def __init__(self, onnx_path):
            super().__init__()
            m = onnx.load(str(onnx_path))
            inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
            def mkc(wn, bn, s, p=0):
                w, b = inits[wn], inits[bn]
                c = nn.Conv2d(w.shape[1], w.shape[0], (w.shape[2], w.shape[3]), stride=s, padding=p)
                with torch.no_grad():
                    c.weight.copy_(torch.from_numpy(w.copy())); c.bias.copy_(torch.from_numpy(b.copy()))
                return c
            def mkl(wn, bn):
                w, b = inits[wn], inits[bn]
                l = nn.Linear(w.shape[0], w.shape[1])
                with torch.no_grad():
                    l.weight.copy_(torch.from_numpy(w.T.copy())); l.bias.copy_(torch.from_numpy(b.copy()))
                return l
            self.stn_conv_1 = mkc("gen_stn/stn_conv_1/Conv2D/ReadVariableOp__42", "gen_stn/stn_conv_1/BiasAdd/ReadVariableOp__43", 2)
            self.stn_conv_2 = mkc("gen_stn/stn_conv_2/Conv2D/ReadVariableOp__44", "gen_stn/stn_conv_2/BiasAdd/ReadVariableOp__45", 2)
            self.stn_conv_3 = mkc("gen_stn/stn_conv_3/Conv2D/ReadVariableOp__46", "gen_stn/stn_conv_3/BiasAdd/ReadVariableOp__47", 2)
            self.stn_fc_1 = mkl("gen_stn/dense_1/MatMul/ReadVariableOp__48", "gen_stn/dense_1/BiasAdd/ReadVariableOp__49")
            self.stn_fc_2 = mkl("gen_stn/W_fc1/read__50", "gen_stn/b_fc1/read__51")
            self.register_buffer("stn_grid", torch.from_numpy(inits["gen_stn/ExpandDims__52"].astype(np.float32).copy()))
            self.dec_conv_1 = mkc("gen_decoder/dec_conv_1/Conv2D/ReadVariableOp__61", "gen_decoder/dec_conv_1/BiasAdd/ReadVariableOp__62", 2)
            self.dec_conv_2 = mkc("gen_decoder/dec_conv_2/Conv2D/ReadVariableOp__63", "gen_decoder/dec_conv_2/BiasAdd/ReadVariableOp__64", 1, 1)
            self.dec_conv_3 = mkc("gen_decoder/dec_conv_3/Conv2D/ReadVariableOp__65", "gen_decoder/dec_conv_3/BiasAdd/ReadVariableOp__66", 2)
            self.dec_conv_4 = mkc("gen_decoder/dec_conv_4/Conv2D/ReadVariableOp__67", "gen_decoder/dec_conv_4/BiasAdd/ReadVariableOp__68", 1, 1)
            self.dec_conv_5 = mkc("gen_decoder/dec_conv_5/Conv2D/ReadVariableOp__69", "gen_decoder/dec_conv_5/BiasAdd/ReadVariableOp__70", 2)
            self.dec_conv_6 = mkc("gen_decoder/dec_conv_6/Conv2D/ReadVariableOp__71", "gen_decoder/dec_conv_6/BiasAdd/ReadVariableOp__72", 2)
            self.dec_conv_7 = mkc("gen_decoder/dec_conv_7/Conv2D/ReadVariableOp__73", "gen_decoder/dec_conv_7/BiasAdd/ReadVariableOp__74", 2, 1)
            self.dec_dense = mkl("gen_decoder/dec_dense/MatMul/ReadVariableOp__75", "gen_decoder/dec_dense/BiasAdd/ReadVariableOp__76")
            self.decoder_out = mkl("gen_decoder/decoder_out/MatMul/ReadVariableOp__77", "gen_decoder/decoder_out/BiasAdd/ReadVariableOp__78")

        def stn_warp(self, x):
            h = F.relu(self.stn_conv_1(F.pad(x - 0.5, (0, 1, 0, 1))))
            h = F.relu(self.stn_conv_2(F.pad(h, (0, 1, 0, 1))))
            h = F.relu(self.stn_conv_3(F.pad(h, (0, 1, 0, 1))))
            h = h.permute(0, 2, 3, 1).contiguous().flatten(1)
            theta = self.stn_fc_2(F.relu(self.stn_fc_1(h))).view(-1, 2, 3)
            B = x.shape[0]
            coords = torch.bmm(theta, self.stn_grid.expand(B, -1, -1)).view(B, 2, 400, 400)
            x_f = (coords[:, 0] + 1.0) * 199.0; y_f = (coords[:, 1] + 1.0) * 199.0
            x0 = torch.floor(x_f).long().clamp(0, 399); x1 = (x0 + 1).clamp(0, 399)
            y0 = torch.floor(y_f).long().clamp(0, 399); y1 = (y0 + 1).clamp(0, 399)
            wx1 = (x_f - x0.float()).unsqueeze(1); wx0 = (x1.float() - x_f).unsqueeze(1)
            wy1 = (y_f - y0.float()).unsqueeze(1); wy0 = (y1.float() - y_f).unsqueeze(1)
            flat = x.view(B, 3, 160000)
            def g(yy, xx):
                idx = (yy * 400 + xx).view(B, 1, 160000).expand(B, 3, 160000)
                return torch.gather(flat, 2, idx).view(B, 3, 400, 400)
            return g(y0, x0)*(wx0*wy0) + g(y1, x0)*(wx0*wy1) + g(y0, x1)*(wx1*wy0) + g(y1, x1)*(wx1*wy1)

        def forward(self, x, use_stn=True):
            w = self.stn_warp(x) if use_stn else x
            h = F.relu(self.dec_conv_1(F.pad(w - 0.5, (0, 1, 0, 1))))
            h = F.relu(self.dec_conv_2(h))
            h = F.relu(self.dec_conv_3(F.pad(h, (0, 1, 0, 1))))
            h = F.relu(self.dec_conv_4(h))
            h = F.relu(self.dec_conv_5(F.pad(h, (0, 1, 0, 1))))
            h = F.relu(self.dec_conv_6(F.pad(h, (0, 1, 0, 1))))
            h = F.relu(self.dec_conv_7(h))
            h = h.permute(0, 2, 3, 1).contiguous().flatten(1)
            return self.decoder_out(F.relu(self.dec_dense(h)))

    def blur2d(x, k=11, sigma=3.5):
        coords = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        ker = (g[:, None] * g[None, :]).view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(F.pad(x, (k // 2,) * 4, mode="reflect"), ker, groups=x.shape[1])

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
                for thr in np.linspace(np.percentile(flat, 3), np.percentile(flat, 97), 30):
                    m = float(np.mean([miou_image(v > thr, g) for v, g in zip(vals, gts)]))
                    if m > best_iou: best_iou = m
            print(f"    {grp:6s} (n={len(sub):2d}): pixel-AUROC={np.mean(aucs):.4f} (med={np.median(aucs):.4f}) | ZERO-SHOT mIoU={best_iou:.4f}", flush=True)

    def make_rise_masks(K=512, grid=12, p_keep=0.25, res=256, seed=42):
        """Standard RISE shifted-bilinear random masks [K, 1, res, res] + [K, 128, 128] float32."""
        rng = torch.Generator(device=dev).manual_seed(seed)
        cell = int(np.ceil(res / grid))
        up_s = (grid + 1) * cell
        small = (torch.rand((K, 1, grid, grid), generator=rng, device=dev) < p_keep).float()
        up = F.interpolate(small, (up_s, up_s), mode="bilinear", align_corners=False)
        # Random sub-cell crop per mask -> smooth isotropic resolution!
        ms = torch.empty((K, 1, res, res), device=dev, dtype=torch.float32)
        for k in range(K):
            dy = int(torch.randint(0, cell, (1,), generator=rng, device=dev).item())
            dx = int(torch.randint(0, cell, (1,), generator=rng, device=dev).item())
            ms[k] = up[k, :, dy:dy + res, dx:dx + res]
        ms128 = F.interpolate(ms, (128, 128), mode="bilinear", align_corners=False)[:, 0]  # [K, 128, 128]
        return ms, ms128

    def run_rise(decode_fn, x_r, x_clean, sign, ms, ms128, bs=128):
        """Computes exact RISE correlation map at 128x128."""
        K = ms.shape[0]
        scores = []
        with torch.no_grad():
            for s in range(0, K, bs):
                m = ms[s:s + bs]
                xb = x_r * m + x_clean * (1.0 - m)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = decode_fn(xb)
                # Use BOTH mean margin AND tanh-clipped margin (more robust to outlier bits)
                sl = lg * sign
                scores.append(sl.mean(1).float())
        sc = torch.cat(scores)  # [K]
        sc_c = sc - sc.mean()
        # Correlation map: sum_k (s_k - s_mean) * (m_k - m_mean) / (var(m_k) + eps)
        m_c = ms128 - ms128.mean(0, keepdim=True)
        num = (sc_c[:, None, None] * m_c).mean(0)
        den = m_c.pow(2).mean(0).clamp(min=1e-4)
        return (num / den).cpu().numpy()

    # ============================================================
    # 1. PIXELSEAL RISE BENCHMARK
    # ============================================================
    print("\n================ 1. PIXELSEAL RISE / SHAPLEY ================", flush=True)
    wm_ps = load_watermark("pixelseal", device=dev)
    ds_ps = LocalisationDataset(str(repo_dir), "train", methods=["pixelseal"])
    ps_all = [ds_ps[i] for i in range(len(ds_ps))]
    sub_ps = (
        [it for it in ps_all if classify_item(it) == "large"][:20] +
        [it for it in ps_all if classify_item(it) == "rect"][:20] +
        [it for it in ps_all if classify_item(it) == "object"][:20]
    )
    def dec_ps(x): return wm_ps.det(x)[:, 1:].float()

    # Test 3 regimes: p=0.15 (de-saturate L!), p=0.35 (balanced), p=0.65 (S-object!), and multi-scale combo!
    cfgs_ps = [
        ("RISE_g12_p15_blur256", 512, 12, 0.15, 256),
        ("RISE_g12_p35_blur256", 512, 12, 0.35, 256),
        ("RISE_g12_p65_blur256", 512, 12, 0.65, 256),
        ("RISE_g16_p25_blur512", 512, 16, 0.25, 512),
    ]
    maps_ps = {c[0]: {} for c in cfgs_ps}
    maps_ps["COMBO_ALL"] = {}
    t0 = time.time()
    pre_masks = {c[0]: make_rise_masks(c[1], c[2], c[3], c[4], seed=100 + i) for i, c in enumerate(cfgs_ps)}
    for it in sub_ps:
        nm = it["image_name"]
        img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
        sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
        x256 = F.interpolate(img01, (256, 256), mode="bilinear", align_corners=False)
        b256 = blur2d(x256, k=11, sigma=3.5)
        b512 = blur2d(img01, k=15, sigma=5.0)
        zsum = np.zeros((128, 128), np.float32)
        for name, K, g, pk, res in cfgs_ps:
            ms, ms128 = pre_masks[name]
            xr, xc = (x256, b256) if res == 256 else (img01, b512)
            mp = run_rise(dec_ps, xr, xc, sign, ms, ms128, bs=128 if res == 256 else 64)
            maps_ps[name][nm] = mp
            zsum += (mp - mp.mean()) / (mp.std() + 1e-6)
        maps_ps["COMBO_ALL"][nm] = zsum / len(cfgs_ps)
    print(f"PixelSeal RISE done in {time.time()-t0:.1f}s ({(time.time()-t0)/len(sub_ps):.2f}s/img)", flush=True)
    for k, m in maps_ps.items():
        eval_maps(f"PS {k}", m, sub_ps)

    # ============================================================
    # 2. TRUSTMARK RISE BENCHMARK (BLUR vs ADOBE OFFICIAL REMOVER!!)
    # ============================================================
    print("\n================ 2. TRUSTMARK RISE / SHAPLEY (BLUR vs ADOBE RM!) ================", flush=True)
    from trustmark import TrustMark
    tm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5)
    tm_dec = tm.decoder.to(dev).eval()
    tm_rm = tm.removal.to(dev).eval()
    for p in tm_dec.parameters(): p.requires_grad_(False)
    for p in tm_rm.parameters(): p.requires_grad_(False)
    def dec_tm(x01): return tm_dec(x01 * 2.0 - 1.0).float()

    ds_tm = LocalisationDataset(str(repo_dir), "train", methods=["trustmark"])
    tm_all = [ds_tm[i] for i in range(len(ds_tm))]
    sub_tm = (
        [it for it in tm_all if classify_item(it) == "large"][:20] +
        [it for it in tm_all if classify_item(it) == "rect"][:20] +
        [it for it in tm_all if classify_item(it) == "object"][:20]
    )

    ms12_15, m128_15 = make_rise_masks(512, 12, 0.15, 256, seed=201)
    ms12_35, m128_35 = make_rise_masks(512, 12, 0.35, 256, seed=202)
    ms12_65, m128_65 = make_rise_masks(512, 12, 0.65, 256, seed=203)
    ms16_25, m128_25 = make_rise_masks(512, 16, 0.25, 256, seed=204)

    maps_tm = {k: {} for k in ("BLUR_p15", "BLUR_p35", "ADOBURM_p15", "ADOBURM_p35", "ADOBURM_p65", "ADOBURM_g16_p25", "ADOBURM_COMBO")}
    t0 = time.time()
    for it in sub_tm:
        nm = it["image_name"]
        img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
        sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
        x256 = F.interpolate(img01, (256, 256), mode="bilinear", align_corners=False)
        b256 = blur2d(x256, k=11, sigma=3.5)
        with torch.no_grad():
            rm_out = tm_rm(x256 * 2.0 - 1.0)
            if isinstance(rm_out, tuple): rm_out = rm_out[0]
            rm01 = ((rm_out + 1.0) * 0.5).clamp(0.0, 1.0).float()

        maps_tm["BLUR_p15"][nm] = run_rise(dec_tm, x256, b256, sign, ms12_15, m128_15)
        maps_tm["BLUR_p35"][nm] = run_rise(dec_tm, x256, b256, sign, ms12_35, m128_35)
        r15 = run_rise(dec_tm, x256, rm01, sign, ms12_15, m128_15)
        r35 = run_rise(dec_tm, x256, rm01, sign, ms12_35, m128_35)
        r65 = run_rise(dec_tm, x256, rm01, sign, ms12_65, m128_65)
        r25 = run_rise(dec_tm, x256, rm01, sign, ms16_25, m128_25)
        maps_tm["ADOBURM_p15"][nm] = r15
        maps_tm["ADOBURM_p35"][nm] = r35
        maps_tm["ADOBURM_p65"][nm] = r65
        maps_tm["ADOBURM_g16_p25"][nm] = r25
        maps_tm["ADOBURM_COMBO"][nm] = sum((r - r.mean()) / (r.std() + 1e-6) for r in (r15, r35, r65, r25)) / 4.0

    print(f"TrustMark RISE done in {time.time()-t0:.1f}s ({(time.time()-t0)/len(sub_tm):.2f}s/img)", flush=True)
    for k, m in maps_tm.items():
        eval_maps(f"TM {k}", m, sub_tm)

    # ============================================================
    # 3. STEGASTAMP RISE BENCHMARK
    # ============================================================
    print("\n================ 3. STEGASTAMP RISE / SHAPLEY ================", flush=True)
    ss_pt = StegaStampDecoderPT(models_dir / "stegastamp" / "stega_stamp.onnx").to(dev).eval()
    for p in ss_pt.parameters(): p.requires_grad_(False)
    def dec_ss(x01): return ss_pt(x01, use_stn=True).float()
    ds_ss = LocalisationDataset(str(repo_dir), "train", methods=["stegastamp"])
    ss_all = [ds_ss[i] for i in range(len(ds_ss))]
    sub_ss = (
        [it for it in ss_all if classify_item(it) == "large"][:15] +
        [it for it in ss_all if classify_item(it) == "rect"][:15] +
        [it for it in ss_all if classify_item(it) == "object"][:15]
    )
    ms_ss15, m128_ss15 = make_rise_masks(512, 12, 0.15, 400, seed=301)
    ms_ss35, m128_ss35 = make_rise_masks(512, 12, 0.35, 400, seed=302)
    maps_ss = {"SS_RISE_p15": {}, "SS_RISE_p35": {}, "SS_COMBO": {}}
    t0 = time.time()
    for it in sub_ss:
        nm = it["image_name"]
        img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
        sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
        x400 = F.interpolate(img01, (400, 400), mode="bilinear", align_corners=False)
        b400 = blur2d(x400, k=15, sigma=5.0)
        r15 = run_rise(dec_ss, x400, b400, sign, ms_ss15, m128_ss15, bs=128)
        r35 = run_rise(dec_ss, x400, b400, sign, ms_ss35, m128_ss35, bs=128)
        maps_ss["SS_RISE_p15"][nm] = r15
        maps_ss["SS_RISE_p35"][nm] = r35
        maps_ss["SS_COMBO"][nm] = ((r15 - r15.mean())/(r15.std()+1e-6) + (r35 - r35.mean())/(r35.std()+1e-6)) * 0.5
    print(f"StegaStamp RISE done in {time.time()-t0:.1f}s ({(time.time()-t0)/len(sub_ss):.2f}s/img)", flush=True)
    for k, m in maps_ss.items():
        eval_maps(f"SS {k}", m, sub_ss)


@app.local_entrypoint()
def main():
    run_rise_test.remote()
