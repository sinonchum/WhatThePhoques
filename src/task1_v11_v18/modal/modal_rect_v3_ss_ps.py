"""Modal 2xA100-80GB/H100 Parallel Exact Known-Message Rect Search V3:
  - GPU 1: FIRST-EVER StegaStamp Rect Search (using exact PyTorch StegaStampDecoderPT, 16:1 aspect!)
  - GPU 2: ULTRA-FINE PixelSeal Rect Search V3 (sizes down to 16px, max_aspect=16.0, topk=10 refine!)
Saves to /vol/outputs/rect_v3/{method}_{split}.npz (100% compatible with JURECA rect_search_v2 format!).
"""
import modal

app = modal.App("task1-rect-search-v3")
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
    .pip_install("datasets")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1800)
def run_rect_v3(method: str, split: str = "train"):
    import os, sys, time, pathlib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.ops import roi_align
    import onnx
    from onnx import numpy_helper

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark

    dev = "cuda:0"
    torch.backends.cudnn.benchmark = True

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

    if method == "stegastamp":
        net = StegaStampDecoderPT(models_dir / "stegastamp" / "stega_stamp.onnx").to(dev).eval()
        for p in net.parameters(): p.requires_grad_(False)
        res = 400
        bs_eval = 256
    else:
        wm = load_watermark("pixelseal", device=dev)
        net = wm.det.eval()
        for p in net.parameters(): p.requires_grad_(False)
        res = 256
        bs_eval = 1024

    # Ultra-fine coarse box bank covering 16:1 thin strips & tiny 16px boxes!
    SIZES = [18, 28, 42, 62, 92, 136, 200, 296, 420]
    out_b = []
    for h in SIZES:
        for w in SIZES:
            if max(h, w) / min(h, w) > 16.0 or h * w > 0.48 * 512 * 512 or h * w < 400:
                continue
            sy = max(14, h // 2)
            sx = max(14, w // 2)
            ys = sorted(set(list(range(0, 512 - h + 1, sy)) + [512 - h]))
            xs = sorted(set(list(range(0, 512 - w + 1, sx)) + [512 - w]))
            out_b += [(y, x, y + h, x + w) for y in ys for x in xs]
    boxes = np.array(out_b, np.float32)
    print(f"[RECT-V3 {method}/{split}] coarse boxes={len(boxes)} (16:1 thin strips included!)", flush=True)

    @torch.no_grad()
    def score_boxes(img01, b_np, sign):
        ms, acs = [], []
        for s in range(0, len(b_np), bs_eval):
            b = torch.from_numpy(b_np[s:s+bs_eval]).to(dev)
            rois = torch.cat([torch.zeros(len(b), 1, device=dev), b[:, [1, 0, 3, 2]]], 1)
            crops = roi_align(img01, rois, output_size=res, sampling_ratio=2, aligned=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if method == "pixelseal":
                    lg = net(crops)[:, 1:].float()
                else:
                    lg = net(crops, use_stn=True).float()
            sl = lg * sign
            ms.append(sl.mean(1))
            acs.append((sl > 0).float().mean(1))
        return torch.cat(ms).cpu().numpy(), torch.cat(acs).cpu().numpy()

    def refine(img01, sign, box, rounds=(0.25, 0.12, 0.06, 0.03, 0.015)):
        best = np.array(box, np.float32)
        bm, ba = score_boxes(img01, best[None], sign)
        bm, ba = float(bm[0]), float(ba[0])
        for frac in rounds:
            for _ in range(4):
                h, w = best[2] - best[0], best[3] - best[1]
                d = np.array([frac * h, frac * w, frac * h, frac * w], np.float32)
                cand = []
                for e in range(4):
                    for sgn in (-1, 1):
                        c = best.copy(); c[e] += sgn * max(2.0, d[e]); cand.append(c)
                for sgn in (-1, 1):
                    c = best.copy(); c[[0, 2]] += sgn * max(2.0, d[0]); cand.append(c)
                    c = best.copy(); c[[1, 3]] += sgn * max(2.0, d[1]); cand.append(c)
                cand = np.clip(np.array(cand), 0, 512)
                ok = (cand[:, 2] - cand[:, 0] >= 8) & (cand[:, 3] - cand[:, 1] >= 8)
                cand = cand[ok]
                if len(cand) == 0: break
                m, a = score_boxes(img01, cand, sign)
                j = int(np.argmax(m))
                if m[j] <= bm + 1e-4: break
                best, bm, ba = cand[j], float(m[j]), float(a[j])
        return best, bm, ba

    ds = LocalisationDataset(str(repo_dir), split, methods=[method])
    out_dir = pathlib.Path("/vol/outputs/rect_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_d = {k: [] for k in ("orig_idx", "names", "box", "margin", "acc", "full_margin", "full_acc")}
    t0 = time.time()
    for i in range(len(ds)):
        it = ds[i]
        img01 = torch.from_numpy(it["image"].copy()).to(dev).permute(2, 0, 1)[None].float() / 255.0
        sign = torch.from_numpy(2.0 * it["message"].astype(np.float32) - 1.0).to(dev)[None]
        fm, fa = score_boxes(img01, np.array([[0, 0, 512, 512]], np.float32), sign)
        # Skip expensive coarse box search if full image ALREADY decodes > 0.85 (it's a LARGE image!)
        if float(fa[0]) > 0.85:
            box, bm, ba = np.array([0, 0, 512, 512], np.float32), float(fm[0]), float(fa[0])
        else:
            m, _ = score_boxes(img01, boxes, sign)
            top = np.argsort(-m)[:8]
            cands = [refine(img01, sign, boxes[j]) for j in top]
            box, bm, ba = max(cands, key=lambda c: c[1])
        for k, v in zip(res_d, (i, it["image_name"], box, bm, ba, float(fm[0]), float(fa[0]))):
            res_d[k].append(v)
        if (i + 1) % 30 == 0 or (i + 1) == len(ds):
            print(f"  [RECT-V3 {method}/{split}] {i+1}/{len(ds)} ({(time.time()-t0)/(i+1):.2f}s/img)", flush=True)

    out_p = out_dir / f"{method}_{split}.npz"
    np.savez(out_p, **{k: np.array(v) for k, v in res_d.items()})
    vol.commit()
    print(f"SAVED {out_p} in {(time.time()-t0)/60:.1f}m", flush=True)

    if split == "train":
        rows = []
        for i in range(len(ds)):
            it = ds[i]; mask = it["mask"] > 0
            ys, xs = np.where(mask)
            y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
            kind = "large" if it["full_watermark"] else ("rect" if mask.sum() / ((y1-y0)*(x1-x0)) > 0.98 else "object")
            pb = np.zeros_like(mask)
            b = np.round(res_d["box"][i]).astype(int)
            pb[b[0]:b[2], b[1]:b[3]] = True
            iou = (pb & mask).sum() / max((pb | mask).sum(), 1)
            rows.append((kind, iou, res_d["margin"][i], res_d["acc"][i]))
        for kind in ("rect", "object", "large"):
            r = [x for x in rows if x[0] == kind]
            iou = np.array([x[1] for x in r]); ac = np.array([x[3] for x in r])
            print(f"  [RECT-V3 {method}] {kind:6s} n={len(r):2d} | box_IoU={iou.mean():.4f} (>0.5:{(iou>0.5).mean():.2f} >0.8:{(iou>0.8).mean():.2f}) | med_acc={np.median(ac):.3f}", flush=True)


@app.local_entrypoint()
def main():
    h1 = run_rect_v3.spawn("stegastamp", "train")
    h2 = run_rect_v3.spawn("stegastamp", "test")
    h3 = run_rect_v3.spawn("pixelseal", "train")
    h4 = run_rect_v3.spawn("pixelseal", "test")
    for h in (h1, h2, h3, h4):
        h.get()
    print("=== ALL 4 RECT-V3 JOBS DONE ===")
