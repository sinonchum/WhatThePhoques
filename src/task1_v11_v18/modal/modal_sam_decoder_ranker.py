"""Modal 2xA100-80GB/H100 White-Box Physics Upgrade (`V7-SAM-DecRank + PixelSeal Super-Rect-V4`):
  - PART A: For ALL 510 images per method (PixelSeal & TrustMark), takes top-16 cached SAM-ViT-H segments
    and computes EXACT WHITE-BOX WATERMARK DECODER OCCLUSION DROP (`base_margin - decode(I with S_j blurred)`)
    in a single batched GPU pass (~50s total!).
  - PART B: PixelSeal Super-Rect-V4: uses our brand-new V6-Fix01 MetaStack proposals (7 thresholds) +
    high-solidity SAM bboxes + fine coordinate refinement through PixelSeal `crop256` decoder (~45s total!).
  - PART C: Strict 4-fold nested DEV CV fusion:
    * Learned multi-feature SAM candidate ranker (combining neural soft_iou, contrast, max_lg, AND white-box
      decoder occlusion drop & drop-per-area!)
    * Super-Rect-V4 + rect_nnref union for PixelSeal
    * Evaluates ONCE on untouched 102 holdout -> saves verified `sub_v7_sam_decrank.pt`!
"""
import modal

app = modal.App("task1-v7-sam-decrank")
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
    .pip_install("git+https://github.com/facebookresearch/segment-anything.git")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1200)
def compute_sam_dec_and_rect_v4(method: str):
    """Compute white-box decoder occlusion drop for top-20 SAM-ViT-H segments of every image in `method`,
    PLUS Super-Rect-V4 for PixelSeal!
    """
    import os, sys, time, pathlib
    import numpy as np
    from scipy import ndimage
    import torch
    import torch.nn.functional as F

    os.environ["WMLOC_MODELS"] = "/vol/models"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark

    dev = "cuda:0"
    t0 = time.time()
    wm = load_watermark(method, device=dev)
    tm_rm = None
    if method == "trustmark":
        from trustmark import TrustMark
        tm_rm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5).removal.to(dev).eval()
        for p in tm_rm.parameters(): p.requires_grad_(False)

    # Load cached SAM-ViT-H shards
    sam_packed = {}
    for r in range(4):
        z = np.load(f"/vol/outputs/sam_h/shard_{r}.npz")
        for k in z.files:
            sam_packed[k] = z[k]

    # Load top-8 neural ensemble (use top-1 + mean of top-4 for fast, leak-free candidate pre-selection)
    ez = np.load("/vol/jureca_ens/ens_top8_v4_fix01.npz", allow_pickle=True)
    lg_by_name = {}
    for sp in ("dev", "hold", "test"):
        nms = [str(x) for x in ez[f"{method}_{sp}_names"]]
        X = ez[f"{method}_X_{sp}"].astype(np.float32)  # [N, M, 128, 128]
        # Simple mean of all clean models (strictly zero label leakage!)
        m_lg = X.mean(axis=1)
        for nm, lg in zip(nms, m_lg):
            lg_by_name[nm] = lg

    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    items = [ds_tr[i] for i in range(len(ds_tr)) if ds_tr[i]["method"] == method] + \
            [ds_te[i] for i in range(len(ds_te)) if ds_te[i]["method"] == method]
    print(f"[{method}] Scoring white-box decoder occlusion on {len(items)} images...", flush=True)

    @torch.no_grad()
    def decode_batch(imgs01_b, sign_t):
        """imgs01_b: [B, 3, H, W] in [0, 1] on GPU -> returns (margin [B], acc [B])"""
        x256 = F.interpolate(imgs01_b, (256, 256), mode="bilinear", align_corners=False) if imgs01_b.shape[-1] != 256 else imgs01_b
        if method == "pixelseal":
            bits = wm.det(x256)[:, 1:].float()
            margin = (bits * sign_t[None, :]).mean(dim=1)
            acc = ((bits > 0) == (sign_t[None, :] > 0)).float().mean(dim=1)
            return margin, acc
        else:
            logits = wm.net(x256 * 2.0 - 1.0).float()
            margin = (logits * sign_t[None, :]).mean(dim=1)
            acc = ((logits > 0) == (sign_t[None, :] > 0)).float().mean(dim=1)
            return margin, acc

    # Gaussian blur kernel for destroying watermark inside candidate segment S_j
    def make_blur_k(sigma=7.0, k=25):
        ax = torch.arange(k, device=dev, dtype=torch.float32) - k // 2
        g = torch.exp(-0.5 * (ax / sigma) ** 2)
        g = g / g.sum()
        k2d = (g[:, None] * g[None, :])[None, None, :, :].repeat(3, 1, 1, 1)
        return k2d
    blur_k = make_blur_k(7.0, 25)

    def blur_img(x01):
        return F.conv2d(F.pad(x01, (12, 12, 12, 12), mode="replicate"), blur_k, groups=3)

    # For TrustMark, use Adobe KBNet_s remover! Removing watermark inside S_j is cleaner than blur!
    @torch.no_grad()
    def clean_version(x01):
        if tm_rm is not None:
            x256 = F.interpolate(x01, (256, 256), mode="bilinear", align_corners=False)
            ro = tm_rm(x256 * 2.0 - 1.0); ro = ro[0] if isinstance(ro, tuple) else ro
            rm256 = ((ro.float() + 1.0) * 0.5).clamp(0, 1)
            rm512 = F.interpolate(rm256, (512, 512), mode="bilinear", align_corners=False)
            return 0.6 * rm512 + 0.4 * blur_img(x01)
        return blur_img(x01)

    K_TOP = 24
    names_out = []
    cand_indices_out = []
    occ_drop_out = []
    occ_acc_drop_out = []
    base_margin_out = []
    base_acc_out = []
    # For PixelSeal Super-Rect-V4
    rv4_box = []
    rv4_margin = []
    rv4_acc = []
    rv4_full_margin = []
    rv4_full_acc = []

    for idx, it in enumerate(items):
        nm = it["image_name"]
        img = it["image"]  # [512, 512, 3] uint8
        msg = np.asarray(it["message"], dtype=np.float32)
        sign_t = torch.from_numpy(np.where(msg > 0.5, 1.0, -1.0)).to(dev, dtype=torch.float32)
        x01 = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(dev, dtype=torch.float32) / 255.0
        x_cln = clean_version(x01)

        # Base decode (full image)
        bm, ba = decode_batch(x01, sign_t)
        bm_f, ba_f = float(bm[0].item()), float(ba[0].item())

        # Unpack SAM candidates
        p = sam_packed.get(nm)
        if p is not None and len(p) > 0:
            cands = np.unpackbits(p, axis=1).reshape(-1, 128, 128).astype(bool)
        else:
            cands = np.zeros((0, 128, 128), dtype=bool)

        lg = lg_by_name[nm]
        isL = (lg >= 0.0).mean() > 0.5
        sig = (-lg) if isL else lg
        prob_min = 1.0 / (1.0 + np.exp(np.clip(-sig, -20, 20)))

        drops = np.zeros(K_TOP, np.float32)
        adrops = np.zeros(K_TOP, np.float32)
        cidx = np.full(K_TOP, -1, np.int32)

        if len(cands) > 0:
            cf = cands.reshape(len(cands), -1).astype(np.float32)
            pf = prob_min.ravel()[None, :]
            sf = sig.ravel()[None, :]
            si = (cf * pf).sum(1) / (cf + pf - cf * pf).sum(1).clip(min=1e-4)
            s_in = (cf * sf).sum(1) / cf.sum(1).clip(min=1.0)
            s_out = ((1.0 - cf) * sf).sum(1) / (1.0 - cf).sum(1).clip(min=1.0)
            ct = s_in - s_out
            # Select diverse top-K candidates: top-12 by soft_iou + top-12 by contrast!
            sel = []
            for j in list(np.argsort(-si)[:14]) + list(np.argsort(-ct)[:14]):
                if int(j) not in sel and len(sel) < K_TOP:
                    sel.append(int(j))
            M_sel = len(sel)
            cidx[:M_sel] = sel
            # Batch-compose all M_sel occluded 512x512 images on GPU and decode in ONE shot!
            m128 = torch.from_numpy(cands[sel].astype(np.float32)).to(dev)[:, None, :, :]
            m512 = F.interpolate(m128, (512, 512), mode="nearest")
            imgs_occ = x01 * (1.0 - m512) + x_cln * m512  # [M_sel, 3, 512, 512]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                om, oa = decode_batch(imgs_occ, sign_t)
            drops[:M_sel] = (bm_f - om.float().cpu().numpy())
            adrops[:M_sel] = (ba_f - oa.float().cpu().numpy())

        names_out.append(nm)
        cand_indices_out.append(cidx)
        occ_drop_out.append(drops)
        occ_acc_drop_out.append(adrops)
        base_margin_out.append(bm_f)
        base_acc_out.append(ba_f)

        # PART B: PixelSeal Super-Rect-V4 (test ALL neural multi-thr bboxes + high-solidity SAM bboxes + refine!)
        if method == "pixelseal":
            if ba_f >= 0.85:
                rv4_box.append(np.array([0, 0, 512, 512], np.float32))
                rv4_margin.append(bm_f); rv4_acc.append(ba_f)
                rv4_full_margin.append(bm_f); rv4_full_acc.append(ba_f)
            else:
                boxes = []
                # 1) Multi-threshold connected-component bboxes from neural ensemble
                for thr_q in (-2.5, -1.5, -0.8, -0.2, 0.3, 0.8, 1.5):
                    bw = lg >= thr_q
                    if bw.sum() < 6 or bw.mean() > 0.55: continue
                    lab, nl = ndimage.label(bw)
                    for c in range(1, min(nl + 1, 6)):
                        ys, xs = np.where(lab == c)
                        if len(ys) < 6: continue
                        y0, y1 = max(0, int(ys.min()*4)), min(512, int((ys.max()+1)*4))
                        x0, x1 = max(0, int(xs.min()*4)), min(512, int((xs.max()+1)*4))
                        if (y1 - y0) >= 8 and (x1 - x0) >= 8:
                            boxes.append([y0, x0, y1, x1])
                            # Also padded & shrunk variants
                            dh, dw = max(4, int(0.1*(y1-y0))), max(4, int(0.1*(x1-x0)))
                            boxes.append([max(0, y0-dh), max(0, x0-dw), min(512, y1+dh), min(512, x1+dw)])
                # 2) High-solidity SAM candidate bboxes (solidity > 0.72 -> likely rectangular outpaintings!)
                for j in range(len(cands)):
                    ys, xs = np.where(cands[j])
                    if len(ys) < 8: continue
                    ba_j = (ys.max()-ys.min()+1)*(xs.max()-xs.min()+1)
                    if len(ys) / max(ba_j, 1) >= 0.72:
                        y0, y1 = int(ys.min()*4), int((ys.max()+1)*4)
                        x0, x1 = int(xs.min()*4), int((xs.max()+1)*4)
                        if (y1 - y0) >= 10 and (x1 - x0) >= 10:
                            boxes.append([y0, x0, y1, x1])
                # 3) Coarse 16:1 strip sweep around neural peak!
                py, px = np.unravel_index(np.argmax(ndimage.gaussian_filter(lg, 1.0)), (128, 128))
                cy, cx = int(py*4+2), int(px*4+2)
                for h in (16, 24, 36, 56, 84, 128, 192, 256):
                    for w in (16, 24, 36, 56, 84, 128, 192, 256):
                        if max(h/w, w/h) > 16.0: continue
                        y0 = max(0, min(512-h, cy - h//2)); x0 = max(0, min(512-w, cx - w//2))
                        boxes.append([y0, x0, y0+h, x0+w])

                def eval_ps_boxes(blist):
                    if not blist: return np.array([]), np.array([])
                    ms, acs = [], []
                    for s in range(0, len(blist), 64):
                        ch = blist[s:s+64]
                        crops = torch.cat([
                            F.interpolate(x01[:, :, max(0,int(y0)):min(512,int(y1)), max(0,int(x0)):min(512,int(x1))],
                                          (256, 256), mode="bilinear", align_corners=False)
                            for (y0, x0, y1, x1) in ch if y1 > y0 + 2 and x1 > x0 + 2
                        ], dim=0)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            m_b, a_b = decode_batch(crops, sign_t)
                        ms.append(m_b.float().cpu().numpy()); acs.append(a_b.float().cpu().numpy())
                    return np.concatenate(ms), np.concatenate(acs)

                boxes = [b for b in boxes if b[2] > b[0] + 4 and b[3] > b[1] + 4]
                ms0, ac0 = eval_ps_boxes(boxes)
                if len(ms0) > 0:
                    top_k = np.argsort(-ms0)[:3]
                    best_b = np.array(boxes[top_k[0]], np.float32)
                    best_m, best_a = float(ms0[top_k[0]]), float(ac0[top_k[0]])
                    # Coordinate refinement on top-2 seeds
                    for seed_i in top_k[:2]:
                        cb = np.array(boxes[seed_i], np.float32)
                        cm, ca = float(ms0[seed_i]), float(ac0[seed_i])
                        for frac in (0.18, 0.09, 0.04, 0.02):
                            h = max(cb[2]-cb[0], 8.0); w = max(cb[3]-cb[1], 8.0)
                            dh = max(2.0, h * frac); dw = max(2.0, w * frac)
                            cands_r = []
                            for side, d in ((0, -dh), (0, dh), (2, -dh), (2, dh), (1, -dw), (1, dw), (3, -dw), (3, dw)):
                                nb = cb.copy(); nb[side] = np.clip(nb[side] + d, 0, 512)
                                if nb[2] - nb[0] >= 8 and nb[3] - nb[1] >= 8:
                                    cands_r.append(nb.tolist())
                            if not cands_r: continue
                            rm_r, ra_r = eval_ps_boxes(cands_r)
                            j_r = int(np.argmax(rm_r))
                            if float(rm_r[j_r]) > cm:
                                cm, ca, cb = float(rm_r[j_r]), float(ra_r[j_r]), np.array(cands_r[j_r], np.float32)
                        if cm > best_m:
                            best_m, best_a, best_b = cm, ca, cb
                else:
                    best_b = np.array([0, 0, 512, 512], np.float32); best_m, best_a = bm_f, ba_f
                rv4_box.append(best_b); rv4_margin.append(best_m); rv4_acc.append(best_a)
                rv4_full_margin.append(bm_f); rv4_full_acc.append(ba_f)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(items):
            print(f"  [{method}] {idx+1}/{len(items)} ({(time.time()-t0)/(idx+1):.2f}s/img)", flush=True)

    out_dir = pathlib.Path("/vol/outputs/sam_dec_feats")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{method}.npz",
        names=np.array(names_out),
        cand_idx=np.stack(cand_indices_out),
        occ_drop=np.stack(occ_drop_out),
        occ_acc_drop=np.stack(occ_acc_drop_out),
        base_margin=np.array(base_margin_out, np.float32),
        base_acc=np.array(base_acc_out, np.float32),
    )
    if method == "pixelseal":
        np.savez_compressed(
            out_dir / "pixelseal_rect_v4.npz",
            names=np.array(names_out),
            box=np.stack(rv4_box),
            margin=np.array(rv4_margin, np.float32),
            acc=np.array(rv4_acc, np.float32),
            full_margin=np.array(rv4_full_margin, np.float32),
            full_acc=np.array(rv4_full_acc, np.float32),
        )
    vol.commit()
    print(f"[{method}] SAVED white-box SAM occlusion features in {time.time()-t0:.1f}s!", flush=True)


@app.function(image=image, gpu=["A100-80GB", "L40S", "A10G", "H100"], volumes={"/vol": vol}, timeout=900)
def eval_v7_sam_decrank():
    """Strict 4-fold nested DEV CV fusion using:
      1. Top-8 clean honest-OOF neural models + GPU MetaStackNet
      2. Adaptive Split-Thr + Hole-Dip Recovery
      3. Super-Rect-V4 + rect_nnref (best per-image by margin!) for PixelSeal, rect_nnref for TrustMark
      4. Learned Multi-Feature SAM-ViT-H Segment Ranker (using neural features + WHITE-BOX DECODER OCCLUSION DROP!)
    """
    import sys, json, time, pathlib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = "cuda:0"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import miou_image, downsample_mask

    sam_packed = {}
    for r in range(4):
        z = np.load(f"/vol/outputs/sam_h/shard_{r}.npz")
        for k in z.files: sam_packed[k] = z[k]

    dec_feats = {}
    for m in ("pixelseal", "trustmark"):
        z = np.load(f"/vol/outputs/sam_dec_feats/{m}.npz", allow_pickle=True)
        dec_feats[m] = {str(z["names"][i]): {
            "cand_idx": z["cand_idx"][i], "occ_drop": z["occ_drop"][i],
            "occ_acc_drop": z["occ_acc_drop"][i], "base_margin": float(z["base_margin"][i]),
            "base_acc": float(z["base_acc"][i]),
        } for i in range(len(z["names"]))}

    ez = np.load("/vol/jureca_ens/ens_top8_v4_fix01.npz", allow_pickle=True)
    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    tr_by_name = {ds_tr[i]["image_name"]: ds_tr[i] for i in range(len(ds_tr))}
    te_names_all = [ds_te[i]["image_name"] for i in range(len(ds_te))]

    def unpack_cands(nm):
        p = sam_packed.get(nm)
        if p is None or len(p) == 0: return np.zeros((0, 128, 128), dtype=bool)
        return np.unpackbits(p, axis=1).reshape(-1, 128, 128).astype(bool)

    def fast_miou_vec(pred, gt):
        p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
        i1 = (p & g).sum(1); u1 = (p | g).sum(1)
        i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
        return 0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0) + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0))

    def load_rect(p):
        z = np.load(p, allow_pickle=True)
        return {str(z["names"][i]): {
            "box": z["box"][i].astype(np.float32), "margin": float(z["margin"][i]),
            "acc": float(z["acc"][i]), "full_margin": float(z["full_margin"][i]),
            "full_acc": float(z["full_acc"][i]),
        } for i in range(len(z["names"]))}

    def box_to_m128(box):
        m = np.zeros((128, 128), dtype=bool)
        y0, x0, y1, x1 = np.clip(np.round(box / 4.0).astype(int), 0, 128)
        if y1 > y0 and x1 > x0: m[y0:y1, x0:x1] = True
        return m

    def rect_solidity(m):
        ys, xs = np.where(m)
        if len(ys) < 16: return 0.0
        return float(len(ys)) / max(float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)), 1.0)

    class MetaStackNet(nn.Module):
        def __init__(self, M, init_w, init_b):
            super().__init__()
            self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
            self.b = nn.Parameter(torch.tensor(float(init_b), dtype=torch.float32))
            in_ch = M + 4
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 24, 3, padding=1), nn.GroupNorm(4, 24), nn.GELU(),
                nn.Conv2d(24, 24, 3, padding=2, dilation=2), nn.GroupNorm(4, 24), nn.GELU(),
                nn.Conv2d(24, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
                nn.Conv2d(16, 1, 1),
            )
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
            self.res_scale = nn.Parameter(torch.tensor(0.25))

        def forward(self, x):
            w = torch.softmax(self.w, dim=0)
            base = (x * w[None, :, None, None]).sum(1, keepdim=True) + self.b
            feat = torch.cat([x, x.mean(1, keepdim=True), x.std(1, keepdim=True),
                              base.mean((2, 3), keepdim=True).expand_as(base),
                              base.std((2, 3), keepdim=True).expand_as(base)], dim=1)
            return (base + self.res_scale * self.net(feat)).squeeze(1)

    def sym_miou_loss(logits, target):
        p = torch.sigmoid(logits).flatten(1); y = target.flatten(1)
        i1 = (p * y).sum(1); u1 = (p + y - p * y).sum(1)
        i0 = ((1 - p) * (1 - y)).sum(1); u0 = ((1 - p) + (1 - y) - (1 - p) * (1 - y)).sum(1)
        return (1.0 - 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0))).mean() + 0.15 * F.binary_cross_entropy_with_logits(logits, target)

    def fit_meta(Xd, Xh, Xt, dg, dL):
        N, M = Xd.shape[0], Xd.shape[1]
        idx_L = np.where(dL)[0]; idx_S = np.where(~dL)[0]
        folds = [[] for _ in range(4)]
        for k, i in enumerate(idx_L): folds[k % 4].append(i)
        for k, i in enumerate(idx_S): folds[k % 4].append(i)
        def opt_w(Xtr, gtr):
            w = np.zeros(M, np.float32); w[0] = 1.0
            best = float(fast_miou_vec(Xtr[:, 0] >= 0.0, gtr).mean())
            for _ in range(3):
                for j in range(M):
                    for step in (-0.3, -0.15, 0.15, 0.3, 0.5, 1.0):
                        wc = w.copy(); wc[j] = max(0.0, wc[j] + step)
                        if wc.sum() <= 0: continue
                        wc /= wc.sum()
                        sc = float(fast_miou_vec(np.tensordot(Xtr, wc, axes=([1], [0])) >= 0.0, gtr).mean())
                        if sc > best + 1e-5: best, w = sc, wc
            lg = np.tensordot(Xtr, w, axes=([1], [0]))
            bb, bsb = 0.0, best
            for b in np.linspace(-0.6, 0.6, 25):
                sc = float(fast_miou_vec(lg + b >= 0.0, gtr).mean())
                if sc > bsb: bsb, bb = sc, float(b)
            return w, bb
        meta_oof = np.zeros((N, 128, 128), np.float32); meta_hs, meta_ts = [], []
        Xd_t = torch.from_numpy(Xd).to(dev); yd_t = torch.from_numpy(dg.astype(np.float32)).to(dev)
        Xh_t = torch.from_numpy(Xh).to(dev); Xt_t = torch.from_numpy(Xt).to(dev)
        for k in range(4):
            vi = np.array(folds[k]); ti = np.array([i for i in range(N) if i not in set(folds[k])])
            wk, bk = opt_w(Xd[ti], dg[ti])
            iw = np.log(np.maximum(wk / max(float(wk.sum()), 1e-6), 1e-2))
            torch.manual_seed(42 + k)
            net = MetaStackNet(M, iw, bk).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=2e-2)
            bsc, bval, bst = -1.0, None, None
            for ep in range(60):
                net.train()
                perm = ti[np.random.RandomState(100*k + ep).permutation(len(ti))]
                for s in range(0, len(perm), 16):
                    bi = perm[s:s+16]; opt.zero_grad()
                    sym_miou_loss(net(Xd_t[bi]), yd_t[bi]).backward(); opt.step()
                if (ep + 1) % 5 == 0:
                    net.eval()
                    with torch.no_grad(): vl = net(Xd_t[vi]).cpu().numpy()
                    sc = float(fast_miou_vec(vl >= 0.0, dg[vi]).mean())
                    if sc > bsc:
                        bsc, bval = sc, vl
                        bst = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
            meta_oof[vi] = bval
            net.load_state_dict(bst); net.eval()
            with torch.no_grad():
                meta_hs.append(net(Xh_t).cpu().numpy()); meta_ts.append(net(Xt_t).cpu().numpy())
        return meta_oof, np.mean(meta_hs, axis=0), np.mean(meta_ts, axis=0)

    # First check PixelSeal Super-Rect-V4 vs rect_nnref on DEV!
    ps_rv4 = load_rect("/vol/outputs/sam_dec_feats/pixelseal_rect_v4.npz")
    ps_nnref_tr = load_rect("/vol/jureca_ens/pixelseal_train_nnref.npz")
    ps_nnref_te = load_rect("/vol/jureca_ens/pixelseal_test_nnref.npz")

    # Merge PixelSeal rect_nnref + Super-Rect-V4: pick whichever box has HIGHER decoder margin!
    def merge_rects(r1, r2):
        out = {}
        for k in r1:
            a, b = r1[k], r2.get(k, r1[k])
            out[k] = b if b["margin"] > a["margin"] else a
        return out
    ps_rect_merged_tr = merge_rects(ps_nnref_tr, ps_rv4)
    ps_rect_merged_te = merge_rects(ps_nnref_te, ps_rv4)

    test_masks = {}
    summary = {}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        mnames = [str(x) for x in ez[f"{method}_names_m"]]
        dn = [str(x) for x in ez[f"{method}_dev_names"]]
        hn = [str(x) for x in ez[f"{method}_hold_names"]]
        tn = [str(x) for x in ez[f"{method}_test_names"]]
        Xd = ez[f"{method}_X_dev"].astype(np.float32)
        Xh = ez[f"{method}_X_hold"].astype(np.float32)
        Xt = ez[f"{method}_X_test"].astype(np.float32)
        dg = np.stack([downsample_mask(tr_by_name[n]["mask"] > 0) > 0 for n in dn])
        hg = np.stack([downsample_mask(tr_by_name[n]["mask"] > 0) > 0 for n in hn])
        dL = np.array([bool(tr_by_name[n]["full_watermark"]) for n in dn])
        hL = np.array([bool(tr_by_name[n]["full_watermark"]) for n in hn])

        m_sc = [float(fast_miou_vec(Xd[:, j] >= 0.0, dg).mean()) for j in range(len(mnames))]
        order = np.argsort(-np.array(m_sc))[:8]
        Xd, Xh, Xt = Xd[:, order], Xh[:, order], Xt[:, order]
        d_lg, h_lg, t_lg = fit_meta(Xd, Xh, Xt, dg, dL)
        base_d = float(fast_miou_vec(d_lg >= 0.0, dg).mean())
        base_h = float(fast_miou_vec(h_lg >= 0.0, hg).mean())
        print(f"\n================ {method.upper()} (V7 META dev={base_d:.4f} hold={base_h:.4f}) ================", flush=True)

        # 1. Adaptive Split-Thr
        def apply_adapt(lg, tL, tS, zL, zS):
            isL = (lg >= 0.0).mean(axis=(1, 2)) > 0.5
            out = np.where(isL[:, None, None], lg >= tL, lg >= tS)
            for i in range(len(lg)):
                m = lg[i]; mu = float(m.mean()); sd = float(m.std()) + 1e-6
                if isL[i] and (~out[i]).mean() < 0.005 and zL < 0: out[i] = (m - mu) / sd >= zL
                elif (not isL[i]) and out[i].mean() < 0.005 and zS > 0: out[i] = (m - mu) / sd >= zS
            return out
        ad_grid = [(tL, tS, zL, zS) for tL in (-0.2, 0.0, 0.4, 0.8) for tS in (0.0, -0.4, -0.8) for zL in (0.0, -2.5, -3.0) for zS in (0.0, 2.5, 3.0)]
        ad_scores = [fast_miou_vec(apply_adapt(d_lg, *c), dg) for c in ad_grid]
        cv_ad = np.zeros(len(dg), np.float32)
        for fold in range(4):
            vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
            cv_ad[vm] = ad_scores[max(range(len(ad_grid)), key=lambda idx: float(ad_scores[idx][tm].mean()))][vm]
        if cv_ad.mean() > base_d + 0.0005:
            ad_cfg = ad_grid[max(range(len(ad_grid)), key=lambda idx: float(ad_scores[idx].mean()))]
            d_m, h_m, t_m = apply_adapt(d_lg, *ad_cfg), apply_adapt(h_lg, *ad_cfg), apply_adapt(t_lg, *ad_cfg)
            print(f"  [1. ADAPT] nested-CV={cv_ad.mean():.4f} cfg={ad_cfg} | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)
        else:
            d_m, h_m, t_m = d_lg >= 0.0, h_lg >= 0.0, t_lg >= 0.0

        # 2. Rect Override (compare rect_nnref vs merged Super-Rect-V4 for PixelSeal strictly on DEV CV!)
        d_rect_trig = np.zeros(len(dn), bool); h_rect_trig = np.zeros(len(hn), bool); t_rect_trig = np.zeros(len(tn), bool)
        if method in ("pixelseal", "trustmark"):
            rect_cands = [("rect_nnref", load_rect(f"/vol/jureca_ens/{method}_train_nnref.npz"), load_rect(f"/vol/jureca_ens/{method}_test_nnref.npz"))]
            if method == "pixelseal":
                rect_cands.append(("rect_v4_merged", ps_rect_merged_tr, ps_rect_merged_te))
            best_r_cv, best_r_pack = float(fast_miou_vec(d_m, dg).mean()), None
            for rname, rd_tr, rd_te in rect_cands:
                d_rec = [rd_tr[n] for n in dn]; h_rec = [rd_tr[n] for n in hn]; t_rec = [rd_te[n] for n in tn]
                d_bm = np.stack([box_to_m128(r["box"]) for r in d_rec])
                h_bm = np.stack([box_to_m128(r["box"]) for r in h_rec])
                t_bm = np.stack([box_to_m128(r["box"]) for r in t_rec])
                base_v = fast_miou_vec(d_m, dg); rep_v = fast_miou_vec(d_bm, dg)
                fa_d = np.array([r["full_acc"] for r in d_rec]); ac_d = np.array([r["acc"] for r in d_rec])
                gn_d = np.array([r["margin"] - r["full_margin"] for r in d_rec])
                rgrid = [(fa, ac, gn) for fa in (0.58, 0.62, 0.68, 0.75) for ac in (0.62, 0.66, 0.70, 0.76) for gn in (0.15, 0.40, 0.80, 1.20)]
                cv_r = np.zeros_like(base_v)
                for fold in range(4):
                    vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
                    bc = max(rgrid, key=lambda c: float(np.where((fa_d < c[0]) & (ac_d >= c[1]) & (gn_d >= c[2]), rep_v, base_v)[tm].mean()))
                    cv_r[vm] = np.where((fa_d < bc[0]) & (ac_d >= bc[1]) & (gn_d >= bc[2]), rep_v, base_v)[vm]
                brc = max(rgrid, key=lambda c: float(np.where((fa_d < c[0]) & (ac_d >= c[1]) & (gn_d >= c[2]), rep_v, base_v).mean()))
                print(f"  [2. {rname:14s}] nested-CV={cv_r.mean():.4f} cfg={brc}", flush=True)
                if cv_r.mean() > best_r_cv + 0.001:
                    best_r_cv = float(cv_r.mean())
                    best_r_pack = (rname, brc, fa_d, ac_d, gn_d, d_rec, h_rec, t_rec, d_bm, h_bm, t_bm)
            if best_r_pack is not None:
                rname, (fa, ac, gn), fa_d, ac_d, gn_d, d_rec, h_rec, t_rec, d_bm, h_bm, t_bm = best_r_pack
                d_rect_trig = (fa_d < fa) & (ac_d >= ac) & (gn_d >= gn)
                h_rect_trig = np.array([r["full_acc"] < fa and r["acc"] >= ac and (r["margin"]-r["full_margin"]) >= gn for r in h_rec])
                t_rect_trig = np.array([r["full_acc"] < fa and r["acc"] >= ac and (r["margin"]-r["full_margin"]) >= gn for r in t_rec])
                d_m = np.where(d_rect_trig[:, None, None], d_bm, d_m)
                h_m = np.where(h_rect_trig[:, None, None], h_bm, h_m)
                t_m = np.where(t_rect_trig[:, None, None], t_bm, t_m)
                print(f"     >> ADOPTED {rname}! trig={d_rect_trig.sum()}/136 hold={h_rect_trig.sum()}/34 | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        # 3. SAM-ViT-H + WHITE-BOX DECODER OCCLUSION RANKER (strict 4-fold nested DEV CV!)
        def match_sam_with_dec(nm, lg, isL_flag, cur_mask, w_dec=0.0):
            cands = unpack_cands(nm)
            cur_min = (~cur_mask) if isL_flag else cur_mask
            sol = rect_solidity(cur_min)
            if len(cands) == 0:
                return None, None, 0.0, 0.0, -99.0, 0.0, sol, 0.0
            if isL_flag:
                prob_min = 1.0 / (1.0 + np.exp(np.clip(lg, -20, 20))); sig = -lg
            else:
                prob_min = 1.0 / (1.0 + np.exp(np.clip(-lg, -20, 20))); sig = lg
            cf = cands.reshape(len(cands), -1).astype(np.float32)
            pf = prob_min.ravel()[None, :]; sf = sig.ravel()[None, :]; cmf = cur_min.ravel().astype(np.float32)[None, :]
            soft_iou = (cf * pf).sum(1) / (cf + pf - cf * pf).sum(1).clip(min=1e-4)

            # Map white-box decoder occlusion drop onto candidate array!
            dec_sig = np.zeros(len(cands), np.float32)
            if method in dec_feats and nm in dec_feats[method]:
                df = dec_feats[method][nm]
                c_i = df["cand_idx"]; dr = df["occ_drop"]; adr = df["occ_acc_drop"]
                bm_f = max(abs(df["base_margin"]), 0.05)
                for k_i in range(len(c_i)):
                    ci = int(c_i[k_i])
                    if 0 <= ci < len(cands):
                        if isL_flag:
                            # For L-hole: a CLEAN HOLE has LOWER drop per area than watermarked objects!
                            ar_c = max(float(cands[ci].mean()), 0.01)
                            dec_sig[ci] = -float(dr[k_i]) / (bm_f * ar_c * 10.0 + 0.2)
                        else:
                            # For S-object: the WATERMARKED OBJECT has POSITIVE occlusion drop!
                            dec_sig[ci] = max(0.0, float(dr[k_i]) / bm_f) + 2.0 * max(0.0, float(adr[k_i]))

            # Combined ranking score = soft_iou + w_dec * clip(dec_sig, -1, 1.5)
            rank_sc = soft_iou + w_dec * np.clip(dec_sig, -1.0, 1.5)
            top_idx = np.argsort(-rank_sc)[:6]
            best_m = cands[top_idx[0]]
            best_si = float(soft_iou[top_idx[0]])
            best_ds = float(dec_sig[top_idx[0]])
            # Check top-2 union
            for j2 in top_idx[1:]:
                u = cands[top_idx[0]] | cands[j2]
                if u.mean() > 0.48: continue
                uf = u.ravel().astype(np.float32)
                si_u = float((uf * pf[0]).sum() / max((uf + pf[0] - uf * pf[0]).sum(), 1e-4))
                if si_u > best_si + 0.025:
                    best_si, best_m = si_u, u
            bmf = best_m.ravel().astype(np.float32)
            bi = float((bmf * cmf[0]).sum() / max((bmf + cmf[0] - bmf * cmf[0]).sum(), 1e-4))
            ct = float((bmf * sf[0]).sum() / max(bmf.sum(), 1.0) - ((1.0 - bmf) * sf[0]).sum() / max((1.0 - bmf).sum(), 1.0))
            ar = float(best_m.mean())
            full_m = (~best_m) if isL_flag else best_m
            return full_m, best_m, bi, best_si, ct, ar, sol, best_ds

        w_dec_list = (0.0, 0.15, 0.35, 0.60) if method in ("pixelseal", "trustmark") else (0.0,)
        # Precompute for each w_dec
        cache = {}
        for wd in w_dec_list:
            def prep(names, lgs, cur_ms):
                isL_a = (lgs >= 0.0).mean(axis=(1, 2)) > 0.5
                sh, sb2, sb4 = cur_ms.copy(), cur_ms.copy(), cur_ms.copy()
                bi = np.zeros(len(names), np.float32); si = np.zeros(len(names), np.float32)
                ct = np.zeros(len(names), np.float32); ar = np.zeros(len(names), np.float32)
                sol = np.zeros(len(names), np.float32); ds_v = np.zeros(len(names), np.float32)
                for i, nm in enumerate(names):
                    cm, bm, bi_, si_, ct_, ar_, sol_, ds_ = match_sam_with_dec(nm, lgs[i], bool(isL_a[i]), cur_ms[i], wd)
                    sol[i] = sol_; ds_v[i] = ds_
                    if cm is not None:
                        sh[i] = cm
                        sgn = (-1.0) if isL_a[i] else 1.0
                        sb2[i] = (lgs[i] + 2.0 * sgn * (2.0 * bm.astype(np.float32) - 1.0)) >= 0.0
                        sb4[i] = (lgs[i] + 4.0 * sgn * (2.0 * bm.astype(np.float32) - 1.0)) >= 0.0
                        bi[i], si[i], ct[i], ar[i] = bi_, si_, ct_, ar_
                return isL_a, {"hard": sh, "b2": sb2, "b4": sb4}, bi, si, ct, ar, sol, ds_v
            cache[wd] = (prep(dn, d_lg, d_m), prep(hn, h_lg, h_m), prep(tn, t_lg, t_m))

        for sub_name, is_L_target in [("SAM_S_OBJ", False), ("SAM_L_HOLE", True)]:
            base_v = fast_miou_vec(d_m, dg)
            # Precompute mIoU vectors for all (wd, mode)
            sv_map = {(wd, md): fast_miou_vec(cache[wd][0][1][md], dg) for wd in w_dec_list for md in ("hard", "b2", "b4")}
            sgrid = [(wd, md, t_si, t_bi, t_ds, s_max)
                     for wd in w_dec_list
                     for md in ("hard", "b2", "b4")
                     for t_si in (0.12, 0.22, 0.35, 0.48)
                     for t_bi in (0.0, 0.25, 0.45)
                     for t_ds in (-9.0, 0.05, 0.20)
                     for s_max in (0.85, 0.92, 1.01)]
            cv_s = np.zeros_like(base_v)
            for fold in range(4):
                vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
                bsc, bc = float(base_v[tm].mean()), None
                for c in sgrid:
                    wd, md, t_si, t_bi, t_ds, s_max = c
                    d_isL, _, d_bi, d_si, d_ct, d_ar, d_sol, d_ds = cache[wd][0]
                    tf = (d_isL == is_L_target) & (~d_rect_trig) & (d_si >= t_si) & (d_bi >= t_bi) & (d_ds >= t_ds) & (d_sol <= s_max) & (d_ar >= 0.008)
                    sc = float(np.where(tf, sv_map[(wd, md)], base_v)[tm].mean())
                    if sc > bsc: bsc, bc = sc, c
                if bc is None: cv_s[vm] = base_v[vm]
                else:
                    wd, md, t_si, t_bi, t_ds, s_max = bc
                    d_isL, _, d_bi, d_si, d_ct, d_ar, d_sol, d_ds = cache[wd][0]
                    tf = (d_isL == is_L_target) & (~d_rect_trig) & (d_si >= t_si) & (d_bi >= t_bi) & (d_ds >= t_ds) & (d_sol <= s_max) & (d_ar >= 0.008)
                    cv_s[vm] = np.where(tf, sv_map[(wd, md)], base_v)[vm]
            bfull, bfc = float(base_v.mean()), None
            for c in sgrid:
                wd, md, t_si, t_bi, t_ds, s_max = c
                d_isL, _, d_bi, d_si, d_ct, d_ar, d_sol, d_ds = cache[wd][0]
                tf = (d_isL == is_L_target) & (~d_rect_trig) & (d_si >= t_si) & (d_bi >= t_bi) & (d_ds >= t_ds) & (d_sol <= s_max) & (d_ar >= 0.008)
                sc = float(np.where(tf, sv_map[(wd, md)], base_v).mean())
                if sc > bfull: bfull, bfc = sc, c
            gain_cv = float(cv_s.mean() - base_v.mean())
            print(f"  [3. {sub_name:10s}] nested-CV={cv_s.mean():.4f} ({gain_cv:+.4f}) full-dev={bfull:.4f} cfg={bfc}", flush=True)
            if bfc is not None and gain_cv > 0.001:
                wd, md, thr_si, thr_bi, thr_ds, sol_max = bfc
                d_isL, d_sams, d_bi, d_si, d_ct, d_ar, d_sol, d_ds = cache[wd][0]
                h_isL, h_sams, h_bi, h_si, h_ct, h_ar, h_sol, h_ds = cache[wd][1]
                t_isL, t_sams, t_bi, t_si, t_ct, t_ar, t_sol, t_ds = cache[wd][2]
                td = (d_isL == is_L_target) & (~d_rect_trig) & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_ds >= thr_ds) & (d_sol <= sol_max) & (d_ar >= 0.008)
                th = (h_isL == is_L_target) & (~h_rect_trig) & (h_si >= thr_si) & (h_bi >= thr_bi) & (h_ds >= thr_ds) & (h_sol <= sol_max) & (h_ar >= 0.008)
                tt = (t_isL == is_L_target) & (~t_rect_trig) & (t_si >= thr_si) & (t_bi >= thr_bi) & (t_ds >= thr_ds) & (t_sol <= sol_max) & (t_ar >= 0.008)
                d_m = np.where(td[:, None, None], d_sams[md], d_m)
                h_m = np.where(th[:, None, None], h_sams[md], h_m)
                t_m = np.where(tt[:, None, None], t_sams[md], t_m)
                print(f"     >> ADOPTED {sub_name} (wd={wd},{md})! trig dev={td.sum()}/136 hold={th.sum()}/34 test={tt.sum()}/340 | hold_now={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        d_off = np.array([miou_image(d_m[i], dg[i]) for i in range(len(dg))])
        h_off = np.array([miou_image(h_m[i], hg[i]) for i in range(len(hg))])
        print(f"  *** FINAL {method.upper()} *** DEV={d_off.mean():.4f} (L {d_off[dL].mean():.4f} S {d_off[~dL].mean():.4f}) | HOLD={h_off.mean():.4f} (L {h_off[hL].mean():.4f} S {h_off[~hL].mean():.4f})", flush=True)
        summary[method] = {"dev": float(d_off.mean()), "hold": float(h_off.mean()), "hold_L": float(h_off[hL].mean()), "hold_S": float(h_off[~hL].mean())}
        for i, n in enumerate(tn): test_masks[n] = t_m[i].astype(np.uint8)

    m_dev = float(np.mean([summary[m]["dev"] for m in summary]))
    m_hld = float(np.mean([summary[m]["hold"] for m in summary]))
    print("\n============================================================", flush=True)
    print(f"FINAL V7-SAM-DECRANK + SUPER-RECT-V4 DEV  OOF : {m_dev:.4f}", flush=True)
    print(f"FINAL V7-SAM-DECRANK + SUPER-RECT-V4 HOLDOUT  : {m_hld:.4f}", flush=True)
    print(f"PER-METHOD HOLDOUT: {summary}", flush=True)
    print("============================================================", flush=True)
    masks_u8 = torch.from_numpy(np.stack([test_masks[n] for n in te_names_all])).to(torch.uint8)
    out_p = pathlib.Path("/vol/outputs/sub_v7_sam_decrank.pt")
    torch.save({"image_names": te_names_all, "masks": masks_u8}, out_p)
    (out_p.with_suffix(".json")).write_text(json.dumps({"dev": m_dev, "hold": m_hld, "methods": summary}, indent=2))
    vol.commit()
    print(f"SAVED {out_p} ({out_p.stat().st_size/1e6:.2f} MB, fg={masks_u8.float().mean():.4f})", flush=True)
    return m_dev, m_hld


@app.local_entrypoint()
def main():
    h1 = compute_sam_dec_and_rect_v4.spawn("pixelseal")
    h2 = compute_sam_dec_and_rect_v4.spawn("trustmark")
    h1.get(); h2.get()
    dev_s, hld_s = eval_v7_sam_decrank.remote()
    print(f"=== V7 COMPLETE: DEV={dev_s:.4f} | HOLD={hld_s:.4f} ===")
