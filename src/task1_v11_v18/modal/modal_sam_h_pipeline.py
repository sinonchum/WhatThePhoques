"""SAM-ViT-H (Huge) 4-GPU Parallel Object & Hole Snapping Pipeline + Strict 4-Fold Nested DEV-CV Fusion!
100% CLEAN (zero holdout tuning, exact HOLDOUT_102 match to JURECA!).
Stage 1 (4x A100-80GB/H100 in parallel, ~2.5 min):
  - Generates ~50-80 pixel-sharp SAM-ViT-H candidates per image for ALL 1,530 images (510 train + 1,020 test):
    (a) SamAutomaticMaskGenerator (32x32 grid)
    (b) Neural-Prompted SamPredictor (prompted by JURECA ensemble peak/dip points & multi-thr boxes!)
Stage 2 (1 GPU, ~30s):
  - Strict 4-fold nested DEV CV optimizing:
      1) Adaptive L/S Split-Threshold + Hole-Dip Recovery (tL, tS, zL, zS)
      2) rect_nnref exact rectangle override (small, rect)
      3) SAM-ViT-H Segment Snapping for S-object AND high-overlap L-hole (Object Addition/Alteration)!
  - Evaluates ONCE on untouched 102 holdout images + writes verified `sub_v5_samh_clean.pt` (<20 MB)!
"""
import modal

app = modal.App("task1-sam-huge-pipeline")
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

HOLDOUT_102 = {
    "01a775d8e589", "01a8a19cb848", "0703932c7705", "07b2717b53ad", "0ae0e6224c29", "0b35f8d56e9f",
    "0f715ead6d99", "124efa616612", "15097a1653aa", "17612bb5282e", "18953a989e61", "1974c13982d2",
    "1a46eb491c4f", "2260d5ffabd1", "22df37a04afe", "23bfdcdaecd3", "2610e4f2a637", "2637e305ead3",
    "279267bc151a", "28e8c53f7a62", "2df125cd8603", "2dfa53b9ecf4", "2e731007efaf", "34db63f9db87",
    "37e62376b989", "3b54f848fd6d", "3d6238419ba9", "3e2ad107d8bd", "44bea055534d", "4571a5b76263",
    "45765a754beb", "46dca85f7506", "483da5e9fc39", "4aa4d78c3968", "4c25603f341a", "4c732277795f",
    "4ca1111f7fa9", "4db78b807f52", "5645ac9a5991", "572b61707c4c", "585c5e22e771", "5b05c1ae340a",
    "5cde0a3be719", "61ebd2521e38", "6633ff9f8192", "682a1d54d9f8", "6f273955efcc", "7168c732ee2c",
    "73478591d121", "7ac4e3df5158", "7aefd4b81812", "7f233578c2dc", "7f74ddf5998b", "81e90c8af665",
    "8be8cdac04c2", "8f4599154c39", "9e05ce4e6a2f", "a463b746a441", "a6610e290ef7", "acf2e6dd8a72",
    "b3466495202a", "b591ddada6e2", "b801c7f31896", "b96b76dce396", "bac4f0abc084", "bbef9fc8f0b7",
    "bc61dd574b14", "bd93c8feb303", "be73d9dde1fe", "c1ff63b4cf9e", "c267b2bf0c13", "c34457504bd2",
    "c499d5109725", "c4cf42463984", "c73e30f4cffc", "c8f688ee9e58", "cfb8cf894a96", "d0611fc637fa",
    "d0a8f1c0ccce", "d53fa7a68289", "d6470fd6a771", "d91c354d3ba8", "dc93a9acd93b", "deb4c0be1388",
    "e04c373f1f19", "e2dab72c9d37", "e3307d3e3a5e", "e5081cd3fcd7", "e7e2022625fa", "e7ece8e6125a",
    "e8485cda4a0b", "e8c512787c95", "ea6792c38efa", "ef86e1e99ad6", "efd9328e27ab", "f1d4d90df0dc",
    "f5b8615ffbf3", "f9e9a9d8fbaa", "fb670b6f1e9a", "fb7cafc7cf8e", "fcbae90b5203", "fda585808eeb",
}


@app.function(image=image, gpu=None, volumes={"/vol": vol}, timeout=600)
def ensure_sam_h():
    import pathlib, urllib.request, time
    ckpt = pathlib.Path("/vol/models/sam_vit_h_4b8939.pth")
    if not ckpt.exists():
        t0 = time.time()
        print("Downloading sam_vit_h_4b8939.pth (2.56 GB)...", flush=True)
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            str(ckpt.with_suffix(".part")),
        )
        ckpt.with_suffix(".part").rename(ckpt)
        vol.commit()
        print(f"Downloaded SAM-ViT-H in {time.time()-t0:.1f}s", flush=True)
    else:
        print(f"SAM-ViT-H already cached ({ckpt.stat().st_size/1e9:.2f} GB)", flush=True)


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1800)
def extract_sam_h_shard(rank: int, world: int = 4):
    """Extract SAM-ViT-H Automatic + Neural-Prompted proposals for shard `rank` of ALL 1,530 images!"""
    import os, sys, time, pathlib
    import numpy as np
    from scipy import ndimage
    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import downsample_mask

    dev = "cuda:0"
    sam = sam_model_registry["vit_h"](checkpoint="/vol/models/sam_vit_h_4b8939.pth").to(dev).eval()
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        points_per_batch=128,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.84,
        box_nms_thresh=0.75,
        min_mask_region_area=64,
    )
    predictor = SamPredictor(sam)

    # Load JURECA ensemble logits for neural-guided SamPredictor prompts!
    ens_z = np.load("/vol/jureca_ens/ens_logits_v3.npz", allow_pickle=True)
    lg_by_name = {}
    for meth in ("pixelseal", "trustmark", "stegastamp"):
        for sp in ("dev", "hold", "test"):
            nms = ens_z[f"{meth}_{sp}_names"]
            lgs = ens_z[f"{meth}_{sp}_lg"].astype(np.float32)
            for nm, lg in zip(nms, lgs):
                lg_by_name[str(nm)] = lg

    # Build unified list of ALL 1,530 items (train + test)
    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    tasks = [("train", i) for i in range(len(ds_tr))] + [("test", i) for i in range(len(ds_te))]
    my_tasks = tasks[rank::world]
    print(f"[SAM-H r{rank}/{world}] Processing {len(my_tasks)} images...", flush=True)

    out_dir = pathlib.Path("/vol/outputs/sam_h")
    out_dir.mkdir(parents=True, exist_ok=True)
    packed_all = {}
    t0 = time.time()

    for ki, (sp, idx) in enumerate(my_tasks):
        it = ds_tr[idx] if sp == "train" else ds_te[idx]
        nm = it["image_name"]
        img = it["image"]
        lg = lg_by_name[nm]  # [128, 128] float32

        # 1) AutomaticMaskGenerator proposals
        anns = mask_gen.generate(img)
        cands = []
        for a in anns:
            m = a["segmentation"]
            ar = m.mean()
            if 0.002 <= ar <= 0.49:
                cands.append(downsample_mask(m) > 0)

        # 2) Neural-Prompted SamPredictor proposals (both peak/dip points AND multi-thr bboxes!)
        predictor.set_image(img)
        isL = (lg >= 0.0).mean() > 0.5
        # Signal to localize: for S -> +lg (watermarked object); for L -> -lg (clean hole!)
        sig = (-lg) if isL else lg
        sig_sm = ndimage.gaussian_filter(sig, sigma=1.5)
        # Positive point = global peak of sig_sm; Negative point = global min of sig_sm!
        py, px = np.unravel_index(np.argmax(sig_sm), (128, 128))
        ny, nx = np.unravel_index(np.argmin(sig_sm), (128, 128))
        pts = np.array([[px * 4 + 2, py * 4 + 2], [nx * 4 + 2, ny * 4 + 2]], dtype=np.float32)
        lbs = np.array([1, 0], dtype=np.int32)
        masks_p, _, _ = predictor.predict(point_coords=pts, point_labels=lbs, multimask_output=True)
        for m in masks_p:
            if 0.002 <= m.mean() <= 0.49:
                cands.append(downsample_mask(m) > 0)

        # Also prompt with top-2 peak points + multi-threshold bounding boxes of largest connected component!
        for thr_q in (0.0, -0.8, 0.8):
            bw = sig >= thr_q
            if bw.sum() < 12 or bw.mean() > 0.48:
                continue
            lab, nl = ndimage.label(bw)
            if nl == 0: continue
            # Pick component with highest summed signal
            best_c = max(range(1, nl + 1), key=lambda c: float(sig[lab == c].clip(0).sum()))
            comp = lab == best_c
            if comp.sum() < 12: continue
            ys, xs = np.where(comp)
            y0, y1, x0, x1 = max(0, ys.min()*4 - 4), min(512, (ys.max()+1)*4 + 4), max(0, xs.min()*4 - 4), min(512, (xs.max()+1)*4 + 4)
            box = np.array([x0, y0, x1, y1], dtype=np.float32)
            cy, cx = np.unravel_index(np.argmax(np.where(comp, sig_sm, -1e9)), (128, 128))
            m_b, _, _ = predictor.predict(
                point_coords=np.array([[cx*4+2, cy*4+2]], np.float32),
                point_labels=np.array([1], np.int32),
                box=box,
                multimask_output=True,
            )
            for m in m_b:
                if 0.002 <= m.mean() <= 0.49:
                    cands.append(downsample_mask(m) > 0)

        # Deduplicate 128x128 binary candidates (IoU > 0.94)
        uniq = []
        for m in cands:
            mf = m.ravel()
            dup = False
            for u in uniq:
                inter = np.logical_and(mf, u).sum()
                union = np.logical_or(mf, u).sum()
                if union > 0 and inter / union > 0.94:
                    dup = True; break
            if not dup:
                uniq.append(mf)
        if uniq:
            arr = np.packbits(np.stack(uniq, axis=0), axis=1)  # [M, 2048] uint8!
        else:
            arr = np.zeros((0, 2048), dtype=np.uint8)
        packed_all[nm] = arr

        if (ki + 1) % 60 == 0 or (ki + 1) == len(my_tasks):
            print(f"  [SAM-H r{rank}] {ki+1}/{len(my_tasks)} ({(time.time()-t0)/(ki+1):.2f}s/img, last_M={len(uniq)})", flush=True)

    np.savez(out_dir / f"shard_{rank}.npz", **packed_all)
    vol.commit()
    print(f"[SAM-H r{rank}] SAVED shard_{rank}.npz in {(time.time()-t0)/60:.1f}m", flush=True)


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1200)
def fuse_and_eval_sam_h():
    """Strict 4-fold nested DEV CV fusion of:
      JURECA ensemble logits + Adaptive Split-Thr + rect_nnref + SAM-ViT-H Segment Snapping!
      ZERO holdout tuning -> evaluates ONCE on 102 untouched holdout images + writes verified submission .pt!
    """
    import os, sys, time, json, pathlib
    import numpy as np
    import torch

    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import miou_image, downsample_mask

    # Load all 4 SAM-H shards into dict: name -> [M, 128, 128] bool (unpack on demand!)
    sam_packed = {}
    for r in range(4):
        z = np.load(f"/vol/outputs/sam_h/shard_{r}.npz")
        for k in z.files:
            sam_packed[k] = z[k]
    print(f"Loaded SAM-ViT-H proposals for {len(sam_packed)}/1530 images", flush=True)

    ens_z = np.load("/vol/jureca_ens/ens_logits_v3.npz", allow_pickle=True)
    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    tr_by_name = {ds_tr[i]["image_name"]: ds_tr[i] for i in range(len(ds_tr))}
    te_names_all = [ds_te[i]["image_name"] for i in range(len(ds_te))]

    def unpack_cands(nm):
        p = sam_packed.get(nm)
        if p is None or len(p) == 0:
            return np.zeros((0, 128, 128), dtype=bool)
        return np.unpackbits(p, axis=1).reshape(-1, 128, 128).astype(bool)

    def fast_miou_vec(pred, gt):
        p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
        i1 = (p & g).sum(1); u1 = (p | g).sum(1)
        i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
        return 0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0) + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0))

    def load_rect(p):
        z = np.load(p, allow_pickle=True)
        return {str(z["names"][i]): {
            "box": z["box"][i].astype(np.float32),
            "margin": float(z["margin"][i]), "acc": float(z["acc"][i]),
            "full_margin": float(z["full_margin"][i]), "full_acc": float(z["full_acc"][i]),
        } for i in range(len(z["names"]))}

    def box_to_m128(box):
        m = np.zeros((128, 128), dtype=bool)
        y0, x0, y1, x1 = np.clip(np.round(box / 4.0).astype(int), 0, 128)
        if y1 > y0 and x1 > x0: m[y0:y1, x0:x1] = True
        return m

    def best_sam_match(nm, lg, isL_flag, cur_mask):
        """Find best SAM-ViT-H segment matching either S foreground (isL=False) or L hole (isL=True)!
        Returns (best_full_mask, best_iou_with_cur, best_soft_iou, best_contrast, best_area).
        """
        cands = unpack_cands(nm)
        if len(cands) == 0:
            return None, 0.0, 0.0, -99.0, 0.0
        # Target binary & soft probability of the MINORITY region (fg for S, hole for L)
        if isL_flag:
            prob_min = 1.0 / (1.0 + np.exp(np.clip(lg, -20, 20)))  # sigmoid(-lg) = hole prob!
            sig = -lg
            cur_min = ~cur_mask
        else:
            prob_min = 1.0 / (1.0 + np.exp(np.clip(-lg, -20, 20)))  # sigmoid(+lg) = fg prob!
            sig = lg
            cur_min = cur_mask

        # Vectorized scoring across all M SAM candidates!
        cf = cands.reshape(len(cands), -1).astype(np.float32)  # [M, P]
        pf = prob_min.ravel()[None, :]                         # [1, P]
        sf = sig.ravel()[None, :]                              # [1, P]
        cmf = cur_min.ravel().astype(np.float32)[None, :]      # [1, P]

        areas = cf.mean(1)
        inter_soft = (cf * pf).sum(1)
        union_soft = (cf + pf - cf * pf).sum(1).clip(min=1e-4)
        soft_iou = inter_soft / union_soft

        inter_bin = (cf * cmf).sum(1)
        union_bin = (cf + cmf - cf * cmf).sum(1).clip(min=1e-4)
        bin_iou = inter_bin / union_bin

        # Contrast: mean(sig inside) - mean(sig outside)
        s_in = (cf * sf).sum(1) / cf.sum(1).clip(min=1.0)
        s_out = ((1.0 - cf) * sf).sum(1) / (1.0 - cf).sum(1).clip(min=1.0)
        contrast = s_in - s_out

        # Composite matching score: primarily soft_iou!
        j = int(np.argmax(soft_iou))
        cand_mask = (~cands[j]) if isL_flag else cands[j]
        return cand_mask, float(bin_iou[j]), float(soft_iou[j]), float(contrast[j]), float(areas[j])

    test_masks = {}
    summary = {}
    for method in ("pixelseal", "trustmark", "stegastamp"):
        dn = [str(x) for x in ens_z[f"{method}_dev_names"]]
        hn = [str(x) for x in ens_z[f"{method}_hold_names"]]
        tn = [str(x) for x in ens_z[f"{method}_test_names"]]
        d_lg = ens_z[f"{method}_dev_lg"].astype(np.float32)
        h_lg = ens_z[f"{method}_hold_lg"].astype(np.float32)
        t_lg = ens_z[f"{method}_test_lg"].astype(np.float32)

        dg = np.stack([downsample_mask(tr_by_name[n]["mask"] > 0) > 0 for n in dn])
        hg = np.stack([downsample_mask(tr_by_name[n]["mask"] > 0) > 0 for n in hn])
        dL = np.array([bool(tr_by_name[n]["full_watermark"]) for n in dn])
        hL = np.array([bool(tr_by_name[n]["full_watermark"]) for n in hn])

        base_d = float(fast_miou_vec(d_lg >= 0.0, dg).mean())
        base_h = float(fast_miou_vec(h_lg >= 0.0, hg).mean())
        print(f"\n================ {method.upper()} (BASE dev={base_d:.4f} hold={base_h:.4f}) ================", flush=True)

        # STEP 1: Adaptive Split-Threshold + Hole-Dip Recovery (strict 4-fold nested DEV CV!)
        def apply_adapt(lg, tL, tS, zL, zS):
            isL = (lg >= 0.0).mean(axis=(1, 2)) > 0.5
            out = np.where(isL[:, None, None], lg >= tL, lg >= tS)
            for i in range(len(lg)):
                m = lg[i]; mu = float(m.mean()); sd = float(m.std()) + 1e-6
                if isL[i] and (~out[i]).mean() < 0.005 and zL < 0:
                    out[i] = (m - mu) / sd >= zL
                elif (not isL[i]) and out[i].mean() < 0.005 and zS > 0:
                    out[i] = (m - mu) / sd >= zS
            return out

        ad_grid = [(tL, tS, zL, zS)
                   for tL in (0.0, 0.4, 0.8, 1.2)
                   for tS in (0.0, -0.4, -0.8, -1.2)
                   for zL in (0.0, -2.0, -2.5, -3.0)
                   for zS in (0.0, 2.5, 3.0)]
        # Precompute masks on dev? Fast enough to do 4-fold nested CV directly!
        ad_scores = [fast_miou_vec(apply_adapt(d_lg, *c), dg) for c in ad_grid]
        cv_ad = np.zeros(len(dg), np.float32)
        for fold in range(4):
            vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
            bi = max(range(len(ad_grid)), key=lambda idx: float(ad_scores[idx][tm].mean()))
            cv_ad[vm] = ad_scores[bi][vm]
        best_ad_i = max(range(len(ad_grid)), key=lambda idx: float(ad_scores[idx].mean()))
        if cv_ad.mean() > base_d + 0.0005:
            ad_cfg = ad_grid[best_ad_i]
            d_m = apply_adapt(d_lg, *ad_cfg)
            h_m = apply_adapt(h_lg, *ad_cfg)
            t_m = apply_adapt(t_lg, *ad_cfg)
            print(f"  [1. ADAPT] nested-CV={cv_ad.mean():.4f} ({cv_ad.mean()-base_d:+.4f}) cfg={ad_cfg} | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)
        else:
            d_m, h_m, t_m = d_lg >= 0.0, h_lg >= 0.0, t_lg >= 0.0
            print(f"  [1. ADAPT] skipped (nested-CV={cv_ad.mean():.4f})", flush=True)

        # STEP 2: rect_nnref override (for pixelseal & trustmark, strict 4-fold nested DEV CV!)
        d_rect_trig = np.zeros(len(dn), bool); h_rect_trig = np.zeros(len(hn), bool); t_rect_trig = np.zeros(len(tn), bool)
        if method in ("pixelseal", "trustmark"):
            rd_tr = load_rect(f"/vol/jureca_ens/{method}_train_nnref.npz")
            rd_te = load_rect(f"/vol/jureca_ens/{method}_test_nnref.npz")
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
                best_c = max(rgrid, key=lambda c: float(np.where((fa_d < c[0]) & (ac_d >= c[1]) & (gn_d >= c[2]), rep_v, base_v)[tm].mean()))
                tr_f = (fa_d < best_c[0]) & (ac_d >= best_c[1]) & (gn_d >= best_c[2])
                cv_r[vm] = np.where(tr_f, rep_v, base_v)[vm]
            best_rc = max(rgrid, key=lambda c: float(np.where((fa_d < c[0]) & (ac_d >= c[1]) & (gn_d >= c[2]), rep_v, base_v).mean()))
            if cv_r.mean() > base_v.mean() + 0.001:
                fa, ac, gn = best_rc
                d_rect_trig = (fa_d < fa) & (ac_d >= ac) & (gn_d >= gn)
                h_rect_trig = np.array([r["full_acc"] < fa and r["acc"] >= ac and (r["margin"]-r["full_margin"]) >= gn for r in h_rec])
                t_rect_trig = np.array([r["full_acc"] < fa and r["acc"] >= ac and (r["margin"]-r["full_margin"]) >= gn for r in t_rec])
                d_m = np.where(d_rect_trig[:, None, None], d_bm, d_m)
                h_m = np.where(h_rect_trig[:, None, None], h_bm, h_m)
                t_m = np.where(t_rect_trig[:, None, None], t_bm, t_m)
                print(f"  [2. RECT_NNREF] nested-CV={cv_r.mean():.4f} ({cv_r.mean()-base_v.mean():+.4f}) cfg={best_rc} trig={d_rect_trig.sum()}/136 | hold={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        # STEP 3: SAM-ViT-H SEGMENT SNAPPING (separately for S non-rect AND L holes, strict 4-fold nested DEV CV!)
        def precompute_sam(names, lgs, cur_ms):
            isL_arr = (lgs >= 0.0).mean(axis=(1, 2)) > 0.5
            sm_masks = cur_ms.copy()
            biou = np.zeros(len(names), np.float32)
            siou = np.zeros(len(names), np.float32)
            cont = np.zeros(len(names), np.float32)
            area = np.zeros(len(names), np.float32)
            for i, nm in enumerate(names):
                cm, bi, si, ct, ar = best_sam_match(nm, lgs[i], bool(isL_arr[i]), cur_ms[i])
                if cm is not None:
                    sm_masks[i] = cm
                    biou[i], siou[i], cont[i], area[i] = bi, si, ct, ar
            return isL_arr, sm_masks, biou, siou, cont, area

        d_isL, d_sam, d_bi, d_si, d_ct, d_ar = precompute_sam(dn, d_lg, d_m)
        h_isL, h_sam, h_bi, h_si, h_ct, h_ar = precompute_sam(hn, h_lg, h_m)
        t_isL, t_sam, t_bi, t_si, t_ct, t_ar = precompute_sam(tn, t_lg, t_m)

        # Evaluate SAM snapping separately on S (non-rect!) and L!
        for sub_name, is_L_target in [("SAM_S_OBJ", False), ("SAM_L_HOLE", True)]:
            base_v = fast_miou_vec(d_m, dg)
            sam_v = fast_miou_vec(d_sam, dg)
            elig_d = (d_isL == is_L_target) & (~d_rect_trig)
            elig_h = (h_isL == is_L_target) & (~h_rect_trig)
            elig_t = (t_isL == is_L_target) & (~t_rect_trig)

            sgrid = [(t_si, t_bi, t_ct, a_min)
                     for t_si in (0.15, 0.22, 0.30, 0.38, 0.46, 0.55, 0.65)
                     for t_bi in (0.0, 0.20, 0.35, 0.50)
                     for t_ct in (0.0, 0.5, 1.0, 1.8, 2.5)
                     for a_min in (0.005, 0.015, 0.03)]
            cv_s = np.zeros_like(base_v)
            for fold in range(4):
                vm = (np.arange(len(dg)) % 4) == fold; tm = ~vm
                best_sc_tr, best_c = float(base_v[tm].mean()), None
                for c in sgrid:
                    tr_f = elig_d & (d_si >= c[0]) & (d_bi >= c[1]) & (d_ct >= c[2]) & (d_ar >= c[3])
                    sc = float(np.where(tr_f, sam_v, base_v)[tm].mean())
                    if sc > best_sc_tr: best_sc_tr, best_c = sc, c
                if best_c is None:
                    cv_s[vm] = base_v[vm]
                else:
                    tr_f = elig_d & (d_si >= best_c[0]) & (d_bi >= best_c[1]) & (d_ct >= best_c[2]) & (d_ar >= best_c[3])
                    cv_s[vm] = np.where(tr_f, sam_v, base_v)[vm]
            # Fit full dev
            best_full, best_fc = float(base_v.mean()), None
            for c in sgrid:
                tr_f = elig_d & (d_si >= c[0]) & (d_bi >= c[1]) & (d_ct >= c[2]) & (d_ar >= c[3])
                sc = float(np.where(tr_f, sam_v, base_v).mean())
                if sc > best_full: best_full, best_fc = sc, c

            gain_cv = float(cv_s.mean() - base_v.mean())
            print(f"  [3. {sub_name:10s}] nested-CV={cv_s.mean():.4f} ({gain_cv:+.4f}) full-dev={best_full:.4f} cfg={best_fc}", flush=True)
            if best_fc is not None and gain_cv > 0.001:
                thr_si, thr_bi, thr_ct, thr_ar = best_fc
                td = elig_d & (d_si >= thr_si) & (d_bi >= thr_bi) & (d_ct >= thr_ct) & (d_ar >= thr_ar)
                th = elig_h & (h_si >= thr_si) & (h_bi >= thr_bi) & (h_ct >= thr_ct) & (h_ar >= thr_ar)
                tt = elig_t & (t_si >= thr_si) & (t_bi >= thr_bi) & (t_ct >= thr_ct) & (t_ar >= thr_ar)
                d_m = np.where(td[:, None, None], d_sam, d_m)
                h_m = np.where(th[:, None, None], h_sam, h_m)
                t_m = np.where(tt[:, None, None], t_sam, t_m)
                print(f"     >> ADOPTED {sub_name}! trig dev={td.sum()}/136 hold={th.sum()}/34 test={tt.sum()}/340 | hold_now={fast_miou_vec(h_m, hg).mean():.4f}", flush=True)

        d_off = np.array([miou_image(d_m[i], dg[i]) for i in range(len(dg))])
        h_off = np.array([miou_image(h_m[i], hg[i]) for i in range(len(hg))])
        print(f"  *** FINAL {method.upper()} *** DEV={d_off.mean():.4f} (L {d_off[dL].mean():.4f} S {d_off[~dL].mean():.4f}) | HOLD={h_off.mean():.4f} (L {h_off[hL].mean():.4f} S {h_off[~hL].mean():.4f}) | test_fg={t_m.mean():.4f}", flush=True)
        summary[method] = {"dev": float(d_off.mean()), "hold": float(h_off.mean()), "hold_L": float(h_off[hL].mean()), "hold_S": float(h_off[~hL].mean())}
        for i, n in enumerate(tn):
            test_masks[n] = t_m[i].astype(np.uint8)

    m_dev = float(np.mean([summary[m]["dev"] for m in summary]))
    m_hld = float(np.mean([summary[m]["hold"] for m in summary]))
    print("\n============================================================", flush=True)
    print(f"FINAL SAM-ViT-H + ADAPT + RECT_NNREF 3-METH DEV  OOF : {m_dev:.4f}", flush=True)
    print(f"FINAL SAM-ViT-H + ADAPT + RECT_NNREF 3-METH HOLDOUT  : {m_hld:.4f}", flush=True)
    print(f"PER-METHOD HOLDOUT: {summary}", flush=True)
    print("============================================================", flush=True)

    masks_u8 = torch.from_numpy(np.stack([test_masks[n] for n in te_names_all])).to(torch.uint8)
    assert masks_u8.shape == (1020, 128, 128)
    out_p = pathlib.Path("/vol/outputs/sub_v5_samh_clean.pt")
    torch.save({"image_names": te_names_all, "masks": masks_u8}, out_p)
    (out_p.with_suffix(".json")).write_text(json.dumps({"dev": m_dev, "hold": m_hld, "methods": summary}, indent=2))
    vol.commit()
    print(f"SAVED {out_p} ({out_p.stat().st_size/1e6:.2f} MB, fg={masks_u8.float().mean():.4f})", flush=True)
    return m_dev, m_hld


@app.local_entrypoint()
def main():
    ensure_sam_h.remote()
    hs = [extract_sam_h_shard.spawn(r, 4) for r in range(4)]
    for h in hs:
        h.get()
    dev_s, hld_s = fuse_and_eval_sam_h.remote()
    print(f"=== SAM-H PIPELINE COMPLETE: DEV={dev_s:.4f} | HOLD={hld_s:.4f} ===")
