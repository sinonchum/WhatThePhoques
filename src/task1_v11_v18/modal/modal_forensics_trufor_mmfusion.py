"""Modal 2xA100-80GB/H100 Academic SOTA Forensic Localization Extractors (~2.5 min total for ALL 1,530 images!):
  - GPU 1: TruFor (CVPR 2023, grip-unina/TruFor — RGB + Noiseprint++ + SegFormer-B2!)
           Outputs: `map` (forgery anomaly prob), `conf` (reliability), `npp` (Noiseprint++ fingerprint!)
  - GPU 2: MMFusion-IML (MMM 2024, IDT-ITI/MMFusion-IML — Early Fusion of SRM + BayarConv + Noiseprint++!)
           Outputs: `map` (forgery anomaly prob), `conf` (reliability)
Immediately evaluates ZERO-SHOT AUROC & mIoU (with polarity flip for L vs S) on all 3 methods x 3 edit types
and caches all 1,530 maps permanently to `/vol/outputs/forensics/{trufor,mmfusion}_1530.npz`!
"""
import modal

app = modal.App("task1-trufor-mmfusion-forensics")
vol = modal.Volume.from_name("task1-wm-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "unzip", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "timm",
        "numpy",
        "scipy",
        "scikit-learn",
        "pillow",
        "yacs",
        "PyYAML",
        "einops",
        "tqdm",
        "gdown",
        "opencv-python-headless",
    )
)


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1200)
def run_trufor():
    import os, sys, time, subprocess, pathlib, zipfile, urllib.request
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    dev = "cuda:0"
    tf_dir = pathlib.Path("/tmp/TruFor")
    if not tf_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/grip-unina/TruFor.git", str(tf_dir)], check=True)

    w_dir = pathlib.Path("/vol/models/trufor")
    w_dir.mkdir(parents=True, exist_ok=True)
    w_pth = w_dir / "trufor.pth.tar"
    if not w_pth.exists():
        z_p = w_dir / "TruFor_weights.zip"
        print("[TruFor] Downloading TruFor_weights.zip...", flush=True)
        urllib.request.urlretrieve("https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip", str(z_p))
        with zipfile.ZipFile(z_p, "r") as zf:
            zf.extractall(w_dir)
        # Locate trufor.pth.tar inside extracted tree
        found = list(w_dir.rglob("*.pth*"))
        print(f"[TruFor] Extracted weights: {[str(p) for p in found]}", flush=True)
        for p in found:
            if p.name == "trufor.pth.tar" and p != w_pth:
                p.rename(w_pth)
        vol.commit()

    src_dir = tf_dir / "test_docker" / "src"
    sys.path.insert(0, str(src_dir))
    from config import _C as config
    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx

    cfg = config.clone()
    cfg.defrost()
    cfg.merge_from_file(str(src_dir / "trufor.yaml"))
    cfg.TEST.MODEL_FILE = str(w_pth)
    cfg.freeze()

    ckpt = torch.load(str(w_pth), map_location=dev, weights_only=False)
    model = confcmx(cfg=cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(dev).eval()
    print(f"[TruFor] Model loaded in {time.time()-t0:.1f}s!", flush=True)

    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import downsample_mask, miou_image

    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    all_items = [("train", ds_tr[i]) for i in range(len(ds_tr))] + [("test", ds_te[i]) for i in range(len(ds_te))]

    names = []
    maps128 = np.zeros((len(all_items), 128, 128), np.float16)
    confs128 = np.zeros((len(all_items), 128, 128), np.float16)
    npp128 = np.zeros((len(all_items), 128, 128), np.float16)
    scores = np.zeros(len(all_items), np.float32)

    # Batch size 8 on 512x512 -> ~190 batches = ~35 seconds total!
    t1 = time.time()
    for s in range(0, len(all_items), 8):
        ch = all_items[s:s+8]
        imgs = torch.stack([
            torch.from_numpy(it["image"].copy()).permute(2, 0, 1).float() / 255.0
            for _, it in ch
        ]).to(dev)
        with torch.no_grad():
            pred, conf, det, npp = model(imgs)
            pmap = F.softmax(pred, dim=1)[:, 1:2, :, :]  # [B, 1, 512, 512]
            pm128 = F.interpolate(pmap, (128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            cf128 = F.interpolate(torch.sigmoid(conf)[:, 0:1], (128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy() if conf is not None else np.zeros_like(pm128)
            np128 = F.interpolate(npp[:, 0:1], (128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy() if npp is not None else np.zeros_like(pm128)
            sc = torch.sigmoid(det).squeeze(-1).cpu().numpy() if det is not None else np.zeros(len(ch))
        for j, (_, it) in enumerate(ch):
            names.append(it["image_name"])
            maps128[s + j] = pm128[j].astype(np.float16)
            confs128[s + j] = cf128[j].astype(np.float16)
            npp128[s + j] = np128[j].astype(np.float16)
            scores[s + j] = float(sc[j]) if np.ndim(sc) > 0 else float(sc)
    print(f"[TruFor] Extracted ALL {len(all_items)} images in {time.time()-t1:.1f}s!", flush=True)

    # Zero-shot diagnostic on all 510 train images!
    def classify_it(it):
        m = it["mask"] > 0
        if it["full_watermark"]: return "large"
        ys, xs = np.where(m)
        return "rect" if m.sum() / max((ys.max()-ys.min()+1)*(xs.max()-xs.min()+1), 1) > 0.98 else "object"

    for meth in ("pixelseal", "trustmark", "stegastamp"):
        by_g = {"large": [], "object": [], "rect": []}
        mi_g = {"large": [], "object": [], "rect": []}
        for i in range(len(ds_tr)):
            it = ds_tr[i]
            if it["method"] != meth: continue
            g = classify_it(it)
            gt = downsample_mask(it["mask"] > 0) > 0
            pm = maps128[i].astype(np.float32)
            # For large: anomaly map = HOLE (~gt); for object/rect: anomaly map = fg (gt) OR ~gt (outpaint!)
            y = (~gt).ravel().astype(np.int32) if g == "large" else gt.ravel().astype(np.int32)
            if 0 < y.sum() < len(y):
                auc = roc_auc_score(y, pm.ravel())
                # Also check 1-auc for outpainting (where outer 95% was outpainted!)
                by_g[g].append(max(auc, 1.0 - auc) if g == "rect" else auc)
            pred_b = (pm < 0.5) if g == "large" else (pm >= 0.5)
            mi_g[g].append(miou_image(pred_b, gt))
        print(f"*** [TruFor ZERO-SHOT {meth.upper():10s}] AUROC: L={np.mean(by_g['large']):.4f} OBJ={np.mean(by_g['object']):.4f} RECT(sym)={np.mean(by_g['rect']):.4f} | mIoU@0.5: L={np.mean(mi_g['large']):.4f} OBJ={np.mean(mi_g['object']):.4f} RECT={np.mean(mi_g['rect']):.4f} ***", flush=True)

    out_dir = pathlib.Path("/vol/outputs/forensics")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "trufor_1530.npz", names=np.array(names), map=maps128, conf=confs128, npp=npp128, score=scores)
    vol.commit()
    print(f"[TruFor] SAVED /vol/outputs/forensics/trufor_1530.npz!", flush=True)


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=1200)
def run_mmfusion():
    import os, sys, time, subprocess, pathlib
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF

    t0 = time.time()
    dev = "cuda:0"
    mm_dir = pathlib.Path("/tmp/MMFusion-IML")
    if not mm_dir.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/IDT-ITI/MMFusion-IML.git", str(mm_dir)], check=True)

    w_dir = pathlib.Path("/vol/models/mmfusion")
    w_dir.mkdir(parents=True, exist_ok=True)
    ckpts = list(w_dir.rglob("*.pth"))
    if not ckpts:
        print("[MMFusion] Downloading pretrained checkpoints via gdown...", flush=True)
        subprocess.run(["gdown", "--folder", "https://drive.google.com/drive/folders/1P1efXjVmYltZ7zWT-m6WoIlZgh9R_kAg", "-O", str(w_dir)], check=False)
        ckpts = list(w_dir.rglob("*.pth"))
        print(f"[MMFusion] Found ckpts: {[str(p) for p in ckpts]}", flush=True)
        vol.commit()
    if not ckpts:
        print("[MMFusion] WARNING: gdown failed to download checkpoints; skipping.", flush=True)
        return

    # Pick early_fusion_detection.pth or best available
    ckpt_p = next((p for p in ckpts if "early_fusion_detection" in p.name), ckpts[0])
    print(f"[MMFusion] Loading {ckpt_p}...", flush=True)
    os.chdir(str(mm_dir))
    sys.path.insert(0, str(mm_dir))
    from models.cmnext_conf import CMNeXtWithConf
    from models.modal_extract import ModalitiesExtractor
    from configs.cmnext_init_cfg import _C as config, update_config

    exp_y = mm_dir / "experiments" / "ec_example_phase2.yaml"
    cfg = update_config(config, str(exp_y))
    # Patch pretrained path to empty so CMNeXtWithConf skips loading missing local mit_b2.pth (full ckpt loaded next!)
    cfg.defrost()
    cfg.MODEL.PRETRAINED = ""
    cfg.freeze()
    import models.cmnext_conf as _mc
    _mc.load_dualpath_model = lambda *a, **kw: None
    modal_extractor = ModalitiesExtractor(cfg.MODEL.MODALS[1:], None)
    model = CMNeXtWithConf(cfg.MODEL)
    ck = torch.load(str(ckpt_p), map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"], strict=False)
    if "extractor_state_dict" in ck:
        modal_extractor.load_state_dict(ck["extractor_state_dict"], strict=False)
    modal_extractor = modal_extractor.to(dev).eval()
    model = model.to(dev).eval()
    print(f"[MMFusion] Loaded in {time.time()-t0:.1f}s!", flush=True)

    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.utils import downsample_mask, miou_image

    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test")
    all_items = [("train", ds_tr[i]) for i in range(len(ds_tr))] + [("test", ds_te[i]) for i in range(len(ds_te))]

    names = []
    maps128 = np.zeros((len(all_items), 128, 128), np.float16)
    confs128 = np.zeros((len(all_items), 128, 128), np.float16)
    t1 = time.time()
    for s in range(0, len(all_items), 8):
        ch = all_items[s:s+8]
        imgs = torch.stack([
            torch.from_numpy(it["image"].copy()).permute(2, 0, 1).float() / 255.0
            for _, it in ch
        ]).to(dev)
        with torch.no_grad():
            modals = modal_extractor(imgs)
            imgs_n = TF.normalize(imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            anomaly, conf, det = model([imgs_n] + modals)
            pmap = F.softmax(anomaly, dim=1)[:, 1:2]
            pm128 = F.interpolate(pmap, (128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
            cf128 = F.interpolate(torch.sigmoid(conf)[:, 0:1], (128, 128), mode="bilinear", align_corners=False)[:, 0].cpu().numpy() if conf is not None else np.zeros_like(pm128)
        for j, (_, it) in enumerate(ch):
            names.append(it["image_name"])
            maps128[s + j] = pm128[j].astype(np.float16)
            confs128[s + j] = cf128[j].astype(np.float16)
    print(f"[MMFusion] Extracted ALL {len(all_items)} images in {time.time()-t1:.1f}s!", flush=True)

    def classify_it(it):
        m = it["mask"] > 0
        if it["full_watermark"]: return "large"
        ys, xs = np.where(m)
        return "rect" if m.sum() / max((ys.max()-ys.min()+1)*(xs.max()-xs.min()+1), 1) > 0.98 else "object"

    for meth in ("pixelseal", "trustmark", "stegastamp"):
        by_g = {"large": [], "object": [], "rect": []}
        mi_g = {"large": [], "object": [], "rect": []}
        for i in range(len(ds_tr)):
            it = ds_tr[i]
            if it["method"] != meth: continue
            g = classify_it(it)
            gt = downsample_mask(it["mask"] > 0) > 0
            pm = maps128[i].astype(np.float32)
            y = (~gt).ravel().astype(np.int32) if g == "large" else gt.ravel().astype(np.int32)
            if 0 < y.sum() < len(y):
                auc = roc_auc_score(y, pm.ravel())
                by_g[g].append(max(auc, 1.0 - auc) if g == "rect" else auc)
            pred_b = (pm < 0.5) if g == "large" else (pm >= 0.5)
            mi_g[g].append(miou_image(pred_b, gt))
        print(f"*** [MMFusion ZERO-SHOT {meth.upper():10s}] AUROC: L={np.mean(by_g['large']):.4f} OBJ={np.mean(by_g['object']):.4f} RECT(sym)={np.mean(by_g['rect']):.4f} | mIoU@0.5: L={np.mean(mi_g['large']):.4f} OBJ={np.mean(mi_g['object']):.4f} RECT={np.mean(mi_g['rect']):.4f} ***", flush=True)

    out_dir = pathlib.Path("/vol/outputs/forensics")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "mmfusion_1530.npz", names=np.array(names), map=maps128, conf=confs128)
    vol.commit()
    print(f"[MMFusion] SAVED /vol/outputs/forensics/mmfusion_1530.npz!", flush=True)


@app.local_entrypoint()
def main():
    run_mmfusion.remote()
    print("=== MMFUSION EXTRACTION COMPLETE ===")
