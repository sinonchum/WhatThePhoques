"""V16 NUCLEAR END-TO-END BREAKTHROUGH:
`DualCLIP-ConvNeXt-Large (200M CLIP LAION-2B)` + Full `ExtAttrSegNet` Watermark Physics (`grad*act` + 6-view
per-bit signed message maps + HighPass(40) + 6ch Adobe KBNet_s residual) trained END-TO-END on the
EXACT COMPETITION-DOMAIN 204-COVER `synth3` (clean dev-only PNG covers — ZERO COCO JPEG domain gap,
ZERO holdout/test leakage!) + real dev (`p_real=0.10`), across 8x Modal A100-80GB/H100 in parallel!

Stage 1 (~60s on 2 GPUs): Build exact 204-cover competition-domain `synth3` into `/vol/synth3_exact/{pixelseal,trustmark}/`.
Stage 2 (~18m on 8 GPUs): Train ALL 4 folds of PixelSeal + ALL 4 folds of TrustMark simultaneously!
  - Uses exact `HOLDOUT_102` split + exact 4-fold stratified DEV CV!
  - Saves honest-OOF `dev_lg`, `hold_lg`, `test_lg` directly to `/vol/outputs/v16_end2end/{method}_clip_large.npz`!
"""
import modal

app = modal.App("task1-v16-clip-end2end")
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


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=900)
def build_exact_synth3(method: str):
    """Build 100% clean, exact competition-domain 204-cover synth3 pool on Modal GPU (~45s!).
    Uses ONLY the 204 clean `dev small` images from `train` (ZERO holdout, ZERO test!).
    """
    import os, sys, time, pathlib
    import numpy as np
    import torch
    import torch.nn.functional as F

    os.environ["WMLOC_MODELS"] = "/vol/models"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark

    out_dir = pathlib.Path(f"/vol/synth3_exact/{method}")
    if (out_dir / "done.flag").exists():
        print(f"[synth3 {method}] Already cached!", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    dev = "cuda:0"
    wm = load_watermark(method, device=dev)
    ds_tr = LocalisationDataset(str(repo_dir), "train")

    # Clean covers = ALL dev `small` images (NOT in HOLDOUT_102, full_watermark == False) -> exact 204 PNG covers!
    covers = []
    rect_hws = []
    for i in range(len(ds_tr)):
        it = ds_tr[i]
        if it["image_name"] in HOLDOUT_102:
            continue
        if not it["full_watermark"]:
            covers.append(it["image"].copy())
            if it["method"] == method:
                m = it["mask"] > 0
                ys, xs = np.where(m)
                if len(ys) > 16:
                    h, w = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
                    if m.sum() / max(h * w, 1) > 0.96:
                        rect_hws.append((h, w))
    covers = np.stack(covers, axis=0)  # [204, 512, 512, 3] uint8
    N = len(covers)
    print(f"[synth3 {method}] Clean dev-small PNG covers: N={N}, rect_hws={len(rect_hws)}", flush=True)

    # Embedder helper (100% verified!)
    nbits = 256 if method == "pixelseal" else 100
    if method == "pixelseal":
        wm.model.eval()
        for p in wm.model.parameters(): p.requires_grad_(False)
        @torch.no_grad()
        def embed_batch(x01, msgs):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return wm.model.embed(x01, msgs=msgs.long(), is_video=False)["imgs_w"].float().clamp(0, 1)
    else:
        from trustmark import TrustMark
        tm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5)
        enc_tm = tm.encoder.to(dev).eval()
        for p in enc_tm.parameters(): p.requires_grad_(False)
        @torch.no_grad()
        def embed_batch(x01, msgs):
            x256 = F.interpolate(x01, (256, 256), mode="bilinear", align_corners=False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = enc_tm(x256 * 2.0 - 1.0, msgs.float())
                stego = out[0] if isinstance(out, tuple) else out
            w256 = ((stego.float() + 1.0) * 0.5).clamp(0, 1)
            delta_full = F.interpolate(w256 - x256, x01.shape[-2:], mode="bilinear", align_corners=False)
            return (x01 + delta_full).clamp(0, 1)

    rng = np.random.RandomState(2026 if method == "pixelseal" else 2027)
    K_FULL = 8
    full_imgs = np.zeros((N, K_FULL, 512, 512, 3), dtype=np.uint8)
    full_msgs = rng.randint(0, 2, size=(N, K_FULL, nbits), dtype=np.uint8)

    # Batch-embed all N x K_FULL full 512x512 images on GPU!
    for k in range(K_FULL):
        for s in range(0, N, 16):
            e = min(N, s + 16)
            xb = torch.from_numpy(covers[s:e]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            mb = torch.from_numpy(full_msgs[s:e, k].astype(np.float32)).to(dev)
            wb = embed_batch(xb, mb)
            full_imgs[s:e, k] = (wb.permute(0, 2, 3, 1).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

    # Also pre-embed K_RECT=32 crop-embedded rectangles per cover (exact Outpainting physics!)
    K_RECT = 32
    rect_meta = np.zeros((N, K_RECT, 4), dtype=np.int16)  # y0, x0, h, w
    rect_msgs = rng.randint(0, 2, size=(N, K_RECT, nbits), dtype=np.uint8)
    rect_patches = []  # store flat uint8 bytes + offsets
    rect_offsets = np.zeros(N * K_RECT + 1, dtype=np.int64)
    cur_off = 0
    for c in range(N):
        cov_t = torch.from_numpy(covers[c]).to(dev).permute(2, 0, 1)[None].float() / 255.0
        for r in range(K_RECT):
            h, w = rect_hws[rng.randint(0, len(rect_hws))]
            h = max(12, min(480, int(round(h * rng.uniform(0.85, 1.15)))))
            w = max(12, min(480, int(round(w * rng.uniform(0.85, 1.15)))))
            y0 = int(rng.randint(0, 512 - h + 1)); x0 = int(rng.randint(0, 512 - w + 1))
            rect_meta[c, r] = (y0, x0, h, w)
            crop = cov_t[:, :, y0:y0+h, x0:x0+w]
            m_t = torch.from_numpy(rect_msgs[c, r:r+1].astype(np.float32)).to(dev)
            if method == "pixelseal":
                # PixelSeal embeds crop resized to 256 -> upsample delta back to (h,w)!
                c256 = F.interpolate(crop, (256, 256), mode="bilinear", align_corners=False)
                w256 = embed_batch(c256, m_t)
                d_hw = F.interpolate(w256 - c256, (h, w), mode="bilinear", align_corners=False)
                wc = (crop + d_hw).clamp(0, 1)
            else:
                wc = embed_batch(crop, m_t)
            arr = (wc[0].permute(1, 2, 0).cpu().numpy() * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
            rect_patches.append(arr.ravel())
            cur_off += arr.size
            rect_offsets[c * K_RECT + r + 1] = cur_off
    rect_blob = np.concatenate(rect_patches)

    np.save(out_dir / "covers.npy", covers)
    np.save(out_dir / "full_imgs.npy", full_imgs)
    np.save(out_dir / "full_msgs.npy", full_msgs)
    np.save(out_dir / "rect_meta.npy", rect_meta)
    np.save(out_dir / "rect_msgs.npy", rect_msgs)
    np.save(out_dir / "rect_offsets.npy", rect_offsets)
    np.save(out_dir / "rect_blob.npy", rect_blob)
    (out_dir / "done.flag").write_text("ok")
    vol.commit()
    print(f"[synth3 {method}] BUILT & SAVED exact 204-cover competition synth3 in {time.time()-t0:.1f}s!", flush=True)


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=2400)
def train_v16_e2e_fold(method: str, fold: int, steps: int = 4200, bs: int = 8):
    """Train 1 fold of `DualCLIP-ConvNeXt-Large (200M CLIP LAION-2B)` + Full `ExtAttrSegNet` End-to-End!"""
    import os, sys, copy, time, pathlib
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm

    os.environ["WMLOC_MODELS"] = "/vol/models"
    repo_dir = pathlib.Path("/vol/repo/starter")
    sys.path.insert(0, str(repo_dir))
    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import downsample_mask

    t0 = time.time()
    dev = "cuda:0"
    torch.manual_seed(3400 + fold * 31 + (0 if method == "pixelseal" else 100))
    rng = np.random.RandomState(3400 + fold * 31 + (0 if method == "pixelseal" else 100))

    # Load exact 204-cover synth3
    sdir = pathlib.Path(f"/vol/synth3_exact/{method}")
    covers = np.load(sdir / "covers.npy", mmap_mode="r")
    full_imgs = np.load(sdir / "full_imgs.npy", mmap_mode="r")
    full_msgs = np.load(sdir / "full_msgs.npy")
    rect_meta = np.load(sdir / "rect_meta.npy")
    rect_msgs = np.load(sdir / "rect_msgs.npy")
    rect_offsets = np.load(sdir / "rect_offsets.npy")
    rect_blob = np.load(sdir / "rect_blob.npy", mmap_mode="r")
    N_cov, K_full = full_imgs.shape[0], full_imgs.shape[1]
    K_rect = rect_meta.shape[1]

    # Load real train & test datasets with EXACT HOLDOUT_102 split
    ds_tr = LocalisationDataset(str(repo_dir), "train")
    ds_te = LocalisationDataset(str(repo_dir), "test", methods=[method])

    holes_bank, objs_bank = [], []
    dev_rows, hold_rows = [], []
    for i in range(len(ds_tr)):
        it = ds_tr[i]
        nm = it["image_name"]
        m = it["mask"] > 0
        if nm not in HOLDOUT_102:
            if it["full_watermark"]:
                holes_bank.append(~m)
            else:
                ys, xs = np.where(m)
                if len(ys) > 16 and m.sum() / max((ys.max()-ys.min()+1)*(xs.max()-xs.min()+1), 1) <= 0.96:
                    objs_bank.append(m)
        if it["method"] == method:
            row = {"name": nm, "img": it["image"].copy(), "msg": np.asarray(it["message"], np.float32),
                   "gt128": downsample_mask(m) > 0, "isL": bool(it["full_watermark"])}
            if nm in HOLDOUT_102: hold_rows.append(row)
            else: dev_rows.append(row)
    test_rows = [{"name": ds_te[i]["image_name"], "img": ds_te[i]["image"].copy(),
                  "msg": np.asarray(ds_te[i]["message"], np.float32)} for i in range(len(ds_te))]

    # Exact 4-fold stratified split inside dev_rows (sorted by name for 100% consistency across folds!)
    dev_rows.sort(key=lambda r: r["name"])
    hold_rows.sort(key=lambda r: r["name"])
    test_rows.sort(key=lambda r: r["name"])
    idx_L = [i for i, r in enumerate(dev_rows) if r["isL"]]
    idx_S = [i for i, r in enumerate(dev_rows) if not r["isL"]]
    val_set = set([idx_L[k] for k in range(len(idx_L)) if k % 4 == fold] +
                  [idx_S[k] for k in range(len(idx_S)) if k % 4 == fold])
    tr_real = [dev_rows[i] for i in range(len(dev_rows)) if i not in val_set]
    va_real = [dev_rows[i] for i in range(len(dev_rows)) if i in val_set]
    print(f"[V16-E2E {method} f{fold}] tr_real={len(tr_real)} va_real={len(va_real)} hold={len(hold_rows)} test={len(test_rows)}", flush=True)

    # Load watermark detector + Adobe KBNet_s remover (for TrustMark!)
    wm = load_watermark(method, device=dev)
    tm_rm = None
    if method == "trustmark":
        from trustmark import TrustMark
        tm_rm = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5).removal.to(dev).eval()
        for p in tm_rm.parameters(): p.requires_grad_(False)

    # Build DualCLIP-ConvNeXt-Large + Full ExtAttrSegNet (FIXED [0,1] ImageNet norm!)
    def blk(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(min(16, co), co), nn.GELU(),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.GroupNorm(min(16, co), co), nn.GELU(),
        )

    class HighPass(nn.Module):
        def forward(self, x):
            b3 = F.avg_pool2d(x, 3, 1, 1)
            b5 = F.avg_pool2d(x, 5, 1, 2)
            return torch.cat([(x - b3) * 40.0, (x - b5) * 40.0, (b3 - b5) * 40.0], dim=1).clamp(-3.0, 3.0)

    class DualCLIPPhysicsNet(nn.Module):
        def __init__(self, meth, wm_obj, rm_obj):
            super().__init__()
            self.method = meth
            self.hp = HighPass()
            self.hp_stem = nn.Sequential(
                nn.Conv2d(9, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.GroupNorm(8, 64), nn.GELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.GroupNorm(16, 128), nn.GELU(),
            )
            # Frozen official detector for known-message 6-view maps + grad*act!
            if meth == "pixelseal":
                self.fdet = copy.deepcopy(wm_obj.det).eval()
                for p in self.fdet.parameters(): p.requires_grad_(False)
                self.nbits = 256
                self.enc = copy.deepcopy(wm_obj.det.convnext)
                echs = [96, 192, 384, 768]
                # 6-view per-bit (256+256) + 2 summary + grad*act
                self.m_proj = nn.Sequential(nn.Conv2d(514, 128, 1), nn.GroupNorm(8, 128), nn.GELU())
            else:
                self.fnet = copy.deepcopy(wm_obj.net).eval()
                for p in self.fnet.parameters(): p.requires_grad_(False)
                self.nbits = 100
                self.enc = copy.deepcopy(wm_obj.net)
                self.rm = rm_obj
                echs = [256, 512, 1024, 2048]
                self.m_proj = nn.Sequential(nn.Conv2d(202, 128, 1), nn.GroupNorm(8, 128), nn.GELU())

            # STREAM 2: MONSTER 200M CLIP LAION-2B ConvNeXt-Large (FIXED [0,1] ImageNet norm!)
            self.vis = timm.create_model(
                "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
                pretrained=True, features_only=True, out_indices=(0, 1, 2, 3),
            )
            vchs = self.vis.feature_info.channels()  # [192, 384, 768, 1536]
            if meth == "trustmark":
                # Expand first conv 3ch -> 6ch (3ch RGB + 3ch Adobe KBNet_s residual!)
                c0 = self.vis.stem_0
                nw = nn.Conv2d(6, c0.out_channels, c0.kernel_size, c0.stride, c0.padding, bias=(c0.bias is not None))
                with torch.no_grad():
                    nw.weight[:, :3].copy_(c0.weight)
                    nw.weight[:, 3:].zero_()
                    if c0.bias is not None: nw.bias.copy_(c0.bias)
                self.vis.stem_0 = nw

            self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            D = 160
            self.p3 = nn.Conv2d(echs[3] + vchs[3] + 128, D, 1)
            self.p2 = nn.Conv2d(echs[2] + vchs[2] + 128, D, 1)
            self.p1 = nn.Conv2d(echs[1] + vchs[1] + 128, D, 1)
            self.p0 = nn.Conv2d(echs[0] + vchs[0] + 128, D, 1)
            self.u2 = blk(D * 2, D); self.u1 = blk(D * 2, D); self.u0 = blk(D * 2, D)
            self.fuse = blk(D + 128, 96)
            self.head = nn.Conv2d(96, 1, 1)
            self.aux2 = nn.Conv2d(D, 1, 1); self.aux1 = nn.Conv2d(D, 1, 1)

        def _enc_feats(self, x01):
            if self.method == "pixelseal":
                x = self.fdet.preprocess(x01)
                feats = []
                for i in range(4):
                    x = self.enc.downsample_layers[i](x)
                    x = self.enc.stages[i](x)
                    feats.append(x)
                return feats
            else:
                x = x01 * 2.0 - 1.0
                e = self.enc
                x = e.maxpool(e.relu(e.bn1(e.conv1(x))))
                f0 = e.layer1(x); f1 = e.layer2(f0); f2 = e.layer3(f1); f3 = e.layer4(f2)
                return [f0, f1, f2, f3]

        @torch.no_grad()
        def _msg_maps(self, x01, sign):
            # 6-view known-message signed per-bit maps (1 global + 5 crops at 256)
            B = x01.shape[0]
            def dec(xb):
                if self.method == "pixelseal":
                    return self.fdet(xb)[:, 1:].float()
                return self.fnet(xb * 2.0 - 1.0).float()
            x256 = F.interpolate(x01, (256, 256), mode="bilinear", align_corners=False)
            g_s = (dec(x256) * sign).clamp(-6.0, 6.0) / 3.0  # [B, nbits]
            # 5 crops: 4 quadrants + center
            crops = [x01[:, :, :256, :256], x01[:, :, :256, 256:],
                     x01[:, :, 256:, :256], x01[:, :, 256:, 256:],
                     x01[:, :, 128:384, 128:384]]
            cb = torch.cat(crops, dim=0)
            cs = (dec(cb) * sign.repeat(5, 1)).clamp(-6.0, 6.0) / 3.0
            cs = cs.view(5, B, self.nbits)
            # Assemble 16x16 spatial local map
            loc = torch.zeros(B, self.nbits, 16, 16, device=x01.device, dtype=torch.float32)
            cnt = torch.zeros(1, 1, 16, 16, device=x01.device, dtype=torch.float32)
            coords = [(0, 8, 0, 8), (0, 8, 8, 16), (8, 16, 0, 8), (8, 16, 8, 16), (4, 12, 4, 12)]
            for ci, (y0, y1, x0, x1) in enumerate(coords):
                loc[:, :, y0:y1, x0:x1] += cs[ci][:, :, None, None]
                cnt[:, :, y0:y1, x0:x1] += 1.0
            loc = loc / cnt
            glob = g_s[:, :, None, None].expand(-1, -1, 16, 16)
            g_m = glob.mean(1, keepdim=True); l_m = loc.mean(1, keepdim=True)
            return torch.cat([glob, loc, g_m, l_m], dim=1)  # [B, 2*nbits + 2, 16, 16]

        def forward(self, x01, msg):
            sign = 2.0 * msg - 1.0
            mm = self.m_proj(self._msg_maps(x01, sign))  # [B, 128, 16, 16]
            hp_f = self.hp_stem(self.hp(x01))            # [B, 128, 128, 128]
            ef = self._enc_feats(x01)

            # FIXED [0,1] ImageNet normalization for CLIP ConvNeXt-Large!
            xn = (x01 - self.im_mean) / self.im_std
            if self.method == "trustmark":
                with torch.no_grad():
                    x256 = F.interpolate(x01, (256, 256), mode="bilinear", align_corners=False)
                    ro = self.rm(x256 * 2.0 - 1.0); ro = ro[0] if isinstance(ro, tuple) else ro
                    rm256 = ((ro.float() + 1.0) * 0.5).clamp(0, 1)
                    res = (x01 - F.interpolate(rm256, (512, 512), mode="bilinear", align_corners=False)) * 30.0
                    res = res.clamp(-4.0, 4.0)
                vf = self.vis(torch.cat([xn, res], dim=1))
            else:
                vf = self.vis(xn)

            def fuse_s(e, v, proj):
                m_s = F.interpolate(mm, e.shape[-2:], mode="bilinear", align_corners=False)
                v_s = F.interpolate(v, e.shape[-2:], mode="bilinear", align_corners=False) if v.shape[-2:] != e.shape[-2:] else v
                return proj(torch.cat([e, v_s, m_s], dim=1))

            x3 = fuse_s(ef[3], vf[3], self.p3)
            x2 = self.u2(torch.cat([F.interpolate(x3, ef[2].shape[-2:], mode="bilinear", align_corners=False), fuse_s(ef[2], vf[2], self.p2)], dim=1))
            x1 = self.u1(torch.cat([F.interpolate(x2, ef[1].shape[-2:], mode="bilinear", align_corners=False), fuse_s(ef[1], vf[1], self.p1)], dim=1))
            x0 = self.u0(torch.cat([F.interpolate(x1, ef[0].shape[-2:], mode="bilinear", align_corners=False), fuse_s(ef[0], vf[0], self.p0)], dim=1))
            x128 = F.interpolate(x0, (128, 128), mode="bilinear", align_corners=False)
            out = self.head(self.fuse(torch.cat([x128, hp_f], dim=1))).squeeze(1)
            if self.training:
                a2 = F.interpolate(self.aux2(x2), (128, 128), mode="bilinear", align_corners=False).squeeze(1)
                a1 = F.interpolate(self.aux1(x1), (128, 128), mode="bilinear", align_corners=False).squeeze(1)
                return out, a2, a1
            return out

    def sym_miou_loss(lg, y):
        p = torch.sigmoid(lg).flatten(1); yf = y.flatten(1)
        i1 = (p * yf).sum(1); u1 = (p + yf - p * yf).sum(1)
        i0 = ((1 - p) * (1 - yf)).sum(1); u0 = ((1 - p) + (1 - yf) - (1 - p) * (1 - yf)).sum(1)
        return (1.0 - 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0))).mean() + 0.15 * F.binary_cross_entropy_with_logits(lg, y)

    def fast_miou_vec(pred, gt):
        p = pred.reshape(len(pred), -1); g = gt.reshape(len(gt), -1)
        i1 = (p & g).sum(1); u1 = (p | g).sum(1)
        i0 = ((~p) & (~g)).sum(1); u0 = ((~p) | (~g)).sum(1)
        return 0.5 * (np.where(u1 > 0, i1 / np.maximum(u1, 1), 1.0) + np.where(u0 > 0, i0 / np.maximum(u0, 1), 1.0))

    def sample_batch():
        imgs, msgs, gts = [], [], []
        for _ in range(bs):
            if rng.rand() < 0.10:
                r = tr_real[rng.randint(0, len(tr_real))]
                imgs.append(r["img"]); msgs.append(r["msg"]); gts.append(r["gt128"].astype(np.float32))
            else:
                c = int(rng.randint(0, N_cov))
                cov = covers[c]
                mode = rng.rand()
                if mode < 0.50:
                    # L-hole (100% same cover!)
                    k = int(rng.randint(0, K_full))
                    hole = holes_bank[rng.randint(0, len(holes_bank))]
                    if rng.rand() < 0.5: hole = hole[:, ::-1]
                    if rng.rand() < 0.5: hole = hole[::-1, :]
                    im = np.where(hole[..., None], cov, full_imgs[c, k])
                    gt = downsample_mask(~hole) > 0
                    msg = full_msgs[c, k].astype(np.float32)
                elif mode < 0.75:
                    # S-rect (exact crop-embedded same cover!)
                    r = int(rng.randint(0, K_rect))
                    y0, x0, h, w = [int(v) for v in rect_meta[c, r]]
                    o0, o1 = int(rect_offsets[c * K_rect + r]), int(rect_offsets[c * K_rect + r + 1])
                    patch = np.array(rect_blob[o0:o1]).reshape(h, w, 3)
                    im = cov.copy(); im[y0:y0+h, x0:x0+w] = patch
                    m512 = np.zeros((512, 512), dtype=bool); m512[y0:y0+h, x0:x0+w] = True
                    gt = downsample_mask(m512) > 0
                    msg = rect_msgs[c, r].astype(np.float32)
                else:
                    # S-object (100% same cover!)
                    k = int(rng.randint(0, K_full))
                    obj = objs_bank[rng.randint(0, len(objs_bank))]
                    if rng.rand() < 0.5: obj = obj[:, ::-1]
                    if rng.rand() < 0.5: obj = obj[::-1, :]
                    im = np.where(obj[..., None], full_imgs[c, k], cov)
                    gt = downsample_mask(obj) > 0
                    msg = full_msgs[c, k].astype(np.float32)
                imgs.append(im); msgs.append(msg); gts.append(gt.astype(np.float32))
        x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        m = torch.from_numpy(np.stack(msgs)).to(dev)
        y = torch.from_numpy(np.stack(gts)).to(dev)
        return x, m, y

    @torch.no_grad()
    def eval_rows(model, rows):
        model.eval()
        out = np.zeros((len(rows), 128, 128), np.float32)
        for s in range(0, len(rows), 12):
            ch = rows[s:s+12]
            xb = torch.from_numpy(np.stack([r["img"] for r in ch])).to(dev).permute(0, 3, 1, 2).float() / 255.0
            mb = torch.from_numpy(np.stack([r["msg"] for r in ch])).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xb, mb).float()
            out[s:s+12] = lg.cpu().numpy()
        return out

    model = DualCLIPPhysicsNet(method, wm, tm_rm).to(dev)
    bb_params = list(model.vis.parameters()) + list(model.enc.parameters())
    bb_ids = {id(p) for p in bb_params}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in bb_ids]
    opt = torch.optim.AdamW([
        {"params": [p for p in bb_params if p.requires_grad], "lr": 2.5e-5, "weight_decay": 1e-2},
        {"params": head_params, "lr": 4e-4, "weight_decay": 1e-2},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-6)

    va_gt = np.stack([r["gt128"] for r in va_real])
    hl_gt = np.stack([r["gt128"] for r in hold_rows])
    hl_L = np.array([r["isL"] for r in hold_rows])

    best_va, best_st = -1.0, None
    for st in range(1, steps + 1):
        model.train()
        xb, mb, yb = sample_batch()
        opt.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg, a2, a1 = model(xb, mb)
            loss = sym_miou_loss(lg.float(), yb) + 0.25 * sym_miou_loss(a2.float(), yb) + 0.25 * sym_miou_loss(a1.float(), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()
        if st % 600 == 0 or st == steps:
            vl = eval_rows(model, va_real)
            hl = eval_rows(model, hold_rows)
            vsc = float(fast_miou_vec(vl >= 0.0, va_gt).mean())
            hsc = float(fast_miou_vec(hl >= 0.0, hl_gt).mean())
            hL = float(fast_miou_vec(hl[hl_L] >= 0.0, hl_gt[hl_L]).mean())
            hS = float(fast_miou_vec(hl[~hl_L] >= 0.0, hl_gt[~hl_L]).mean())
            print(f"[V16-E2E {method} f{fold} step {st:4d}/{steps}] val_OOF={vsc:.4f} | hold={hsc:.4f} (L {hL:.4f} S {hS:.4f}) | {(time.time()-t0)/60:.1f}m", flush=True)
            if vsc > best_va:
                best_va = vsc
                best_st = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_st)
    va_lg = eval_rows(model, va_real)
    hl_lg = eval_rows(model, hold_rows)
    te_lg = eval_rows(model, test_rows)
    out_dir = pathlib.Path(f"/vol/outputs/v16_e2e/{method}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"fold_{fold}.npz",
        val_names=np.array([r["name"] for r in va_real]),
        val_lg=va_lg.astype(np.float16),
        hold_names=np.array([r["name"] for r in hold_rows]),
        hold_lg=hl_lg.astype(np.float16),
        test_names=np.array([r["name"] for r in test_rows]),
        test_lg=te_lg.astype(np.float16),
        best_va=np.float32(best_va),
    )
    vol.commit()
    print(f"[V16-E2E {method} f{fold}] DONE! best_val={best_va:.4f} in {(time.time()-t0)/60:.1f}m", flush=True)
    return float(best_va)


@app.local_entrypoint()
def main():
    # Stage 1: Build exact 204-cover competition synth3 for both methods in parallel (~50s!)
    b1 = build_exact_synth3.spawn("pixelseal")
    b2 = build_exact_synth3.spawn("trustmark")
    b1.get(); b2.get()
    # Stage 2: Launch ALL 8 FOLDS (4 PixelSeal + 4 TrustMark) in parallel on 8x Modal GPUs (~18 min!)
    hs = []
    for method in ("pixelseal", "trustmark"):
        for f in range(4):
            hs.append((method, f, train_v16_e2e_fold.spawn(method, f, 4200, 8)))
    for m, f, h in hs:
        print(f"FINISHED {m} f{f}: val_OOF={h.get():.4f}")
