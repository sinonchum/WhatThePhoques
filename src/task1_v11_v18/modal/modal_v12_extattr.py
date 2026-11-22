"""V12 Modal 2xA100-80GB/H100 Breakthrough:
Exact proven `ExtAttrSegNet` (`grad*act` + 6-view per-bit message maps + HighPass(40) stem + unfrozen detector)
trained on 2,500 CLEAN EXTERNAL COCO COVERS (12.3x larger than synth3's 204 covers!)
with 100% PURE SAME-COVER `cov` compositing (ZERO c2 / blur shortcuts!) + INFINITE fresh random messages!
Uses the EXACT 102 `holdout_names(seed=2026)` matching JURECA 100% (ZERO holdout leakage!).
Evaluates BOTH pure-synth (p_real=0.0) and DEV-finetuned (p_real=0.08, strictly excluding holdout!) with 4x TTA!
"""
import modal

app = modal.App("task1-v12-extattr-2500")
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

EXACT_HOLDOUT = {
    "01a775d8e589", "01a8a19cb848", "0703932c7705", "07b2717b53ad", "0ae0e6224c29",
    "0b35f8d56e9f", "0f715eadd6d99"[:12], # wait - fix typo below!
}
# Clean exact 102 holdout names (no terminal line-wrap duplicates!)
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


@app.function(image=image, gpu=["A100-80GB", "H100", "H200"], volumes={"/vol": vol}, timeout=2400)
def train_v12_extattr(method: str = "pixelseal", steps: int = 3500, bs: int = 12, fold: int = 0, p_real: float = 0.08):
    import os, sys, time, math, pathlib, copy
    import numpy as np
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    repo_dir = pathlib.Path("/vol/repo/starter")
    models_dir = pathlib.Path("/vol/models")
    os.environ["WMLOC_MODELS"] = str(models_dir)
    sys.path.insert(0, str(repo_dir))

    from wmloc.data import LocalisationDataset
    from wmloc.watermarks import load_watermark
    from wmloc.utils import miou_image, downsample_mask

    dev = "cuda:0"
    torch.backends.cudnn.benchmark = True
    seed = (1300 if method == "pixelseal" else 1400) + fold
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # 1. Load 2,500 clean external COCO covers!
    covers = np.load("/vol/clean_covers.npy")  # [2500, 512, 512, 3] uint8
    print(f"[{method}-V12 f{fold}] Loaded {len(covers)} clean COCO covers (12.3x synth3!)", flush=True)

    # 2. Load real train set + verify EXACT 102 holdout match + build DEV-only ShapeBank (ZERO holdout leak!)
    ds_all = LocalisationDataset(str(repo_dir), "train")
    all_items = [ds_all[i] for i in range(len(ds_all))]
    matched_h = [it["image_name"] for it in all_items if it["image_name"] in HOLDOUT_102]
    print(f"[{method}-V12 f{fold}] Matched exact holdout images: {len(matched_h)}/102", flush=True)
    assert len(matched_h) == 102, f"Expected 102 holdout matches, got {len(matched_h)}"

    # Build DEV-only ShapeBank (matching wmtask/synth.py exact physics!)
    holes, objs, rect_hws = [], [], []
    n_small, n_rect = 0, 0
    max_aspect = 2.0 if method == "trustmark" else 16.0
    as_grp = {"trustmark"} if method == "trustmark" else {"pixelseal", "stegastamp"}
    for it in all_items:
        if it["image_name"] in HOLDOUT_102:
            continue
        m = it["mask"] > 0
        if it["full_watermark"]:
            hole = ~m
            if hole.sum() >= 32:
                ys, xs = np.where(hole)
                holes.append(hole[ys.min():ys.max()+1, xs.min():xs.max()+1].copy())
        else:
            n_small += 1
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
            if m.sum() / max((y1 - y0) * (x1 - x0), 1) > 0.98:
                n_rect += 1
                if it["method"] in as_grp:
                    rect_hws.append((y1 - y0, x1 - x0))
            else:
                objs.append(m[y0:y1, x0:x1].copy())
    rect_share = n_rect / max(1, n_small)
    print(f"[{method}-V12 f{fold}] DEV-only ShapeBank: holes={len(holes)} objs={len(objs)} rect_hws={len(rect_hws)} rect_share={rect_share:.3f}", flush=True)

    def place_region(bank, min_a, max_a):
        reg = bank[int(rng.integers(len(bank)))]
        if rng.random() < 0.5: reg = reg[:, ::-1]
        if rng.random() < 0.5: reg = reg[::-1, :]
        if rng.random() < 0.5: reg = reg.T
        h, w = reg.shape
        sc = float(rng.uniform(0.65, 1.45))
        nh = int(np.clip(round(h * sc), 12, 480))
        nw = int(np.clip(round(w * sc), 12, 480))
        reg_r = cv2.resize(reg.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        # Area clamp
        area = reg_r.mean() * (nh * nw) / (512 * 512)
        if area > max_a:
            f = math.sqrt(max_a / area)
            nh, nw = max(12, int(nh * f)), max(12, int(nw * f))
            reg_r = cv2.resize(reg.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) > 0
        out = np.zeros((512, 512), dtype=bool)
        y0 = int(rng.integers(0, 512 - nh + 1))
        x0 = int(rng.integers(0, 512 - nw + 1))
        out[y0:y0+nh, x0:x0+nw] = reg_r
        return out

    def sample_rect_hw():
        for _ in range(100):
            h, w = rect_hws[int(rng.integers(len(rect_hws)))]
            h = int(np.clip(round(h * rng.uniform(0.75, 1.30)), 8, 496))
            w = int(np.clip(round(w * rng.uniform(0.75, 1.30)), 8, 496))
            if max(h, w) / min(h, w) <= max_aspect and h * w <= 0.45 * 512 * 512:
                return h, w
        return 96, 96

    # Split method's real items into DEV (136) and HOLDOUT (34) + 4-fold stratified split inside DEV!
    m_items = [it for it in all_items if it["method"] == method]
    dev_items = [it for it in m_items if it["image_name"] not in HOLDOUT_102]
    hold_items = [it for it in m_items if it["image_name"] in HOLDOUT_102]

    # Exact stratified 4-fold inside DEV (same logic as train_v3_attr.py)
    val_names = set()
    for full in (False, True):
        sub = [it for it in dev_items if bool(it["full_watermark"]) == full]
        sub.sort(key=lambda it: (float(downsample_mask(it["mask"] > 0).mean()), it["image_name"]))
        frng = np.random.default_rng(2026 + (1 if full else 0))
        for st in range(0, len(sub), 4):
            blk = sub[st:st+4]
            perm = frng.permutation(len(blk))
            if fold < len(blk):
                val_names.add(blk[int(perm[fold])]["image_name"])
    tr_real = [it for it in dev_items if it["image_name"] not in val_names]
    va_real = [it for it in dev_items if it["image_name"] in val_names]
    print(f"[{method}-V12 f{fold}] Real split: tr_dev={len(tr_real)}, val_dev={len(va_real)}, untouched_hold={len(hold_items)}", flush=True)

    def pack_eval(items):
        im = np.stack([it["image"] for it in items])
        ms = np.stack([it["message"] for it in items]).astype(np.float32)
        gt = np.stack([(downsample_mask(it["mask"] > 0) > 0) for it in items])
        lg = np.array([bool(it["full_watermark"]) for it in items])
        return im, ms, gt, lg

    va_im, va_ms, va_gt, va_L = pack_eval(va_real)
    ho_im, ho_ms, ho_gt, ho_L = pack_eval(hold_items)
    tr_im, tr_ms, tr_gt, tr_L = pack_eval(tr_real)

    # 3. Load official embedder + detector (and Adobe KBNet_s remover for TrustMark!)
    wm = load_watermark(method, device=dev)
    nbits = 256 if method == "pixelseal" else 100
    tm_enc = None; tm_rm = None
    if method == "trustmark":
        from trustmark import TrustMark
        tm_obj = TrustMark(verbose=False, secret_len=100, model_type="Q", encoding_type=TrustMark.Encoding.BCH_5)
        tm_enc = tm_obj.encoder.to(dev).eval()
        tm_rm = tm_obj.removal.to(dev).eval()
        for m_ in (tm_enc, tm_rm):
            for p in m_.parameters(): p.requires_grad_(False)
    else:
        wm.model.eval()
        for p in wm.model.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def embed_full(c1_01, msgs):
        if method == "pixelseal":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return wm.model.embed(c1_01, msgs=msgs.long(), is_video=False)["imgs_w"].float().clamp(0, 1)
        else:
            c256 = F.interpolate(c1_01, (256, 256), mode="bilinear", align_corners=False)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = tm_enc(c256 * 2.0 - 1.0, msgs.float())
                stego = out[0] if isinstance(out, tuple) else out
            w256 = ((stego.float() + 1.0) * 0.5).clamp(0, 1)
            return (c1_01 + F.interpolate(w256 - c256, (512, 512), mode="bilinear", align_corners=False)).clamp(0, 1)

    @torch.no_grad()
    def embed_rect_crop(crop01, msg_1):
        h, w = crop01.shape[-2], crop01.shape[-1]
        c256 = F.interpolate(crop01, (256, 256), mode="bilinear", align_corners=False)
        if method == "pixelseal":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                w256 = wm.model.embed(c256, msgs=msg_1.long(), is_video=False)["imgs_w"].float()
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = tm_enc(c256 * 2.0 - 1.0, msg_1.float())
                stego = out[0] if isinstance(out, tuple) else out
            w256 = ((stego.float() + 1.0) * 0.5)
        delta = F.interpolate(w256 - c256, (h, w), mode="bilinear", align_corners=False)
        return (crop01 + delta).clamp(0, 1)

    # 100% PURE SAME-COVER `cov` ON-THE-FLY GENERATOR (ZERO c2 / blur shortcuts!)
    @torch.no_grad()
    def sample_batch(B: int, cur_p_real: float):
        idx = rng.integers(0, len(covers), size=B)
        c1 = torch.from_numpy(covers[idx].copy()).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if rng.random() < 0.5: c1 = torch.flip(c1, [-1])
        if rng.random() < 0.5: c1 = torch.flip(c1, [-2])
        msgs = torch.randint(0, 2, (B, nbits), device=dev).float()
        wm_full = embed_full(c1, msgs)

        xs = torch.empty_like(c1)
        ys = torch.zeros((B, 128, 128), device=dev, dtype=torch.float32)
        is_L = torch.zeros(B, device=dev, dtype=torch.float32)
        out_msgs = msgs.clone()

        for b in range(B):
            if cur_p_real > 0.0 and rng.random() < cur_p_real:
                ri = int(rng.integers(len(tr_im)))
                xi = torch.from_numpy(tr_im[ri].copy()).to(dev).permute(2, 0, 1).float() / 255.0
                yi = torch.from_numpy(tr_gt[ri].astype(np.float32)).to(dev)
                if rng.random() < 0.5: xi = torch.flip(xi, [-1]); yi = torch.flip(yi, [-1])
                if rng.random() < 0.5: xi = torch.flip(xi, [-2]); yi = torch.flip(yi, [-2])
                xs[b] = xi; ys[b] = yi; is_L[b] = float(tr_L[ri])
                out_msgs[b] = torch.from_numpy(tr_ms[ri]).to(dev)
                continue

            if rng.random() < 0.50:
                # LARGE (50%): SAME-COVER cov!
                is_L[b] = 1.0
                hole = place_region(holes, 0.002, 0.45)
                h_t = torch.from_numpy(hole.astype(np.float32)).to(dev)[None, None]
                xs[b] = (wm_full[b:b+1] * (1.0 - h_t) + c1[b:b+1] * h_t)[0]
                ys[b] = (F.avg_pool2d(1.0 - h_t, 4) >= 0.5).float()[0, 0]
            elif rng.random() < rect_share:
                # SMALL-RECT (~25%): SAME-COVER cov, exact crop-embedded watermark!
                h, w = sample_rect_hw()
                y0 = int(rng.integers(0, 512 - h + 1))
                x0 = int(rng.integers(0, 512 - w + 1))
                crop = c1[b:b+1, :, y0:y0+h, x0:x0+w]
                wcrop = embed_rect_crop(crop, msgs[b:b+1])
                img = c1[b].clone()
                img[:, y0:y0+h, x0:x0+w] = wcrop[0]
                xs[b] = img
                m512 = torch.zeros((1, 1, 512, 512), device=dev)
                m512[0, 0, y0:y0+h, x0:x0+w] = 1.0
                ys[b] = (F.avg_pool2d(m512, 4) >= 0.5).float()[0, 0]
            else:
                # SMALL-OBJECT (~25%): SAME-COVER cov!
                obj = place_region(objs, 0.004, 0.44)
                o_t = torch.from_numpy(obj.astype(np.float32)).to(dev)[None, None]
                xs[b] = (wm_full[b:b+1] * o_t + c1[b:b+1] * (1.0 - o_t))[0]
                ys[b] = (F.avg_pool2d(o_t, 4) >= 0.5).float()[0, 0]

        xs = torch.round(xs.clamp(0, 1) * 255.0) / 255.0
        # Return [-1, 1] matching ExtAttrSegNet exact convention!
        return xs * 2.0 - 1.0, ys, is_L, out_msgs

    # 4. Build EXACT Proven `ExtAttrSegNet` (+ Dual-Scale Adobe KBNet_s residual for TrustMark!)
    def block(cin, cout, stride=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(inplace=True),
        )

    class HighPass(nn.Module):
        def __init__(self, scale=40.0, clip=3.0):
            super().__init__()
            self.scale, self.clip = scale, clip
        def forward(self, x):
            b3 = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1)
            b5 = F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5, 1)
            hp = torch.cat([x - b3, x - b5, b3 - b5], 1) * self.scale
            return torch.cat([x, hp.clamp(-self.clip, self.clip)], 1)

    def norm_attr(g, f):
        ga = g * f
        scale = ga.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        ga_n = (ga / scale).clamp(-6.0, 6.0)
        pos = ga_n.mean(dim=1, keepdim=True)
        absm = ga_n.abs().mean(dim=1, keepdim=True)
        gn = g.norm(dim=1, keepdim=True)
        gn = (gn / gn.mean(dim=(2, 3), keepdim=True).clamp(min=1e-6)).clamp(0.0, 6.0)
        return torch.cat([ga_n, pos, absm, gn], dim=1)

    class ExtAttrV12Net(nn.Module):
        def __init__(self, method, wm, tm_rm=None, dec=192):
            super().__init__()
            self.method = method
            self.tm_rm = tm_rm
            if method == "trustmark":
                self.frozen = wm.net
                self.enc = copy.deepcopy(wm.net)
                self.enc.fc = nn.Identity()
                chs, self.nbits = [256, 512, 1024, 2048], 100
            else:
                self.frozen = wm.det
                self.enc = copy.deepcopy(wm.det.convnext)
                chs, self.nbits = [96, 192, 384, 768], 256
            for p in self.frozen.parameters(): p.requires_grad_(False)
            for p in self.enc.parameters(): p.requires_grad_(True)

            self.msg_glob = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(True))
            self.msg_loc = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(True))
            self.msg_mid = nn.Sequential(nn.Conv2d(self.nbits + 2, dec, 1), nn.SiLU(True))
            self.attr1 = nn.Sequential(nn.Conv2d(chs[1] + 3, dec, 1), nn.SiLU(True))
            self.attr2 = nn.Sequential(nn.Conv2d(chs[2] + 3, dec, 1), nn.SiLU(True))
            self.attr3 = nn.Sequential(nn.Conv2d(chs[3] + 3, dec, 1), nn.SiLU(True))
            self.lat = nn.ModuleList(nn.Conv2d(c, dec, 1) for c in chs)
            self.top = block(4 * dec, dec)
            self.glob = nn.Sequential(nn.Linear(dec, dec), nn.SiLU(True))
            self.smooth = nn.ModuleList(block(dec, dec) for _ in range(3))

            self.hp_front = HighPass(scale=40.0, clip=3.0)
            hp_in_c = 12 + (6 if method == "trustmark" else 0)  # +6ch dual-scale Adobe KBNet_s residual (256+512)!
            self.hp_stem = nn.Sequential(block(hp_in_c, 48, 1), block(48, 96, 2), block(96, dec, 2))
            self.fuse_hp = block(2 * dec, dec)
            self.head = nn.Sequential(block(dec, 64), nn.Conv2d(64, 1, 1))
            self.cls = nn.Linear(dec, 1)

        def train(self, mode=True):
            super().train(mode)
            self.frozen.eval()
            for m in self.enc.modules():
                if isinstance(m, nn.BatchNorm2d): m.eval()
            return self

        def encode(self, x01):
            if self.method == "trustmark":
                n = self.enc
                x = n.maxpool(n.relu(n.bn1(n.conv1(x01 * 2.0 - 1.0))))
                out = []
                for k in ("layer1", "layer2", "layer3", "layer4"):
                    x = getattr(n, k)(x); out.append(x)
                return out
            h = self.frozen.preprocess(x01)
            out = []
            for down, stage in zip(self.enc.downsample_layers, self.enc.stages):
                h = stage(down(h)); out.append(h)
            return out

        @torch.no_grad()
        def _single_msg_map(self, x01_256, bits):
            with torch.autocast("cuda", enabled=False):
                x01_256 = x01_256.float()
                if self.method == "trustmark":
                    n = self.frozen
                    x = n.maxpool(n.relu(n.bn1(n.conv1(x01_256 * 2.0 - 1.0))))
                    x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
                    per = torch.einsum("bchw,oc->bohw", x, n.fc.weight) + n.fc.bias[None, :, None, None]
                else:
                    d = self.frozen
                    lat = d.convnext(d.preprocess(x01_256))
                    emb = d.pixel_decoder.output_upscaling(lat)
                    lin = d.pixel_decoder.linear
                    per = (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
                s = per * (2.0 * bits.float() - 1.0)[:, :, None, None]
                return torch.cat([(s / 4.0).clamp(-4, 4), s.mean(1, keepdim=True) / 4.0, (s > 0).float().mean(1, keepdim=True) * 2.0 - 1.0], dim=1)

        @torch.no_grad()
        def multi_crop_msg(self, x01_512, bits):
            B = x01_512.shape[0]
            g256 = F.interpolate(x01_512.float(), (256, 256), mode="bilinear", align_corners=False, antialias=True)
            all_x = torch.cat([
                g256,
                x01_512[:, :, 0:256, 0:256], x01_512[:, :, 0:256, 256:512],
                x01_512[:, :, 256:512, 0:256], x01_512[:, :, 256:512, 256:512],
                x01_512[:, :, 128:384, 128:384],
            ], dim=0)
            all_m = self._single_msg_map(all_x, bits.repeat(6, 1))
            mg, m00, m01, m10, m11, mc = all_m.chunk(6, dim=0)
            C = mg.shape[1]
            cv = torch.zeros((B, C, 16, 16), device=mg.device, dtype=mg.dtype)
            wt = torch.zeros((1, 1, 16, 16), device=mg.device, dtype=mg.dtype)
            cv[:, :, 0:8, 0:8] += m00; wt[:, :, 0:8, 0:8] += 1
            cv[:, :, 0:8, 8:16] += m01; wt[:, :, 0:8, 8:16] += 1
            cv[:, :, 8:16, 0:8] += m10; wt[:, :, 8:16, 0:8] += 1
            cv[:, :, 8:16, 8:16] += m11; wt[:, :, 8:16, 8:16] += 1
            cv[:, :, 4:12, 4:12] += mc; wt[:, :, 4:12, 4:12] += 1
            return mg, cv / wt

        def _frozen_grad_attr(self, x01_512, bits):
            sign = 2.0 * bits.float() - 1.0
            with torch.autocast("cuda", enabled=False):
                x_in = F.interpolate(x01_512.detach().float(), (256, 256), mode="bilinear", align_corners=False, antialias=True)
                with torch.enable_grad():
                    xr = x_in.requires_grad_(True)
                    if self.method == "trustmark":
                        n = self.frozen
                        h = n.maxpool(n.relu(n.bn1(n.conv1(xr * 2.0 - 1.0))))
                        s0 = n.layer1(h); s1 = n.layer2(s0); s2 = n.layer3(s1); s3 = n.layer4(s2)
                        logits = n.fc(s3.mean((2, 3)))
                    else:
                        d = self.frozen
                        h = d.preprocess(xr); feats = []
                        for down, stage in zip(d.convnext.downsample_layers, d.convnext.stages):
                            h = stage(down(h)); feats.append(h)
                        s1, s2, s3 = feats[1], feats[2], feats[3]
                        emb = d.pixel_decoder.output_upscaling(s3); lin = d.pixel_decoder.linear
                        per = (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
                        logits = per.mean((2, 3))
                    obj = (logits * sign).mean(1).sum()
                    g1, g2, g3 = torch.autograd.grad(obj, [s1, s2, s3])
                with torch.no_grad():
                    return norm_attr(g1.detach(), s1.detach()), norm_attr(g2.detach(), s2.detach()), norm_attr(g3.detach(), s3.detach())

        def forward(self, x, bits):
            with torch.autocast("cuda", enabled=False):
                xf = x.float()
                x01 = (xf + 1.0) * 0.5
            mg, mloc = self.multi_crop_msg(x01, bits)
            a1, a2, a3 = self._frozen_grad_attr(x01, bits)
            f = self.encode(x01)
            h3, w3 = f[3].shape[-2:]
            y = self.top(torch.cat([
                self.lat[3](f[3]),
                F.interpolate(self.msg_glob(mg), (h3, w3), mode="bilinear", align_corners=False),
                F.interpolate(self.msg_loc(mloc), (h3, w3), mode="bilinear", align_corners=False),
                F.interpolate(self.attr3(a3), (h3, w3), mode="bilinear", align_corners=False),
            ], dim=1))
            g = self.glob(y.mean((2, 3)))
            y = y + g[:, :, None, None]
            y = self.smooth[2](F.interpolate(y, f[2].shape[-2:], mode="nearest") + self.lat[2](f[2])
                               + F.interpolate(self.msg_mid(mloc), f[2].shape[-2:], mode="bilinear", align_corners=False)
                               + F.interpolate(self.attr2(a2), f[2].shape[-2:], mode="bilinear", align_corners=False))
            y = self.smooth[1](F.interpolate(y, f[1].shape[-2:], mode="nearest") + self.lat[1](f[1])
                               + F.interpolate(self.attr1(a1), f[1].shape[-2:], mode="bilinear", align_corners=False))
            y = self.smooth[0](F.interpolate(y, f[0].shape[-2:], mode="nearest") + self.lat[0](f[0]))
            y128 = F.interpolate(y, (128, 128), mode="bilinear", align_corners=False)

            with torch.autocast("cuda", enabled=False):
                hp_in = self.hp_front(xf)
                if self.method == "trustmark":
                    with torch.no_grad():
                        x256 = F.interpolate(xf, (256, 256), mode="bilinear", align_corners=False)
                        r256 = self.tm_rm(x256); r256 = r256[0] if isinstance(r256, tuple) else r256
                        res256 = F.interpolate((x256 - r256).float(), (512, 512), mode="bilinear", align_corners=False)
                        r512 = self.tm_rm(xf); r512 = r512[0] if isinstance(r512, tuple) else r512
                        res512 = (xf - r512).float()
                        hp_in = torch.cat([hp_in, (res256 * 12.0).clamp(-3, 3), (res512 * 12.0).clamp(-3, 3)], dim=1)
            hp128 = self.hp_stem(hp_in.to(y128.dtype))
            y128 = self.fuse_hp(torch.cat([y128, hp128], dim=1))
            return self.head(y128)[:, 0], self.cls(g)[:, 0]

    def sym_miou_loss(logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits).flatten(1); t = target.flatten(1)
        i1 = (p * t).sum(1); u1 = (p + t - p * t).sum(1)
        i0 = ((1 - p) * (1 - t)).sum(1); u0 = ((1 - p) + (1 - t) - (1 - p) * (1 - t)).sum(1)
        sym = 0.5 * ((i1 + 1.0) / (u1 + 1.0) + (i0 + 1.0) / (u0 + 1.0)).mean()
        return 0.45 * bce + 0.55 * (1.0 - sym)

    net = ExtAttrV12Net(method, wm, tm_rm=tm_rm, dec=192).to(dev)
    bb_p = list(net.enc.parameters())
    bb_ids = {id(p) for p in bb_p} | {id(p) for p in net.frozen.parameters()}
    hd_p = [p for p in net.parameters() if p.requires_grad and id(p) not in bb_ids]
    print(f"[{method}-V12 f{fold}] Trainable params: bb={sum(p.numel() for p in bb_p)/1e6:.1f}M head={sum(p.numel() for p in hd_p)/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW([{"params": bb_p, "lr": 1e-4}, {"params": hd_p, "lr": 1e-3}], weight_decay=1e-4)
    warm = 250
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    )

    @torch.no_grad()
    def eval_tta(imgs_np, msgs_np, gt_np, L_np):
        net.eval()
        probs = []
        for s in range(0, len(imgs_np), 12):
            xr = torch.from_numpy(imgs_np[s:s+12].copy()).to(dev).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            mr = torch.from_numpy(msgs_np[s:s+12]).to(dev)
            acc_p = 0.0
            for hf, vf in [(False, False), (True, False), (False, True), (True, True)]:
                xt = torch.flip(xr, [-1]) if hf else xr
                xt = torch.flip(xt, [-2]) if vf else xt
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = net(xt, mr)
                if vf: lg = torch.flip(lg, [-2])
                if hf: lg = torch.flip(lg, [-1])
                acc_p = acc_p + torch.sigmoid(lg.float())
            probs.append((acc_p / 4.0).cpu().numpy())
        probs = np.concatenate(probs)
        best_t, best_sc, best_ious = 0.5, 0.0, None
        for t in (0.35, 0.45, 0.50, 0.55, 0.65):
            ious = np.array([miou_image(probs[i] >= t, gt_np[i]) for i in range(len(gt_np))])
            if ious.mean() > best_sc:
                best_sc, best_t, best_ious = float(ious.mean()), t, ious
        return best_sc, best_t, float(best_ious[L_np].mean()), float(best_ious[~L_np].mean()), probs

    t0 = time.time()
    best_va = 0.0
    for step in range(1, steps + 1):
        net.train()
        # First 1500 steps: pure 2,500-cover synthetic (zero real overfit); after 1500: mix p_real from tr_dev!
        cur_pr = 0.0 if step <= 1500 else p_real
        xb, yb, lb, mb = sample_batch(bs, cur_pr)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            seg, cls = net(xb, mb)
            loss = sym_miou_loss(seg.float(), yb) + 0.2 * F.binary_cross_entropy_with_logits(cls.float(), lb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % 500 == 0 or step == steps:
            v_sc, v_t, vL, vS, _ = eval_tta(va_im, va_ms, va_gt, va_L)
            # Evaluate holdout using val-selected threshold v_t (ZERO holdout tuning!)
            h_sc, _, hL, hS, h_pr = eval_tta(ho_im, ho_ms, ho_gt, ho_L)
            h_at_vt = np.array([miou_image(h_pr[i] >= v_t, ho_gt[i]) for i in range(len(ho_gt))])
            h_strict = float(h_at_vt.mean())
            if v_sc > best_va:
                best_va = v_sc
                torch.save(net.state_dict(), f"/vol/outputs/v12_{method}_f{fold}_best.pt")
            print(f"[{method}-V12 f{fold} step {step:4d}/{steps}] ({time.time()-t0:5.1f}s pr={cur_pr:.2f}) | "
                  f"VAL_OOF={v_sc:.4f} (thr={v_t:.2f} L {vL:.4f} S {vS:.4f}) | "
                  f"UNTOUCHED_HOLD={h_strict:.4f} (L {h_at_vt[ho_L].mean():.4f} S {h_at_vt[~ho_L].mean():.4f})", flush=True)
    vol.commit()
    return best_va


@app.local_entrypoint()
def main():
    h_ps = train_v12_extattr.spawn(method="pixelseal", steps=3500, bs=12, fold=0, p_real=0.08)
    h_tm = train_v12_extattr.spawn(method="trustmark", steps=3500, bs=12, fold=0, p_real=0.08)
    print(f"=== V12 PILOT DONE: PS_VAL={h_ps.get():.4f} | TM_VAL={h_tm.get():.4f} ===")
