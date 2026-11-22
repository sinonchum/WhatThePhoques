# Task 1 — StegaStamp White-Box Localization Analysis & Results

## 1. Evaluation Protocol (Team-Unified)

- **Split**: `holdout_names(rows, seed=2026)` from `ding7`'s `wmtask/common.py` — 34 held-out images per watermark method (17 `full_watermark=True` / large, 17 `full_watermark=False` / small), 102 held-out images total.
- **Strict isolation**: The 34 held-out StegaStamp images are never used for training, checkpoint selection, or threshold tuning.
- **Metric**: Official per-image 2-class mean IoU (`wmloc.utils.miou_image`) on the `128x128` grid.
- **Reference baseline**: Constant per-image oracle (choosing all-0 or all-1 per image) caps at **0.4547** across all 510 train images (**0.4601** on StegaStamp).

## 2. Diagnosis of Earlier Bottlenecks

Prior experiments established why naive baselines stalled below 0.50:

- **Decoder-only spatial maps & thresholds**: PixelSeal 8x8 per-position agreement + global threshold scored ~0.47; TrustMark official bbox detector (`trustmark==0.9.2`, `localize()`) scored **0.3543** across 170 train images; window-crop decoding on TrustMark scored **0.2914**; 4x4 occlusion drop scored **0.3924**.
- **Blind image-only heads**: Small RGB/high-pass CNNs, bounding-box regressors, and frozen ImageNet ConvNeXt FPNs scored **0.26–0.35** on held-out images because post-hoc watermark residuals are ~0.7–1.3 gray levels and mask boundaries do not align with visible RGB edges (only 6/72 solid-rectangle masks had strongest gradient energy on the true box boundary).
- **StegaStamp-specific gap in `ding7`'s pipeline**: `ding7`'s `ExtSegNet` (`extnet.py`) boosted PixelSeal (`0.6316` held-out) and TrustMark (`0.6004` held-out) by pairing the official PyTorch extractor with the known message, but did not support StegaStamp because StegaStamp is distributed as an ONNX graph (`stega_stamp.onnx`). Their blind StegaStamp model (`stegastamp_v3`) reached **0.6064** on `small` patches but collapsed to **0.1569** on `large` (`full_watermark=True`), giving **0.3816** overall on held-out.

## 3. Method: Native PyTorch StegaStamp Port + `StegaExtSegNet`

To close the StegaStamp gap on both `large` (full-image watermark with clean hole) and `small` (embedded rectangle / transplanted object), we implemented:

1. **Exact PyTorch CUDA port of the official StegaStamp decoder (`src/task1_stegastamp/stegastamp_pt.py`)**
   - Loads weights directly from `models/stegastamp/stega_stamp.onnx` (`gen_stn` + 7-conv `gen_decoder` + `dec_dense` + `decoder_out`).
   - Verified numerical parity against `onnxruntime` on 6 real StegaStamp train images: worst-case absolute logit difference **3.72e-5** (bit-exact predictions).
   - Runs natively on CUDA 12.4 without ONNXRuntime CUDA 13 library issues, and enables exact autograd attribution.

2. **`StegaExtSegNet` (`src/task1_stegastamp/stega_extnet.py`)**
   - **Frozen official StegaStamp decoder**:
     - Global known-message summary (STN-warped and unwarped): 100 signed bit logits, mean signed margin, and bit accuracy (204 channels). On StegaStamp `large` images, global bit accuracy is typically 0.83–1.00 vs ~0.50 on `small` patches, eliminating the `large` collapse.
     - Unwarped spatially aligned first-order message attribution (`grad * act` + normalized gradient norms) computed via one autograd pass of `sum((2*bits - 1) * logits)` at `100x100`, `50x50`, `25x25`, and `13x13`.
   - **Trainable copy of the official 7-conv StegaStamp decoder backbone**: initialized from the official weights (`lr=1e-4`), preserving StegaStamp-tuned residual filters from step 0.
   - **Full-resolution `512 -> 128` HighPass residual stem**: fixed truncated high-pass front end (`scale=40, clip=3`) + 3 residual conv blocks fused at `128x128` to capture sharp patch/hole boundaries.

3. **Training & Leakage-Free Selection (`src/task1_stegastamp/train_stega_ext.py`, `train_stega_5fold.py`)**
   - Reuses `ding7`'s low-resolution synthetic pool (`/p/scratch/training2666/ding7/claude_task1/runs/synth2/stegastamp`) and cover bank (`runs/covers`).
   - Loss: `BCE + soft 2-class per-image mIoU` + `0.2 * BCE(cls, is_large)`.
   - When mixing real DEV images (`p_real=0.25`), an internal 20% stratified slice of DEV (28 images) is held back from training (`--dev-val-frac 0.2`) so checkpoint and threshold selection remain unbiased without touching the 34 seed-2026 holdout images.

## 4. Verified Results (Seed-2026 Holdout, Official `miou_image`)

### Completed & Exported for Fusion

| Run | Setup | Dev (n=136) | Holdout Large (n=17) | Holdout Small (n=17) | Holdout Both (n=34) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `ding7` `stegastamp_v3` (ref) | Blind `WMSegNet`, synth only | 0.3737 | 0.1569 | 0.6064 | 0.3816 | Baseline before extractor port |
| `qin7_stegaext_live_stegastamp` | `StegaExtSegNet` (dec=128, `p_real=0.25`, `dev_val=20%`), step 2000, `@0.5` | 0.8661 (sel=0.6745) | 0.6689 | 0.6113 | **0.6401** | `dev_in_sample=True` |
| `qin7_stegaext_live_stegastamp` | Same checkpoint, dev-val tuned `thr=0.20` | 0.8617 | **0.6672** | **0.6478** | **0.6575** | Exported & schema-verified |

Export path on JURECA (group/world readable):
`/p/scratch/training2666/qin7/task1/runs/fusion/qin7_stegaext_live_stegastamp/{dev,holdout,test}.npz`
- Verified shapes/dtypes: `dev.npz` `(136, 128, 128) float32`, `holdout.npz` `(34, 128, 128) float32`, `test.npz` `(340, 128, 128) float32`, `threshold=0.20`, `dev_in_sample=True`.

### In-Progress Runs on JURECA (Intermediate Checkpoints @0.5, Unfinalized)

- **`stegaext_real25_a100`** (A100 job `15902088`, `dec=160`, `p_real=0.25`, `dev_val=20%`): step 2500/6000 reached `sel 0.7184` | holdout **0.6732** (`large 0.6735`, `small 0.6729`).
- **`stegaext_synth_a100`** (A100 job `15902088`, `dec=160`, **pure synthetic `p_real=0.0`**, `dev_in_sample=False`): step 3000/6000 reached `dev 0.6420` | holdout **0.6370** (`large 0.5902`, `small 0.6838`).
- **`stegaext_real35_a100`** (A100 job `15902088`, `dec=192`, `p_real=0.35`, `dev_val=20%`): step 2500/6000 reached `sel 0.6515` | holdout **0.6541** (`large 0.6740`, `small 0.6342`).
- **`stegaext_5fold`** (RTX 8000, 5-fold stratified CV over DEV for 100% honest OOF `dev.npz` + 5-fold ensemble): running fold 0 (`oof_val 0.6033` at step 500/2500).

## 5. Impact on Team-Wide Task 1 Holdout

Combining `ding7`'s current PixelSeal (`0.6316`) and TrustMark (`0.6004`) runs with `qin7`'s exported StegaStamp run (`0.6575`) raises the unweighted 3-method holdout average from **0.538** to **0.630** before multi-model fusion, with the in-flight A100 and 5-fold StegaStamp runs already trending toward **0.67+** on StegaStamp alone.
