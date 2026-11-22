# Task 1 — V2/V3/V4 Architectures, Clean `synth3` Retraining & Pure-Dev OOF CV

## 1. Model Iterations (`src/task1_v2_v4/`)

- **V2 (`extnet_v2.py`, `stega_extnet_v2.py`, `train_v2.py`)**:
  - Upgrades frozen known-message map from single global 256×256 (`8×8`) to **6-view 5-crop stitched `16×16`** (`global + 4 quadrants + center`) so small patches (`<15%` area) keep readable known-message evidence.
  - Adds **512×512 → 128×128 HighPass residual stem** + EMA checkpoint selection on unbiased validation split.
- **V3 (`extnet_v3_attr.py`, `train_v3_attr.py`, `assemble_kfold.py`)**:
  - Brings **multi-scale `grad * act` known-message spatial attribution** (exact gradient of known-message alignment through the frozen official decoder at 3 spatial scales, L2-normalized + channel mean/max/pos-ratio) to **PixelSeal** and **TrustMark**, paired with **4-fold stratified OOF CV** (`dev_in_sample=False`).
- **V4-RM TrustMark Specialist (`extnet_v4_tm_rm.py`, `train_v4_tm_rm.py`)**:
  - Adds Adobe's official TrustMark neural watermark remover (`tm.removal`, `trustmark_rm_Q.ckpt`) across 6 views to produce:
    1. Full-resolution 512×512 removal residual `x - rm_Q(x)` (measured `1.87×–2.34×` fg/bg SNR on small real TrustMark dev patches before any training) fed into the HighPass stem.
    2. Differential known-message map `message_map(x) - message_map(rm_Q(x))` isolating pixels where the official remover erased known bits.
- **Clean `synth3` StegaStamp K-Fold (`train_stega_s3_kfold.py`)**:
  - 4-fold stratified OOF trainer for both `StegaExtSegNet` (V1) and `StegaExtSegNetV2` (V2), trained **strictly on clean `synth3/stegastamp`** (train-only non-holdout covers + `--lowres` 256-embed upsampled-delta synthesis).
- **Pure-Dev OOF CV Fusion (`fuse_clean_cv.py`)**:
  - Explicitly filters to **clean `synth3` models only** (excludes legacy `synth2` / `ext1` / `ext3real` / `ext4_512real_testcovers`).
  - Fits threshold-aligned logit weights & bias **strictly on honest dev OOF** (`dev_in_sample=False`), never using seed-2026 holdout for model selection or weight tuning.

## 2. Verified Milestones

- **First live submission (`#1249`, 2026-11-21)**:
  - Local seed-2026 holdout: `0.6864` (PixelSeal `0.6839` / TrustMark `0.6457` / StegaStamp `0.7295`)
  - Official public leaderboard: **`0.6813`** (Task 1 #3; team Overall #2)
- **Clean `synth3` 4-Fold OOF Pipeline (active on JURECA)**:
  - `slurm/job_v4_tmrm_s3.sbatch`: TrustMark V4-RM 4-fold OOF on clean `synth3/trustmark`
  - `slurm/job_v3_ps_s3.sbatch`: PixelSeal V3 `grad*act` 4-fold OOF on clean `synth3/pixelseal`
  - `slurm/job_ss_s3_4fold_cispa.sbatch`: StegaStamp V2 + V1 4-fold OOF on clean `synth3/stegastamp`
