# Task 1 — V11–V18 Clean Pipeline (JURECA + Modal Dual-Platform Sprint)

Final public leaderboard progression for team **WhatThePhoques** (100% clean zero-leakage training covers, strict 4-fold nested DEV OOF CV, zero holdout tuning):

- **Sub `1546`**: `0.7362` (baseline clean `synth3` ensemble)
- **Sub `2261`**: `0.7634` (`V5.2` + `MetaStackNet` + `rect_nnref`)
- **Sub `3751`**: `0.7710` (`+0.0076`) — `[0, 1]` input normalization bug fix (`V11-Fix01`) + L/S split-threshold + adaptive hole-dip recovery
- **Sub `3821`**: `0.7868` (`+0.0158`) — `SAM-1 ViT-H` semantic snapping (rectangularity guard + Top-2 union + soft `b2` L-hole blend)
- **Sub `4256` (Final)**: **`0.7897`** (`+0.0029`, **DEV OOF `0.7943` / Untouched Holdout `0.7871`**) — 43 clean JURECA models + 4 cross-method full-resolution refiners (`v14_cnxl_clip384`, `v15_cnxl_clip512`, `v16_cnxl_occ512`, `v17_dinov2l_ms_apex518` with 4-stage multi-scale DINOv2-Large + `TruFor`/`Noiseprint++` + `MMFusion-IML` + dual-scale white-box dense occlusion maps) + `Spatial MetaStacker-V3` + joint `SAM-1 vs Axis-Rect` arbitration (`RECT_SNAP`)

## Directory Layout

- `jureca/`:
  - `job_v11_jureca_fix01.sbatch`, `job_v11_tm_and_sub4.sbatch`: `[0, 1]` normalization fix retrains (`TrustMark V11-Fix01` `0.6935` dev / `0.6908` hold, `PixelSeal V7.2-Fix01` `0.6809` hold, `StegaStamp V11-Fix01` `0.8711` dev).
  - `job_v12_fix01_ps.sbatch`, `job_v12_fix01_tm_ss.sbatch`: Heterogeneous Fix01 retrains (`TrustMark Swin-Base-384 Fix01` `0.6811` dev, `PixelSeal ConvNeXt-Base Fix01` `0.6796` hold / `S 0.7110`, `PixelSeal Swin-Base Fix01`, `StegaStamp Swin-Base Fix01`).
  - `job_v13_eva_fix01_ps.sbatch`, `job_v13_eva_fix01_tm.sbatch`: `PixelSeal ConvNeXt-HP6 Fix01` (`0.6811` hold / `S 0.7148`), `PixelSeal EVA02-Base-448 Fix01` (`0.6789` hold), `TrustMark EVA02-Base-448 Fix01`, `TrustMark ConvNeXt-Base dec=256 Fix01` (`0.6806` hold).
  - `export_top8_v4.py`: Clean honest-OOF model packager (`43` models).
- `modal/`:
  - `modal_wb_probe*.py`, `modal_carrier_test.py`, `modal_rise_shapley_test.py`, `modal_forensic_probe.py`, `modal_sam_oracle_test.py`: White-box diagnostic & oracle probes.
  - `modal_sam_h_pipeline.py`, `modal_sam_h_v2_upgrade.py`, `modal_sam1_v17_reprompt.py`, `modal_sam2_pipeline.py`, `modal_sam_decoder_ranker.py`: `SAM-1 ViT-H` & `SAM 2.1 Hiera-Large` multi-GPU proposal extraction, white-box decoder occlusion ranking, and `mask_input` re-prompting.
  - `modal_dense_occ_maps.py`: Dual-scale (`12x12` + `8x8`) white-box decoder dense patch occlusion heatmap extractor for PixelSeal & TrustMark (`1,020` images).
  - `modal_forensics_trufor_mmfusion.py`: Zero-shot academic forensic feature extraction (`TruFor` CVPR 2023 `map + conf + Noiseprint++` & `MMFusion-IML` MMM 2024 `SRM + BayarConv + Noiseprint++`) across all `1,530` images.
  - `modal_v14_cross_refiner.py`, `modal_v15_monster_refiners.py`, `modal_v16_dense_occ_refiner.py`, `modal_v17_apex_parallel.py`, `modal_v17b_giant_and_clip.py`: Cross-method 8x Dihedral-TTA neural refiners (`CLIP LAION-2B ConvNeXt-Large/XXLarge`, `Meta DINOv2-Large Multi-Scale`, `Meta DINOv2-Giant 1.1B`).
  - `modal_v8_grand_fusion.py`, `modal_v18_champion_fusion.py`: Strict 4-fold nested DEV CV fusion engines (`Spatial MetaStacker-V3`, `rect_nnref`, joint `SAM-1 vs Axis-Rect` arbitration, speck cleaning).

## Final Per-Method Honest Metrics (`sub_v18_dino_ms.pt`, ID `4256`)

- **PixelSeal**: DEV `0.7193` (`L 0.7102 / S 0.7285`) | Holdout `0.7148` (`L 0.6830 / S 0.7466`)
- **TrustMark**: DEV `0.7628` (`L 0.6895 / S 0.8360`) | Holdout `0.7342` (`L 0.6800 / S 0.7884`)
- **StegaStamp**: DEV `0.9008` (`L 0.8890 / S 0.9125`) | Holdout `0.9122` (`L 0.8726 / S 0.9517`)
- **3-Method Average**: DEV OOF **`0.7943`** | Untouched Holdout **`0.7871`** | Public LB **`0.7897`**
