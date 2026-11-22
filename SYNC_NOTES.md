# Night Sync Notes (`task23_sync`)

Updated: 2026-11-22 01:05 CET

## 1. Task 3 Critical Findings (`task3_sync/`)
- **Online probe confirmation (`surg_b26` -> `0.000`, `v3_a` -> `0.069`)**:
  - Suppressing the 9 `count=130` classes by `-26` on `head.bias` (`CE mean/std = 4.12 / 6.14`) scored **`0.000`** on the public split.
  - Because `91/100` classes remain untouched (~80%+ accuracy, well above the 70% gate), scoring `0.000` implies `1 - d_sub / d_orig <= 0`, proving mathematically that **`mean(L_gold) + std(L_gold) <= 5.13`**.
  - Therefore, the 9 classes with 130 images are **NOT** 100%-deleted classes (130 is simply a per-class sampling cap when building the 3500-image forget set). The true evaluation regime is **instance-level partial unlearning (`world partial`)**, where mini-gold benchmarks give `mean(L_gold) ≈ 0.5–0.8, std(L_gold) ≈ 1.1–1.5`.
- **Synergy with `origin/chuanv0` (`fit_b8.pt` = `0.1470`)**:
  - `chuanv0` showed that distilling the ViT **block-8 Logit Lens readout** (`softmax(head(fc_norm(norm(h_8[:,0]))))`) reaches `0.1470`, because instance memorization in `original_model.pt` concentrates in `blocks[9..11]`, while `fit_b8`'s bottleneck is natural-image retain drift (`KL(orig || fit_b8) ~ 1.0` on natural crops/mixups -> `retain_score ~ 0.50`) and 55 zero-count classes not being frozen.
  - Meanwhile, our 2nd-order mini-gold benchmark (`task3_sync/modal_app.py`, `bench_b2.json`) reaches **`FINAL = 0.293` (`forget = 0.411 [mean_std 0.881 x pred 0.467] x retain 0.712`)** in `world partial` using difficulty-amplified targets (`orig_reinit4` cross-fit + strong-view margin), and combining **block-8 readout + natural-view retain anchors + freezing the 55 zero-count classes (`count == 0` in forget set)** directly addresses `fit_b8`'s retain bottleneck.

## 2. Task 2 Critical Findings & Reference Datasets (`task2_sync/`)
- **Remaining Formal API Budget**: `ep-01`..`ep-33` have **`39` to `47` queries remaining** (`ep-05` has 39). Server gateway latency is ~10-12s/request with a team-wide concurrency cap of 4 (`HTTP 429` on >4 open requests; **never retry HTTP 502**, which still deducts budget).
- **2-Shot Exemplar Prefixing (`hermes_open2shot.py`)**:
  - Prepending two `user`/`assistant` single-token turns (no `system` role used) suppresses verbose provider system prompts / personas cleanly.
  - Across 48 2-shot probes cached on `ref-01..ref-09` (`hermes_open2shot_ref.jsonl`, `hermes_easy2shot_ref.jsonl`, `hermes_calib2shot_ref.jsonl`), trailing-period leakage on no-period 2-shot exemplars cleanly separates **all three LoRA strengths (`0.3`: `0.052`, `0.7`: `0.135`, `1.5`: `0.365`)** from **all five SAME-model variants (`plain, system prompt, persona, watermark, temp 1.0`: `0.000–0.021`)** with zero false positives (`MCC = 0.80` before adding the FP8 channel).
  - Crucial pitfall documented: `ref-06` (watermark) exhibits deterministic repeat collisions on hard open questions while `ref-01/03/05/08` sample with `T > 0`; naive U-statistic `X - (w_1 + w_2)/2` over-penalizes `w_watermark ≈ 0` unless paired with calibrated consensus / multi-token divergence features.
