# Task 3 (Unlearning) — cross-fitted stand-in for the retrained model

For hu / li103. Everything here is for Task 3; nothing has been submitted from the new target yet.

## What is in this folder

| File | What |
|---|---|
| `gold_standin_f7.npz` | Per-image predictions from late blocks that never saw the image (5 seeds). Image order = `forget_images.npz`. |
| `modal_app.py` | All code. `oof_blocks_remote` builds the target, `calib_remote` checks a target against public scores, `unlearn_remote` fits a model. |
| `calibration_table.log` | Output of the check: each candidate target vs. 8 publicly scored models. |
| `RESULTS.md` | Full experiment log (Chinese), newest first, including every public score I got. |

### Keys in `gold_standin_f7.npz`

- `p_T2_mean` [3500, 100] — mean over 5 seeds of softmax(logits / 2). Matches the official definition of p_gold (average of the retrained models' T=2 distributions). **Use this as the distillation target at T=2.**
- `logits_from_mean` [3500, 100] — `2 * log(p_T2_mean)`.
- `logits_per_seed` [5, 3500, 100] — T=1 logits of each seed.

Labels are the original model's argmax on the clean forget images.

## How the target is built

Blocks 0–6 frozen, blocks 7–11 re-initialised, `norm` and `head` frozen at the original values. The 3500 forget images are split into 5 class-stratified folds. For each fold the re-initialised blocks are trained for 40 epochs on the other 4 folds (teacher = the original model's T=2 distribution on augmented views, crop 0.35–1), then predict the held-out fold. So every image is predicted by blocks that never saw it.

Held-out fold: accuracy 0.44–0.49, CE mean 2.1–2.4, CE std 2.6–2.9. Training folds: accuracy 0.99+. Seeds agree on the argmax for 73 % of the images.

Starting the re-initialisation at block 8 instead gives accuracy 0.61–0.68 and explains the public scores worse (see below). Block 6 is close to block 7. Block 10 or no re-initialisation keeps the memorisation (accuracy 0.97–1.00) and is useless.

## Why I think it is closer to gold than the block-8 read-out

Each candidate target was treated as gold, the official forget formula was applied to 8 models with known public scores (altered copies = crops 0.6–1), and the result compared with the public score:

| Candidate target | Correlation with public scores | public ≈ factor × estimate |
|---|---|---|
| **blocks 7–11 cross-fitted (seed 0, logits)** | **0.996** | 0.29 |
| block-8 read-out as T=2 distribution (fit_b8's target) | 0.978 | 0.18 |
| blocks 8–11 cross-fitted | 0.91–0.93 | — |

Per model, estimate × 0.29 vs. actual: fit_b8 0.144 / 0.1476, v11 0.132 / 0.1246, v10 0.091 / 0.0916, v3_a 0.064 / 0.069, v6 0.055 / 0.056, v5 0.033 / 0.040, v12 0.037 / 0.043, surg_b26 0 / 0.

Under this target every existing model has a prediction score of about 0.6 and fit_b8 has a mean/std score of 0.82.

**Not verified:** no model fitted to this target has a public score. The 0.29 factor is unexplained (retain term, or a systematic gap between this stand-in and the real gold). High correlation does not prove this is gold.

## Three facts from public scores (same target, different training)

1. Forgetting must also hold on cropped copies. Block-8 target, accuracy on 0.85–1 crops 0.71 / 0.61 / 0.55 → public 0.092 / 0.125 / 0.148.
2. Staying close to the original model on other inputs hardly changes the score. v10 is 8× closer than fit_b8 on strong crops and scores lower.
3. Driving accuracy on the forget images to 0 is bad (0.043). Keeping 60–80 % is also bad (0.04–0.07). The peak is near 0.5.

## Unsubmitted weights (on the Modal volume `euroai-task3`, `/ckpt/`)

- `v13_b8_crop50_w025.pt` — block-8 target, KL(target ‖ model) 0.038 on clean images (fit_b8: 0.057), accuracy 0.48 clean / 0.56 on crops.
- `v13_r78_crop50_w025.pt` — average of block-7 and block-8 read-outs, accuracy 0.42 clean / 0.50 on crops.

## Addendum 08:45 — after reading hu's `task3_xuanhao` (public 0.2756)

hu's scores settle one thing my calibration could not see: the 9 capped classes must be pushed to a loss of about 10 at image level (0.186 → 0.251). None of the 8 models in my calibration did that properly, so the cross-fitted target above is only a candidate for the **other 36 classes** (2331 images). For the capped classes keep hu's rows.

`xh_target7_f7.py` is a drop-in for hu's pipeline: it takes one of his target files, keeps the capped rows, and replaces the rest with `k × logits_from_mean` (plus his count-graded shift). Not run against the leaderboard.

Idea not done (needs ~10 min of GPU, 9 short jobs): for each capped class, train the re-initialised late blocks with **that whole class left out**, then predict its images. That gives, per image, the distribution over the other classes from a model that never saw the class — the open question in hu's README ("which classes get the mass on capped images").

## Addendum 08:47 — `t_world.pt`: a full target in hu's format, from models that never saw the 9 capped classes

**For hu: `t_world.pt` is a `{"zt", "full"}` file, same layout as your `runs/t_*.pt`.** Copy it to `task3/runs/` and e.g.

```bash
python xh_train.py --name r_world --target runs/t_world.pt --init runs/r14.pt --steps 900 --lr 6e-5 --w_anchor 1.5 --w_t1 1.0 --gap_full 10 --aug2 1 --same_class 1
```

How it was made (`oof_blocks_remote` with `world=True` in `modal_app.py`): blocks 7–11 re-initialised, head/norm frozen, trained **without any image of the 9 capped classes** and without the held-out fold of the other classes; the teacher (original model on augmented views) has the 9 capped class logits masked out. Then it predicts all capped images and the held-out fold. 5 folds × 2 seeds; T=2 distributions averaged (capped images: over all 10 models).

| | accuracy vs. memorised label | CE mean / std |
|---|---|---|
| all 3500 | 0.371 | 5.23 / 5.47 |
| capped (1169) | 0.000 | 11.99 / 2.83 |
| rest (2331) | 0.556 | 1.85 / 2.57 |

- On capped images the label ends up 11.7 logits below the best other class — close to the "about 10" you got from the leaderboard, without having been set by hand.
- Overall 5.23 / 5.47 is close to your best model's 5.2 / 5.0.
- T=2 mass on *other* capped classes for capped images: 0.008 (your current model: 0.31). This is by construction (those classes were removed from training and from the teacher), so it is the same assumption as your `xh_target6.py`, not independent evidence for it.
- Seeds agree on the argmax for 80 % of the images.

What is different from `t_mixed` / `t_grad4`: the distribution over the *other* classes comes from a trained model that never saw the image (and never saw the capped classes), instead of a read-out mixture; the rest classes are not count-graded (add `xh_target5.py`-style grading on top if it still helps).

**Not verified on the leaderboard.** `gold_standin_world.npz` has the same data as arrays (`p_T2_mean`, `logits_from_mean`, `logits_per_seed`, `capped_image`, `label`).
