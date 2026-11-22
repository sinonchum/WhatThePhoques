# Extending Simon's Task 1 localization models

**Keep the scored submission.** The [pinned upstream experiment notes](https://github.com/sinonchum/euro-ai-championship-paris/blob/00d38f7807b93dabe3b622a0d684c004cb9e808d/docs/tasks/task1-v2-v4-experiments.md) record #1249 at 0.6813 public and 0.6864 on the seed-2026 holdout (PS 0.6839, TM 0.6457, SS 0.7295). These numbers belong to that submission, **not** to every clean four-fold checkpoint in the same branch. No new public score. No automatic submission. `80` is the target, not a result.

## What was tested

Simon's `feat/task1-stegastamp-extnet` provides decoder features and later `synth3` four-fold experts. The shared `sub_v1.pt` has 1,020 named boolean grids. We checked its SHA-256 (`06ea399cbefc63b51bace702fb2ac4a69d47a5e21643d429badacf86a8c41ce2`); the archive alone does not prove the scorer received those exact bytes. Training uses clean development predictions only; each source used for stacking must declare `dev_in_sample=False`. Per method: 136 development + 34 untouched-by-fitting holdout; 340 unseen test masks. Compare variants on development folds, freeze, then report holdout. The same holdout has been reported multiple times, so it is not a fresh blind score.

| JURECA experiment | Paired holdout comparison | Decision |
|---|---|---|
| 15902730: clean single expert per method | 0.65993 overall, same 102 IDs as team | control, not #1249 |
| Shape repair / auxiliary blend | 0.66248; TM 0.62497 → 0.63262 | experimental; no full-team upgrade |
| Image-conditioned residual adapter | 0.66357; PS 0.67587 → 0.68362, TM 0.62497 → 0.62812, SS unchanged at 0.67896 | complete candidate; below 0.6864 |
| 15902741: restore original four TM V3 checkpoints; refit boundary correction on each fold's original training IDs | TM meta-validation 0.64489 → 0.64284 | reject extra head; other 680 archived predictions byte-for-byte equal as mask arrays |
| 15902790: two clean experts per method plus RGB boundary and adaptive-threshold comparison | plain blend 0.66659 → 0.66974 overall (PS 0.68451 / TM 0.63363 / SS 0.69108) | correction passed development gate only on SS; not a replacement for #1249 |

The best experimental full artifact is `simon80-region-15902790/submission.pt` (SHA-256 `f1a0896cfd2e2211876590bfd9ea6f234d2627ddf0ebbe4d8b06aa4871e05d63`).

## Running / using the results

The two files in `upstream/` are unchanged copies of Simon's V2/V3 implementations at the pinned commit. Team model and prediction inputs in `refine_oof.ROOTS` and `warm_trustmark.TEAM` are read-only; weights and release data are **not** in Git. On JURECA, from the prepared scratch root containing `train.zip`, `test.zip`, `native-env/` (scikit-learn 1.7.2, NumPy, SciPy, Pillow, cluster PyTorch), put this folder under `native75-src/simon80` and `task1/forensic_features.py` under `native75-src/`. Jobs write to a new, job-specific output directory. For the warm start, load torchvision 0.24.1 too.

```bash
sbatch native75-src/simon80/run.sbatch     # shape and forensic-adapter comparison
sbatch native75-src/simon80/warm.sbatch    # original TM fold weights, strict parity check
sbatch native75-src/simon80/region.sbatch  # paired clean experts + color-guided boundaries and image-specific thresholds
```

Follow-up `15902808` is conditional on the team's existing V4-TMF producer `15902588` completing successfully. `sbatch --dependency=afterok:<producer-id> --kill-on-invalid-dep=yes native75-src/simon80/region.sbatch --trustmark-v4` selects the assembled V4-TMF TrustMark bundle instead of the auxiliary 512 model; no new training is claimed while the dependency is pending. Missing bundles or in-sample flags do not silently fall back.

For analysis without retraining, use `simon80-15902730/{adapter,shape}/report.json`, `submission.pt`, and per-method `{method}-{holdout,test}.npz` under the prepared scratch root. For the warm start, inspect `simon80-warm-15902741/report.json`. Region outputs use `simon80-region-$SLURM_JOB_ID/`. Read `report.json` and verify its file hash before accepting a candidate. Each full `.pt` has `image_names` and `masks`, official filenames in manifest order and `torch.bool[1020,128,128]`. A diagnostic oracle that chooses a configuration using each dev image's true mask is **not an achievable score**. The default handoff recommendation remains #1249 unless a challenger clears its own paired comparison.
