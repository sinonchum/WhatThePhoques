# Task 1 — use our StegaStamp predictions with your two heads

**Agent entry point (current: v2):** use `stegastamp-test-v2.npz` for all **340 StegaStamp** test images. Keep your own **340 PixelSeal + 340 TrustMark** masks. No weights, training cache, Git LFS, network access or credentials are required at merge time. Do not merge entire submissions by index or stack three method arrays in blocks: the official metadata interleaves the methods.

## Merge (Python, NumPy, PyTorch)

From this branch checkout, run:

```bash
python task1/merge_stegastamp.py --base /path/to/teammate.pt --metadata /path/to/test.zip --output /path/to/merged-submission.pt
```

`--metadata` also accepts the extracted official `test/metadata.json`. `--base` must be a PyTorch dictionary `{'image_names': list[str], 'masks': Tensor[N,128,128]}` with 0/1 values. It may contain all 1020 names or **only** the other 680 names. Name strings are the official `image_name`, not paths/filenames. Output must be new. The script checks bundle SHA-256, restores official order, reloads the result, and checks every non-StegaStamp mask was preserved and every StegaStamp mask came from this bundle. It **doesn't submit** anything. Submit the resulting `.pt`, not the `.npz` or the source archive.

## What is actually measured

**v2 (recommended):** `stega-native-refine-v1-global-3701-56` improves the same 34-image holdout from **0.60144 to 0.61633**. It freezes v1 and learns a local correction using whole-image StegaStamp decoder attribution for the supplied message. The arm was selected on the 16-image inner development set only: global `0.65616`, magnitude-only `0.64901`, all scales `0.64866`, no correction `0.63505`. The held-out labels were then evaluated separately, and the exported 34 named masks were rescored independently against the archive. The 340 test predictions are ready; there is **no official score** for v2 or a merged team submission. `merge_stegastamp.py` now defaults to v2. To reproduce v1 instead, pass `--bundle task1/handoff/stegastamp-test-v1.npz`.

To reproduce the v2 extension after setting up the v1 cache, run `python task1/stega_native_cam.py ROOT --check`, `python task1/stega_native_cam.py ROOT` and `python task1/stega_refine.py ROOT --arm global --seed 3701 --epochs 56`. The decoder check compares native ONNX logits and signed/complement maps. The export tool `task1/export_stega_head.py` validates the official test names and independently recomputes the fixed holdout score before writing an immutable versioned bundle.

**Earlier v1 benchmark:**

- Version `v1`: `neural-message-v1-1451-120-r1t1`. An encoder residual for known vs opposite payloads, image forensics and local interpolation/noise statistics feed a mask head and a separate polarity head. Dev picked epoch 76 and threshold 0.5. The test prediction uses image, method and message; no `full_watermark` or hidden labels.
- **StegaStamp alone:** 0.63505 inner dev (16 images), **0.60144 independent local holdout** (34 images). The arrays in `stegastamp-hold-v1.npz` reproduce the latter against the corresponding 34 training labels. They are NOT test predictions. The fixed split is defined by `task1/train_grid.py::split_ids`.
- **Three-method baseline:** 0.55263 on our 102-image holdout before combining with any teammate. The teammate's heads have not been evaluated here; the merged score is **unknown**. No official score is claimed.
- Both bundles contain Unicode `image_names` and NumPy boolean `masks`; `True` = **watermark remains**. Their matching `.json` files give split, counts, source, exact metric, dataset revision and SHA-256. `np.load(..., allow_pickle=False)` is sufficient. `v1` is immutable; new experiments should publish a new version, not silently change these files.

## Reproduction / experiments

Use revision `8d27b2c9cc37c4829c8501e43e11fcb1b5db58c0` of `SprintML/Hackathon_Watermark_Localisation_Task_Paris`. The main trainer was run as `python task1/train_cue_net.py ROOT --rewrite --tile --seed 1451 --epochs 120` after extracting the forensic, message, semantic, rewrite and tile feature caches. See `task1/stega_pair.py`, `rewrite_evidence.py`, `tile_evidence.py` and `train_cue_net.py`. The historical feature/training scripts target Apple MPS; the **merge script is CPU-only and portable**. Pretrained models and original competition data aren't redistributed here.
