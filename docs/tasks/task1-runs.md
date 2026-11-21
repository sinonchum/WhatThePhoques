# Task 1: competition runs

Local scores are **not** public scores. The official server requires the team's own API key. None of these scripts uploads a submission, reads a team credential or optimizes against public test feedback. Keep models, data, credentials and competition outputs out of Git.

## Data and models

`ROOT` below is a private cache directory. Put the official `train.zip` and `test.zip` there, pinned to [dataset revision 8d27b2c9](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris/tree/8d27b2c9cc37c4829c8501e43e11fcb1b5db58c0). Dependencies: Python, NumPy, Pillow, SciPy, scikit-learn, threadpoolctl, PyTorch, torchvision, timm, einops, onnx and onnxruntime. Neural paths were tested on Mac MPS. For the message paths:

- `ROOT/videoseal`: [Meta Video Seal](https://github.com/facebookresearch/videoseal) at `b1bf4c0`; `ROOT/pixelseal.pth`: official `https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth`. Load weights only.
- `ROOT/decoder_Q.ckpt`: [Adobe's Q decoder](https://cai-watermark.adobe.net/watermarking/trustmark-models/decoder_Q.ckpt), publisher MD5 `4ced90e9cfe13e3295ad082887fe9187`; loaded strictly into a ResNet-50 with 100 outputs.
- `ROOT/stega_stamp.onnx`: original [WAVES LFS object](https://media.githubusercontent.com/media/umd-huang-lab/WAVES/main/decoders/stega_stamp.onnx), SHA-256 `369f8134f4e35da9659777f02468f4200c256d14742b9702f2ac28808da675e2`. The small Git LFS pointer itself is not a model.
- Semantic prior: torchvision's `LRASPP_MobileNet_V3_Large_Weights.DEFAULT`, used only as evidence of object extent, not as watermark ground truth.

## Reproduce

Run from the repository root with the dependencies installed:

```bash
for split in train test; do
  python task1/forensic_features.py "$ROOT" "$split"
  for m in trustmark pixelseal; do python task1/message_cam.py "$ROOT" "$split" "$m"; done
  python task1/stega_pair.py "$ROOT" "$split"
done
python task1/semantic_prior.py "$ROOT"
python task1/train_grid.py "$ROOT" --epochs 24
for m in trustmark pixelseal stegastamp; do
  python task1/fit_message_fusion.py "$ROOT" "$m"
  python task1/polarity.py "$ROOT" "$m"
done
python task1/train_cue_net.py "$ROOT" --epochs 40 --seed 203
python task1/pack_best.py "$ROOT" "$ROOT/submission.pt"
```

The prediction file contains exactly `image_names` and `masks`. The packer verifies the 1,020 official names by index and method, uniqueness, `[1020,128,128]` binary masks, size under 20 MB, and `weights_only` round trip; it writes a sibling JSON with SHA-256 and method selection. Neural checkpoints are selected on 48 development images; the 102-image holdout is scored separately. The 360 fit, 48 development and 102 holdout images do not overlap by filename; source-level near-duplicates are not ruled out. `full_watermark` is a training target/stratification variable, not an inference feature.

First local runs on the same 102 images: residual ridge 0.3150, forensic-grid U-Net 0.3985, three calibrated boosted message heads ~0.488, message/semantic neural model (40 epochs, seed 203) 0.5031. Per-method scores for the last model: PixelSeal 0.5204, TrustMark 0.4697, StegaStamp 0.5191. The margin is too small to infer public rank. Later candidates store their exact selections and local scores in their sibling JSON. A verified official result needs the organizers' returned score and an updated Task 1 leaderboard.
