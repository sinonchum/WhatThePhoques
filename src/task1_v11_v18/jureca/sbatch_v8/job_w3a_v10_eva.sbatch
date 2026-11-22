#!/bin/bash -l
#SBATCH --job-name=w3a-v10
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:55:00
#SBATCH --output=/p/home/jusers/qin7/jureca/sbatch_v8/w3a_v10_%j.out
#SBATCH --error=/p/home/jusers/qin7/jureca/sbatch_v8/w3a_v10_%j.out

set -euo pipefail
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models

jutil env activate -p training2666 >/dev/null
module load Stages/2025
module load PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

echo "=== W3A STAGE 1: PixelSeal V10 EVA02-Base-448 + RGB (3rd Orthogonal MIM-ViT Family, p_real=0.08, bf16) ==="
export V5_VIS_BACKBONE=eva02_base_patch14_448.mim_in22k_ft_in22k_in1k
export V5_VIS_MODE=rgb
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method pixelseal \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/pixelseal \
    --out runs/v10_pseva_rgb_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v10_pseva_rgb_f${F}_pixelseal \
    --steps 5000 --bs 6 --lr 5e-4 --lr-backbone 2e-5 --dec 160 --enc-res 512 \
    --p-real 0.08 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((1601 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method pixelseal \
  --folds runs/v10_pseva_rgb_f0 runs/v10_pseva_rgb_f1 runs/v10_pseva_rgb_f2 runs/v10_pseva_rgb_f3 \
  --out runs/fusion/qin7_s3_v10_pseva_rgb_4fold_pixelseal
rm -f runs/v10_pseva_rgb_f*/best*.pt runs/v10_pseva_rgb_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v10_pseva_rgb_f*

echo "=== W3A STAGE 2: StegaStamp V10 High-Res dec=256 ConvNeXt-Base-88M (p_real=0.12, bf16) ==="
export V5_VIS_BACKBONE=convnext_base.fb_in22k_ft_in1k_384
export V5_VIS_MODE=rgb
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method stegastamp \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/stegastamp \
    --out runs/v10_ssb256_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v10_ssb256_f${F}_stegastamp \
    --steps 5000 --bs 4 --lr 5e-4 --lr-backbone 2.5e-5 --dec 256 --enc-res 512 \
    --p-real 0.12 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((1701 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method stegastamp \
  --folds runs/v10_ssb256_f0 runs/v10_ssb256_f1 runs/v10_ssb256_f2 runs/v10_ssb256_f3 \
  --out runs/fusion/qin7_s3_v10_ssb256_4fold_stegastamp
rm -f runs/v10_ssb256_f*/best*.pt runs/v10_ssb256_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v10_ssb256_f*
echo "=== W3A DONE ==="
