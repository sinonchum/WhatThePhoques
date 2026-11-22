#!/bin/bash -l
#SBATCH --job-name=w2b-v9hr
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:55:00
#SBATCH --output=/p/home/jusers/qin7/jureca/sbatch_v8/w2b_v9hr_%j.out
#SBATCH --error=/p/home/jusers/qin7/jureca/sbatch_v8/w2b_v9hr_%j.out

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

find /p/scratch/training2666/qin7/task1/runs -mindepth 2 -maxdepth 3 -name "best*.pt" -mmin +15 ! -path "*/submissions/*" -delete 2>/dev/null || true

echo "=== W2B STAGE 1: TrustMark V9 High-Res dec=256 ConvNeXt-Base-88M + 6ch Adobe-RM (p_real=0.10, bf16) ==="
export V5_VIS_BACKBONE=convnext_base.fb_in22k_ft_in1k_384
export V5_VIS_MODE=rm
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method trustmark \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/trustmark \
    --out runs/v9_tmb256_rm6_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v9_tmb256_rm6_f${F}_trustmark \
    --steps 5000 --bs 4 --lr 5e-4 --lr-backbone 2.5e-5 --dec 256 --enc-res 512 \
    --p-real 0.10 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((1401 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method trustmark \
  --folds runs/v9_tmb256_rm6_f0 runs/v9_tmb256_rm6_f1 runs/v9_tmb256_rm6_f2 runs/v9_tmb256_rm6_f3 \
  --out runs/fusion/qin7_s3_v9_tmb256_rm6_4fold_trustmark
rm -f runs/v9_tmb256_rm6_f*/best*.pt runs/v9_tmb256_rm6_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v9_tmb256_rm6_f*

echo "=== W2B STAGE 2: PixelSeal V9 High-Res dec=256 ConvNeXt-Base-88M + RGB (p_real=0.06 anti-overfit, bf16) ==="
export V5_VIS_BACKBONE=convnext_base.fb_in22k_ft_in1k_384
export V5_VIS_MODE=rgb
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method pixelseal \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/pixelseal \
    --out runs/v9_psb256_rgb_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v9_psb256_rgb_f${F}_pixelseal \
    --steps 5000 --bs 4 --lr 5e-4 --lr-backbone 2.5e-5 --dec 256 --enc-res 512 \
    --p-real 0.06 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((1501 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method pixelseal \
  --folds runs/v9_psb256_rgb_f0 runs/v9_psb256_rgb_f1 runs/v9_psb256_rgb_f2 runs/v9_psb256_rgb_f3 \
  --out runs/fusion/qin7_s3_v9_psb256_rgb_4fold_pixelseal
rm -f runs/v9_psb256_rgb_f*/best*.pt runs/v9_psb256_rgb_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v9_psb256_rgb_f*
echo "=== W2B DONE ==="
