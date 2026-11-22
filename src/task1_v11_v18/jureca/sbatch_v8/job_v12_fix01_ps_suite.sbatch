#!/bin/bash -l
#SBATCH --job-name=v12-ps-fix
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:55:00
#SBATCH --output=/p/home/jusers/qin7/jureca/sbatch_v8/v12_ps_fix_%j.out
#SBATCH --error=/p/home/jusers/qin7/jureca/sbatch_v8/v12_ps_fix_%j.out

set -euo pipefail
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models
jutil env activate -p training2666 >/dev/null
module load Stages/2025 PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

echo "=== V12-PS STAGE 1: PixelSeal ConvNeXt-Base-88M + RGB (FIXED [0,1] ImageNet norm!, p_real=0.08, 4-fold OOF) ==="
export V5_VIS_BACKBONE=convnext_base.fb_in22k_ft_in1k_384
export V5_VIS_MODE=rgb
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method pixelseal \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/pixelseal \
    --out runs/v12_psb_fix01_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v12_psb_fix01_f${F}_pixelseal \
    --steps 5500 --bs 8 --lr 5e-4 --lr-backbone 2.5e-5 --dec 160 --enc-res 512 \
    --p-real 0.08 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((2401 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method pixelseal \
  --folds runs/v12_psb_fix01_f0 runs/v12_psb_fix01_f1 runs/v12_psb_fix01_f2 runs/v12_psb_fix01_f3 \
  --out runs/fusion/qin7_s3_v12_psb_fix01_4fold_pixelseal
rm -f runs/v12_psb_fix01_f*/best*.pt runs/v12_psb_fix01_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v12_psb_fix01_f*

echo "=== V12-PS STAGE 2: PixelSeal Swin-Base-384 + RGB (FIXED [0,1] ImageNet norm!, p_real=0.08, 4-fold OOF) ==="
export V5_VIS_BACKBONE=swin_base_patch4_window12_384.ms_in22k_ft_in1k
export V5_VIS_MODE=rgb
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method pixelseal \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/pixelseal \
    --out runs/v12_psswin_fix01_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v12_psswin_fix01_f${F}_pixelseal \
    --steps 5500 --bs 8 --lr 5e-4 --lr-backbone 2.5e-5 --dec 160 --enc-res 512 \
    --p-real 0.08 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((2501 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method pixelseal \
  --folds runs/v12_psswin_fix01_f0 runs/v12_psswin_fix01_f1 runs/v12_psswin_fix01_f2 runs/v12_psswin_fix01_f3 \
  --out runs/fusion/qin7_s3_v12_psswin_fix01_4fold_pixelseal
rm -f runs/v12_psswin_fix01_f*/best*.pt runs/v12_psswin_fix01_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v12_psswin_fix01_f*
echo "=== V12_PS_FIX_DONE ==="
