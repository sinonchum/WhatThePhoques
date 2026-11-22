#!/bin/bash -l
#SBATCH --job-name=v12-tm-fix
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:55:00
#SBATCH --output=/p/home/jusers/qin7/jureca/sbatch_v8/v12_tm_fix_%j.out
#SBATCH --error=/p/home/jusers/qin7/jureca/sbatch_v8/v12_tm_fix_%j.out

set -euo pipefail
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models
jutil env activate -p training2666 >/dev/null
module load Stages/2025 PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

echo "=== V12-TM STAGE 1: TrustMark Swin-Base-384 + 6ch Adobe-RM (FIXED [0,1] ImageNet norm!, p_real=0.10, 4-fold OOF) ==="
export V5_VIS_BACKBONE=swin_base_patch4_window12_384.ms_in22k_ft_in1k
export V5_VIS_MODE=rm
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method trustmark \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/trustmark \
    --out runs/v12_tmswin_fix01_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v12_tmswin_fix01_f${F}_trustmark \
    --steps 5500 --bs 8 --lr 5e-4 --lr-backbone 2.5e-5 --dec 160 --enc-res 512 \
    --p-real 0.10 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((2601 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method trustmark \
  --folds runs/v12_tmswin_fix01_f0 runs/v12_tmswin_fix01_f1 runs/v12_tmswin_fix01_f2 runs/v12_tmswin_fix01_f3 \
  --out runs/fusion/qin7_s3_v12_tmswin_fix01_4fold_trustmark
rm -f runs/v12_tmswin_fix01_f*/best*.pt runs/v12_tmswin_fix01_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v12_tmswin_fix01_f*

echo "=== V12-TM STAGE 2: StegaStamp Swin-Base-384 + RGB (FIXED [0,1] ImageNet norm!, p_real=0.15, 4-fold OOF) ==="
export V5_VIS_BACKBONE=swin_base_patch4_window12_384.ms_in22k_ft_in1k
export V5_VIS_MODE=rgb
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v5_dual.py \
    --method stegastamp \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/stegastamp \
    --out runs/v12_ssswin_fix01_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v12_ssswin_fix01_f${F}_stegastamp \
    --steps 6000 --bs 8 --lr 5e-4 --lr-backbone 3e-5 --dec 160 --enc-res 512 \
    --p-real 0.15 --n-folds 4 --fold $F --workers 8 --amp bf16 --eval-every 500 --seed $((2701 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method stegastamp \
  --folds runs/v12_ssswin_fix01_f0 runs/v12_ssswin_fix01_f1 runs/v12_ssswin_fix01_f2 runs/v12_ssswin_fix01_f3 \
  --out runs/fusion/qin7_s3_v12_ssswin_fix01_4fold_stegastamp
rm -f runs/v12_ssswin_fix01_f*/best*.pt runs/v12_ssswin_fix01_f*/last*.pt 2>/dev/null || true
chmod -R a+rX runs/fusion runs/v12_ssswin_fix01_f*
echo "=== V12_TM_SS_DONE ==="
