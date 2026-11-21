#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:55:00
#SBATCH --job-name=v3-ps384
#SBATCH --output=output/v3_ps384_%j.out

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models

jutil env activate -p training2666 >/dev/null
module load Stages/2025
module load PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

# Clean synth3 PixelSeal V3 with higher-resolution 384x384 ConvNeXt grad*act attribution + p_real=0.35 (4-Fold OOF)
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v3_attr.py \
    --method pixelseal \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/pixelseal \
    --out runs/v3_ps384_s3_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v3_ps384_f${F}_pixelseal \
    --steps 6000 --bs 8 --lr 8e-4 --lr-backbone 1e-4 --dec 160 --enc-res 512 --attr-res 384 \
    --p-real 0.35 --n-folds 4 --fold $F --workers 10 --eval-every 500 --seed $((751 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method pixelseal \
  --folds runs/v3_ps384_s3_f0 runs/v3_ps384_s3_f1 runs/v3_ps384_s3_f2 runs/v3_ps384_s3_f3 \
  --out runs/fusion/qin7_s3_v3_ps384_4fold_pixelseal
chmod -R a+rX runs/fusion runs/v3_ps384_s3_f*
