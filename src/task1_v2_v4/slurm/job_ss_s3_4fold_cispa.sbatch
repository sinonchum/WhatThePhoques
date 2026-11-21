#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:55:00
#SBATCH --job-name=ss-s3-4f
#SBATCH --output=output/ss_s3_4f_%j.out

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models

jutil env activate -p training2666 >/dev/null
module load Stages/2025
module load PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

POOL=/p/scratch/training2666/ding7/claude_task1/runs/synth3/stegastamp
echo "Waiting for clean synth3/stegastamp/pool.json..."
while [ ! -f "$POOL/pool.json" ]; do
  sleep 10
done
echo "Clean synth3/stegastamp ready!"

# Phase A: 4-Fold OOF StegaExtSegNetV2 (5-crop STN + grad*act + HighPass + EMA) on CLEAN synth3/stegastamp
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_stega_s3_kfold.py \
    --arch v2 --pool "$POOL" \
    --out runs/s3_ss_v2_f${F} \
    --fusion-dir runs/fusion/qin7_s3_ss_v2_f${F}_stegastamp \
    --steps 4500 --bs 8 --lr 1e-3 --p-real 0.25 --n-folds 4 --fold $F --workers 10 --eval-every 500 --seed $((801 + F)) &
done
wait

.venv/bin/python assemble_kfold.py \
  --method stegastamp \
  --folds runs/s3_ss_v2_f0 runs/s3_ss_v2_f1 runs/s3_ss_v2_f2 runs/s3_ss_v2_f3 \
  --out runs/fusion/qin7_s3_ss_v2_4fold_stegastamp

# Phase B: 4-Fold OOF StegaExtSegNet V1 (classic grad*act + HighPass + EMA) on CLEAN synth3/stegastamp for diversity!
for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_stega_s3_kfold.py \
    --arch v1 --pool "$POOL" \
    --out runs/s3_ss_v1_f${F} \
    --fusion-dir runs/fusion/qin7_s3_ss_v1_f${F}_stegastamp \
    --steps 4500 --bs 8 --lr 1e-3 --p-real 0.25 --n-folds 4 --fold $F --workers 10 --eval-every 500 --seed $((901 + F)) &
done
wait

.venv/bin/python assemble_kfold.py \
  --method stegastamp \
  --folds runs/s3_ss_v1_f0 runs/s3_ss_v1_f1 runs/s3_ss_v1_f2 runs/s3_ss_v1_f3 \
  --out runs/fusion/qin7_s3_ss_v1_4fold_stegastamp

chmod -R a+rX runs/fusion runs/s3_ss_*
