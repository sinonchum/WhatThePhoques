#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
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

# Stage 1 (~14 min): Clean synth3 StegaStamp V2 4-Fold OOF
for F in 0 1 2 3; do
  STEGA_ARCH=v2 CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_stega_s3_kfold.py \
    --method stegastamp \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/stegastamp \
    --out runs/s3_ss_v2_f${F} \
    --fusion-dir runs/fusion/qin7_s3_ss_v2_f${F}_stegastamp \
    --steps 4500 --bs 8 --lr 1e-3 --lr-backbone 3e-4 --dec 128 \
    --p-real 0.25 --n-folds 4 --fold $F --workers 10 --eval-every 500 --seed $((801 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method stegastamp \
  --folds runs/s3_ss_v2_f0 runs/s3_ss_v2_f1 runs/s3_ss_v2_f2 runs/s3_ss_v2_f3 \
  --out runs/fusion/qin7_s3_ss_v2_4fold_stegastamp
chmod -R a+rX runs/fusion runs/s3_ss_v2_f*

# Stage 2 (~12 min): Clean synth3 StegaStamp V1 4-Fold OOF (architectural diversity)
for F in 0 1 2 3; do
  STEGA_ARCH=v1 CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_stega_s3_kfold.py \
    --method stegastamp \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/stegastamp \
    --out runs/s3_ss_v1_f${F} \
    --fusion-dir runs/fusion/qin7_s3_ss_v1_f${F}_stegastamp \
    --steps 4500 --bs 8 --lr 1e-3 --lr-backbone 1e-3 --dec 128 \
    --p-real 0.25 --n-folds 4 --fold $F --workers 10 --eval-every 500 --seed $((901 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method stegastamp \
  --folds runs/s3_ss_v1_f0 runs/s3_ss_v1_f1 runs/s3_ss_v1_f2 runs/s3_ss_v1_f3 \
  --out runs/fusion/qin7_s3_ss_v1_4fold_stegastamp
chmod -R a+rX runs/fusion runs/s3_ss_v1_f*

# Stage 3 (~12 min): Clean synth3 StegaStamp V2 pure-synth (p_real=0.0, 100% zero-real diversity!)
for F in 0 1; do
  STEGA_ARCH=v2 CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_stega_s3_kfold.py \
    --method stegastamp \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/stegastamp \
    --out runs/s3_ss_v2_synth_s${F} \
    --fusion-dir runs/fusion/qin7_s3_ss_v2_synth_s${F}_stegastamp \
    --steps 4500 --bs 8 --lr 1e-3 --lr-backbone 3e-4 --dec 128 \
    --p-real 0.0 --workers 10 --eval-every 500 --seed $((951 + F)) &
done
wait
chmod -R a+rX runs/fusion runs/s3_ss_v2_synth_*

.venv/bin/python fuse_clean_cv.py
