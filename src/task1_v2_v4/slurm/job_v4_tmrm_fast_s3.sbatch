#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:55:00
#SBATCH --job-name=v4-tmf-s3
#SBATCH --output=output/v4_tmf_s3_%j.out

export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models

jutil env activate -p training2666 >/dev/null
module load Stages/2025
module load PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

for F in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$F .venv/bin/python train_v4_tm_rm_fast.py \
    --method trustmark \
    --pool /p/scratch/training2666/ding7/claude_task1/runs/synth3/trustmark \
    --out runs/v4_tmf_s3_f${F} \
    --fusion-dir runs/fusion/qin7_s3_v4_tmf_f${F}_trustmark \
    --steps 6500 --bs 8 --lr 1e-3 --lr-backbone 1e-4 --dec 160 --enc-res 512 --attr-res 256 \
    --p-real 0.25 --n-folds 4 --fold $F --workers 10 --eval-every 500 --seed $((671 + F)) &
done
wait
.venv/bin/python assemble_kfold.py --method trustmark \
  --folds runs/v4_tmf_s3_f0 runs/v4_tmf_s3_f1 runs/v4_tmf_s3_f2 runs/v4_tmf_s3_f3 \
  --out runs/fusion/qin7_s3_v4_tmf_4fold_trustmark
chmod -R a+rX runs/fusion runs/v4_tmf_s3_f*
.venv/bin/python fuse_clean_cv.py
