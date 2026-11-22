#!/bin/bash
# Run from the existing personal message_heatmaps directory.
#SBATCH --job-name=stega-occlusion
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/stegastamp_occlusion-%j.log

set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
srun .venv/bin/python -u stegastamp_occlusion.py \
  --data release --models-dir models \
  --out output/fusion/codex_occlusion_stegastamp --batch-size 32
