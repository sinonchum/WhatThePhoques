#!/bin/bash
# Submit from the existing message_heatmaps task directory.
#SBATCH --job-name=small-segmenter
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=output/small_segmenter-%j.log

set -euo pipefail
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
srun .venv/bin/python -u train_small_segmenter.py \
  --data release --heatmaps output/message_heatmaps --out output/small_segmenter
