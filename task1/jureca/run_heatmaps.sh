#!/bin/bash
# Submit from the task directory after: mkdir -p output
# sbatch jureca/run_heatmaps.sh
#SBATCH --job-name=message-heatmaps
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/message_heatmaps-%j.log

set -euo pipefail
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
srun .venv/bin/python -u message_heatmaps.py \
  --data release --models-dir models --out output/message_heatmaps
