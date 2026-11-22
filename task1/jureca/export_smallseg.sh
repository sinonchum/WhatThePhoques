#!/bin/bash
# Submit from /p/scratch/training2666/ding7/message_heatmaps.
#SBATCH --job-name=export-smallseg
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=output/export_smallseg-%j.log

set -euo pipefail
export XDG_CACHE_HOME="$PWD/.cache"
export HF_HOME="$PWD/.cache/huggingface"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
srun .venv/bin/python -u export_smallseg.py \
  --data release --smallseg output/small_segmenter --models-dir models \
  --test-zip /p/scratch/training2666/qin7/task1/Hackathon_Watermark_Localisation_Task/test.zip \
  --out output/fusion
