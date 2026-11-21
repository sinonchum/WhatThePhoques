#!/bin/bash
#SBATCH --job-name=stega-cues-dev
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/stega_cues_dev-%j.log

set -euo pipefail
export XDG_CACHE_HOME="$PWD/.cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
srun .venv/bin/python -u extract_stega_cues.py \
  --data release --models-dir models \
  --split output/fusion/codex_occlusion_stegastamp/split.json \
  --out output/stega_cues --stage dev
srun .venv/bin/python -u train_stega_cues.py \
  --data release --features output/stega_cues \
  --out output/fusion/codex_stegacues_stegastamp --stage dev
