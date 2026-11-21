#!/bin/bash
#SBATCH --job-name=stega-template
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=output/stega_template_probe-%j.log

set -euo pipefail
export XDG_CACHE_HOME="$PWD/.cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
srun .venv/bin/python -u stega_template_probe.py \
  --data release --models-dir models \
  --split output/fusion/codex_occlusion_stegastamp/split.json \
  --out output/stega_template_probe --pilot-images 24
