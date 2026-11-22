#!/bin/bash
#SBATCH --job-name=ps-crop-full
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=output/export_pixelseal_crops-%j.log

set -euo pipefail
export XDG_CACHE_HOME="$PWD/.cache"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
srun .venv/bin/python -u export_crop_search.py --method pixelseal --data release \
  --models-dir models --batch-size 32 --gpus 4 --out output/fusion/codex_cropsearch_pixelseal
