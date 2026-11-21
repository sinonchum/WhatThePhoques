#!/bin/bash
#SBATCH --job-name=crop-search-dev
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/crop_search-%j.log

set -euo pipefail
export XDG_CACHE_HOME="$PWD/.cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
srun .venv/bin/python -u crop_search.py \
  --data release --models-dir models --split-root output \
  --out output/crop_search_pilot --methods pixelseal trustmark stegastamp \
  --pilot-images 6 --batch-size 32
