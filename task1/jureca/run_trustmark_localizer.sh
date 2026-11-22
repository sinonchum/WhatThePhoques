#!/bin/bash
#SBATCH --job-name=tm-localizer
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/trustmark_localizer-%j.log

set -euo pipefail
export XDG_CACHE_HOME="$PWD/.cache"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
srun .venv/bin/python -u trustmark_localizer.py \
  --data release --trustmark-repo models/trustmark_new \
  --out output/fusion/codex_tm_localizer_trustmark
