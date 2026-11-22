#!/bin/bash -l
#SBATCH --job-name=t1-region-cv
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=cloud-%j.log
set -eo pipefail
module load PyTorch/2.9.1 Pillow/11.3.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$PWD/native-env/bin/python" -u "$PWD/native75-src/simon80/region.py" "$PWD" "$PWD/simon80-region-$SLURM_JOB_ID" "$@"
