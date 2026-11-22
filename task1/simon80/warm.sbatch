#!/bin/bash -l
#SBATCH --job-name=t1-simon80-warm
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:10:00
#SBATCH --output=cloud-%j.log
set -eo pipefail
module load PyTorch/2.9.1 torchvision/0.24.1 Pillow/11.3.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$PWD/native75-src${PYTHONPATH:+:$PYTHONPATH}"
"$PWD/native-env/bin/python" -u "$PWD/native75-src/simon80/warm_trustmark.py" "$PWD" "$PWD/simon80-warm-$SLURM_JOB_ID"
