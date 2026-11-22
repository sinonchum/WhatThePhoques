#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=output/%j.out
#
# Generic GPU job for Task 1. Submit FROM THE TASK DIRECTORY (so output/%j.out resolves there):
#   cd <TASK_DIR> && mkdir -p output && sbatch task1/jureca/job.sbatch task1/some_script.py --arg value
# Override resources on the command line, e.g.  sbatch --gres=gpu:4 --time=06:00:00 task1/jureca/job.sbatch ...
# Compute nodes have no internet: everything must already be downloaded.

set -eo pipefail
TASK_DIR="${SLURM_SUBMIT_DIR:?}"
cd "$TASK_DIR"
source .venv/bin/activate

export PYTHONPATH="$TASK_DIR:$TASK_DIR/task1${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WMLOC_MODELS="$TASK_DIR/models"
# No network on compute nodes: make libraries fail fast instead of hanging on downloads.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$(dirname "$TASK_DIR")/.cache"

echo "[job $SLURM_JOB_ID] host=$(hostname) start=$(date '+%F %T') cmd: python $*"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
srun python "$@"
echo "[job $SLURM_JOB_ID] done $(date '+%F %T')"
