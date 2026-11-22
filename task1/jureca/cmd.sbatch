#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=output/%j.out
#
# Run an arbitrary shell command line on a GPU node, with the Task 1 environment set up.
# Submit from the work dir:  sbatch task1/jureca/cmd.sbatch 'python task1/a.py --x 1 && python task1/b.py'

set -eo pipefail
TASK_DIR="${SLURM_SUBMIT_DIR:?}"
cd "$TASK_DIR"
source .venv/bin/activate
export PYTHONPATH="$TASK_DIR:$TASK_DIR/task1" PYTHONUNBUFFERED=1 WMLOC_MODELS="$TASK_DIR/models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$(dirname "$TASK_DIR")/.cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "[job $SLURM_JOB_ID] host=$(hostname) start=$(date '+%F %T') cmd: $*"
eval "$@"
echo "[job $SLURM_JOB_ID] done $(date '+%F %T')"
