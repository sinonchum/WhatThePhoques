#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=04:00:00
#SBATCH --output=output/%j.out
#
# One node, three methods in parallel (one A100 each): build the synthetic pool, then train the segmentation net.
# Submit from the work dir:  sbatch task1/jureca/synth_all.sbatch <tag> [extra train_seg.py args]
#   e.g. sbatch task1/jureca/synth_all.sbatch v1 --steps 8000
# Per-method logs: output/<jobid>_<method>.log

set -eo pipefail
TAG="${1:?tag missing}"; shift
TASK_DIR="${SLURM_SUBMIT_DIR:?}"
cd "$TASK_DIR"
source .venv/bin/activate
export PYTHONPATH="$TASK_DIR:$TASK_DIR/task1" PYTHONUNBUFFERED=1 WMLOC_MODELS="$TASK_DIR/models"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$(dirname "$TASK_DIR")/.cache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "[job $SLURM_JOB_ID] host=$(hostname) start=$(date '+%F %T') tag=$TAG extra: $*"
if [ ! -f runs/covers/covers_meta.json ]; then
    CUDA_VISIBLE_DEVICES=0 python task1/prep_covers.py --out runs/covers
fi

gpu=0
for m in pixelseal trustmark stegastamp; do
    (
        export CUDA_VISIBLE_DEVICES=$gpu
        if [ ! -f "runs/synth/$m/pool.json" ]; then
            python task1/gen_synth.py --method "$m" --out "runs/synth/$m" --k-full 4 --k-rect 16
        fi
        python task1/train_seg.py --method "$m" --pool "runs/synth/$m" --out "runs/seg/${m}_${TAG}" "$@"
    ) > "output/${SLURM_JOB_ID}_$m.log" 2>&1 &
    gpu=$((gpu + 1))
done
wait
echo "[job $SLURM_JOB_ID] done $(date '+%F %T')"
tail -n 25 output/${SLURM_JOB_ID}_*.log
