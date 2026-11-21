#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=03:00:00
#SBATCH --output=output/%j.out
#
# Keeps one 4-GPU node for a series of short experiments, so each one does not have to queue again.
# Experiments are started as steps inside this allocation:
#   srun --jobid=<id> --overlap --ntasks=1 --cpus-per-task=N --gres=gpu:0 <script>   (script sets CUDA_VISIBLE_DEVICES)
# Cancel it as soon as it is no longer needed: scancel <id>
echo "[job $SLURM_JOB_ID] holding $(hostname) from $(date '+%F %T')"
sleep 10700
