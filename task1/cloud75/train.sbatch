#!/bin/bash -l
#SBATCH --job-name=t1-native75
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --time=03:00:00
#SBATCH --output=cloud-%j.log
set -eo pipefail
module load PyTorch/2.9.1 torchvision/0.24.1 Pillow/11.3.0
export PYTHONPATH="$PWD/native75-src${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=2
py="$PWD/native-env/bin/python"; src="$PWD/native75-src/cloud75"; rc=0; pids=()
for m in pixelseal trustmark stegastamp; do
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 "$py" -u "$src/native_windows.py" "$PWD" "$m" > "native75-$m.log" 2>&1 & pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if (( rc )); then python3 -c 'from pathlib import Path as P;print([(p.name,p.read_text()[-650:]) for p in P.cwd().glob("native75-*.log") if p.name!="native75-prepare.log"])'; exit 1; fi
"$py" -u "$src/collect.py"
