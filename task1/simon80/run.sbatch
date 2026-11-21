#!/bin/bash -l
#SBATCH --job-name=t1-simon80
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=cloud-%j.log
set -eo pipefail
module load PyTorch/2.9.1 Pillow/11.3.0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$PWD/native75-src${PYTHONPATH:+:$PYTHONPATH}"
py="$PWD/native-env/bin/python"; src="$PWD/native75-src/simon80"; out="$PWD/simon80-$SLURM_JOB_ID"; mkdir -p "$out"
"$py" -u "$src/refine_oof.py" "$PWD" "$out/shape" > "$out/shape.log" 2>&1 & ranker=$!
rc=0; "$py" -u "$src/adapter.py" "$PWD" "$out/adapter" > "$out/adapter.log" 2>&1 || rc=1
wait "$ranker" || rc=1
"$py" -c 'import pathlib,sys;p=pathlib.Path(sys.argv[1]);print([(f.name,f.read_text()[-1100:]) for f in sorted(p.glob("*.log"))])' "$out"
exit "$rc"
