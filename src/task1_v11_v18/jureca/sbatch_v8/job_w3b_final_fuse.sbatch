#!/bin/bash -l
#SBATCH --job-name=w3b-fuse
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --output=/p/home/jusers/qin7/jureca/sbatch_v8/w3b_fuse_%j.out
#SBATCH --error=/p/home/jusers/qin7/jureca/sbatch_v8/w3b_fuse_%j.out

set -euo pipefail
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WMLOC_MODELS=/p/scratch/training2666/qin7/task1/models

jutil env activate -p training2666 >/dev/null
module load Stages/2025
module load PyTorch/2.5.1
cd /p/scratch/training2666/qin7/task1

find /p/scratch/training2666/qin7 -maxdepth 3 -name "core.*" -type f -delete 2>/dev/null || true
find /p/scratch/training2666/qin7/task1/runs -mindepth 2 -maxdepth 3 -name "best*.pt" -mmin +5 ! -path "*/submissions/*" -delete 2>/dev/null || true

echo "=== 1. RE-RUN NN-RECT-REFINE WITH FULL V8+V9+V10 3-FAMILY ENSEMBLE ==="
DATA="/p/scratch/training2666/ding7/claude_task1/release"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python nn_rect_refine.py --data "$DATA" --method pixelseal --split train --rect-npz runs/rect/pixelseal_train_nnref.npz --out runs/rect/pixelseal_train_nnref.npz &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python nn_rect_refine.py --data "$DATA" --method pixelseal --split test  --rect-npz runs/rect/pixelseal_test_nnref.npz  --out runs/rect/pixelseal_test_nnref.npz &
CUDA_VISIBLE_DEVICES=2 .venv/bin/python nn_rect_refine.py --data "$DATA" --method trustmark --split train --rect-npz runs/rect/trustmark_train_nnref.npz --out runs/rect/trustmark_train_nnref.npz &
CUDA_VISIBLE_DEVICES=3 .venv/bin/python nn_rect_refine.py --data "$DATA" --method trustmark --split test  --rect-npz runs/rect/trustmark_test_nnref.npz  --out runs/rect/trustmark_test_nnref.npz &
wait
chmod -R a+rX runs/rect

echo "=== 2. FINAL OVERNIGHT PURE-DEV CV FUSION + METASTACK + QUEUE AUTO-SUBMIT ==="
CUDA_VISIBLE_DEVICES=0 .venv/bin/python build_sub_v3_clean.py | tee /tmp/w3b_fuse.log
.venv/bin/python - << 'PY'
import json, pathlib, re, shutil
txt = pathlib.Path("/tmp/w3b_fuse.log").read_text()
m_dev = re.search(r"FINAL 100% CLEAN 3-METHOD DEV\s+OOF\s*:\s*([0-9.]+)", txt)
m_hld = re.search(r"FINAL 100% CLEAN 3-METHOD HOLDOUT\s*:\s*([0-9.]+)", txt)
assert m_dev and m_hld, "Could not parse final scores!"
dev_s, hld_s = float(m_dev.group(1)), float(m_hld.group(1))
qdir = pathlib.Path.home() / "sbatch_v8/auto_sub_queue"
qdir.mkdir(parents=True, exist_ok=True)
dst = qdir / "sub_v5_wave3_v10.pt"
shutil.copy2("runs/submissions/sub_v3_clean_nnref.pt", dst)
(qdir / "sub_v5_wave3_v10.json").write_text(json.dumps({"file": dst.name, "dev_oof": dev_s, "holdout": hld_s}, indent=2))
print(f"QUEUED FOR AUTO-SUBMIT: {dst} (dev_oof={dev_s:.4f}, holdout={hld_s:.4f})")
PY
echo "=== W3B DONE ==="
