#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export XDG_CACHE_HOME="$PWD/.cache_codex_local"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

.venv_codex_mps/bin/python -u extract_stega_cues.py \
  --data data_codex_final --models-dir models_codex_local \
  --split output/fusion/codex_occlusion_stegastamp/split.json \
  --out output/stega_cues --stage final --device mps
.venv_codex_mps/bin/python -u stega_lowpass_cues.py \
  --data data_codex_final --models-dir models_codex_local \
  --base output/stega_cues --out output/stega_cues_lowpass \
  --stage final --device mps
.venv_codex_mps/bin/python -u train_stega_cues.py \
  --data data_codex_final --features output/stega_cues_lowpass \
  --out output/fusion/codex_stegacutmix_stegastamp --stage final --device mps
