#!/bin/bash
# Fill in the missing pieces of the Task 1 environment. Run on a JURECA LOGIN node (needs internet).
# Idempotent: never rebuilds an existing .venv, never overwrites existing files, deletes nothing.
#
# From the Mac:  ssh jureca 'bash -ls -- <TASK_DIR> <step>' < task1/jureca/setup_env.sh
#   steps: kit     unzip starter_kit.zip (only files that do not exist yet)
#          venv    create .venv with Python 3.11 and install requirements.txt (only if .venv is absent)
#          models  python download_models.py  (videoseal clone, PixelSeal checkpoint, StegaStamp ONNX)
#          data    unzip train.zip / test.zip into release/
#          all     kit, venv, models, data in that order

PROJECT_ID="training2666"

main() {
    local TASK_DIR="${1:?usage: setup_env.sh <TASK_DIR> <kit|venv|models|data|all>}"
    local STEP="${2:?step missing: kit|venv|models|data|all}"
    local TEAM_DIR; TEAM_DIR="$(dirname "$TASK_DIR")"

    command -v jutil >/dev/null 2>&1 && jutil env activate -p "$PROJECT_ID"
    set -eo pipefail

    # Same shared cache locations the official setup scripts write to ~/.bashrc.
    # UV_BIN_DIR (optional): directory holding an existing uv binary, if uv is not on PATH.
    export PATH="${UV_BIN_DIR:+$UV_BIN_DIR:}$HOME/.local/bin:$PATH"
    export UV_CACHE_DIR="$TEAM_DIR/.uv/cache" UV_TOOL_DIR="$TEAM_DIR/.uv/tools" UV_PYTHON_INSTALL_DIR="$TEAM_DIR/.uv/python"
    export HF_HOME="$TEAM_DIR/.cache" HUGGINGFACE_HUB_CACHE="$TEAM_DIR/.cache/hub"

    cd "$TASK_DIR"
    echo "[setup] host=$(hostname) user=$(id -un) dir=$PWD step=$STEP"

    step_kit() {
        if [ -f wmloc/watermarks.py ] && [ -f requirements.txt ] && [ -f download_models.py ]; then
            echo "[kit] already unpacked - skipping"; return 0
        fi
        [ -f starter_kit.zip ] || { echo "[kit] starter_kit.zip missing"; return 1; }
        unzip -n starter_kit.zip        # -n: never overwrite existing files
        echo "[kit] done"
    }

    step_venv() {
        if [ -d .venv ]; then
            echo "[venv] .venv already exists (owner $(stat -c %U .venv)) - not touching it"; return 0
        fi
        [ -f requirements.txt ] || { echo "[venv] requirements.txt missing - run step 'kit' first"; return 1; }
        command -v uv >/dev/null 2>&1 || { echo "[venv] uv not found on PATH - install it first (see README of SprintML/hackathon)"; return 1; }
        uv venv -p 3.11 .venv
        uv pip install --python .venv/bin/python -r requirements.txt
        .venv/bin/python -c "import torch, torchvision; print('[venv] torch', torch.__version__, 'torchvision', torchvision.__version__)"
        echo "[venv] done"
    }

    step_models() {
        [ -x .venv/bin/python ] || { echo "[models] .venv missing - run step 'venv' first"; return 1; }
        .venv/bin/python download_models.py
        echo "[models] done"
    }

    step_data() {
        [ -x .venv/bin/python ] || { echo "[data] .venv missing - run step 'venv' first"; return 1; }
        local s
        for s in train test; do
            if [ -f "release/$s/metadata.json" ]; then
                echo "[data] release/$s already unpacked - skipping"
            else
                [ -f "$s.zip" ] || { echo "[data] $s.zip missing"; return 1; }
                .venv/bin/python -c "from wmloc.data import unzip; unzip('$s.zip', 'release')"
                echo "[data] unpacked $s.zip"
            fi
        done
        mkdir -p output
        echo "[data] train images: $(find release/train/images -name '*.png' | wc -l), test images: $(find release/test/images -name '*.png' | wc -l)"
    }

    case "$STEP" in
        kit)    step_kit ;;
        venv)   step_venv ;;
        models) step_models ;;
        data)   step_data ;;
        all)    step_kit && step_venv && step_models && step_data ;;
        *)      echo "unknown step: $STEP"; return 1 ;;
    esac
}

# The script arrives on stdin; keep child processes from reading (and eating) the rest of it.
main "$@" </dev/null
