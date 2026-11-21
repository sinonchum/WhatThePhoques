#!/bin/bash
# Read-only status check of the Task 1 environment on JURECA. Changes nothing.
#
# (-l = login shell, so /etc/profile is sourced and jutil is available)
# From the Mac:   ssh jureca 'bash -ls' < task1/jureca/check_env.sh
# With a path:    ssh jureca 'bash -ls -- /p/scratch/training2666/<lead>/<team>/Hackathon_Watermark_Localisation_Task' < task1/jureca/check_env.sh

PROJECT_ID="training2666"
TASK_NAME="Hackathon_Watermark_Localisation_Task"
SCRATCH_BASE="/p/scratch/$PROJECT_ID"

sec()  { printf '\n=== %s ===\n' "$1"; }
have() { if [ -e "$1" ]; then printf '  [ok]      %s\n' "$1"; else printf '  [MISSING] %s\n' "$1"; fi; }

main() {
    local TASK_DIR="${1:-}"

    sec "who / where"
    echo "user=$(id -un)  host=$(hostname)  date=$(date '+%F %T %Z')"

    sec "project activation"
    if command -v jutil >/dev/null 2>&1; then
        jutil env activate -p "$PROJECT_ID" && echo "PROJECT=${PROJECT:-<empty>}  SCRATCH=${SCRATCH:-<empty>}"
    else
        echo "jutil not found (not on JURECA?)"
    fi

    sec "task directories visible to this account"
    # Team folders are ACL-locked, so only our own team's folder should show up here.
    mapfile -t CANDIDATES < <(ls -d "$SCRATCH_BASE"/*/*/"$TASK_NAME" 2>/dev/null)
    if [ "${#CANDIDATES[@]}" -eq 0 ]; then
        echo "none found under $SCRATCH_BASE/*/*/$TASK_NAME"
    else
        printf '  %s\n' "${CANDIDATES[@]}"
    fi
    if [ -z "$TASK_DIR" ] && [ "${#CANDIDATES[@]}" -eq 1 ]; then
        TASK_DIR="${CANDIDATES[0]}"
    fi
    if [ -z "$TASK_DIR" ] || [ ! -d "$TASK_DIR" ]; then
        echo "No unique task dir (got '${TASK_DIR}'). Pass it as the first argument."
        echo "Own scratch dir: $(ls -d "$SCRATCH_BASE/$(id -un)" 2>/dev/null || echo '<none>')"
        return 2
    fi
    local TEAM_DIR; TEAM_DIR="$(dirname "$TASK_DIR")"
    echo "TASK_DIR=$TASK_DIR"
    echo "TEAM_DIR=$TEAM_DIR   (lead=$(basename "$(dirname "$TEAM_DIR")"), owner=$(stat -c %U "$TEAM_DIR" 2>/dev/null))"

    sec "access"
    local d
    for d in "$TEAM_DIR" "$TASK_DIR"; do
        printf '  %s  r=%s w=%s x=%s\n' "$d" "$([ -r "$d" ] && echo y || echo n)" "$([ -w "$d" ] && echo y || echo n)" "$([ -x "$d" ] && echo y || echo n)"
    done

    sec "team folder (top level)"
    ls -la "$TEAM_DIR"

    sec "task folder (top level)"
    ls -la "$TASK_DIR"

    cd "$TASK_DIR" || return 2

    sec "expected files"
    local f
    for f in README.txt starter_kit.zip train.zip test.zip \
             requirements.txt download_models.py submission_template.py \
             wmloc/watermarks.py wmloc/data.py wmloc/utils.py \
             .venv/bin/python output \
             models/videoseal models/pixelseal/checkpoint.pth models/pixelseal_card.yaml \
             models/stegastamp/stega_stamp.onnx models/stegastamp/stega_stamp_logits.onnx \
             release/train/metadata.json release/train/masks_128.npz release/train/images release/train/masks \
             release/test/metadata.json release/test/images; do
        have "$f"
    done

    sec ".venv"
    if [ -x .venv/bin/python ]; then
        .venv/bin/python --version
        echo "owner=$(stat -c %U .venv)  created=$(stat -c %y .venv | cut -d. -f1)"
        ls .venv/lib/*/site-packages 2>/dev/null | grep -i -E '^(torch|torchvision|trustmark|numpy|pillow|PIL|timm|omegaconf|onnx|onnxruntime_gpu|onnxruntime|einops|av|decord|pycocotools|safetensors)(-[0-9].*)?$' | sort | tr '\n' ' '
        echo
        echo "trustmark weights: $(find .venv/lib/*/site-packages/trustmark -maxdepth 2 \( -name '*.ckpt' -o -name '*.yaml' \) 2>/dev/null | wc -l) files under site-packages/trustmark"
    else
        echo "no usable .venv"
    fi

    sec "data (release/)"
    if [ -d release ]; then
        for d in release/train/images release/train/masks release/test/images; do
            [ -d "$d" ] && printf '  %-28s png=%s\n' "$d" "$(find "$d" -maxdepth 1 -name '*.png' | wc -l)"
        done
        echo "  expect: train images 510, train masks 510, test images 1020"
    else
        echo "release/ not present -> data not unzipped"
    fi

    sec "models/ (files > 1 MB)"
    if [ -d models ]; then
        find models -maxdepth 3 -type f -size +1M -printf '  %12s bytes  %p\n' 2>/dev/null | sort -k3 | head -20
        [ -d models/videoseal/.git ] && echo "  videoseal commit: $(git -C models/videoseal log -1 --format='%h %ad' --date=short 2>/dev/null)  (expect b1bf4c0)"
    else
        echo "models/ not present -> download_models.py not run"
    fi

    sec "caches / tools"
    echo "uv=$(command -v uv || ls "$HOME/.local/bin/uv" 2>/dev/null || echo '<not found>')"
    echo "UV_CACHE_DIR=${UV_CACHE_DIR:-<unset>}  HF_HOME=${HF_HOME:-<unset>}"
    [ -d "$TEAM_DIR/.uv" ] && echo "$TEAM_DIR/.uv: $(ls "$TEAM_DIR/.uv" | tr '\n' ' ')"
    [ -d "$TEAM_DIR/.cache" ] && echo "$TEAM_DIR/.cache: $(ls "$TEAM_DIR/.cache" | tr '\n' ' ')"

    sec "setup scripts / shell config"
    if [ -f "$HOME/hackathon-scripts/teammate.sh" ]; then
        grep -n -E '^(PROJECT_NAME|TEAM_LEAD|TEAM_FOLDER)=' "$HOME/hackathon-scripts/teammate.sh"
    else
        echo "~/hackathon-scripts/teammate.sh not found"
    fi
    grep -n "TEAM SHARED UV/HF CONFIG" "$HOME/.bashrc" 2>/dev/null || echo "no shared-cache block in ~/.bashrc (setup script not run from this account?)"
    if [ -n "${HACKATHON_API_KEY:-}" ]; then echo "HACKATHON_API_KEY: set in this shell"; else echo "HACKATHON_API_KEY: NOT set in this (non-interactive login) shell"; fi
    grep -l "HACKATHON_API_KEY" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.profile" 2>/dev/null | sed 's/^/  mentioned in: /'

    sec "slurm"
    scontrol show reservation cispahack 2>&1 | head -12
    echo "--- my jobs ---"
    squeue -u "$(id -un)" 2>&1 | head -20
    echo "--- filesystem ---"
    df -h "$TASK_DIR" 2>/dev/null | tail -1

    echo
    echo "check_env done."
}

# The script arrives on stdin; keep child processes from reading (and eating) the rest of it.
main "$@" </dev/null
