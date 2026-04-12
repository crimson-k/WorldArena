#!/bin/bash
set -euo pipefail
# Usage: run_evaluation_JEPA.sh <MODEL_NAME> <GEN_VIDEO_DIR> <GT_VIDEO_DIR>

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
GT_VIDEO_DIR=${3:-}
SCRIPT_START_TIME=$(date +%s)

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT="$ROOT_DIR/data/${MODEL_NAME}/output_JEDi"

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$GT_VIDEO_DIR" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <GT_VIDEO_DIR>"
    exit 1
fi

cd ./video_quality/JEDi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate WorldArena_JEPA
export PATH="your absolute path/WorldArena_JEPA/bin:$PATH"

echo ">>> JEPA output root: $OUTPUT_ROOT"

echo ">>> Running JEPA evaluation..."
STEP_START_TIME=$(date +%s)
python batch.py \
	--real_dir "$GT_VIDEO_DIR" \
	--gen_dir "$GEN_VIDEO_DIR" \
    --model_dir "$ROOT_DIR/JEDi/pretrained_models" \
    --config_path "$ROOT_DIR/JEDi/configs/vith16_ssv2_16x2x3.yaml" \
    --output_root "$OUTPUT_ROOT" 
STEP_END_TIME=$(date +%s)

SCRIPT_END_TIME=$(date +%s)
echo ">>> JEPA evaluation finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"
echo ">>> Total elapsed time: $(format_duration $((SCRIPT_END_TIME - SCRIPT_START_TIME)))"
echo ">>> ✅ JEPA evaluation finished"
