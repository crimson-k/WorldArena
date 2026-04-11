#!/bin/bash
set -euo pipefail

# Usage: run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH]
# METRIC_LIST example: "image_quality,photometric_smoothness,action_following"

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
RAW_METRICS=${4:-}
CONFIG_PATH=${5:-"./config/config.yaml"}
if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH]"
    exit 1
fi

SCRIPT_START_TIME=$(date +%s)
TEMP_CONFIG_PATH=""

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

render_config_for_model() {
    local src_config=$1
    local model_name=$2
    local dst_config
    dst_config=$(mktemp "/tmp/worldarena_${model_name}_config_XXXXXX.yaml")
    sed "s|__MODEL_NAME__|${model_name}|g" "$src_config" > "$dst_config"
    echo "$dst_config"
}

cleanup() {
    if [ -n "$TEMP_CONFIG_PATH" ] && [ -f "$TEMP_CONFIG_PATH" ]; then
        rm -f "$TEMP_CONFIG_PATH"
    fi
}

trap cleanup EXIT

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate WorldArena
export PATH="your absolute path:$PATH"

# Parse metrics
CLEAN_METRICS=$(echo "$RAW_METRICS" | tr ',' ' ' | tr '"' ' ')
METRIC_ARRAY=($CLEAN_METRICS)
echo ">>> Input metrics: $RAW_METRICS"
echo ">>> Formatted for evaluate.py: ${METRIC_ARRAY[*]}"

DATA_DIR="./data/$MODEL_NAME"
CONFIG_DIR="./config"
OUTPUT_DIR="./output"
OUTPUT_DIR_ACTION="./output_action_following"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR_ACTION"

TEMP_CONFIG_PATH=$(render_config_for_model "$CONFIG_PATH" "$MODEL_NAME")
CONFIG_PATH="$TEMP_CONFIG_PATH"
echo ">>> Resolved config for model '$MODEL_NAME': $CONFIG_PATH"

# Split metrics
EVAL_METRICS=()
RUN_ACTION=false
for metric in "${METRIC_ARRAY[@]}"; do
    if [ "$metric" == "action_following" ]; then
        RUN_ACTION=true
    else
        EVAL_METRICS+=("$metric")
    fi
done

# Standard metrics
if [ ${#EVAL_METRICS[@]} -gt 0 ]; then
    echo ">>> Running Preprocessing for standard metrics..."
    STEP_START_TIME=$(date +%s)
    python preprocess_datasets.py --summary_json "$SUMMARY_JSON" --gen_video_dir "$GEN_VIDEO_DIR" --output_base "$DATA_DIR"
    STEP_END_TIME=$(date +%s)
    echo ">>> Preprocessing finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"

    echo ">>> Running video resize..."
    STEP_START_TIME=$(date +%s)
    python ./processing/video_resize.py --config_path "$CONFIG_PATH"
    STEP_END_TIME=$(date +%s)
    echo ">>> Video resize finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"

    echo ">>> Running detection and tracking..."
    STEP_START_TIME=$(date +%s)
    python ./processing/detection_tracking.py --config_path "$CONFIG_PATH" --detect_gt
    STEP_END_TIME=$(date +%s)
    echo ">>> Detection & tracking finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"

    echo ">>> Starting Standard Evaluation: ${EVAL_METRICS[*]}"
    STEP_START_TIME=$(date +%s)
    python evaluate.py --dimension ${EVAL_METRICS[@]} --config "$CONFIG_PATH" --overwrite
    STEP_END_TIME=$(date +%s)
    echo ">>> Standard evaluation finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"
fi

SCRIPT_END_TIME=$(date +%s)
echo ">>> Total elapsed time: $(format_duration $((SCRIPT_END_TIME - SCRIPT_START_TIME)))"
echo ">>> ✅ All evaluations finished"
