#!/bin/bash
set -euo pipefail

# Usage:
#   run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH] [SHARD_INDEX] [NUM_SHARDS]
#   run_evaluation.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [SHARD_INDEX] [NUM_SHARDS]
# METRIC_LIST example: "image_quality,photometric_smoothness,action_following"

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
RAW_METRICS=${4:-}
CONFIG_PATH="./config/config.yaml"
SHARD_INDEX=""
NUM_SHARDS=""
if [ $# -ge 5 ]; then
    if [[ "${5}" =~ ^[0-9]+$ ]]; then
        SHARD_INDEX="${5}"
        NUM_SHARDS=${6:-}
    else
        CONFIG_PATH="${5}"
        SHARD_INDEX=${6:-}
        NUM_SHARDS=${7:-}
    fi
fi
if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH] [SHARD_INDEX] [NUM_SHARDS]"
    echo "   or: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [SHARD_INDEX] [NUM_SHARDS]"
    exit 1
fi

SCRIPT_START_TIME=$(date +%s)
TEMP_CONFIG_PATH=""
EFFECTIVE_MODEL_NAME="$MODEL_NAME"
SHARD_TAG=""
PROJECT_ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

is_non_negative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

format_shard_name() {
    local shard_index=$1
    local num_shards=$2
    local max_index=$((num_shards - 1))
    local width=${#max_index}
    if [ "$width" -lt 2 ]; then
        width=2
    fi
    printf "shard_%0${width}d" "$shard_index"
}

resolve_sharded_inputs() {
    if [ -n "$SHARD_INDEX" ] || [ -n "$NUM_SHARDS" ]; then
        if [ -z "$SHARD_INDEX" ] || [ -z "$NUM_SHARDS" ]; then
            echo "[ERROR] SHARD_INDEX and NUM_SHARDS must be provided together."
            exit 1
        fi
        if ! is_non_negative_int "$SHARD_INDEX" || ! is_non_negative_int "$NUM_SHARDS"; then
            echo "[ERROR] SHARD_INDEX and NUM_SHARDS must be non-negative integers."
            exit 1
        fi
        if [ "$NUM_SHARDS" -le 1 ]; then
            echo "[ERROR] NUM_SHARDS must be greater than 1 when shard mode is enabled."
            exit 1
        fi
        if [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
            echo "[ERROR] SHARD_INDEX ($SHARD_INDEX) must be smaller than NUM_SHARDS ($NUM_SHARDS)."
            exit 1
        fi

        local summary_parent
        summary_parent=$(cd -- "$(dirname "$SUMMARY_JSON")" && pwd)
        local summary_file
        summary_file=$(basename "$SUMMARY_JSON")
        local expected_shard_name
        expected_shard_name=$(format_shard_name "$SHARD_INDEX" "$NUM_SHARDS")

        local shard_dir
        if [[ "$(basename "$summary_parent")" =~ ^shard_[0-9]+$ ]] && [ "$summary_file" = "summary.json" ]; then
            shard_dir="$summary_parent"
            if [ "$(basename "$shard_dir")" != "$expected_shard_name" ]; then
                echo "[ERROR] Summary already points to $(basename "$shard_dir"), but shard args request $expected_shard_name."
                exit 1
            fi
        else
            shard_dir="${summary_parent}/shards/${expected_shard_name}"
        fi

        local shard_summary="${shard_dir}/summary.json"
        if [ ! -f "$shard_summary" ]; then
            echo "[ERROR] Shard summary not found: $shard_summary"
            exit 1
        fi

        local shard_gen_dir="${shard_dir}/$(basename "$GEN_VIDEO_DIR")"
        if [ ! -d "$shard_gen_dir" ]; then
            local fallback_gen="${shard_dir}/${MODEL_NAME}_test"
            if [ -d "$fallback_gen" ]; then
                shard_gen_dir="$fallback_gen"
            else
                echo "[ERROR] Shard generated video dir not found under: $shard_dir"
                echo "        Tried: $shard_gen_dir"
                echo "        Tried: $fallback_gen"
                exit 1
            fi
        fi

        SUMMARY_JSON="$shard_summary"
        GEN_VIDEO_DIR="$shard_gen_dir"
    fi

    local summary_parent
    summary_parent=$(cd -- "$(dirname "$SUMMARY_JSON")" && pwd)
    if [[ "$(basename "$summary_parent")" =~ ^shard_[0-9]+$ ]] && [ "$(basename "$SUMMARY_JSON")" = "summary.json" ]; then
        SHARD_TAG="$(basename "$summary_parent")"
        EFFECTIVE_MODEL_NAME="${MODEL_NAME}_${SHARD_TAG}"
    fi
}

normalize_config_legacy_root() {
    local config_file=$1
    local legacy_root="/data/liuwenhao/WorldArena"
    if [ ! -d "$legacy_root" ] && grep -q "$legacy_root" "$config_file"; then
        sed -i "s|$legacy_root|$PROJECT_ROOT|g" "$config_file"
        echo ">>> Rewrote legacy config root: $legacy_root -> $PROJECT_ROOT"
    fi
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

resolve_sharded_inputs

if [ ! -f "$SUMMARY_JSON" ]; then
    echo "[ERROR] Summary JSON not found: $SUMMARY_JSON"
    exit 1
fi
if [ ! -d "$GEN_VIDEO_DIR" ]; then
    echo "[ERROR] Generated video dir not found: $GEN_VIDEO_DIR"
    exit 1
fi
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[ERROR] Config path not found: $CONFIG_PATH"
    exit 1
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate WorldArena
export PATH="your absolute path:$PATH"

# Parse metrics
CLEAN_METRICS=$(echo "$RAW_METRICS" | tr ',' ' ' | tr '"' ' ')
METRIC_ARRAY=($CLEAN_METRICS)
echo ">>> Input metrics: $RAW_METRICS"
echo ">>> Formatted for evaluate.py: ${METRIC_ARRAY[*]}"

DATA_DIR="./data/$EFFECTIVE_MODEL_NAME"
CONFIG_DIR="./config"
OUTPUT_DIR="./output"
OUTPUT_DIR_ACTION="./output_action_following"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR_ACTION"

TEMP_CONFIG_PATH=$(render_config_for_model "$CONFIG_PATH" "$EFFECTIVE_MODEL_NAME")
CONFIG_PATH="$TEMP_CONFIG_PATH"
normalize_config_legacy_root "$CONFIG_PATH"
echo ">>> Base model: $MODEL_NAME"
echo ">>> Effective model for this run: $EFFECTIVE_MODEL_NAME"
if [ -n "$SHARD_TAG" ]; then
    echo ">>> Shard mode: $SHARD_TAG"
fi
echo ">>> Input generated video dir: $GEN_VIDEO_DIR"
echo ">>> Input summary json: $SUMMARY_JSON"
echo ">>> Resolved config for model '$EFFECTIVE_MODEL_NAME': $CONFIG_PATH"

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

if [ "$RUN_ACTION" = true ]; then
    echo ">>> [WARN] action_following is not executed in run_evaluation.sh; please use run_action_following.sh separately."
fi

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
