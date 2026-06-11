#!/bin/bash
set -euo pipefail

# Usage:
#   run_evaluation_8metrics.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH] [SHARD_INDEX] [NUM_SHARDS]
#   run_evaluation_8metrics.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [SHARD_INDEX] [NUM_SHARDS]
#
# Supported metrics (exactly these 8 only):
#   aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality
#
# Difference vs run_evaluation.sh:
#   - Skips detection_tracking.py

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
RAW_METRICS=${4:-}
CONFIG_PATH="./config/config_ALL.yaml"
SHARD_INDEX=""
NUM_SHARDS=""
TEMP_CONFIG_PATH=""
EFFECTIVE_MODEL_NAME="$MODEL_NAME"
SHARD_TAG=""

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

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$ROOT_DIR/.." && pwd)
SCRIPT_START_TIME=$(date +%s)

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
    # if [ "$NUM_SHARDS" -le 1 ]; then
    #   echo "[ERROR] NUM_SHARDS must be greater than 1 when shard mode is enabled."
    #   exit 1
    # fi
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

render_config_for_model() {
  local src_config=$1
  local model_name=$2
  local dst_config
  dst_config=$(mktemp "/tmp/worldarena_${model_name}_8metrics_config_XXXXXX.yaml")
  sed "s|__MODEL_NAME__|${model_name}|g" "$src_config" > "$dst_config"
  echo "$dst_config"
}

normalize_config_legacy_root() {
  local config_file=$1
  local legacy_root="/data/liuwenhao/WorldArena"
  if [ ! -d "$legacy_root" ] && grep -q "$legacy_root" "$config_file"; then
    sed -i "s|$legacy_root|$PROJECT_ROOT|g" "$config_file"
    echo ">>> Rewrote legacy config root: $legacy_root -> $PROJECT_ROOT"
  fi
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

CLEAN_METRICS=$(echo "$RAW_METRICS" | tr ',' ' ' | tr '"' ' ')
METRIC_ARRAY=($CLEAN_METRICS)
if [ ${#METRIC_ARRAY[@]} -eq 0 ]; then
  echo "[ERROR] METRIC_LIST is empty."
  exit 1
fi

ALLOWED_METRICS=(
  "semantic_alignment"
  "aesthetic_quality"
  "background_consistency"
  "dynamic_degree"
  "flow_score"
  "photometric_smoothness"
  "motion_smoothness"
  "subject_consistency"
  "image_quality"
  "depth_accuracy"
  "trajectory_accuracy"
)

# ALLOWED_METRICS=(
#   "aesthetic_quality"
#   "background_consistency"
#   "dynamic_degree"
#   "flow_score"
#   "photometric_smoothness"
#   "motion_smoothness"
#   "subject_consistency"
#   "image_quality"
# )

for metric in "${METRIC_ARRAY[@]}"; do
  found=0
  for allowed in "${ALLOWED_METRICS[@]}"; do
    if [ "$metric" = "$allowed" ]; then
      found=1
      break
    fi
  done
  if [ "$found" -ne 1 ]; then
    echo "[ERROR] Unsupported metric for this script: $metric"
    echo "        Allowed: ${ALLOWED_METRICS[*]}"
    exit 1
  fi
done

# 这里不再自己重新用 conda info --base 找环境。
# 直接使用调用者已经准备好的环境（外层 one_step_ok_8metrics.sh 已在容器内激活 WorldArena_test）。
echo ">>> Python executable: $(command -v python)"
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  echo ">>> Current conda env: ${CONDA_DEFAULT_ENV}"
else
  echo ">>> Current conda env: (not set; using current PATH)"
fi

TEMP_CONFIG_PATH=$(render_config_for_model "$CONFIG_PATH" "$EFFECTIVE_MODEL_NAME")
CONFIG_PATH="$TEMP_CONFIG_PATH"
normalize_config_legacy_root "$CONFIG_PATH"

OUTPUT_BASE="$ROOT_DIR/data/$EFFECTIVE_MODEL_NAME"
mkdir -p "$OUTPUT_BASE"

echo ">>> Base model: $MODEL_NAME"
echo ">>> Effective model: $EFFECTIVE_MODEL_NAME"
if [ -n "$SHARD_TAG" ]; then
  echo ">>> Shard mode: $SHARD_TAG"
fi
echo ">>> Input generated video dir: $GEN_VIDEO_DIR"
echo ">>> Input summary json: $SUMMARY_JSON"
echo ">>> Resolved config: $CONFIG_PATH"
echo ">>> Metrics: ${METRIC_ARRAY[*]}"
echo ">>> NOTE: detection_tracking.py is now included."

echo ">>> Step 1/4: preprocess_datasets.py"
STEP_START_TIME=$(date +%s)
python "$ROOT_DIR/preprocess_datasets.py" \
  --summary_json "$SUMMARY_JSON" \
  --gen_video_dir "$GEN_VIDEO_DIR" \
  --output_base "$OUTPUT_BASE"
STEP_END_TIME=$(date +%s)
echo "$SUMMARY_JSON"
echo "$GEN_VIDEO_DIR"
echo "$OUTPUT_BASE"
echo ">>> Preprocessing finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"

echo ">>> Step 2/4: processing/video_resize.py"
STEP_START_TIME=$(date +%s)
python "$ROOT_DIR/processing/video_resize_ALL.py" \
  --config_path "$CONFIG_PATH" \
  --resize_gt
STEP_END_TIME=$(date +%s)
echo ">>> Video resize finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"

# echo ">>> Step 3/4: processing/detection_tracking.py"
# STEP_START_TIME=$(date +%s)
# python "$ROOT_DIR/processing/detection_tracking.py" \
#   --config_path "$CONFIG_PATH" \
#   --detect_gt
# STEP_END_TIME=$(date +%s)
# echo ">>> Detection and tracking finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"

echo ">>> Step 4/4: evaluate.py"
STEP_START_TIME=$(date +%s)
EVAL_ARGS=(--config "$CONFIG_PATH" --dimension "${METRIC_ARRAY[@]}")
if [ "${WORLD_ARENA_OVERWRITE_RESULTS:-0}" = "1" ]; then
  EVAL_ARGS+=(--overwrite)
  echo ">>> Evaluation mode: overwrite (WORLD_ARENA_OVERWRITE_RESULTS=1)"
else
  echo ">>> Evaluation mode: incremental merge"
fi
python "$ROOT_DIR/evaluate.py" "${EVAL_ARGS[@]}"
STEP_END_TIME=$(date +%s)
echo ">>> Evaluation finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"

SCRIPT_END_TIME=$(date +%s)
echo ">>> Total elapsed time: $(format_duration $((SCRIPT_END_TIME - SCRIPT_START_TIME)))"
echo ">>> ✅ Done. Results are under: $ROOT_DIR/data/$EFFECTIVE_MODEL_NAME/output"