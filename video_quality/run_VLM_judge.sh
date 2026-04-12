#!/bin/bash
set -euo pipefail

# Usage:
#   run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> [METRICS] [CONFIG_PATH] [SHARD_INDEX] [NUM_SHARDS]
#   run_VLM_judge.sh <MODEL_NAME> <VIDEO_DIR> <SUMMARY_JSON> [METRICS] [SHARD_INDEX] [NUM_SHARDS]

MODEL_NAME=${1:?model name required}
VIDEO_DIR=${2:?video dir required}
SUMMARY_JSON=${3:?summary json required}
METRICS=${4:-all}
# Optional: pass a custom config path; defaults to the repo config
CONFIG_PATH=""
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
TEMP_CONFIG_PATH=""
SCRIPT_START_TIME=$(date +%s)
EFFECTIVE_MODEL_NAME="$MODEL_NAME"
SHARD_TAG=""

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

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$ROOT_DIR/.." && pwd)
PY="$ROOT_DIR/VLM_judge.py"
OUTPUT_ROOT="$ROOT_DIR/output_VLM"
TMP_ROOT="$ROOT_DIR/tmp_VLM"
DEFAULT_CONFIG="$ROOT_DIR/config/config.yaml"

# Use provided config path if set, otherwise default
CONFIG_ARG=${CONFIG_PATH:-$DEFAULT_CONFIG}

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

    local shard_video_dir="${shard_dir}/$(basename "$VIDEO_DIR")"
    if [ ! -d "$shard_video_dir" ]; then
      local fallback_video="${shard_dir}/${MODEL_NAME}_test_vlm"
      if [ -d "$fallback_video" ]; then
        shard_video_dir="$fallback_video"
      else
        echo "[ERROR] Shard video dir not found under: $shard_dir"
        echo "        Tried: $shard_video_dir"
        echo "        Tried: $fallback_video"
        exit 1
      fi
    fi

    SUMMARY_JSON="$shard_summary"
    VIDEO_DIR="$shard_video_dir"
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
  dst_config=$(mktemp "/tmp/worldarena_${model_name}_vlm_config_XXXXXX.yaml")
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
if [ ! -d "$VIDEO_DIR" ]; then
  echo "[ERROR] Video dir not found: $VIDEO_DIR"
  exit 1
fi
if [ ! -f "$CONFIG_ARG" ]; then
  echo "[ERROR] Config path not found: $CONFIG_ARG"
  exit 1
fi

TEMP_CONFIG_PATH=$(render_config_for_model "$CONFIG_ARG" "$EFFECTIVE_MODEL_NAME")
CONFIG_ARG="$TEMP_CONFIG_PATH"
normalize_config_legacy_root "$CONFIG_ARG"
OUTPUT_ROOT="$ROOT_DIR/data/${EFFECTIVE_MODEL_NAME}/output_VLM"
TMP_ROOT="$ROOT_DIR/data/${EFFECTIVE_MODEL_NAME}/tmp_VLM"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate WorldArena_VLM
export PATH="your absolute path:$PATH"

echo ">>> Base model: $MODEL_NAME"
echo ">>> Effective model for this run: $EFFECTIVE_MODEL_NAME"
if [ -n "$SHARD_TAG" ]; then
  echo ">>> Shard mode: $SHARD_TAG"
fi
echo ">>> Input video dir: $VIDEO_DIR"
echo ">>> Input summary json: $SUMMARY_JSON"
echo ">>> Resolved config for model '$EFFECTIVE_MODEL_NAME': $CONFIG_ARG"
echo ">>> VLM output root: $OUTPUT_ROOT"

echo ">>> Running VLM judge..."
STEP_START_TIME=$(date +%s)
python3 "$PY" \
  --model_name "$EFFECTIVE_MODEL_NAME" \
  --video_dir "$VIDEO_DIR" \
  --summary_json "$SUMMARY_JSON" \
  --metrics "$METRICS" \
  --num_frames 16 \
  --output_root "$OUTPUT_ROOT" \
  --tmp_root "$TMP_ROOT" \
  --config_path "$CONFIG_ARG"
STEP_END_TIME=$(date +%s)

SCRIPT_END_TIME=$(date +%s)
echo ">>> VLM judge finished in $(format_duration $((STEP_END_TIME - STEP_START_TIME)))"
echo ">>> Total elapsed time: $(format_duration $((SCRIPT_END_TIME - SCRIPT_START_TIME)))"
echo ">>> ✅ VLM judge finished"
