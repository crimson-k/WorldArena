#!/bin/bash
set -euo pipefail

MODEL_NAME=${1:?model name required}
VIDEO_DIR=${2:?video dir required}
SUMMARY_JSON=${3:?summary json required}
METRICS=${4:-all}
# Optional: pass a custom config path; defaults to the repo config
CONFIG_PATH=${5:-}
TEMP_CONFIG_PATH=""
SCRIPT_START_TIME=$(date +%s)

format_duration() {
  local total_seconds=$1
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="$ROOT_DIR/VLM_judge.py"
OUTPUT_ROOT="$ROOT_DIR/output_VLM"
TMP_ROOT="$ROOT_DIR/tmp_VLM"
DEFAULT_CONFIG="$ROOT_DIR/config/config.yaml"

# Use provided config path if set, otherwise default
CONFIG_ARG=${CONFIG_PATH:-$DEFAULT_CONFIG}

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

TEMP_CONFIG_PATH=$(render_config_for_model "$CONFIG_ARG" "$MODEL_NAME")
CONFIG_ARG="$TEMP_CONFIG_PATH"
OUTPUT_ROOT="$ROOT_DIR/data/${MODEL_NAME}/output_VLM"
TMP_ROOT="$ROOT_DIR/data/${MODEL_NAME}/tmp_VLM"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate WorldArena_VLM
export PATH="your absolute path:$PATH"

echo ">>> Resolved config for model '$MODEL_NAME': $CONFIG_ARG"
echo ">>> VLM output root: $OUTPUT_ROOT"

echo ">>> Running VLM judge..."
STEP_START_TIME=$(date +%s)
python3 "$PY" \
  --model_name "$MODEL_NAME" \
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
