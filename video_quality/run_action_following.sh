#!/bin/bash
set -euo pipefail

# Usage: run_action_following.sh <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [CONFIG_PATH]

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
CONFIG_PATH=${4:-"./config/config.yaml"}
TEMP_CONFIG_PATH=""

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ]; then
  echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [CONFIG_PATH]"
  exit 1
fi

render_config_for_model() {
  local src_config=$1
  local model_name=$2
  local dst_config
  dst_config=$(mktemp "/tmp/worldarena_${model_name}_action_config_XXXXXX.yaml")
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

DATA_DIR="./data_action_following/$MODEL_NAME"
CONFIG_DIR="./config"
OUTPUT_DIR_ACTION="./output_action_following"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR_ACTION"

TEMP_CONFIG_PATH=$(render_config_for_model "$CONFIG_PATH" "$MODEL_NAME")
CONFIG_PATH="$TEMP_CONFIG_PATH"
echo ">>> Resolved config for model '$MODEL_NAME': $CONFIG_PATH"

echo ">>> Running action_following preprocessing..."
python preprocess_datasets_diversity.py --summary_json "$SUMMARY_JSON" --gen_video_dir "$GEN_VIDEO_DIR" --output_base "$DATA_DIR"

echo ">>> Running action_following evaluation..."
python evaluate.py --dimension "action_following" --config "$CONFIG_PATH" --overwrite || echo ">>> [WARNING] evaluate.py (action_following) returned non-zero code"

echo ">>> ✅ Action Following Script Finished"
