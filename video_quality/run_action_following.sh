#!/bin/bash
set -euo pipefail

SELF_PATH="$(cd -- "$(dirname "$0")" && pwd)/$(basename "$0")"
SIF_PATH="/share/home/usersht/images/ubuntu_2204.sif"
CONDA_SH="/share/apps/miniconda3/etc/profile.d/conda.sh"

IN_CONTAINER=0
if [ -n "${APPTAINER_NAME:-}" ] || [ -n "${SINGULARITY_NAME:-}" ] || [ -n "${SINGULARITY_CONTAINER:-}" ]; then
  IN_CONTAINER=1
fi

if [ "${WORLD_ARENA_ACTION_INNER:-0}" != "1" ] && [ "$IN_CONTAINER" != "1" ]; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load Singularity/4.2.1

  unset -f module 2>/dev/null || true
  unset -f ml 2>/dev/null || true

  exec singularity exec --cleanenv --nv \
    -B /ssdfs:/ssdfs \
    -B /share:/share \
    "$SIF_PATH" \
    env WORLD_ARENA_ACTION_INNER=1 bash "$SELF_PATH" "$@"
fi

source "$CONDA_SH"
conda activate WorldArena_test

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
SUMMARY_JSON=${3:-}
SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG_PATH="$SCRIPT_DIR/config/config.yaml"
SHARD_INDEX=""
NUM_SHARDS=""
TEMP_CONFIG_PATH=""
EFFECTIVE_MODEL_NAME="$MODEL_NAME"
SHARD_TAG=""

if [ $# -ge 4 ]; then
  if [[ "${4}" =~ ^[0-9]+$ ]]; then
    SHARD_INDEX="${4}"
    NUM_SHARDS=${5:-}
  else
    CONFIG_PATH="${4}"
    SHARD_INDEX=${5:-}
    NUM_SHARDS=${6:-}
  fi
fi

infer_inputs_from_model_name() {
  local data_base="$SCRIPT_DIR/data"
  local model_no_formal="${MODEL_NAME%_formal}"
  local selected_root=""
  local -a root_candidates=()
  local candidate_root

  if [[ "$MODEL_NAME" == *_formal ]]; then
    root_candidates+=("${data_base}/${MODEL_NAME}" "${data_base}/${model_no_formal}")
  else
    root_candidates+=("${data_base}/${MODEL_NAME}_formal" "${data_base}/${MODEL_NAME}")
  fi

  for candidate_root in "${root_candidates[@]}"; do
    if [ -d "$candidate_root" ]; then
      selected_root="$candidate_root"
      break
    fi
  done

  if [ -z "$selected_root" ]; then
    echo "[ERROR] Cannot infer data root from MODEL_NAME=${MODEL_NAME}"
    echo "        Tried:"
    printf "        - %s\n" "${root_candidates[@]}"
    echo "        Please pass GEN_VIDEO_DIR and SUMMARY_JSON explicitly."
    exit 1
  fi

  local inferred_summary="${selected_root}/summary.json"
  if [ ! -f "$inferred_summary" ]; then
    echo "[ERROR] Cannot find inferred summary: $inferred_summary"
    echo "        Please pass SUMMARY_JSON explicitly."
    exit 1
  fi

  local root_name
  root_name=$(basename "$selected_root")
  local inferred_gen=""
  local gen_candidates=(
    "${selected_root}/${MODEL_NAME}_test"
    "${selected_root}/${model_no_formal}_test"
    "${selected_root}/${root_name}_test"
  )
  local candidate
  for candidate in "${gen_candidates[@]}"; do
    if [ -d "$candidate" ]; then
      inferred_gen="$candidate"
      break
    fi
  done

  if [ -z "$inferred_gen" ]; then
    local -a test_dirs=()
    mapfile -t test_dirs < <(find "$selected_root" -maxdepth 1 -mindepth 1 -type d -name "*_test" ! -name "*_test_1" ! -name "*_test_2" ! -name "*_test_vlm" | sort)
    if [ "${#test_dirs[@]}" -eq 1 ]; then
      inferred_gen="${test_dirs[0]}"
    elif [ "${#test_dirs[@]}" -gt 1 ]; then
      echo "[ERROR] Multiple *_test folders found under $selected_root:"
      printf "        - %s\n" "${test_dirs[@]}"
      echo "        Please pass GEN_VIDEO_DIR explicitly."
      exit 1
    else
      echo "[ERROR] Cannot infer generated video dir under $selected_root"
      echo "        Expected one of:"
      printf "        - %s\n" "${gen_candidates[@]}"
      echo "        or exactly one folder matching *_test."
      exit 1
    fi
  fi

  SUMMARY_JSON="$inferred_summary"
  GEN_VIDEO_DIR="$inferred_gen"
  echo ">>> Auto-resolved SUMMARY_JSON: $SUMMARY_JSON"
  echo ">>> Auto-resolved GEN_VIDEO_DIR: $GEN_VIDEO_DIR"
}

if [ -n "$MODEL_NAME" ] && [ -z "$GEN_VIDEO_DIR" ] && [ -z "$SUMMARY_JSON" ] && [ $# -eq 1 ]; then
  infer_inputs_from_model_name
fi

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ]; then
  echo "Usage: $0 <MODEL_NAME>"
  echo "   or: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [CONFIG_PATH] [SHARD_INDEX] [NUM_SHARDS]"
  echo "   or: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> [SHARD_INDEX] [NUM_SHARDS]"
  exit 1
fi

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

render_config_for_model() {
  local src_config=$1
  local model_name=$2
  local dst_config
  dst_config=$(mktemp "/tmp/worldarena_${model_name}_action_config_XXXXXX.yaml")
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

DATA_DIR="$SCRIPT_DIR/data_action_following/$EFFECTIVE_MODEL_NAME"
CONFIG_DIR="$SCRIPT_DIR/config"
OUTPUT_DIR_ACTION="$SCRIPT_DIR/output_action_following"
mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR_ACTION"

TEMP_CONFIG_PATH=$(render_config_for_model "$CONFIG_PATH" "$EFFECTIVE_MODEL_NAME")
CONFIG_PATH="$TEMP_CONFIG_PATH"
normalize_config_legacy_root "$CONFIG_PATH"

echo ">>> Python executable: $(command -v python)"
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
  echo ">>> Current conda env: ${CONDA_DEFAULT_ENV}"
else
  echo ">>> Current conda env: (not set; using current PATH)"
fi

echo ">>> Base model: $MODEL_NAME"
echo ">>> Effective model for this run: $EFFECTIVE_MODEL_NAME"
if [ -n "$SHARD_TAG" ]; then
  echo ">>> Shard mode: $SHARD_TAG"
fi
echo ">>> Input generated video dir: $GEN_VIDEO_DIR"
echo ">>> Input summary json: $SUMMARY_JSON"
echo ">>> Resolved config: $CONFIG_PATH"

echo ">>> Running action_following preprocessing..."
python "$SCRIPT_DIR/preprocess_datasets_diversity.py" \
  --summary_json "$SUMMARY_JSON" \
  --gen_video_dir "$GEN_VIDEO_DIR" \
  --output_base "$DATA_DIR"

echo ">>> Running action_following evaluation..."
EVAL_ARGS=(--dimension "action_following" --config "$CONFIG_PATH")
if [ "${WORLD_ARENA_OVERWRITE_RESULTS:-0}" = "1" ]; then
  EVAL_ARGS+=(--overwrite)
  echo ">>> Evaluation mode: overwrite (WORLD_ARENA_OVERWRITE_RESULTS=1)"
else
  echo ">>> Evaluation mode: incremental merge"
fi
python "$SCRIPT_DIR/evaluate.py" "${EVAL_ARGS[@]}" || echo ">>> [WARNING] evaluate.py (action_following) returned non-zero code"

echo ">>> ✅ Action Following Script Finished"