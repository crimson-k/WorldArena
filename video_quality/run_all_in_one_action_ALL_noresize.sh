#!/usr/bin/env bash
set -euo pipefail

SELF_PATH="$(cd -- "$(dirname "$0")" && pwd)/$(basename "$0")"

CPU_PARTITION="intel"
GPU_PARTITION="L40"

EXCLUDE_NODES_DEFAULT=""

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"

PREP_SCRIPT="${VQ_DIR}/prepare_eval_package_ALL.py"
SPLIT_SCRIPT="${VQ_DIR}/split_formal_package_shards.py"
AGG_SCRIPT="${VQ_DIR}/csv_results/aggregate_and_build_leaderboard_action_ALL.py"

CONFIG_PATH="${VQ_DIR}/config/config_ALL.yaml"
SIF_PATH="/share/home/usersht/images/ubuntu_2204.sif"
CONDA_SH="/share/apps/miniconda3/etc/profile.d/conda.sh"

# 只跑这两个指标
METRICS="image_quality,aesthetic_quality"

GT_VIDEOS_DIR_DEFAULT="/ssdfs/datahome/usersht/dev/lwh/GT_replay_240p_videos"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "${1}" -gt 0 ]
}

is_non_negative_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

print_gpu_allocation() {
  echo "---------------- GPU allocation info ----------------"
  echo "[GPU-INFO] HOSTNAME=$(hostname)"
  echo "[GPU-INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
  echo "[GPU-INFO] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-NA}"
  echo "[GPU-INFO] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-NA}"
  echo "[GPU-INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
  echo "[GPU-INFO] NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-NA}"
  echo "[GPU-INFO] nvidia-smi visible GPUs:"
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader 2>/dev/null || true
  echo "-----------------------------------------------------"
}

usage() {
  cat <<EOF
Usage:
  bash ${SELF_PATH} <base_model_name> <num_servers> [shards_per_server] [mail_user] [exclude_nodes] [gt_videos_dir]
  bash ${SELF_PATH} submit <base_model_name> <num_servers> [shards_per_server] [mail_user] [exclude_nodes] [gt_videos_dir]

Internal modes:
  bash ${SELF_PATH} prepare <base_model_name> [gt_videos_dir]
  bash ${SELF_PATH} split <base_model_name> <num_servers> [shards_per_server]
  bash ${SELF_PATH} worker <formal_model_name_with_server_suffix> [shards_per_server]
  bash ${SELF_PATH} eval_noresize <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH] [SHARD_INDEX] [NUM_SHARDS]
  bash ${SELF_PATH} aggregate <base_model_name>

Examples:
  bash ${SELF_PATH} runid_xxx 5
  bash ${SELF_PATH} runid_xxx 5 4
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com gpu4041
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com none
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com gpu4041 /path/to/GT_replay_240p_videos
EOF
}

derive_names() {
  local base_model_name="$1"

  if [[ "${base_model_name}" == *_formal ]]; then
    RAW_MODEL_NAME="${base_model_name%_formal}"
    FORMAL_MODEL_NAME="${base_model_name}"
  else
    RAW_MODEL_NAME="${base_model_name}"
    FORMAL_MODEL_NAME="${base_model_name}_formal"
  fi

  INPUT_ROOT="${VQ_DIR}/data/${RAW_MODEL_NAME}"
  OUTPUT_ROOT="${VQ_DIR}/data/${FORMAL_MODEL_NAME}"
  LOG_DIR="${VQ_DIR}/nohup_log/pipeline_${FORMAL_MODEL_NAME}"
}

has_action_following_dirs() {
  local root_dir="$1"
  local root_name
  root_name="$(basename "$root_dir")"

  if [ -d "${root_dir}/example_test_1" ] && [ -d "${root_dir}/example_test_2" ]; then
    return 0
  fi

  if [ -d "${root_dir}/${root_name}_test_1" ] && [ -d "${root_dir}/${root_name}_test_2" ]; then
    return 0
  fi

  return 1
}

run_prepare() {
  local base_model_name="${1:?missing base model name}"
  local gt_videos_dir="${2:-${GT_VIDEOS_DIR_DEFAULT}}"

  derive_names "${base_model_name}"

  source "${CONDA_SH}"
  conda activate WorldArena

  local -a prep_args=(
    python "${PREP_SCRIPT}"
    --input_root "${INPUT_ROOT}"
    --output_root "${OUTPUT_ROOT}"
    --model_name "${FORMAL_MODEL_NAME}"
    --src_main example_test
    --id1 usersht
    --image_mode keep
    --gt_videos_dir "${gt_videos_dir}"
    --overwrite
  )

  if has_action_following_dirs "${INPUT_ROOT}"; then
    echo "[prepare] Found example_test_1 and example_test_2. Package them, although action_following will not be evaluated."
    prep_args+=(
      --src_alt1 example_test_1
      --src_alt2 example_test_2
    )
  else
    echo "[prepare] example_test_1/example_test_2 not found. Continue with main videos only."
  fi

  "${prep_args[@]}"
}

run_split() {
  local base_model_name="${1:?missing base model name}"
  local num_servers="${2:?missing num_servers}"
  local shards_per_server="${3:-4}"

  is_positive_int "${num_servers}" || die "num_servers must be a positive integer, got: ${num_servers}"
  is_positive_int "${shards_per_server}" || die "shards_per_server must be a positive integer, got: ${shards_per_server}"

  derive_names "${base_model_name}"

  source "${CONDA_SH}"
  conda activate WorldArena

  WORLD_ARENA_NUM_SERVERS="${num_servers}" \
  WORLD_ARENA_SHARDS_PER_SERVER="${shards_per_server}" \
  python "${SPLIT_SCRIPT}" "${OUTPUT_ROOT}"
}

run_worker() {
  local shard_model="${1:?missing shard model name}"
  local shards_per_server="${2:-4}"

  is_positive_int "${shards_per_server}" || die "shards_per_server must be a positive integer, got: ${shards_per_server}"

  local summary_json="${VQ_DIR}/data/${shard_model}/summary.json"
  local gen_video_dir="${VQ_DIR}/data/${shard_model}/${shard_model}_test"

  local ALL_GPUS=(0 1 2 3 4 5 6 7)
  if [ "${shards_per_server}" -gt "${#ALL_GPUS[@]}" ]; then
    die "shards_per_server (${shards_per_server}) exceeds available GPUs (${#ALL_GPUS[@]})"
  fi

  local GPUS=("${ALL_GPUS[@]:0:${shards_per_server}}")
  local NUM_SHARDS=${#GPUS[@]}

  mkdir -p "${VQ_DIR}/nohup_log/${shard_model}"

  echo "=================================================="
  echo "[START] MODEL_NAME=${shard_model}"
  echo "[INFO] Only run metrics: ${METRICS}"
  echo "[INFO] NUM_SHARDS=${NUM_SHARDS}"
  echo "[INFO] GPUS(logical inside job)=(${GPUS[*]})"
  echo "[INFO] Resize is disabled."
  echo "[INFO] VLM / action_following / JEPA are disabled."
  echo "=================================================="

  print_gpu_allocation

  cd "${VQ_DIR}"

  source /etc/profile.d/modules.sh 2>/dev/null || true
  module load Singularity/4.2.1

  unset -f module 2>/dev/null || true
  unset -f ml 2>/dev/null || true

  singularity exec --cleanenv --nv \
    -B /ssdfs:/ssdfs \
    -B /share:/share \
    "${SIF_PATH}" \
    bash -lc "
set -euo pipefail

source '${CONDA_SH}'
conda activate WorldArena_test
cd '${VQ_DIR}'

GPUS=(${GPUS[*]})
NUM_SHARDS=${NUM_SHARDS}

HOST_SLURM_JOB_ID='${SLURM_JOB_ID:-NA}'
HOST_SLURM_JOB_NODELIST='${SLURM_JOB_NODELIST:-NA}'
HOST_SLURM_JOB_GPUS='${SLURM_JOB_GPUS:-NA}'
HOST_CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-NA}'

echo '---------------- Singularity GPU allocation info ----------------'
echo \"[SINGULARITY] HOSTNAME=\$(hostname)\"
echo \"[SINGULARITY] HOST_SLURM_JOB_ID=\${HOST_SLURM_JOB_ID}\"
echo \"[SINGULARITY] HOST_SLURM_JOB_NODELIST=\${HOST_SLURM_JOB_NODELIST}\"
echo \"[SINGULARITY] HOST_SLURM_JOB_GPUS=\${HOST_SLURM_JOB_GPUS}\"
echo \"[SINGULARITY] HOST_CUDA_VISIBLE_DEVICES=\${HOST_CUDA_VISIBLE_DEVICES}\"
echo \"[SINGULARITY] inner CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-NA}\"
echo \"[SINGULARITY] logical GPUS=(${GPUS[*]})\"
echo \"[SINGULARITY] nvidia-smi visible GPUs:\"
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader 2>/dev/null || true
echo '-----------------------------------------------------------------'

EVAL_PIDS=()

for shard_idx in \"\${!GPUS[@]}\"; do
  gpu=\${GPUS[\$shard_idx]}

  {
    echo \"---------------- image/aesthetic shard GPU info ----------------\"
    echo \"[IMG-AES] HOSTNAME=\$(hostname)\"
    echo \"[IMG-AES] shard_idx=\${shard_idx}\"
    echo \"[IMG-AES] NUM_SHARDS=\${NUM_SHARDS}\"
    echo \"[IMG-AES] logical gpu=\${gpu}\"
    echo \"[IMG-AES] HOST_SLURM_JOB_ID=\${HOST_SLURM_JOB_ID}\"
    echo \"[IMG-AES] HOST_SLURM_JOB_NODELIST=\${HOST_SLURM_JOB_NODELIST}\"
    echo \"[IMG-AES] HOST_SLURM_JOB_GPUS=\${HOST_SLURM_JOB_GPUS}\"
    echo \"[IMG-AES] HOST_CUDA_VISIBLE_DEVICES=\${HOST_CUDA_VISIBLE_DEVICES}\"
    echo \"[IMG-AES] inherited inner CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-NA}\"
    echo \"[IMG-AES] set CUDA_VISIBLE_DEVICES=\${gpu}\"
    echo \"[IMG-AES] nvidia-smi visible GPUs before launch:\"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader 2>/dev/null || true
    echo \"---------------------------------------------------------------\"

    WORLD_ARENA_OVERWRITE_RESULTS=1 CUDA_VISIBLE_DEVICES=\$gpu \
    bash '${SELF_PATH}' eval_noresize \
      '${shard_model}' \
      '${gen_video_dir}' \
      '${summary_json}' \
      '${METRICS}' \
      '${CONFIG_PATH}' \
      \"\$shard_idx\" \"\$NUM_SHARDS\"
  } > '${VQ_DIR}/nohup_log/${shard_model}/output_image_aesthetic_${shard_model}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &

  EVAL_PIDS+=(\"\$!\")
done

eval_failed=0
for pid in \"\${EVAL_PIDS[@]}\"; do
  if ! wait \"\$pid\"; then
    eval_failed=1
  fi
done

if [ \"\$eval_failed\" -ne 0 ]; then
  echo '[ERROR] One or more image/aesthetic shards failed for ${shard_model}.'
  exit 1
fi

echo '[DONE] Image Quality + Aesthetic Quality finished for ${shard_model}'
"

  echo "=================================================="
  echo "[DONE] MODEL_NAME=${shard_model}"
  echo "=================================================="
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

run_eval_noresize() {
  local MODEL_NAME="${1:-}"
  local GEN_VIDEO_DIR="${2:-}"
  local SUMMARY_JSON="${3:-}"
  local RAW_METRICS="${4:-}"
  local CONFIG_PATH_LOCAL="./config/config.yaml"
  local SHARD_INDEX=""
  local NUM_SHARDS=""

  if [ $# -ge 5 ]; then
    if [[ "${5}" =~ ^[0-9]+$ ]]; then
      SHARD_INDEX="${5}"
      NUM_SHARDS="${6:-}"
    else
      CONFIG_PATH_LOCAL="${5}"
      SHARD_INDEX="${6:-}"
      NUM_SHARDS="${7:-}"
    fi
  fi

  if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$SUMMARY_JSON" ] || [ -z "$RAW_METRICS" ]; then
    echo "Usage: $0 eval_noresize <MODEL_NAME> <GEN_VIDEO_DIR> <SUMMARY_JSON> <METRIC_LIST> [CONFIG_PATH] [SHARD_INDEX] [NUM_SHARDS]"
    exit 1
  fi

  echo "---------------- eval_noresize runtime GPU info ----------------"
  echo "[EVAL-NORESIZE] HOSTNAME=$(hostname)"
  echo "[EVAL-NORESIZE] MODEL_NAME=${MODEL_NAME}"
  echo "[EVAL-NORESIZE] SHARD_INDEX=${SHARD_INDEX:-NA}"
  echo "[EVAL-NORESIZE] NUM_SHARDS=${NUM_SHARDS:-NA}"
  echo "[EVAL-NORESIZE] SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
  echo "[EVAL-NORESIZE] SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-NA}"
  echo "[EVAL-NORESIZE] SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-NA}"
  echo "[EVAL-NORESIZE] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
  echo "[EVAL-NORESIZE] NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-NA}"
  echo "[EVAL-NORESIZE] nvidia-smi visible GPUs:"
  nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader 2>/dev/null || true
  echo "----------------------------------------------------------------"

  local TEMP_CONFIG_PATH=""
  local EFFECTIVE_MODEL_NAME="$MODEL_NAME"
  local SHARD_TAG=""
  local PROJECT_ROOT
  PROJECT_ROOT="$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

  cleanup_eval_config() {
    if [ -n "${TEMP_CONFIG_PATH:-}" ] && [ -f "${TEMP_CONFIG_PATH:-}" ]; then
      rm -f "${TEMP_CONFIG_PATH}"
    fi
  }

  trap cleanup_eval_config EXIT

  resolve_sharded_inputs_for_eval() {
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
      summary_parent="$(cd -- "$(dirname "$SUMMARY_JSON")" && pwd)"

      local summary_file
      summary_file="$(basename "$SUMMARY_JSON")"

      local expected_shard_name
      expected_shard_name="$(format_shard_name "$SHARD_INDEX" "$NUM_SHARDS")"

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
    summary_parent="$(cd -- "$(dirname "$SUMMARY_JSON")" && pwd)"

    if [[ "$(basename "$summary_parent")" =~ ^shard_[0-9]+$ ]] && [ "$(basename "$SUMMARY_JSON")" = "summary.json" ]; then
      SHARD_TAG="$(basename "$summary_parent")"
      EFFECTIVE_MODEL_NAME="${MODEL_NAME}_${SHARD_TAG}"
    fi
  }

  normalize_config_legacy_root_for_eval() {
    local config_file="$1"
    local legacy_root="/data/liuwenhao/WorldArena"

    if [ ! -d "$legacy_root" ] && grep -q "$legacy_root" "$config_file"; then
      sed -i "s|$legacy_root|$PROJECT_ROOT|g" "$config_file"
      echo ">>> Rewrote legacy config root: $legacy_root -> $PROJECT_ROOT"
    fi
  }

  render_config_for_model_for_eval() {
    local src_config="$1"
    local model_name="$2"
    local dst_config

    dst_config="$(mktemp "/tmp/worldarena_${model_name}_config_XXXXXX.yaml")"
    sed "s|__MODEL_NAME__|${model_name}|g" "$src_config" > "$dst_config"

    echo "$dst_config"
  }

  resolve_sharded_inputs_for_eval

  if [ ! -f "$SUMMARY_JSON" ]; then
    echo "[ERROR] Summary JSON not found: $SUMMARY_JSON"
    exit 1
  fi

  if [ ! -d "$GEN_VIDEO_DIR" ]; then
    echo "[ERROR] Generated video dir not found: $GEN_VIDEO_DIR"
    exit 1
  fi

  if [ ! -f "$CONFIG_PATH_LOCAL" ]; then
    echo "[ERROR] Config path not found: $CONFIG_PATH_LOCAL"
    exit 1
  fi

  source "${CONDA_SH}"
  conda activate WorldArena_test

  local CLEAN_METRICS
  CLEAN_METRICS="$(echo "$RAW_METRICS" | tr ',' ' ' | tr '"' ' ')"

  local METRIC_ARRAY=($CLEAN_METRICS)
  local EVAL_METRICS=()

  if [ "${#METRIC_ARRAY[@]}" -eq 0 ]; then
    echo "[ERROR] Empty metric list."
    exit 1
  fi

  for metric in "${METRIC_ARRAY[@]}"; do
    case "$metric" in
      image_quality|aesthetic_quality)
        EVAL_METRICS+=("$metric")
        ;;
      *)
        echo "[ERROR] This all-in-one no-resize script only supports: image_quality,aesthetic_quality"
        echo "        Got unsupported metric: $metric"
        exit 1
        ;;
    esac
  done

  echo ">>> Input metrics: $RAW_METRICS"
  echo ">>> Formatted for evaluate.py: ${EVAL_METRICS[*]}"

  local DATA_DIR="./data/$EFFECTIVE_MODEL_NAME"
  local CONFIG_DIR="./config"
  local OUTPUT_DIR="./output"

  mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$OUTPUT_DIR"

  TEMP_CONFIG_PATH="$(render_config_for_model_for_eval "$CONFIG_PATH_LOCAL" "$EFFECTIVE_MODEL_NAME")"
  CONFIG_PATH_LOCAL="$TEMP_CONFIG_PATH"
  normalize_config_legacy_root_for_eval "$CONFIG_PATH_LOCAL"

  echo ">>> Base model: $MODEL_NAME"
  echo ">>> Effective model for this run: $EFFECTIVE_MODEL_NAME"

  if [ -n "$SHARD_TAG" ]; then
    echo ">>> Shard mode: $SHARD_TAG"
  fi

  echo ">>> Input generated video dir: $GEN_VIDEO_DIR"
  echo ">>> Input summary json: $SUMMARY_JSON"
  echo ">>> Resolved config for model '$EFFECTIVE_MODEL_NAME': $CONFIG_PATH_LOCAL"

  echo ">>> Running preprocessing for Image Quality + Aesthetic Quality..."
  python preprocess_datasets.py \
    --summary_json "$SUMMARY_JSON" \
    --gen_video_dir "$GEN_VIDEO_DIR" \
    --output_base "$DATA_DIR"

  echo ">>> Skip video resize because resize flow has been removed."
  echo ">>> Skip detection and tracking because image_quality/aesthetic_quality do not need it."

  echo ">>> Starting Evaluation: ${EVAL_METRICS[*]}"

  local EVAL_ARGS=(--dimension "${EVAL_METRICS[@]}" --config "$CONFIG_PATH_LOCAL")

  if [ "${WORLD_ARENA_OVERWRITE_RESULTS:-0}" = "1" ]; then
    EVAL_ARGS+=(--overwrite)
    echo ">>> Evaluation mode: overwrite"
  else
    echo ">>> Evaluation mode: incremental merge"
  fi

  python evaluate.py "${EVAL_ARGS[@]}"

  echo ">>> ✅ Image Quality + Aesthetic Quality evaluation finished"
}

run_aggregate() {
  local base_model_name="${1:?missing base model name}"

  derive_names "${base_model_name}"

  if [ ! -f "${AGG_SCRIPT}" ]; then
    echo "[aggregate] Skip aggregate because script not found: ${AGG_SCRIPT}"
    exit 0
  fi

  source "${CONDA_SH}"
  conda activate WorldArena

  echo "[aggregate] Aggregate existing metrics for ${FORMAL_MODEL_NAME}"
  python "${AGG_SCRIPT}" --model_name "${FORMAL_MODEL_NAME}"
}

submit_pipeline() {
  local base_model_name="${1:?missing base model name}"
  local num_servers="${2:?missing num_servers}"
  local shards_per_server="${3:-4}"
  local mail_user="${4:-}"
  local exclude_nodes="${5:-${EXCLUDE_NODES_DEFAULT}}"
  local gt_videos_dir="${6:-${GT_VIDEOS_DIR_DEFAULT}}"

  if [ "${mail_user}" = "none" ] || [ "${mail_user}" = "NONE" ]; then
    mail_user=""
  fi

  is_positive_int "${num_servers}" || die "num_servers must be a positive integer, got: ${num_servers}"
  is_positive_int "${shards_per_server}" || die "shards_per_server must be a positive integer, got: ${shards_per_server}"

  if [ "${exclude_nodes}" = "none" ] || [ "${exclude_nodes}" = "NONE" ]; then
    exclude_nodes=""
  fi

  derive_names "${base_model_name}"
  mkdir -p "${LOG_DIR}"

  local worker_cpus=$((shards_per_server * 7))

  local -a GPU_EXCLUDE_ARGS=()
  if [ -n "${exclude_nodes}" ]; then
    GPU_EXCLUDE_ARGS+=(--exclude="${exclude_nodes}")
  fi

  echo "=================================================="
  echo "[PIPELINE] RAW_MODEL_NAME=${RAW_MODEL_NAME}"
  echo "[PIPELINE] FORMAL_MODEL_NAME=${FORMAL_MODEL_NAME}"
  echo "[PIPELINE] ONLY_METRICS=${METRICS}"
  echo "[PIPELINE] NUM_SERVERS=${num_servers}"
  echo "[PIPELINE] SHARDS_PER_SERVER=${shards_per_server}"
  echo "[PIPELINE] WORKER_CPUS=${worker_cpus}"
  echo "[PIPELINE] SCRIPT=${SELF_PATH}"
  if [ -n "${mail_user}" ]; then
    echo "[PIPELINE] MAIL_USER=${mail_user}"
  else
    echo "[PIPELINE] MAIL_USER=(disabled)"
  fi
  if [ -n "${exclude_nodes}" ]; then
    echo "[PIPELINE] EXCLUDE_NODES=${exclude_nodes}"
  else
    echo "[PIPELINE] EXCLUDE_NODES=(none)"
  fi
  echo "[PIPELINE] LOG_DIR=${LOG_DIR}"
  echo "[PIPELINE] GT_VIDEOS_DIR=${gt_videos_dir}"
  echo "=================================================="

  local prep_wrap
  printf -v prep_wrap "bash %q prepare %q %q" "${SELF_PATH}" "${RAW_MODEL_NAME}" "${gt_videos_dir}"

  local PREP_JOB_ID
  PREP_JOB_ID=$(sbatch --parsable \
    -J "prep_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    -o "${LOG_DIR}/prep_%j.out" \
    --wrap="${prep_wrap}")

  echo "[SUBMIT] prepare job id: ${PREP_JOB_ID}"

  local split_wrap
  printf -v split_wrap "bash %q split %q %q %q" "${SELF_PATH}" "${RAW_MODEL_NAME}" "${num_servers}" "${shards_per_server}"

  local SPLIT_JOB_ID
  SPLIT_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${PREP_JOB_ID} \
    -J "split_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    -o "${LOG_DIR}/split_%j.out" \
    --wrap="${split_wrap}")

  echo "[SUBMIT] split job id: ${SPLIT_JOB_ID}"

  local -a WORKER_JOB_IDS=()
  local sid

  for ((sid=0; sid<num_servers; sid++)); do
    local shard_model="${FORMAL_MODEL_NAME}_${sid}"
    local worker_wrap
    local jid

    printf -v worker_wrap "bash %q worker %q %q" "${SELF_PATH}" "${shard_model}" "${shards_per_server}"

    jid=$(sbatch --parsable \
      --dependency=afterok:${SPLIT_JOB_ID} \
      -J "iq_aes_${FORMAL_MODEL_NAME}_${sid}" \
      -p "${GPU_PARTITION}" \
      -N 1 -n 1 -c "${worker_cpus}" --gres=gpu:l40:${shards_per_server} \
      "${GPU_EXCLUDE_ARGS[@]}" \
      -o "${LOG_DIR}/worker_${sid}_%j.out" \
      --wrap="${worker_wrap}")

    WORKER_JOB_IDS+=("${jid}")
    echo "[SUBMIT] worker shard=${sid} job id: ${jid}"
  done

  local WORKER_DEP
  WORKER_DEP=$(IFS=:; echo "${WORKER_JOB_IDS[*]}")

  local -a AGG_MAIL_OPTS=()
  if [ -n "${mail_user}" ]; then
    AGG_MAIL_OPTS+=(--mail-type=END,FAIL --mail-user="${mail_user}")
  fi

  local agg_wrap
  printf -v agg_wrap "bash %q aggregate %q" "${SELF_PATH}" "${RAW_MODEL_NAME}"

  local AGG_JOB_ID
  AGG_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${WORKER_DEP} \
    -J "agg_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    "${AGG_MAIL_OPTS[@]}" \
    -o "${LOG_DIR}/aggregate_%j.out" \
    --wrap="${agg_wrap}")

  echo "[SUBMIT] aggregate job id: ${AGG_JOB_ID}"

  echo "=================================================="
  echo "[DONE] Pipeline submitted successfully."
  echo
  echo "Prepare job:   ${PREP_JOB_ID}"
  echo "Split job:     ${SPLIT_JOB_ID}"
  echo "Worker jobs:   ${WORKER_JOB_IDS[*]}"
  echo "Aggregate job: ${AGG_JOB_ID}"
  if [ -n "${mail_user}" ]; then
    echo "Mail notify:   enabled (${mail_user})"
  else
    echo "Mail notify:   disabled"
  fi
  if [ -n "${exclude_nodes}" ]; then
    echo "Exclude nodes: ${exclude_nodes}"
  else
    echo "Exclude nodes: none"
  fi
  echo
  echo "Check status:"
  echo "  squeue -j ${PREP_JOB_ID},${SPLIT_JOB_ID},${AGG_JOB_ID}$(printf ",%s" "${WORKER_JOB_IDS[@]}")"
  echo
  echo "Scancel jobs:"
  echo "  scancel ${PREP_JOB_ID} ${SPLIT_JOB_ID} ${AGG_JOB_ID}$(printf " %s" "${WORKER_JOB_IDS[@]}")"
  echo
  echo "Logs:"
  echo "  ${LOG_DIR}"
  echo "  worker detail logs: ${VQ_DIR}/nohup_log/${FORMAL_MODEL_NAME}_*/"
  echo "=================================================="
}

MODE="submit"

if [ $# -gt 0 ]; then
  case "$1" in
    submit|prepare|split|worker|eval_noresize|aggregate)
      MODE="$1"
      shift
      ;;
  esac
fi

case "${MODE}" in
  submit)
    [ $# -ge 2 ] || { usage; exit 1; }
    submit_pipeline "$@"
    ;;
  prepare)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_prepare "$@"
    ;;
  split)
    [ $# -ge 2 ] || { usage; exit 1; }
    run_split "$@"
    ;;
  worker)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_worker "$@"
    ;;
  eval_noresize)
    [ $# -ge 4 ] || { usage; exit 1; }
    run_eval_noresize "$@"
    ;;
  aggregate)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_aggregate "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac