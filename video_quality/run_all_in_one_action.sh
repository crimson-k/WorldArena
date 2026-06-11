#!/usr/bin/env bash
set -euo pipefail

SELF_PATH="$(cd -- "$(dirname "$0")" && pwd)/$(basename "$0")"

CPU_PARTITION="intel"
GPU_PARTITION="L40"

# 默认排除节点。可以通过第 5 个命令行参数覆盖。
EXCLUDE_NODES_DEFAULT=""

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"

PREP_SCRIPT="${VQ_DIR}/prepare_eval_package.py"
SPLIT_SCRIPT="${VQ_DIR}/split_formal_package_shards.py"
AGG_SCRIPT="${VQ_DIR}/csv_results/aggregate_and_build_leaderboard_action_ALL.py"
# AGG_SCRIPT="${VQ_DIR}/csv_results/aggregate_and_build_leaderboard_partial_1000.py"

CONFIG_PATH="${VQ_DIR}/config/config.yaml"
SIF_PATH="/share/home/usersht/images/ubuntu_2204.sif"
CONDA_SH="/share/apps/miniconda3/etc/profile.d/conda.sh"

METRICS="aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality"
# METRICS="semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy"

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "${1}" -gt 0 ]
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
  bash ${SELF_PATH} <base_model_name> <num_servers> [shards_per_server] [mail_user] [exclude_nodes]
  bash ${SELF_PATH} submit <base_model_name> <num_servers> [shards_per_server] [mail_user] [exclude_nodes]

Internal modes (normally do not call manually):
  bash ${SELF_PATH} prepare <base_model_name>
  bash ${SELF_PATH} split <base_model_name> <num_servers> [shards_per_server]
  bash ${SELF_PATH} worker <formal_model_name_with_server_suffix> [shards_per_server]
  bash ${SELF_PATH} aggregate <base_model_name>

Examples:
  bash ${SELF_PATH} runid_xxx 5
  bash ${SELF_PATH} runid_xxx 5 4
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com gpu4041
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com gpu4041,gpu4008
  bash ${SELF_PATH} runid_xxx 5 4 your_email@qq.com none
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

  # 原始数据目录格式：example_test_1 / example_test_2
  if [ -d "${root_dir}/example_test_1" ] && [ -d "${root_dir}/example_test_2" ]; then
    return 0
  fi

  # formal / split 后目录格式：<model_name>_test_1 / <model_name>_test_2
  if [ -d "${root_dir}/${root_name}_test_1" ] && [ -d "${root_dir}/${root_name}_test_2" ]; then
    return 0
  fi

  return 1
}

run_prepare() {
  local base_model_name="${1:?missing base model name}"
  derive_names "${base_model_name}"

  source "${CONDA_SH}"
  conda activate WorldArena

  local -a prep_args=(
    python "${PREP_SCRIPT}"
    --input_root "${INPUT_ROOT}"
    --output_root "${OUTPUT_ROOT}"
    --model_name "${FORMAL_MODEL_NAME}"
    --src_main example_test
    --id1 fixed_scene_task
    --image_mode keep
    --overwrite
  )

  if has_action_following_dirs "${INPUT_ROOT}"; then
    echo "[prepare] Found example_test_1 and example_test_2. Enable action_following package fields."
    prep_args+=(
      --src_alt1 example_test_1
      --src_alt2 example_test_2
    )
  else
    echo "[prepare] example_test_1/example_test_2 not found. Skip action_following package fields."
  fi

  "${prep_args[@]}"
}

run_split() {
  local base_model_name="${1:?missing base model name}"
  local num_servers="${2:?missing num_servers}"
  local shards_per_server="${3:-8}"

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
  local shards_per_server="${2:-8}"

  is_positive_int "${shards_per_server}" || die "shards_per_server must be a positive integer, got: ${shards_per_server}"

  local summary_json="${VQ_DIR}/data/${shard_model}/summary.json"
  local video_dir="${VQ_DIR}/data/${shard_model}/${shard_model}_test_vlm"
  local gen_video_dir="${VQ_DIR}/data/${shard_model}/${shard_model}_test"

  local shard_root="${VQ_DIR}/data/${shard_model}"
  local action_following_enabled=0

  if has_action_following_dirs "${shard_root}"; then
    action_following_enabled=1
    echo "[worker] action_following detected for ${shard_model}, but action_following is disabled in this script."
  else
    echo "[worker] action_following skipped for ${shard_model}: *_test_1 and *_test_2 not found"
  fi

  local ALL_GPUS=(0 1 2 3 4 5 6 7)
  if [ "${shards_per_server}" -gt "${#ALL_GPUS[@]}" ]; then
    die "shards_per_server (${shards_per_server}) exceeds available GPUs (${#ALL_GPUS[@]})"
  fi

  local GPUS=("${ALL_GPUS[@]:0:${shards_per_server}}")
  local NUM_SHARDS=${#GPUS[@]}

  mkdir -p "${VQ_DIR}/nohup_log/${shard_model}"

  echo "=================================================="
  echo "[START] MODEL_NAME=${shard_model}"
  echo "[INFO] NUM_SHARDS=${NUM_SHARDS}"
  echo "[INFO] GPUS(logical inside job)=(${GPUS[*]})"
  echo "=================================================="

  print_gpu_allocation

  cd "${VQ_DIR}"

  # -------------------------------
  # Stage 0: data_process
  # -------------------------------
  echo "[0/3] Skip data_process.py for ${shard_model} (assume already prepared)"

  # -------------------------------
  # Stage 1: VLM judge on host
  # -------------------------------
  echo "[1/3] Run VLM judge for ${shard_model}"

  source "${CONDA_SH}"
  conda activate WorldArena

  VLM_PIDS=()

  for shard_idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$shard_idx]}

    {
      echo "---------------- VLM shard GPU info ----------------"
      echo "[VLM] HOSTNAME=$(hostname)"
      echo "[VLM] shard_idx=${shard_idx}"
      echo "[VLM] NUM_SHARDS=${NUM_SHARDS}"
      echo "[VLM] logical gpu=${gpu}"
      echo "[VLM] parent SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
      echo "[VLM] parent SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-NA}"
      echo "[VLM] parent SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-NA}"
      echo "[VLM] parent CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-NA}"
      echo "[VLM] set CUDA_VISIBLE_DEVICES=${gpu}"
      echo "[VLM] nvidia-smi visible GPUs before launch:"
      nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader 2>/dev/null || true
      echo "----------------------------------------------------"

      env CUDA_VISIBLE_DEVICES=$gpu bash "${VQ_DIR}/run_VLM_judge.sh" \
        "${shard_model}" \
        "${video_dir}" \
        "${summary_json}" \
        all \
        "${CONFIG_PATH}" \
        "${shard_idx}" "${NUM_SHARDS}"
    } > "${VQ_DIR}/nohup_log/${shard_model}/output_run_VLM_judge_${shard_model}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &

    VLM_PIDS+=("$!")
  done

  # -------------------------------
  # Stage 2: 8-metrics in singularity
  # -------------------------------
  echo "[2/3] Run 8-metrics in singularity for ${shard_model}"
  echo "[2/3] NOTE: action_following is disabled in this script."

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

# 2A: 8-metrics
for shard_idx in \"\${!GPUS[@]}\"; do
  gpu=\${GPUS[\$shard_idx]}

  {
    echo \"---------------- 8-metrics shard GPU info ----------------\"
    echo \"[8METRICS] HOSTNAME=\$(hostname)\"
    echo \"[8METRICS] shard_idx=\${shard_idx}\"
    echo \"[8METRICS] NUM_SHARDS=\${NUM_SHARDS}\"
    echo \"[8METRICS] logical gpu=\${gpu}\"
    echo \"[8METRICS] HOST_SLURM_JOB_ID=\${HOST_SLURM_JOB_ID}\"
    echo \"[8METRICS] HOST_SLURM_JOB_NODELIST=\${HOST_SLURM_JOB_NODELIST}\"
    echo \"[8METRICS] HOST_SLURM_JOB_GPUS=\${HOST_SLURM_JOB_GPUS}\"
    echo \"[8METRICS] HOST_CUDA_VISIBLE_DEVICES=\${HOST_CUDA_VISIBLE_DEVICES}\"
    echo \"[8METRICS] inherited inner CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-NA}\"
    echo \"[8METRICS] set CUDA_VISIBLE_DEVICES=\${gpu}\"
    echo \"[8METRICS] nvidia-smi visible GPUs before launch:\"
    nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv,noheader 2>/dev/null || true
    echo \"-----------------------------------------------------------\"

    WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=\$gpu \
    bash '${VQ_DIR}/run_evaluation_8metrics.sh' \
      '${shard_model}' \
      '${gen_video_dir}' \
      '${summary_json}' \
      '${METRICS}' \
      '${CONFIG_PATH}' \
      \"\$shard_idx\" \"\$NUM_SHARDS\"
  } > '${VQ_DIR}/nohup_log/${shard_model}/output_run_evaluation_8metrics_${shard_model}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &

  EVAL_PIDS+=(\"\$!\")
done

# 2B: action_following disabled
echo '[2B] action_following disabled in this script for ${shard_model}.'

eval_failed=0
for pid in \"\${EVAL_PIDS[@]}\"; do
  if ! wait \"\$pid\"; then
    eval_failed=1
  fi
done

if [ \"\$eval_failed\" -ne 0 ]; then
  echo '[ERROR] One or more 8-metrics shards failed for ${shard_model}.'
  exit 1
fi

echo '[2/3] 8-metrics finished for ${shard_model}'
"

  # -------------------------------
  # Stage 3: wait host-side VLM jobs
  # -------------------------------
  echo "[3/3] Wait host-side VLM judge for ${shard_model}"

  vlm_failed=0
  for pid in "${VLM_PIDS[@]}"; do
    if ! wait "$pid"; then
      vlm_failed=1
    fi
  done

  if [ "$vlm_failed" -ne 0 ]; then
    echo "[ERROR] One or more VLM judge shards failed for ${shard_model}."
    exit 1
  fi

  echo "=================================================="
  echo "[DONE] MODEL_NAME=${shard_model}"
  echo "=================================================="
}

run_aggregate() {
  local base_model_name="${1:?missing base model name}"
  derive_names "${base_model_name}"
  mkdir -p "${LOG_DIR}"

  source "${CONDA_SH}"
  conda activate WorldArena

  python "${AGG_SCRIPT}" --model_name "${FORMAL_MODEL_NAME}"
}

submit_pipeline() {
  local base_model_name="${1:?missing base model name}"
  local num_servers="${2:?missing num_servers}"
  local shards_per_server="${3:-8}"
  local mail_user="${4:-}"
  local exclude_nodes="${5:-${EXCLUDE_NODES_DEFAULT}}"

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
  echo "[PIPELINE] NUM_SERVERS=${num_servers}"
  echo "[PIPELINE] SHARDS_PER_SERVER=${shards_per_server}"
  echo "[PIPELINE] WORKER_CPUS=${worker_cpus}"
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
  echo "=================================================="

  PREP_JOB_ID=$(sbatch --parsable \
    -J "prep_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    -o "${LOG_DIR}/prep_%j.out" \
    --wrap="bash ${SELF_PATH} prepare ${RAW_MODEL_NAME}")

  echo "[SUBMIT] prepare job id: ${PREP_JOB_ID}"

  SPLIT_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${PREP_JOB_ID} \
    -J "split_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    -o "${LOG_DIR}/split_%j.out" \
    --wrap="bash ${SELF_PATH} split ${RAW_MODEL_NAME} ${num_servers} ${shards_per_server}")

  echo "[SUBMIT] split job id: ${SPLIT_JOB_ID}"

  local -a WORKER_JOB_IDS=()
  local sid

  for ((sid=0; sid<num_servers; sid++)); do
    local shard_model="${FORMAL_MODEL_NAME}_${sid}"
    local jid

    jid=$(sbatch --parsable \
      --dependency=afterok:${SPLIT_JOB_ID} \
      -J "eval_${FORMAL_MODEL_NAME}_${sid}" \
      -p "${GPU_PARTITION}" \
      -N 1 -n 1 -c "${worker_cpus}" --gres=gpu:l40:${shards_per_server} \
      "${GPU_EXCLUDE_ARGS[@]}" \
      -o "${LOG_DIR}/worker_${sid}_%j.out" \
      --wrap="bash ${SELF_PATH} worker ${shard_model} ${shards_per_server}")

    WORKER_JOB_IDS+=("${jid}")
    echo "[SUBMIT] worker shard=${sid} job id: ${jid}"
  done

  local WORKER_DEP
  WORKER_DEP=$(IFS=:; echo "${WORKER_JOB_IDS[*]}")

  local -a AGG_MAIL_OPTS=()
  if [ -n "${mail_user}" ]; then
    AGG_MAIL_OPTS+=(--mail-type=END,FAIL --mail-user="${mail_user}")
  fi

  AGG_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${WORKER_DEP} \
    -J "agg_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    "${AGG_MAIL_OPTS[@]}" \
    -o "${LOG_DIR}/aggregate_%j.out" \
    --wrap="bash ${SELF_PATH} aggregate ${RAW_MODEL_NAME}")

  echo "[SUBMIT] aggregate job id: ${AGG_JOB_ID}"

  echo "=================================================="
  echo "[DONE] Pipeline submitted successfully."
  echo
  echo "Prepare job:   ${PREP_JOB_ID}"
  echo "Split job:     ${SPLIT_JOB_ID}"
  echo "Aggregate job: ${AGG_JOB_ID}"
  echo "Worker jobs:   ${WORKER_JOB_IDS[*]}"
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
  echo "Scancel Job:"
  echo "  scancel ${PREP_JOB_ID} ${SPLIT_JOB_ID} ${AGG_JOB_ID}$(printf " %s" "${WORKER_JOB_IDS[@]}")"
  echo
  echo "Logs:"
  echo "  ${LOG_DIR}"
  echo "=================================================="
}

MODE="submit"

if [ $# -gt 0 ]; then
  case "$1" in
    submit|prepare|split|worker|aggregate)
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
  aggregate)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_aggregate "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac