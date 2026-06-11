#!/usr/bin/env bash
set -euo pipefail

SELF_PATH="$(cd -- "$(dirname "$0")" && pwd)/$(basename "$0")"

CPU_PARTITION="intel"
GPU_PARTITION="L40"

# 默认排除节点。可以通过第 5 个命令行参数覆盖。
# 例如最后一个参数传 gpu4003 或 gpu4003,gpu4041
# 如果不想排除任何节点，传 none。
EXCLUDE_NODES_DEFAULT=""

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"

PREP_SCRIPT="${VQ_DIR}/prepare_eval_package_ALL.py"
SPLIT_SCRIPT="${VQ_DIR}/split_formal_package_shards.py"
AGG_SCRIPT="${VQ_DIR}/csv_results/aggregate_and_build_leaderboard_action_ALL.py"
DATA_PROCESS_SCRIPT="${ROOT}/data_process.py"
RUN_JEPA_SCRIPT="${VQ_DIR}/run_evaluation_JEPA.sh"

CONFIG_PATH="${VQ_DIR}/config/config_ALL.yaml"
SIF_PATH="/share/home/usersht/images/ubuntu_2204.sif"
CONDA_SH="/share/apps/miniconda3/etc/profile.d/conda.sh"

# 这里只放非 JEPA 的 3 个指标。
# JEPA Similarity 由 run_jepa() 单独执行。
METRICS="semantic_alignment,depth_accuracy,trajectory_accuracy"

# 这个脚本固定不跑 action_following。
RUN_ACTION_FOLLOWING=0

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

is_positive_int() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && [ "${1}" -gt 0 ]
}

usage() {
  cat <<EOF
Usage:
  bash ${SELF_PATH} <base_model_name> <num_servers> [shards_per_server] [mail_user] [exclude_nodes]
  bash ${SELF_PATH} submit <base_model_name> <num_servers> [shards_per_server] [mail_user] [exclude_nodes]

Internal modes:
  bash ${SELF_PATH} prepare <base_model_name>
  bash ${SELF_PATH} split <base_model_name> <num_servers> [shards_per_server]
  bash ${SELF_PATH} worker <formal_model_name_with_server_suffix> [shards_per_server]
  bash ${SELF_PATH} jepa <base_model_name>
  bash ${SELF_PATH} aggregate <base_model_name>

Examples:
  bash ${SELF_PATH} runid_xxx 25 4
  bash ${SELF_PATH} runid_xxx 25 4 your_email@qq.com
  bash ${SELF_PATH} runid_xxx 25 4 your_email@qq.com gpu4003
  bash ${SELF_PATH} runid_xxx 25 4 your_email@qq.com gpu4003,gpu4041
  bash ${SELF_PATH} runid_xxx 25 4 your_email@qq.com none
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
  LOG_DIR="${VQ_DIR}/nohup_log/pipeline_${FORMAL_MODEL_NAME}_4metrics"
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
    --id1 usersht
    --image_mode keep
    --overwrite
  )

  echo "[prepare] RUN_ACTION_FOLLOWING=${RUN_ACTION_FOLLOWING}; skip example_test_1/example_test_2."

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
  echo "[INFO] METRICS=${METRICS}"
  echo "[INFO] action_following disabled"
  echo "=================================================="

  cd "${VQ_DIR}"

  echo "[0/3] Skip data_process.py for ${shard_model} (assume already prepared)"

  # -------------------------------
  # Stage 1: VLM judge
  # -------------------------------
  # semantic_alignment 依赖 VLM judge 结果。
  # 这里仍然使用 all，保持和你原始流程兼容。
  echo "[1/3] Run VLM judge for ${shard_model}"

  source "${CONDA_SH}"
  conda activate WorldArena

  VLM_PIDS=()

  for shard_idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$shard_idx]}
    env CUDA_VISIBLE_DEVICES=$gpu bash "${VQ_DIR}/run_VLM_judge.sh" \
      "${shard_model}" \
      "${video_dir}" \
      "${summary_json}" \
      all \
      "${CONFIG_PATH}" \
      "${shard_idx}" "${NUM_SHARDS}" \
      > "${VQ_DIR}/nohup_log/${shard_model}/output_run_VLM_judge_${shard_model}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
    VLM_PIDS+=("$!")
  done

  # -------------------------------
  # Stage 2: selected metrics in singularity
  # -------------------------------
  echo "[2/3] Run selected metrics in singularity for ${shard_model}"

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

EVAL_PIDS=()

for shard_idx in \"\${!GPUS[@]}\"; do
  gpu=\${GPUS[\$shard_idx]}
  WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=\$gpu \
  bash '${VQ_DIR}/run_evaluation_8metrics_ALL.sh' \
    '${shard_model}' \
    '${gen_video_dir}' \
    '${summary_json}' \
    '${METRICS}' \
    '${CONFIG_PATH}' \
    \"\$shard_idx\" \"\$NUM_SHARDS\" \
    > '${VQ_DIR}/nohup_log/${shard_model}/output_run_selected_metrics_ALL_${shard_model}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &
  EVAL_PIDS+=(\"\$!\")
done

eval_failed=0
for pid in \"\${EVAL_PIDS[@]}\"; do
  if ! wait \"\$pid\"; then
    eval_failed=1
  fi
done

if [ \"\$eval_failed\" -ne 0 ]; then
  echo '[ERROR] One or more selected metric shards failed for ${shard_model}.'
  exit 1
fi

echo '[2/3] selected metrics finished for ${shard_model}'
"

  # -------------------------------
  # Stage 3: wait VLM judge
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

run_jepa() {
  local base_model_name="${1:?missing base model name}"
  derive_names "${base_model_name}"

  mkdir -p "${LOG_DIR}"

  source "${CONDA_SH}"
  conda activate WorldArena

  # echo "[JEPA] Run data_process.py for formal model: ${FORMAL_MODEL_NAME}"
  # python "${DATA_PROCESS_SCRIPT}" \
  #   --results_dir "${OUTPUT_ROOT}" \
  #   --model_name "${FORMAL_MODEL_NAME}" \
  #   > "${LOG_DIR}/output_data_process_JEPA_${FORMAL_MODEL_NAME}.log" 2>&1 || true

  # echo "[JEPA] data_process log: ${LOG_DIR}/output_data_process_JEPA_${FORMAL_MODEL_NAME}.log"

  echo "[JEPA] Run JEPA for formal model: ${FORMAL_MODEL_NAME}"
  cd "${ROOT}"

  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID SLURM_NTASKS

  env CUDA_VISIBLE_DEVICES=0 bash "${RUN_JEPA_SCRIPT}" \
    "${FORMAL_MODEL_NAME}" \
    "${OUTPUT_ROOT}/${FORMAL_MODEL_NAME}_test_vlm" \
    "${OUTPUT_ROOT}/gt_videos" \
    > "${LOG_DIR}/output_run_evaluation_JEPA_${FORMAL_MODEL_NAME}.log" 2>&1
}

run_aggregate() {
  local base_model_name="${1:?missing base model name}"
  derive_names "${base_model_name}"

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
  echo "[PIPELINE] METRICS=${METRICS}"
  echo "[PIPELINE] JEPA=enabled"
  echo "[PIPELINE] ACTION_FOLLOWING=disabled"
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
    -J "prep4_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    -o "${LOG_DIR}/prep_%j.out" \
    --wrap="bash ${SELF_PATH} prepare ${RAW_MODEL_NAME}")

  echo "[SUBMIT] prepare job id: ${PREP_JOB_ID}"

  SPLIT_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${PREP_JOB_ID} \
    -J "split4_${FORMAL_MODEL_NAME}" \
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
      -J "eval4_${FORMAL_MODEL_NAME}_${sid}" \
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

  JEPA_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${WORKER_DEP} \
    -J "jepa4_${FORMAL_MODEL_NAME}" \
    -p "${GPU_PARTITION}" \
    -N 1 -n 1 -c 7 --gres=gpu:l40:1 \
    "${GPU_EXCLUDE_ARGS[@]}" \
    -o "${LOG_DIR}/jepa_%j.out" \
    --wrap="bash ${SELF_PATH} jepa ${RAW_MODEL_NAME}")

  echo "[SUBMIT] JEPA job id: ${JEPA_JOB_ID}"

  local -a AGG_MAIL_OPTS=()
  if [ -n "${mail_user}" ]; then
    AGG_MAIL_OPTS+=(--mail-type=END,FAIL --mail-user="${mail_user}")
  fi

  AGG_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${JEPA_JOB_ID} \
    -J "agg4_${FORMAL_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    "${AGG_MAIL_OPTS[@]}" \
    -o "${LOG_DIR}/aggregate_%j.out" \
    --wrap="bash ${SELF_PATH} aggregate ${RAW_MODEL_NAME}")

  echo "[SUBMIT] aggregate job id: ${AGG_JOB_ID}"

  echo "=================================================="
  echo "[DONE] 4-metrics pipeline submitted successfully."
  echo
  echo "Prepare job:   ${PREP_JOB_ID}"
  echo "Split job:     ${SPLIT_JOB_ID}"
  echo "Worker jobs:   ${WORKER_JOB_IDS[*]}"
  echo "JEPA job:      ${JEPA_JOB_ID}"
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
  echo "  squeue -j ${PREP_JOB_ID},${SPLIT_JOB_ID},${JEPA_JOB_ID},${AGG_JOB_ID}$(printf ",%s" "${WORKER_JOB_IDS[@]}")"
  echo "Scancel Job:"
  echo "  scancel ${PREP_JOB_ID} ${SPLIT_JOB_ID} ${JEPA_JOB_ID} ${AGG_JOB_ID}$(printf " %s" "${WORKER_JOB_IDS[@]}")"
  echo
  echo "Logs:"
  echo "  ${LOG_DIR}"
  echo "=================================================="
}

MODE="submit"
if [ $# -gt 0 ]; then
  case "$1" in
    submit|prepare|split|worker|jepa|aggregate)
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
  jepa)
    [ $# -ge 1 ] || { usage; exit 1; }
    run_jepa "$@"
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