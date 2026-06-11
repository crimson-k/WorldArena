#!/usr/bin/env bash
set -euo pipefail

SELF_PATH="$(cd -- "$(dirname "$0")" && pwd)/$(basename "$0")"

CPU_PARTITION="intel"
GPU_PARTITION="L40"

# 默认排除节点。可以通过第 5 个命令行参数覆盖。
# 如果不想排除任何节点，运行时传 none。
EXCLUDE_NODES_DEFAULT="gpu4041,gpu4008"

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"

SPLIT_SCRIPT="${ROOT}/split.py"
# DATA_PROCESS_SCRIPT="${ROOT}/data_process_gtgt.py"
DATA_PROCESS_SCRIPT="${ROOT}/data_process.py"
AGG_SCRIPT="${VQ_DIR}/csv_results/aggregate_and_build_leaderboard.py"
RUN_VLM_SCRIPT="${VQ_DIR}/run_VLM_judge.sh"
RUN_EVAL_SCRIPT="${VQ_DIR}/run_evaluation.sh"
RUN_JEPA_SCRIPT="${VQ_DIR}/run_evaluation_JEPA.sh"

CONFIG_PATH="${VQ_DIR}/config/config.yaml"
SIF_PATH="/share/home/usersht/images/ubuntu_2204.sif"
CONDA_SH="/share/apps/miniconda3/etc/profile.d/conda.sh"
METRICS="semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim,mse,lpips,fid,fvd"

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
  bash ${SELF_PATH} <base_model_name> <num_splits> [worker_gpus] [mail_user] [exclude_nodes]
  bash ${SELF_PATH} <base_model_name> <num_splits> [mail_user] [exclude_nodes]
  bash ${SELF_PATH} submit <base_model_name> <num_splits> [worker_gpus] [mail_user] [exclude_nodes]

Internal modes:
  bash ${SELF_PATH} split <base_model_name> <num_splits>
  bash ${SELF_PATH} worker <shard_model_name> [worker_gpus]
  bash ${SELF_PATH} jepa <base_model_name>
  bash ${SELF_PATH} aggregate <base_model_name>

Examples:
  bash ${SELF_PATH} 500_480p_GTGT 12
  bash ${SELF_PATH} 500_480p_GTGT 12 4
  bash ${SELF_PATH} 500_480p_GTGT 12 4 your_email@qq.com
  bash ${SELF_PATH} 500_480p_GTGT 12 4 your_email@qq.com gpu4041
  bash ${SELF_PATH} 500_480p_GTGT 12 4 your_email@qq.com gpu4041,gpu4008
  bash ${SELF_PATH} 500_480p_GTGT 12 4 your_email@qq.com none
EOF
}

derive_names() {
  local base_model_name="$1"
  BASE_MODEL_NAME="${base_model_name}"
  BASE_MODEL_DIR="${VQ_DIR}/data/${BASE_MODEL_NAME}"
  LOG_DIR="${VQ_DIR}/nohup_log/pipeline_${BASE_MODEL_NAME}"
}

run_split() {
  local base_model_name="${1:?missing base model name}"
  local num_splits="${2:?missing num_splits}"

  is_positive_int "${num_splits}" || die "num_splits must be a positive integer, got: ${num_splits}"

  derive_names "${base_model_name}"
  mkdir -p "${LOG_DIR}"

  source "${CONDA_SH}"
  conda activate WorldArena

  python "${SPLIT_SCRIPT}" \
    "${BASE_MODEL_DIR}" \
    --num-splits "${num_splits}" \
    --overwrite
}

run_worker() {
  local shard_model="${1:?missing shard model name}"
  local worker_gpus="${2:-8}"

  is_positive_int "${worker_gpus}" || die "worker_gpus must be a positive integer, got: ${worker_gpus}"
  if [ "${worker_gpus}" -gt 8 ]; then
    die "worker_gpus must be <= 8, got: ${worker_gpus}"
  fi

  local summary_json="${VQ_DIR}/data/${shard_model}/worldarena_auto_eval/summary.json"
  local video_dir="${VQ_DIR}/data/${shard_model}/worldarena_auto_eval/${shard_model}_test_vlm"
  local gen_video_dir="${VQ_DIR}/data/${shard_model}/worldarena_auto_eval/${shard_model}_test"

  local ALL_GPUS=(0 1 2 3 4 5 6 7)
  local GPUS=("${ALL_GPUS[@]:0:${worker_gpus}}")
  local NUM_SHARDS=${#GPUS[@]}

  mkdir -p "${VQ_DIR}/nohup_log/${shard_model}"

  echo "=================================================="
  echo "[START] MODEL_NAME=${shard_model}"
  echo "[INFO] WORKER_GPUS=${worker_gpus}"
  echo "[INFO] NUM_SHARDS=${NUM_SHARDS}"
  echo "=================================================="

  cd "${VQ_DIR}"

  # -------------------------------
  # Stage 0: data_process
  # -------------------------------
  echo "[0/2] Run data_process.py for ${shard_model}"

  source "${CONDA_SH}"
  conda activate WorldArena

  python "${DATA_PROCESS_SCRIPT}" \
    --results_dir "${VQ_DIR}/data/${shard_model}" \
    --model_name "${shard_model}" \
    --num_shards "${NUM_SHARDS}" \
    > "${VQ_DIR}/nohup_log/${shard_model}/output_data_process_${shard_model}.log" 2>&1

  echo "[0/2] data_process.py finished for ${shard_model}"
  echo "[0/2] data_process log: ${VQ_DIR}/nohup_log/${shard_model}/output_data_process_${shard_model}.log"

  # -------------------------------
  # Stage 1: VLM judge
  # -------------------------------
  echo "[1/2] Run VLM judge for ${shard_model}"

  VLM_PIDS=()

  for shard_idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$shard_idx]}
    env CUDA_VISIBLE_DEVICES=$gpu bash "${RUN_VLM_SCRIPT}" \
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
  # Stage 2: WorldArena eval in singularity
  # -------------------------------
  echo "[2/2] Run WorldArena eval for ${shard_model}"

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
  bash '${RUN_EVAL_SCRIPT}' \
    '${shard_model}' \
    '${gen_video_dir}' \
    '${summary_json}' \
    '${METRICS}' \
    '${CONFIG_PATH}' \
    \"\$shard_idx\" \"\$NUM_SHARDS\" \
    > '${VQ_DIR}/nohup_log/${shard_model}/output_run_evaluation_${shard_model}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &
  EVAL_PIDS+=(\"\$!\")
done

eval_failed=0
for pid in \"\${EVAL_PIDS[@]}\"; do
  if ! wait \"\$pid\"; then
    eval_failed=1
  fi
done

if [ \"\$eval_failed\" -ne 0 ]; then
  echo '[ERROR] One or more WorldArena eval shards failed for ${shard_model}.'
  exit 1
fi

echo '[2/2] WorldArena eval finished for ${shard_model}'
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

run_jepa() {
  local base_model_name="${1:?missing base model name}"
  derive_names "${base_model_name}"

  source "${CONDA_SH}"
  conda activate WorldArena

  echo "[JEPA] Run data_process.py for base model: ${BASE_MODEL_NAME}"
  python "${DATA_PROCESS_SCRIPT}" \
    --results_dir "${BASE_MODEL_DIR}" \
    --model_name "${BASE_MODEL_NAME}" \
    > "${VQ_DIR}/nohup_log/output_data_process_JEPA_${BASE_MODEL_NAME}.log" 2>&1

  echo "[JEPA] data_process log: ${VQ_DIR}/nohup_log/output_data_process_JEPA_${BASE_MODEL_NAME}.log"

  echo "[JEPA] Run JEPA for base model: ${BASE_MODEL_NAME}"
  cd "${ROOT}"

  unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT SLURM_PROCID SLURM_LOCALID SLURM_NTASKS

  env CUDA_VISIBLE_DEVICES=0 bash "${RUN_JEPA_SCRIPT}" \
    "${BASE_MODEL_NAME}" \
    "${BASE_MODEL_DIR}/worldarena_auto_eval/${BASE_MODEL_NAME}_test_vlm" \
    "${BASE_MODEL_DIR}/worldarena_auto_eval/gt_videos" \
    > "${VQ_DIR}/nohup_log/output_run_evaluation_JEPA_${BASE_MODEL_NAME}.log" 2>&1
}

run_aggregate() {
  local base_model_name="${1:?missing base model name}"
  derive_names "${base_model_name}"

  source "${CONDA_SH}"
  conda activate WorldArena

  python "${AGG_SCRIPT}" --model_name "${BASE_MODEL_NAME}"
}

submit_pipeline() {
  local base_model_name="${1:?missing base model name}"
  local num_splits="${2:?missing num_splits}"
  local worker_gpus="8"
  local mail_user=""
  local exclude_nodes="${EXCLUDE_NODES_DEFAULT}"

  is_positive_int "${num_splits}" || die "num_splits must be a positive integer, got: ${num_splits}"

  # Compatible argument parsing:
  #   bash script.sh model 12
  #   bash script.sh model 12 email
  #   bash script.sh model 12 email exclude_nodes
  #   bash script.sh model 12 4
  #   bash script.sh model 12 4 email
  #   bash script.sh model 12 4 email exclude_nodes
  if [ $# -ge 3 ]; then
    if is_positive_int "${3}"; then
      worker_gpus="${3}"
      mail_user="${4:-}"
      exclude_nodes="${5:-${EXCLUDE_NODES_DEFAULT}}"
    else
      worker_gpus="8"
      mail_user="${3}"
      exclude_nodes="${4:-${EXCLUDE_NODES_DEFAULT}}"
    fi
  fi

  is_positive_int "${worker_gpus}" || die "worker_gpus must be a positive integer, got: ${worker_gpus}"
  if [ "${worker_gpus}" -gt 8 ]; then
    die "worker_gpus must be <= 8, got: ${worker_gpus}"
  fi

  if [ "${exclude_nodes}" = "none" ] || [ "${exclude_nodes}" = "NONE" ]; then
    exclude_nodes=""
  fi

  derive_names "${base_model_name}"
  mkdir -p "${LOG_DIR}"

  local worker_cpus=$((worker_gpus * 7))

  local -a GPU_EXCLUDE_ARGS=()
  if [ -n "${exclude_nodes}" ]; then
    GPU_EXCLUDE_ARGS+=(--exclude="${exclude_nodes}")
  fi

  echo "=================================================="
  echo "[PIPELINE] BASE_MODEL_NAME=${BASE_MODEL_NAME}"
  echo "[PIPELINE] NUM_SPLITS=${num_splits}"
  echo "[PIPELINE] WORKER_GPUS=${worker_gpus}"
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

  SPLIT_JOB_ID=$(sbatch --parsable \
    -J "split_${BASE_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    -o "${LOG_DIR}/split_%j.out" \
    --wrap="bash ${SELF_PATH} split ${BASE_MODEL_NAME} ${num_splits}")

  echo "[SUBMIT] split job id: ${SPLIT_JOB_ID}"

  local -a WORKER_JOB_IDS=()
  local sid
  for ((sid=0; sid<num_splits; sid++)); do
    local shard_model="${BASE_MODEL_NAME}_${sid}"
    local jid
    jid=$(sbatch --parsable \
      --dependency=afterok:${SPLIT_JOB_ID} \
      -J "eval_${BASE_MODEL_NAME}_${sid}" \
      -p "${GPU_PARTITION}" \
      -N 1 -n 1 -c "${worker_cpus}" --gres=gpu:l40:${worker_gpus} \
      "${GPU_EXCLUDE_ARGS[@]}" \
      -o "${LOG_DIR}/worker_${sid}_%j.out" \
      --wrap="bash ${SELF_PATH} worker ${shard_model} ${worker_gpus}")
    WORKER_JOB_IDS+=("${jid}")
    echo "[SUBMIT] worker shard=${sid} job id: ${jid}"
  done

  local WORKER_DEP
  WORKER_DEP=$(IFS=:; echo "${WORKER_JOB_IDS[*]}")

  JEPA_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${WORKER_DEP} \
    -J "jepa_${BASE_MODEL_NAME}" \
    -p "${GPU_PARTITION}" \
    -N 1 -n 1 -c 7 --gres=gpu:l40:1 \
    "${GPU_EXCLUDE_ARGS[@]}" \
    -o "${LOG_DIR}/jepa_%j.out" \
    --wrap="bash ${SELF_PATH} jepa ${BASE_MODEL_NAME}")

  echo "[SUBMIT] JEPA job id: ${JEPA_JOB_ID}"

  local -a AGG_MAIL_OPTS=()
  if [ -n "${mail_user}" ]; then
    AGG_MAIL_OPTS+=(--mail-type=END,FAIL --mail-user="${mail_user}")
  fi

  AGG_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${JEPA_JOB_ID} \
    -J "agg_${BASE_MODEL_NAME}" \
    -p "${CPU_PARTITION}" \
    -N 1 -n 1 -c 7 \
    "${AGG_MAIL_OPTS[@]}" \
    -o "${LOG_DIR}/aggregate_%j.out" \
    --wrap="bash ${SELF_PATH} aggregate ${BASE_MODEL_NAME}")

  echo "[SUBMIT] aggregate job id: ${AGG_JOB_ID}"

  echo "=================================================="
  echo "[DONE] Pipeline submitted successfully."
  echo
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
  echo "  squeue -j ${SPLIT_JOB_ID},${JEPA_JOB_ID},${AGG_JOB_ID}$(printf ",%s" "${WORKER_JOB_IDS[@]}")"
  echo "Scancel Job:"
  echo "  scancel ${SPLIT_JOB_ID} ${JEPA_JOB_ID} ${AGG_JOB_ID}$(printf " %s" "${WORKER_JOB_IDS[@]}")"
  echo
  echo "Logs:"
  echo "  ${LOG_DIR}"
  echo "=================================================="
}

MODE="submit"
if [ $# -gt 0 ]; then
  case "$1" in
    submit|split|worker|jepa|aggregate)
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