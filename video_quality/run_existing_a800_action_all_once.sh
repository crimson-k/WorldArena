#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run WorldArena ALL pipeline on existing A800 interactive allocations.
#
# Usage:
#   bash run_existing_a800_action_all_once.sh <base_model_name>
#
# Example:
#   bash /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/run_existing_a800_action_all_once.sh \
#     runid_xxx
#
# This script:
#   1. Does NOT sbatch new jobs.
#   2. Does NOT salloc new resources.
#   3. Does NOT scancel anything.
#   4. Reuses existing A800 allocations through srun --jobid.
#   5. Runs:
#        prepare -> split -> workers -> jepa -> aggregate
# ============================================================

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"

# 这里填你的 ALL 主脚本路径，也就是你刚才贴的那个脚本
PIPELINE_SCRIPT="${VQ_DIR}/run_all_in_one_action_ALL.sh"

BASE_MODEL_NAME="${1:-}"

NUM_SERVERS=10
SHARDS_PER_SERVER=8

if [ -z "${BASE_MODEL_NAME}" ]; then
  echo "[ERROR] Missing base_model_name."
  echo
  echo "Usage:"
  echo "  bash $0 <base_model_name>"
  exit 1
fi

if [ ! -f "${PIPELINE_SCRIPT}" ]; then
  echo "[ERROR] PIPELINE_SCRIPT not found:"
  echo "  ${PIPELINE_SCRIPT}"
  echo
  echo "Please check whether your ALL pipeline script is named run_all_in_one_action_ALL.sh."
  exit 1
fi

FORMAL_MODEL_NAME="${BASE_MODEL_NAME}_formal"
LOG_DIR="${VQ_DIR}/nohup_log/manual_existing_a800_ALL_${FORMAL_MODEL_NAME}"

mkdir -p "${LOG_DIR}"

# ============================================================
# Existing A800 allocations
#
# sid 0-9 对应：
#   ${FORMAL_MODEL_NAME}_0
#   ${FORMAL_MODEL_NAME}_1
#   ...
#   ${FORMAL_MODEL_NAME}_9
#
# 这些 jobid/node 来自你已经申请好的 A800 interact 资源。
# 如果以后 A800 资源变了，只需要改这里。
# ============================================================

JOB_IDS=(
  1509864
  1499037
  1499021
  1495536
  1495535
  1480013
  1361071
  1361071
  1361071
  1360018
)

NODES=(
  gpu8009
  gpu8012
  gpu8011
  gpu8014
  gpu8013
  gpu8004
  gpu8005
  gpu8006
  gpu8008
  gpu8007
)

if [ "${#JOB_IDS[@]}" -ne "${NUM_SERVERS}" ]; then
  echo "[ERROR] JOB_IDS length != NUM_SERVERS"
  echo "JOB_IDS length=${#JOB_IDS[@]}"
  echo "NUM_SERVERS=${NUM_SERVERS}"
  exit 1
fi

if [ "${#NODES[@]}" -ne "${NUM_SERVERS}" ]; then
  echo "[ERROR] NODES length != NUM_SERVERS"
  echo "NODES length=${#NODES[@]}"
  echo "NUM_SERVERS=${NUM_SERVERS}"
  exit 1
fi

print_header() {
  echo "=================================================="
  echo "[MANUAL-A800-ALL] BASE_MODEL_NAME=${BASE_MODEL_NAME}"
  echo "[MANUAL-A800-ALL] FORMAL_MODEL_NAME=${FORMAL_MODEL_NAME}"
  echo "[MANUAL-A800-ALL] NUM_SERVERS=${NUM_SERVERS}"
  echo "[MANUAL-A800-ALL] SHARDS_PER_SERVER=${SHARDS_PER_SERVER}"
  echo "[MANUAL-A800-ALL] PIPELINE_SCRIPT=${PIPELINE_SCRIPT}"
  echo "[MANUAL-A800-ALL] LOG_DIR=${LOG_DIR}"
  echo
  echo "[MANUAL-A800-ALL] This script will NOT sbatch new resources."
  echo "[MANUAL-A800-ALL] This script will NOT salloc new resources."
  echo "[MANUAL-A800-ALL] This script will NOT scancel existing jobs."
  echo "[MANUAL-A800-ALL] Existing A800 allocations will remain alive after this script exits."
  echo "=================================================="
}

print_mapping() {
  echo
  echo "================ A800 mapping ================"
  for sid in $(seq 0 $((NUM_SERVERS - 1))); do
    echo "sid=${sid}  jobid=${JOB_IDS[$sid]}  node=${NODES[$sid]}  shard=${FORMAL_MODEL_NAME}_${sid}"
  done
  echo "=============================================="
  echo
}

run_prepare() {
  echo
  echo "=================================================="
  echo "[STEP 1/5] prepare"
  echo "=================================================="

  bash "${PIPELINE_SCRIPT}" prepare "${BASE_MODEL_NAME}" \
    > "${LOG_DIR}/prepare.out" 2>&1

  echo "[DONE] prepare finished."
  echo "Log: ${LOG_DIR}/prepare.out"
}

run_split() {
  echo
  echo "=================================================="
  echo "[STEP 2/5] split"
  echo "=================================================="

  bash "${PIPELINE_SCRIPT}" split "${BASE_MODEL_NAME}" "${NUM_SERVERS}" "${SHARDS_PER_SERVER}" \
    > "${LOG_DIR}/split.out" 2>&1

  echo "[DONE] split finished."
  echo "Log: ${LOG_DIR}/split.out"
}

launch_one_worker() {
  local sid="$1"
  local jobid="$2"
  local node="$3"

  local shard_model="${FORMAL_MODEL_NAME}_${sid}"
  local worker_log="${LOG_DIR}/worker_${sid}_${node}.out"

  echo "[LAUNCH-WORKER] sid=${sid}, jobid=${jobid}, node=${node}, shard_model=${shard_model}"
  echo "                log=${worker_log}"

  srun \
    --jobid="${jobid}" \
    --overlap \
    -w "${node}" \
    -N 1 \
    -n 1 \
    -c 56 \
    bash -lc "
set -euo pipefail

echo '=================================================='
echo '[EXISTING-A800-WORKER] START'
echo '[EXISTING-A800-WORKER] shard_id=${sid}'
echo '[EXISTING-A800-WORKER] shard_model=${shard_model}'
echo '[EXISTING-A800-WORKER] expected_node=${node}'
echo '[EXISTING-A800-WORKER] hostname='\"\$(hostname)\"
echo '[EXISTING-A800-WORKER] external_jobid=${jobid}'
echo '[EXISTING-A800-WORKER] SLURM_JOB_ID='\${SLURM_JOB_ID:-NA}
echo '[EXISTING-A800-WORKER] SLURM_STEP_ID='\${SLURM_STEP_ID:-NA}
echo '[EXISTING-A800-WORKER] SLURM_JOB_NODELIST='\${SLURM_JOB_NODELIST:-NA}
echo '[EXISTING-A800-WORKER] SLURM_STEP_NODELIST='\${SLURM_STEP_NODELIST:-NA}
echo '[EXISTING-A800-WORKER] SLURM_JOB_GPUS='\${SLURM_JOB_GPUS:-NA}
echo '[EXISTING-A800-WORKER] CUDA_VISIBLE_DEVICES='\${CUDA_VISIBLE_DEVICES:-NA}
echo '[EXISTING-A800-WORKER] NVIDIA_VISIBLE_DEVICES='\${NVIDIA_VISIBLE_DEVICES:-NA}
echo '[EXISTING-A800-WORKER] nvidia-smi -L:'
nvidia-smi -L || true
echo '=================================================='

bash '${PIPELINE_SCRIPT}' worker '${shard_model}' '${SHARDS_PER_SERVER}'

status=\$?

echo '=================================================='
echo '[EXISTING-A800-WORKER] END'
echo '[EXISTING-A800-WORKER] shard_id=${sid}'
echo '[EXISTING-A800-WORKER] shard_model=${shard_model}'
echo '[EXISTING-A800-WORKER] status='\${status}
echo '=================================================='

exit \${status}
" > "${worker_log}" 2>&1 &
}

run_workers() {
  echo
  echo "=================================================="
  echo "[STEP 3/5] launch workers on existing A800 allocations"
  echo "=================================================="

  WORKER_PIDS=()

  for sid in $(seq 0 $((NUM_SERVERS - 1))); do
    launch_one_worker "${sid}" "${JOB_IDS[$sid]}" "${NODES[$sid]}"
    WORKER_PIDS+=("$!")
  done

  echo
  echo "[INFO] All workers launched."
  echo "[INFO] Waiting for all workers to finish..."
  echo

  local worker_failed=0

  for idx in "${!WORKER_PIDS[@]}"; do
    local pid="${WORKER_PIDS[$idx]}"
    local sid="${idx}"
    local node="${NODES[$idx]}"
    local log="${LOG_DIR}/worker_${sid}_${node}.out"

    if wait "${pid}"; then
      echo "[DONE] worker sid=${sid}, node=${node}"
    else
      echo "[FAILED] worker sid=${sid}, node=${node}"
      echo "         log=${log}"
      worker_failed=1
    fi
  done

  if [ "${worker_failed}" -ne 0 ]; then
    echo
    echo "=================================================="
    echo "[ERROR] One or more workers failed."
    echo "[INFO] Existing A800 allocations are NOT cancelled by this script."
    echo "[INFO] Check logs:"
    echo "  ${LOG_DIR}"
    echo "=================================================="
    exit 1
  fi

  echo
  echo "[DONE] All workers finished successfully."
}

run_jepa_on_existing_a800() {
  echo
  echo "=================================================="
  echo "[STEP 4/5] run JEPA on existing A800 allocation"
  echo "=================================================="

  # JEPA 只需要 1 张卡，这里复用第一个 A800 节点。
  local jepa_jobid="${JOB_IDS[0]}"
  local jepa_node="${NODES[0]}"
  local jepa_log="${LOG_DIR}/jepa_${jepa_node}.out"

  echo "[LAUNCH-JEPA] jobid=${jepa_jobid}, node=${jepa_node}"
  echo "              log=${jepa_log}"

  srun \
    --jobid="${jepa_jobid}" \
    --overlap \
    -w "${jepa_node}" \
    -N 1 \
    -n 1 \
    -c 7 \
    bash -lc "
set -euo pipefail

echo '=================================================='
echo '[EXISTING-A800-JEPA] START'
echo '[EXISTING-A800-JEPA] base_model=${BASE_MODEL_NAME}'
echo '[EXISTING-A800-JEPA] formal_model=${FORMAL_MODEL_NAME}'
echo '[EXISTING-A800-JEPA] expected_node=${jepa_node}'
echo '[EXISTING-A800-JEPA] hostname='\"\$(hostname)\"
echo '[EXISTING-A800-JEPA] external_jobid=${jepa_jobid}'
echo '[EXISTING-A800-JEPA] SLURM_JOB_ID='\${SLURM_JOB_ID:-NA}
echo '[EXISTING-A800-JEPA] SLURM_STEP_ID='\${SLURM_STEP_ID:-NA}
echo '[EXISTING-A800-JEPA] SLURM_JOB_NODELIST='\${SLURM_JOB_NODELIST:-NA}
echo '[EXISTING-A800-JEPA] SLURM_STEP_NODELIST='\${SLURM_STEP_NODELIST:-NA}
echo '[EXISTING-A800-JEPA] SLURM_JOB_GPUS='\${SLURM_JOB_GPUS:-NA}
echo '[EXISTING-A800-JEPA] CUDA_VISIBLE_DEVICES before main script='\${CUDA_VISIBLE_DEVICES:-NA}
echo '[EXISTING-A800-JEPA] nvidia-smi -L:'
nvidia-smi -L || true
echo '=================================================='

bash '${PIPELINE_SCRIPT}' jepa '${BASE_MODEL_NAME}'

status=\$?

echo '=================================================='
echo '[EXISTING-A800-JEPA] END'
echo '[EXISTING-A800-JEPA] status='\${status}
echo '=================================================='

exit \${status}
" > "${jepa_log}" 2>&1

  echo "[DONE] JEPA finished."
  echo "Log: ${jepa_log}"
}

run_aggregate() {
  echo
  echo "=================================================="
  echo "[STEP 5/5] aggregate"
  echo "=================================================="

  bash "${PIPELINE_SCRIPT}" aggregate "${BASE_MODEL_NAME}" \
    > "${LOG_DIR}/aggregate.out" 2>&1

  echo "[DONE] aggregate finished."
  echo "Log: ${LOG_DIR}/aggregate.out"
}

print_finish() {
  echo
  echo "=================================================="
  echo "[ALL DONE]"
  echo "BASE_MODEL_NAME=${BASE_MODEL_NAME}"
  echo "FORMAL_MODEL_NAME=${FORMAL_MODEL_NAME}"
  echo
  echo "Logs:"
  echo "  ${LOG_DIR}"
  echo
  echo "Worker detail logs:"
  echo "  ${VQ_DIR}/nohup_log/${FORMAL_MODEL_NAME}_*/"
  echo
  echo "Important:"
  echo "  This script did not sbatch any new resource."
  echo "  This script did not scancel any existing A800 allocation."
  echo "  Your interact jobs should still be alive unless they ended for other reasons."
  echo "=================================================="
}

main() {
  print_header
  print_mapping
  run_prepare
  run_split
  run_workers
  run_jepa_on_existing_a800
  run_aggregate
  print_finish
}

main