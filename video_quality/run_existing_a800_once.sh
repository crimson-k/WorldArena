#!/usr/bin/env bash
set -uo pipefail

# ============================================================
# Run WorldArena workers on existing A800 interactive allocations.
#
# Usage:
#   bash run_existing_a800_once.sh <base_model_name>
#
# Example:
#   bash /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/run_existing_a800_once.sh \
#     runid_35_equal_f121_step_16000_worldara_5_6_robotwin_state_480p_replay_5e_5_text_off_81_seed_43
#
# 注意：
#   1. 这个脚本不会 sbatch 新资源。
#   2. 这个脚本不会 scancel，因此不会释放你已有的 A800 interact 资源。
#   3. 它只是用 srun --jobid 把任务投到你已经占住的 A800 allocation 里。
# ============================================================

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"

PIPELINE_SCRIPT="${VQ_DIR}/run_all_in_one_action.sh"

CONDA_SH="/share/apps/miniconda3/etc/profile.d/conda.sh"

NUM_SERVERS=10
SHARDS_PER_SERVER=8

BASE_MODEL_NAME="${1:-}"

if [ -z "${BASE_MODEL_NAME}" ]; then
  echo "[ERROR] Missing base_model_name."
  echo
  echo "Usage:"
  echo "  bash $0 <base_model_name>"
  exit 1
fi

if [ ! -f "${PIPELINE_SCRIPT}" ]; then
  echo "[ERROR] PIPELINE_SCRIPT not found: ${PIPELINE_SCRIPT}"
  exit 1
fi

FORMAL_MODEL_NAME="${BASE_MODEL_NAME}_formal"

LOG_DIR="${VQ_DIR}/nohup_log/manual_existing_a800_${FORMAL_MODEL_NAME}"
mkdir -p "${LOG_DIR}"

# ============================================================
# 你当前已经占住的 A800 资源映射
# sid 0-9 分别对应 FORMAL_MODEL_NAME_0 到 FORMAL_MODEL_NAME_9
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
  echo "JOB_IDS length=${#JOB_IDS[@]}, NUM_SERVERS=${NUM_SERVERS}"
  exit 1
fi

if [ "${#NODES[@]}" -ne "${NUM_SERVERS}" ]; then
  echo "[ERROR] NODES length != NUM_SERVERS"
  echo "NODES length=${#NODES[@]}, NUM_SERVERS=${NUM_SERVERS}"
  exit 1
fi

echo "=================================================="
echo "[MANUAL-A800] BASE_MODEL_NAME=${BASE_MODEL_NAME}"
echo "[MANUAL-A800] FORMAL_MODEL_NAME=${FORMAL_MODEL_NAME}"
echo "[MANUAL-A800] NUM_SERVERS=${NUM_SERVERS}"
echo "[MANUAL-A800] SHARDS_PER_SERVER=${SHARDS_PER_SERVER}"
echo "[MANUAL-A800] PIPELINE_SCRIPT=${PIPELINE_SCRIPT}"
echo "[MANUAL-A800] LOG_DIR=${LOG_DIR}"
echo
echo "[MANUAL-A800] This script will NOT sbatch new resources."
echo "[MANUAL-A800] This script will NOT scancel existing jobs."
echo "[MANUAL-A800] Existing A800 allocations will remain allocated after this script exits."
echo "=================================================="

# ============================================================
# Step 1: prepare
# ============================================================

echo
echo "=================================================="
echo "[STEP 1/4] prepare"
echo "=================================================="

bash "${PIPELINE_SCRIPT}" prepare "${BASE_MODEL_NAME}" \
  > "${LOG_DIR}/prepare.out" 2>&1

PREP_STATUS=$?

if [ "${PREP_STATUS}" -ne 0 ]; then
  echo "[ERROR] prepare failed. Log:"
  echo "  ${LOG_DIR}/prepare.out"
  exit "${PREP_STATUS}"
fi

echo "[DONE] prepare finished."
echo "Log: ${LOG_DIR}/prepare.out"

# ============================================================
# Step 2: split
# ============================================================

echo
echo "=================================================="
echo "[STEP 2/4] split"
echo "=================================================="

bash "${PIPELINE_SCRIPT}" split "${BASE_MODEL_NAME}" "${NUM_SERVERS}" "${SHARDS_PER_SERVER}" \
  > "${LOG_DIR}/split.out" 2>&1

SPLIT_STATUS=$?

if [ "${SPLIT_STATUS}" -ne 0 ]; then
  echo "[ERROR] split failed. Log:"
  echo "  ${LOG_DIR}/split.out"
  exit "${SPLIT_STATUS}"
fi

echo "[DONE] split finished."
echo "Log: ${LOG_DIR}/split.out"

# ============================================================
# Step 3: run workers on existing A800 allocations
# ============================================================

echo
echo "=================================================="
echo "[STEP 3/4] launch workers on existing A800 allocations"
echo "=================================================="

WORKER_PIDS=()

launch_worker() {
  local sid="$1"
  local jobid="$2"
  local node="$3"
  local shard_model="${FORMAL_MODEL_NAME}_${sid}"
  local worker_log="${LOG_DIR}/worker_${sid}_${node}.out"

  echo "[LAUNCH] sid=${sid}, jobid=${jobid}, node=${node}, shard_model=${shard_model}"
  echo "         log=${worker_log}"

  srun \
    --jobid="${jobid}" \
    --overlap \
    -w "${node}" \
    -N 1 \
    -n 1 \
    -c 56 \
    bash -lc "
set -uo pipefail

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

  WORKER_PIDS+=("$!")
}

for sid in $(seq 0 $((NUM_SERVERS - 1))); do
  launch_worker "${sid}" "${JOB_IDS[$sid]}" "${NODES[$sid]}"
done

echo
echo "[INFO] All workers launched."
echo "[INFO] Waiting for all workers to finish..."
echo

worker_failed=0

for idx in "${!WORKER_PIDS[@]}"; do
  pid="${WORKER_PIDS[$idx]}"
  sid="${idx}"
  node="${NODES[$idx]}"
  log="${LOG_DIR}/worker_${sid}_${node}.out"

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

# ============================================================
# Step 4: aggregate
# ============================================================

echo
echo "=================================================="
echo "[STEP 4/4] aggregate"
echo "=================================================="

bash "${PIPELINE_SCRIPT}" aggregate "${BASE_MODEL_NAME}" \
  > "${LOG_DIR}/aggregate.out" 2>&1

AGG_STATUS=$?

if [ "${AGG_STATUS}" -ne 0 ]; then
  echo "[ERROR] aggregate failed. Log:"
  echo "  ${LOG_DIR}/aggregate.out"
  echo "[INFO] Existing A800 allocations are NOT cancelled by this script."
  exit "${AGG_STATUS}"
fi

echo "[DONE] aggregate finished."
echo "Log: ${LOG_DIR}/aggregate.out"

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
echo "  This script did not scancel any existing A800 allocation."
echo "  Your interact jobs should still be alive unless they ended for other reasons."
echo "=================================================="