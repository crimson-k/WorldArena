#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:?请传入 MODEL_NAME，例如: bash one_step_ok_8metrics.sh your_model_name}"

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"
CONFIG_PATH="${VQ_DIR}/config/config.yaml"
SUMMARY_JSON="${VQ_DIR}/data/${MODEL_NAME}/summary.json"
VIDEO_DIR="${VQ_DIR}/data/${MODEL_NAME}/${MODEL_NAME}_test_vlm"
GEN_VIDEO_DIR="${VQ_DIR}/data/${MODEL_NAME}/${MODEL_NAME}_test"
METRICS="aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality"

SIF_PATH="/share/home/usersht/images/ubuntu_2204.sif"
CONDA_SH="/share/apps/miniconda3/etc/profile.d/conda.sh"

GPUS=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#GPUS[@]}

mkdir -p "${VQ_DIR}/nohup_log/${MODEL_NAME}"

echo "=================================================="
echo "[START] MODEL_NAME=${MODEL_NAME}"
echo "=================================================="

cd "${VQ_DIR}"

# -------------------------------
# Stage 0: data_process
# -------------------------------
echo "[0/3] Skip data_process.py for ${MODEL_NAME} (assume already prepared)"

# -------------------------------
# Stage 1: VLM judge on host
# -------------------------------
echo "[1/3] Run VLM judge for ${MODEL_NAME}"

source "${CONDA_SH}"
conda activate WorldArena

for shard_idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_idx]}
  env CUDA_VISIBLE_DEVICES=$gpu bash "${VQ_DIR}/run_VLM_judge.sh" \
    "${MODEL_NAME}" \
    "${VIDEO_DIR}" \
    "${SUMMARY_JSON}" \
    all \
    "${CONFIG_PATH}" \
    "${shard_idx}" "${NUM_SHARDS}" \
    > "${VQ_DIR}/nohup_log/${MODEL_NAME}/output_run_VLM_judge_${MODEL_NAME}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
done

# -------------------------------
# Stage 2: 8-metrics + action_following in singularity
# -------------------------------
echo "[2/3] Run 8-metrics + action_following in singularity for ${MODEL_NAME}"

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

GPUS=(0 1 2 3 4 5 6 7)
NUM_SHARDS=\${#GPUS[@]}

# ---------------------------
# 2A: 8-metrics
# ---------------------------

for shard_idx in \"\${!GPUS[@]}\"; do
  gpu=\${GPUS[\$shard_idx]}
  WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=\$gpu \
  bash '${VQ_DIR}/run_evaluation_8metrics.sh' \
    '${MODEL_NAME}' \
    '${GEN_VIDEO_DIR}' \
    '${SUMMARY_JSON}' \
    '${METRICS}' \
    '${CONFIG_PATH}' \
    \"\$shard_idx\" \"\$NUM_SHARDS\" \
    > '${VQ_DIR}/nohup_log/${MODEL_NAME}/output_run_evaluation_8metrics_${MODEL_NAME}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &
done

# ---------------------------
# 2B: action_following
# ---------------------------
for shard_idx in \"\${!GPUS[@]}\"; do
  gpu=\${GPUS[\$shard_idx]}
  WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=\$gpu \
  bash '${VQ_DIR}/run_action_following.sh' \
    '${MODEL_NAME}' \
    '${GEN_VIDEO_DIR}' \
    '${SUMMARY_JSON}' \
    '${CONFIG_PATH}' \
    \"\$shard_idx\" \"\$NUM_SHARDS\" \
    > '${VQ_DIR}/nohup_log/${MODEL_NAME}/output_run_action_following_${MODEL_NAME}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &
done

# wait
echo '[2/3] 8-metrics + action_following finished for ${MODEL_NAME}'
"

# -------------------------------
# Stage 3: wait host-side VLM jobs
# -------------------------------
echo "[3/3] Wait host-side VLM judge for ${MODEL_NAME}"
# wait

echo "=================================================="
echo "[DONE] MODEL_NAME=${MODEL_NAME}"
echo "=================================================="