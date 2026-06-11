#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${1:?请传入 MODEL_NAME，例如: bash run_worldarena_pipeline_one_split.sh your_model_name}"

ROOT="/ssdfs/datahome/usersht/dev/lwh/WorldArena"
VQ_DIR="${ROOT}/video_quality"
CONFIG_PATH="${VQ_DIR}/config/config.yaml"
SUMMARY_JSON="${VQ_DIR}/data/${MODEL_NAME}/worldarena_auto_eval/summary.json"
VIDEO_DIR="${VQ_DIR}/data/${MODEL_NAME}/worldarena_auto_eval/${MODEL_NAME}_test_vlm"
GEN_VIDEO_DIR="${VQ_DIR}/data/${MODEL_NAME}/worldarena_auto_eval/${MODEL_NAME}_test"
METRICS="semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim,mse,lpips,fid,fvd"

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
echo "[0/2] Run data_process.py for ${MODEL_NAME}"
source "${CONDA_SH}"
conda activate WorldArena

python "${ROOT}/data_process.py" \
  --results_dir "${VQ_DIR}/data/${MODEL_NAME}" \
  --model_name "${MODEL_NAME}" \
  --num_shards "${NUM_SHARDS}"

echo "[0/2] data_process.py finished for ${MODEL_NAME}"

# -------------------------------
# Stage 1: VLM judge
# -------------------------------
echo "[1/2] Run VLM judge for ${MODEL_NAME}"

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

# wait
echo "[1/2] VLM judge finished for ${MODEL_NAME}"

# -------------------------------
# Stage 2: WorldArena eval in singularity
# -------------------------------
echo "[2/2] Run WorldArena eval for ${MODEL_NAME}"
module load Singularity/4.2.1

singularity exec --nv \
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

for shard_idx in \"\${!GPUS[@]}\"; do
  gpu=\${GPUS[\$shard_idx]}
  WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=\$gpu \
  bash '${VQ_DIR}/run_evaluation.sh' \
    '${MODEL_NAME}' \
    '${GEN_VIDEO_DIR}' \
    '${SUMMARY_JSON}' \
    '${METRICS}' \
    '${CONFIG_PATH}' \
    \"\$shard_idx\" \"\$NUM_SHARDS\" \
    > '${VQ_DIR}/nohup_log/${MODEL_NAME}/output_run_evaluation_${MODEL_NAME}_shard'\${shard_idx}'_gpu'\${gpu}'.log' 2>&1 &
done


echo '[2/2] WorldArena eval finished for ${MODEL_NAME}'
"

echo "=================================================="
echo "[DONE] MODEL_NAME=${MODEL_NAME}"
echo "=================================================="