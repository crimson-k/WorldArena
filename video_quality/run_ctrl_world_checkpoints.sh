#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_COMMAND="${CONDA_EXE:-conda}"
CORE_ENV="${CORE_ENV:-WorldArena}"
CORE_PYTHON="${CORE_PYTHON:-$("${CONDA_COMMAND}" run -n "${CORE_ENV}" python -c 'import sys; print(sys.executable)')}"
INPUT_BASE="${INPUT_BASE:-/data1/liuwenhao/Projects/world_simulator_baseline/outputs/ctrl_world_robotwin/inference}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/evaluation_runs/ctrl_world_robotwin}"
GPUS="${GPUS:-0,1,2,3}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-4}"
HEAVY_PROCESSES_PER_GPU="${HEAVY_PROCESSES_PER_GPU:-2}"
JEPA_GPU="${JEPA_GPU:-0}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"
CHECKPOINTS=(8000 12000 20000 22000 24000 26000)

mkdir -p "${OUTPUT_BASE}"

for checkpoint in "${CHECKPOINTS[@]}"; do
  input_root="${INPUT_BASE}/checkpoint-${checkpoint}/full-validation-500"
  output_root="${OUTPUT_BASE}/checkpoint-${checkpoint}/full-validation-500"
  reference_gt_trajectory_root=""
  if [ "${checkpoint}" != "8000" ]; then
    reference_gt_trajectory_root="${OUTPUT_BASE}/checkpoint-8000/full-validation-500/cache/jpeg_frames/gt_dataset/default"
  fi

  if [ ! -d "${input_root}" ]; then
    echo "Missing input directory: ${input_root}" >&2
    exit 1
  fi

  mkdir -p "${output_root}/logs"
  date -Is > "${output_root}/logs/batch.started"
  rm -f "${output_root}/logs/batch.completed" "${output_root}/logs/batch.failed"
  echo "Starting checkpoint-${checkpoint}: ${input_root}"

  if INPUT_ROOT="${input_root}" \
    OUTPUT_ROOT="${output_root}" \
    EXPECTED_COUNT="${EXPECTED_COUNT}" \
    GPUS="${GPUS}" \
    PROCESSES_PER_GPU="${PROCESSES_PER_GPU}" \
    HEAVY_PROCESSES_PER_GPU="${HEAVY_PROCESSES_PER_GPU}" \
    JEPA_GPU="${JEPA_GPU}" \
    RESUME_PREPARED=1 \
    REFERENCE_GT_TRAJECTORY_ROOT="${reference_gt_trajectory_root}" \
    bash "${PROJECT_ROOT}/video_quality/run_full_validation_500.sh"; then
    date -Is > "${output_root}/logs/batch.completed"
    echo "Completed checkpoint-${checkpoint}: ${output_root}"
  else
    status=$?
    printf '%s exit=%s\n' "$(date -Is)" "${status}" \
      > "${output_root}/logs/batch.failed"
    echo "Failed checkpoint-${checkpoint} with exit ${status}" >&2
    exit "${status}"
  fi
done

"${CORE_PYTHON}" "${PROJECT_ROOT}/video_quality/summarize_checkpoints.py" \
  --output-base "${OUTPUT_BASE}"
echo "All checkpoints completed: ${OUTPUT_BASE}"
