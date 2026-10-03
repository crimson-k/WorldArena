#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_COMMAND="${CONDA_EXE:-conda}"
CORE_ENV="${CORE_ENV:-WorldArena}"
JEPA_ENV="${JEPA_ENV:-WorldArena_JEPA}"
CORE_PYTHON="${CORE_PYTHON:-$("${CONDA_COMMAND}" run -n "${CORE_ENV}" python -c 'import sys; print(sys.executable)')}"
JEPA_PYTHON="${JEPA_PYTHON:-$("${CONDA_COMMAND}" run -n "${JEPA_ENV}" python -c 'import sys; print(sys.executable)')}"
: "${INPUT_ROOT:?Set INPUT_ROOT to the directory containing <task>/*.mp4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/evaluation_runs/full-validation-500}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"
GPUS="${GPUS:-0,1,2,3}"
PROCESSES_PER_GPU="${PROCESSES_PER_GPU:-1}"
HEAVY_PROCESSES_PER_GPU="${HEAVY_PROCESSES_PER_GPU:-1}"
JEPA_GPU="${JEPA_GPU:-0}"
RESUME_PREPARED="${RESUME_PREPARED:-0}"
REFERENCE_GT_TRAJECTORY_ROOT="${REFERENCE_GT_TRAJECTORY_ROOT:-}"
SUMMARY="${OUTPUT_ROOT}/stacked_summary.json"
MANIFEST="${OUTPUT_ROOT}/run_manifest.json"
CONFIG="${PROJECT_ROOT}/video_quality/config/config.yaml"
METRICS="psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,subject_consistency,trajectory_accuracy,depth_accuracy"
LIGHT_METRICS=(psnr ssim aesthetic_quality image_quality subject_consistency depth_accuracy)
HEAVY_METRICS=(trajectory_accuracy)

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${PROJECT_ROOT}"

if [ "${RESUME_PREPARED}" = "1" ] && [ -f "${MANIFEST}" ]; then
  echo "Reusing prepared manifest: ${MANIFEST}"
else
  "${CORE_PYTHON}" video_quality/build_stacked_summary.py \
    --input-root "${INPUT_ROOT}" \
    --output "${SUMMARY}" \
    --expected-count "${EXPECTED_COUNT}"

  "${CORE_PYTHON}" -m video_quality.cli prepare \
    --summary "${SUMMARY}" \
    --output-dir "${OUTPUT_ROOT}" \
    --metrics "${METRICS}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/prepare.log"
fi

pending_metrics() {
  local metric
  local pending=()
  for metric in "$@"; do
    if [ ! -s "${OUTPUT_ROOT}/results/metrics/${metric}.json" ]; then
      pending+=("${metric}")
    fi
  done
  local IFS=,
  echo "${pending[*]}"
}

LIGHT_PENDING="$(pending_metrics "${LIGHT_METRICS[@]}")"
if [ -n "${LIGHT_PENDING}" ]; then
  "${CORE_PYTHON}" -m video_quality.cli evaluate \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_ROOT}" \
    --config "${CONFIG}" \
    --metrics "${LIGHT_PENDING}" \
    --gpus "${GPUS}" \
    --processes-per-gpu "${PROCESSES_PER_GPU}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/evaluate-light.log"
else
  echo "All light metrics already completed; skipping"
fi

if [ -n "${REFERENCE_GT_TRAJECTORY_ROOT}" ]; then
  linked_trajectories=0
  current_gt_root="${OUTPUT_ROOT}/cache/jpeg_frames/gt_dataset/default"
  while IFS= read -r -d '' sample_root; do
    sample_id="$(basename "${sample_root}")"
    source_traj="${REFERENCE_GT_TRAJECTORY_ROOT}/${sample_id}/traj"
    target_traj="${sample_root}/traj"
    if [ -f "${source_traj}/traj.npy" ] && [ ! -e "${target_traj}" ]; then
      ln -s "${source_traj}" "${target_traj}"
      linked_trajectories=$((linked_trajectories + 1))
    fi
  done < <(find "${current_gt_root}" -mindepth 1 -maxdepth 1 -type d -print0)
  echo "Reused ${linked_trajectories} GT trajectories from ${REFERENCE_GT_TRAJECTORY_ROOT}"
fi

HEAVY_PENDING="$(pending_metrics "${HEAVY_METRICS[@]}")"
if [ -n "${HEAVY_PENDING}" ]; then
  "${CORE_PYTHON}" -m video_quality.cli evaluate \
    --manifest "${MANIFEST}" \
    --output-dir "${OUTPUT_ROOT}" \
    --config "${CONFIG}" \
    --metrics "${HEAVY_PENDING}" \
    --gpus "${GPUS}" \
    --processes-per-gpu "${HEAVY_PROCESSES_PER_GPU}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/evaluate-heavy.log"
else
  echo "All heavy metrics already completed; skipping"
fi

"${CORE_PYTHON}" -m video_quality.cli jepa \
  --stacked-summary "${SUMMARY}" \
  --output-dir "${OUTPUT_ROOT}" \
  --config "${CONFIG}" \
  --jepa-python "${JEPA_PYTHON}" \
  --gpu "${JEPA_GPU}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/jepa.log"

"${CORE_PYTHON}" -m video_quality.cli aggregate \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}" \
  --metrics "${METRICS}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/aggregate.log"
