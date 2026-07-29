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
SUMMARY="${OUTPUT_ROOT}/stacked_summary.json"
CONFIG="${PROJECT_ROOT}/video_quality/config/config.yaml"
METRICS="psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,subject_consistency,trajectory_accuracy,depth_accuracy"

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${PROJECT_ROOT}"

"${CORE_PYTHON}" video_quality/build_stacked_summary.py \
  --input-root "${INPUT_ROOT}" \
  --output "${SUMMARY}" \
  --expected-count "${EXPECTED_COUNT}"

"${CORE_PYTHON}" -m video_quality.cli prepare \
  --summary "${SUMMARY}" \
  --output-dir "${OUTPUT_ROOT}" \
  --metrics "${METRICS}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/prepare.log"

"${CORE_PYTHON}" -m video_quality.cli evaluate \
  --manifest "${OUTPUT_ROOT}/run_manifest.json" \
  --output-dir "${OUTPUT_ROOT}" \
  --config "${CONFIG}" \
  --metrics "${METRICS}" \
  --gpus 0,1,2,3 \
  --processes-per-gpu 1 \
  2>&1 | tee "${OUTPUT_ROOT}/logs/evaluate.log"

"${CORE_PYTHON}" -m video_quality.cli jepa \
  --stacked-summary "${SUMMARY}" \
  --output-dir "${OUTPUT_ROOT}" \
  --config "${CONFIG}" \
  --jepa-python "${JEPA_PYTHON}" \
  --gpu 0 \
  2>&1 | tee "${OUTPUT_ROOT}/logs/jepa.log"

"${CORE_PYTHON}" -m video_quality.cli aggregate \
  --manifest "${OUTPUT_ROOT}/run_manifest.json" \
  --output-dir "${OUTPUT_ROOT}" \
  --metrics "${METRICS}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/aggregate.log"
