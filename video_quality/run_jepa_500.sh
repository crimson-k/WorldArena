#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_COMMAND="${CONDA_EXE:-conda}"
CORE_ENV="${CORE_ENV:-WorldArena}"
JEPA_ENV="${JEPA_ENV:-WorldArena_JEPA}"
CORE_PYTHON="${CORE_PYTHON:-$("${CONDA_COMMAND}" run -n "${CORE_ENV}" python -c 'import sys; print(sys.executable)')}"
JEPA_PYTHON="${JEPA_PYTHON:-$("${CONDA_COMMAND}" run -n "${JEPA_ENV}" python -c 'import sys; print(sys.executable)')}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/evaluation_runs/full-validation-500}"
SUMMARY="${OUTPUT_ROOT}/stacked_summary.json"
CONFIG="${PROJECT_ROOT}/video_quality/config/config.yaml"
MANIFEST="${OUTPUT_ROOT}/run_manifest.json"
METRICS="psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,subject_consistency,trajectory_accuracy,depth_accuracy"

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${PROJECT_ROOT}"

# This executes a CUDA kernel, so an old wheel that merely imports successfully
# cannot pass the preflight on RTX 5090.
"${JEPA_PYTHON}" -c \
  "import torch; x=torch.ones(1, device='cuda'); print('JEPA CUDA OK:', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), float(x.sum()))"

date -Is > "${OUTPUT_ROOT}/logs/jepa.started"
rm -f "${OUTPUT_ROOT}/logs/jepa.completed" "${OUTPUT_ROOT}/logs/jepa.failed"
if ! "${CORE_PYTHON}" -m video_quality.cli jepa \
  --stacked-summary "${SUMMARY}" \
  --output-dir "${OUTPUT_ROOT}" \
  --config "${CONFIG}" \
  --jepa-python "${JEPA_PYTHON}" \
  --gpu 0 \
  2>&1 | tee "${OUTPUT_ROOT}/logs/jepa.log"; then
  status="${PIPESTATUS[0]}"
  printf '%s exit=%s\n' "$(date -Is)" "${status}" \
    > "${OUTPUT_ROOT}/logs/jepa.failed"
  exit "${status}"
fi

if ! "${CORE_PYTHON}" -m video_quality.cli aggregate \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}" \
  --metrics "${METRICS}" \
  2>&1 | tee "${OUTPUT_ROOT}/logs/aggregate.log"; then
  status="${PIPESTATUS[0]}"
  printf '%s aggregate_exit=%s\n' "$(date -Is)" "${status}" \
    > "${OUTPUT_ROOT}/logs/jepa.failed"
  exit "${status}"
fi

date -Is > "${OUTPUT_ROOT}/logs/jepa.completed"
