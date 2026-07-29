#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/evaluation_runs/full-validation-500}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
EVAL_COMPLETED="${LOG_ROOT}/evaluate.completed"
EVAL_FAILED="${LOG_ROOT}/evaluate.failed"
QUEUE_LOG="${LOG_ROOT}/jepa-queue.log"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[$(date -Is)] JEPA queued; waiting for non-JEPA evaluation."
while [[ ! -f "${EVAL_COMPLETED}" ]]; do
  if [[ -f "${EVAL_FAILED}" ]]; then
    echo "[$(date -Is)] Non-JEPA evaluation failed; JEPA will not start."
    cat "${EVAL_FAILED}"
    exit 1
  fi
  if ! tmux has-session -t worldarena_eval 2>/dev/null; then
    echo "[$(date -Is)] worldarena_eval exited without a completion marker."
    exit 1
  fi
  sleep 30
done

echo "[$(date -Is)] Non-JEPA evaluation completed; starting JEPA on GPU 0."
exec "${PROJECT_ROOT}/video_quality/run_jepa_500.sh"
