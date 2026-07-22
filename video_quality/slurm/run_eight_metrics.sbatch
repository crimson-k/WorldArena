#!/bin/bash
#SBATCH --job-name=worldarena-8metrics
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=worldarena-8metrics-%j.log
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "Usage: sbatch $0 <image.sif> <summary.json> <config.yaml> <output_dir> <jepa_real_dir> <jepa_gen_dir>"
  exit 2
fi

IMAGE=$1
SUMMARY=$2
CONFIG=$3
OUTPUT_DIR=$4
JEPA_REAL_DIR=$5
JEPA_GEN_DIR=$6
PROJECT_ROOT=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
BIND_PATHS="$PROJECT_ROOT:$PROJECT_ROOT,$(dirname "$SUMMARY"):$(dirname "$SUMMARY"),$(dirname "$CONFIG"):$(dirname "$CONFIG"),$OUTPUT_DIR:$OUTPUT_DIR,$JEPA_REAL_DIR:$JEPA_REAL_DIR,$JEPA_GEN_DIR:$JEPA_GEN_DIR"

singularity exec --nv --bind "$BIND_PATHS" "$IMAGE" \
  /opt/micromamba/envs/WorldArena/bin/python -m video_quality.cli all \
  --summary "$SUMMARY" --output-dir "$OUTPUT_DIR" --config "$CONFIG" \
  --jepa-real-dir "$JEPA_REAL_DIR" --jepa-gen-dir "$JEPA_GEN_DIR" \
  --jepa-python /opt/micromamba/envs/WorldArena_JEPA/bin/python
