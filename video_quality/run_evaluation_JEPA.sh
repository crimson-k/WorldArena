#!/bin/bash
set -euo pipefail
# Usage: run_evaluation_JEPA.sh <MODEL_NAME> <GEN_VIDEO_DIR> <GT_VIDEO_DIR>

MODEL_NAME=${1:-}
GEN_VIDEO_DIR=${2:-}
GT_VIDEO_DIR=${3:-}

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT="$ROOT_DIR/${MODEL_NAME}/output_JEDi"

if [ -z "$MODEL_NAME" ] || [ -z "$GEN_VIDEO_DIR" ] || [ -z "$GT_VIDEO_DIR" ]; then
    echo "Usage: $0 <MODEL_NAME> <GEN_VIDEO_DIR> <GT_VIDEO_DIR>"
    exit 1
fi

cd ./video_quality/JEDi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate WorldArena_JEPA
export PATH="your absolute path/WorldArena_JEPA/bin:$PATH"

echo ">>> JEPA output root: $OUTPUT_ROOT"

python batch.py \
	--real_dir "$GT_VIDEO_DIR" \
	--gen_dir "$GEN_VIDEO_DIR" \
    --output_root "$OUTPUT_ROOT" 
