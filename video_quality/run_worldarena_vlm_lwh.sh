
MODEL_NAME="${1:?请传入 MODEL_NAME，例如: bash run_eval_all.sh your_model_name}"
VIDEO_DIR=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/${MODEL_NAME}_test_vlm
SUMMARY_JSON=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/summary.json
CONFIG_PATH=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/config/config.yaml

GPUS=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#GPUS[@]}

for shard_idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_idx]}
  nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_VLM_judge.sh \
    "$MODEL_NAME" \
    "$VIDEO_DIR" \
    "$SUMMARY_JSON" \
    all \
    "$CONFIG_PATH" \
    "$shard_idx" "$NUM_SHARDS" \
    > "./nohup_log/${MODEL_NAME}/output_run_VLM_judge_${MODEL_NAME}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
done