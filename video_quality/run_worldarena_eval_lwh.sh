# python /ssdfs/datahome/usersht/dev/lwh/WorldArena/data_process.py \
#     --results_dir /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500 \
#     --model_name runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500 \
#     --num_shards 8


#!/usr/bin/env bash

MODEL_NAME="${1:?请传入 MODEL_NAME，例如: bash run_eval_all.sh your_model_name}"

GEN_VIDEO_DIR="/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/${MODEL_NAME}_test"
SUMMARY_JSON="/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/summary.json"
CONFIG_PATH="/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/config/config.yaml"
METRICS="semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim,mse,lpips,fid,fvd"
GPUS=(0 1 2 3 4 5 6 7)
NUM_SHARDS=${#GPUS[@]}

mkdir -p "./nohup_log/${MODEL_NAME}/"

for shard_idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_idx]}
  WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=$gpu \
  bash /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/run_evaluation.sh \
    "$MODEL_NAME" \
    "$GEN_VIDEO_DIR" \
    "$SUMMARY_JSON" \
    "$METRICS" \
    "$CONFIG_PATH" \
    "$shard_idx" "$NUM_SHARDS" \
    > "./nohup_log/${MODEL_NAME}/output_run_evaluation_${MODEL_NAME}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
done

# MODEL_NAME=runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500
# VIDEO_DIR=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/${MODEL_NAME}_test_vlm
# SUMMARY_JSON=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/summary.json
# CONFIG_PATH=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/config/config.yaml

# GPUS=(0 1 2 3 4 5 6 7)
# NUM_SHARDS=${#GPUS[@]}

# for shard_idx in "${!GPUS[@]}"; do
#   gpu=${GPUS[$shard_idx]}
#   nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_VLM_judge.sh \
#     "$MODEL_NAME" \
#     "$VIDEO_DIR" \
#     "$SUMMARY_JSON" \
#     all \
#     "$CONFIG_PATH" \
#     "$shard_idx" "$NUM_SHARDS" \
#     > "./nohup_log/${MODEL_NAME}/output_run_VLM_judge_${MODEL_NAME}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
# done

# nohup env CUDA_VISIBLE_DEVICES=1 bash /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/run_evaluation_JEPA.sh \
#     runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500 \
#     /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500/worldarena_auto_eval/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500_test_vlm \
#     /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500/worldarena_auto_eval/gt_videos > /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/nohup_log/output_run_evaluation_JEPA_runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500.log 2>&1 &
