# 处理数据（可选分片）
## 左右拼接视频，左为 GT，右为 Gen，summary.json 为指令；--num_shards 会生成 worldarena_auto_eval/shards/shard_xx/
python /ssdfs/datahome/usersht/dev/lwh/WorldArena/data_process.py \
    --results_dir /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_9500_split_stage_1_0 \
    --model_name runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_9500_split_stage_1_0 \
    --num_shards 8

# 普通指标评测（按分片多卡：每个卡处理不同 shard）
<!-- cd /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality
for shard in 0 1 2 3; do
    gpu=$shard
    nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_evaluation.sh \
        test_60 \
        /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test \
        /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/test_60/worldarena_auto_eval/summary.json \
        "semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim" \
        /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/config/config.yaml \
        $shard 4 > ./nohup_log/output_run_evaluation_60_shard${shard}.log 2>&1 &
done -->
## 
### WORLD_ARENA_OVERWRITE_RESULTS=0     累计结果
### WORLD_ARENA_OVERWRITE_RESULTS=1     覆盖结果

<!-- MODEL_NAME=GTGT
GEN_VIDEO_DIR=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/${MODEL_NAME}_test
SUMMARY_JSON=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/summary.json
CONFIG_PATH=/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/config/config.yaml
METRICS="semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim,mse,lpips,fid,fvd"

GPUS=(0 1 3)
NUM_SHARDS=${#GPUS[@]}

for shard_idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_idx]}
  nohup env WORLD_ARENA_OVERWRITE_RESULTS=0 CUDA_VISIBLE_DEVICES=$gpu bash run_evaluation.sh \
    "$MODEL_NAME" \
    "$GEN_VIDEO_DIR" \
    "$SUMMARY_JSON" \
    "$METRICS" \
    "$CONFIG_PATH" \
    "$shard_idx" "$NUM_SHARDS" \
    > "./nohup_log/output_run_evaluation_${MODEL_NAME}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
done -->

## 单卡
<!-- nohup env CUDA_VISIBLE_DEVICES=2 bash run_evaluation.sh \
    RoboTwin_Clean_50_multi_node_4_202604111721 \
    /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721/worldarena_auto_eval/RoboTwin_Clean_50_multi_node_4_202604111721_test \
    /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721/worldarena_auto_eval/summary.json \
    "semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim,mse,lpips,fid,fvd" > ./nohup_log/output_run_evaluation_RoboTwin_Clean_50_multi_node_4_202604111721_mse_lpips_fid_fvd.log 2>&1 & -->

# VLM评测（按分片多卡：每个卡处理不同 shard）
<!-- cd /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality

for shard in 0 1 2 3; do
    gpu=$shard
    nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_VLM_judge.sh \
        test_60 \
        /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test_vlm \
        /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/test_60/worldarena_auto_eval/summary.json \
        all \
        /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/config/config.yaml \
        $shard 4 > ./nohup_log/output_run_VLM_judge_60_shard${shard}.log 2>&1 &
done -->

<!-- MODEL_NAME=ckpt80000
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
    > "./nohup_log/output_run_VLM_judge_${MODEL_NAME}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
done -->

## 单卡
<!-- cd video_quality
nohup env CUDA_VISIBLE_DEVICES=3 bash run_VLM_judge.sh \
    RoboTwin_Clean_50_multi_node_4_202604111721_step_17500 \
    /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/worldarena_auto_eval/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_test_vlm \
    /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/worldarena_auto_eval/summary.json \
    all \
    /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/config/config.yaml > ./nohup_log/output_run_VLM_judge_RoboTwin_Clean_50_multi_node_4_202604111721_step_17500.log 2>&1 & -->

# JEPA评测（保持单卡，不做分片）
## 5090 环境不适配时可先升级
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision

nohup env CUDA_VISIBLE_DEVICES=1 bash /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/run_evaluation_JEPA.sh \
    ckpt80000 \
    /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/ckpt80000/worldarena_auto_eval/ckpt80000_test_vlm \
    /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/ckpt80000/worldarena_auto_eval/gt_videos > /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/nohup_log/output_run_evaluation_JEPA_ckpt80000.log 2>&1 &

## 单独一个文件保存所有视频的 JEPA 结果
/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/test_60/output_JEDi/results.json

# 结果集成（分片模式）
<!-- for shard in 00 01 02 03; do
    python /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/csv_results/aggregate_results.py \
        --base_dir /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --model_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --vlm_model_dir RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --csv_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard}_aggregated_results.csv \
        --jepa_result_path /tmp/not_exists_jepa.json
done

for shard in 00 01; do
    python /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/csv_results/aggregate_results.py \
        --base_dir /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --model_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --vlm_model_dir RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --csv_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard}_aggregated_results.csv \
        --jepa_result_path /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/output_JEDi/results.json
done -->

python /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/csv_results/aggregate_results.py \
    --model_name runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500 \
    --video_quality_root /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality \
    --csv_name runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500.csv \
    --num_shards 8 \
    --shard_ids 00,01,02,03,04,05,06,07 \
    --jepa_result_path /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500/output_JEDi/results.json

python /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/csv_results/aggregate_results.py \
  --base_dir /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500 \
  --model_name runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500 \
  --csv_name aggregated_results.csv \
  --jepa_result_path /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500/output_JEDi/results.json

# 结果聚合

python /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/csv_results/build_ewm_leaderboard.py \
    --aggregated_csv /ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500/csv_results/runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500.csv \
    --leaderboard_csv /ssdfs/datahome/usersht/dev/lwh/WorldArena/worldarena_leaderboard.csv \
    --my_model_name runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500 \
    --my_open_source Open-source \
    --my_year 2026 \
    --output_csv /ssdfs/datahome/usersht/dev/lwh/WorldArena/worldarena_leaderboard_with_runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500.csv \
    --output_md /ssdfs/datahome/usersht/dev/lwh/WorldArena/worldarena_leaderboard_with_runid_2_robotwin_aloha_clean_50_multi_node_2_20260413_122421_step_28500.md