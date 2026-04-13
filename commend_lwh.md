# 处理数据（可选分片）
## 左右拼接视频，左为 GT，右为 Gen，summary.json 为指令；--num_shards 会生成 worldarena_auto_eval/shards/shard_xx/
python /data/liuwenhao/WorldArena/data_process.py \
    --results_dir /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500 \
    --model_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500 \
    --num_shards 2

# 普通指标评测（按分片多卡：每个卡处理不同 shard）
cd /data/liuwenhao/WorldArena/video_quality
for shard in 0 1 2 3; do
    gpu=$shard
    nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_evaluation.sh \
        test_60 \
        /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test \
        /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/summary.json \
        "semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim" \
        /data/liuwenhao/WorldArena/video_quality/config/config.yaml \
        $shard 4 > ./nohup_log/output_run_evaluation_60_shard${shard}.log 2>&1 &
done
## 
GPUS=(1 2)
NUM_SHARDS=${#GPUS[@]}

for shard_idx in "${!GPUS[@]}"; do
    gpu=${GPUS[$shard_idx]}
    nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_evaluation.sh \
        RoboTwin_Clean_50_multi_node_4_202604111721_step_17500 \
        /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/worldarena_auto_eval/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_test \
        /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/worldarena_auto_eval/summary.json \
        "semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim,mse,lpips,fid,fvd" \
        /data/liuwenhao/WorldArena/video_quality/config/config.yaml \
        "$shard_idx" "$NUM_SHARDS" \
        > "./nohup_log/output_run_evaluation_RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
done

## 单卡
nohup env CUDA_VISIBLE_DEVICES=2 bash run_evaluation.sh \
    RoboTwin_Clean_50_multi_node_4_202604111721 \
    /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721/worldarena_auto_eval/RoboTwin_Clean_50_multi_node_4_202604111721_test \
    /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721/worldarena_auto_eval/summary.json \
    "semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim" > ./nohup_log/output_run_evaluation_RoboTwin_Clean_50_multi_node_4_202604111721.log 2>&1 &

# VLM评测（按分片多卡：每个卡处理不同 shard）
cd /data/liuwenhao/WorldArena/video_quality
for shard in 0 1 2 3; do
    gpu=$shard
    nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_VLM_judge.sh \
        test_60 \
        /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test_vlm \
        /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/summary.json \
        all \
        /data/liuwenhao/WorldArena/video_quality/config/config.yaml \
        $shard 4 > ./nohup_log/output_run_VLM_judge_60_shard${shard}.log 2>&1 &
done

## 单卡
cd video_quality
nohup env CUDA_VISIBLE_DEVICES=3 bash run_VLM_judge.sh \
    RoboTwin_Clean_50_multi_node_4_202604111721_step_17500 \
    /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/worldarena_auto_eval/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_test_vlm \
    /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/worldarena_auto_eval/summary.json \
    all \
    /data/liuwenhao/WorldArena/video_quality/config/config.yaml > ./nohup_log/output_run_VLM_judge_RoboTwin_Clean_50_multi_node_4_202604111721_step_17500.log 2>&1 &

# JEPA评测（保持单卡，不做分片）
## 5090 环境不适配时可先升级
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision

nohup env CUDA_VISIBLE_DEVICES=1 bash /data/liuwenhao/WorldArena/video_quality/run_evaluation_JEPA.sh \
    RoboTwin_Clean_50_multi_node_4_202604111721_step_17500 \
    /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/worldarena_auto_eval/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_test_vlm \
    /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/worldarena_auto_eval/gt_videos > /data/liuwenhao/WorldArena/video_quality/nohup_log/output_run_evaluation_JEPA_RoboTwin_Clean_50_multi_node_4_202604111721_step_17500.log 2>&1 &

## 单独一个文件保存所有视频的 JEPA 结果
/data/liuwenhao/WorldArena/video_quality/data/test_60/output_JEDi/results.json

# 结果集成（分片模式）
<!-- for shard in 00 01 02 03; do
    python /data/liuwenhao/WorldArena/video_quality/csv_results/aggregate_results.py \
        --base_dir /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --model_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --vlm_model_dir RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --csv_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard}_aggregated_results.csv \
        --jepa_result_path /tmp/not_exists_jepa.json
done

for shard in 00 01; do
    python /data/liuwenhao/WorldArena/video_quality/csv_results/aggregate_results.py \
        --base_dir /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --model_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --vlm_model_dir RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard} \
        --csv_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_${shard}_aggregated_results.csv \
        --jepa_result_path /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/output_JEDi/results.json
done -->

python /data/liuwenhao/WorldArena/video_quality/csv_results/aggregate_results.py \
    --model_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500 \
    --video_quality_root /data/liuwenhao/WorldArena/video_quality \
    --csv_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500.csv \
    --num_shards 2 \
    --shard_ids 00,01 \
    --jepa_result_path /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/output_JEDi/results.json

# 结果聚合

python /data/liuwenhao/WorldArena/video_quality/csv_results/build_ewm_leaderboard.py \
    --aggregated_csv /data/liuwenhao/WorldArena/video_quality/data/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/csv_results/RoboTwin_Clean_50_multi_node_4_202604111721_step_17500.csv \
    --leaderboard_csv /data/liuwenhao/WorldArena/worldarena_leaderboard.csv \
    --my_model_name RoboTwin_Clean_50_multi_node_4_202604111721_step_17500 \
    --my_open_source Open-source \
    --my_year 2026 \
    --output_csv /data/liuwenhao/WorldArena/worldarena_leaderboard_with_my_model.csv \
    --output_md /data/liuwenhao/WorldArena/worldarena_leaderboard_with_my_model.md