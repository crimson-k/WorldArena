# 处理数据（可选分片）
## 左右拼接视频，左为 GT，右为 Gen，summary.json 为指令；--num_shards 会生成 worldarena_auto_eval/shards/shard_xx/
python /data/liuwenhao/WM/WorldArena/data_process.py \
    --results_dir /data/liuwenhao/WM/WorldArena/video_quality/data/test_60 \
    --model_name test_60 \
    --num_shards 4

# 普通指标评测（按分片多卡：每个卡处理不同 shard）
cd /data/liuwenhao/WM/WorldArena/video_quality
for shard in 0 1 2 3; do
    gpu=$shard
    nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_evaluation.sh \
        test_60 \
        /data/liuwenhao/WM/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test \
        /data/liuwenhao/WM/WorldArena/video_quality/data/test_60/worldarena_auto_eval/summary.json \
        "semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim" \
        /data/liuwenhao/WM/WorldArena/video_quality/config/config.yaml \
        $shard 4 > ./nohup_log/output_run_evaluation_60_shard${shard}.log 2>&1 &
done

# VLM评测（按分片多卡：每个卡处理不同 shard）
cd /data/liuwenhao/WM/WorldArena/video_quality
for shard in 0 1 2 3; do
    gpu=$shard
    nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_VLM_judge.sh \
        test_60 \
        /data/liuwenhao/WM/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test_vlm \
        /data/liuwenhao/WM/WorldArena/video_quality/data/test_60/worldarena_auto_eval/summary.json \
        all \
        /data/liuwenhao/WM/WorldArena/video_quality/config/config.yaml \
        $shard 4 > ./nohup_log/output_run_VLM_judge_60_shard${shard}.log 2>&1 &
done

# JEPA评测（保持单卡，不做分片）
## 5090 环境不适配时可先升级
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision

nohup env CUDA_VISIBLE_DEVICES=2 bash /data/liuwenhao/WM/WorldArena/video_quality/run_evaluation_JEPA.sh \
    test_60 \
    /data/liuwenhao/WM/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test_vlm \
    /data/liuwenhao/WM/WorldArena/video_quality/data/test_60/worldarena_auto_eval/gt_videos > /data/liuwenhao/WM/WorldArena/video_quality/nohup_log/output_run_evaluation_JEPA_60.log 2>&1 &

## 单独一个文件保存所有视频的 JEPA 结果
/data/liuwenhao/WM/WorldArena/video_quality/data/test_60/output_JEDi/results.json

# 结果集成（分片模式）
for shard in 00 01 02 03; do
    python /data/liuwenhao/WM/WorldArena/video_quality/csv_results/aggregate_results.py \
        --base_dir /data/liuwenhao/WM/WorldArena/video_quality/data/test_60_shard_${shard} \
        --model_name test_60_shard_${shard} \
        --vlm_model_dir test_60_shard_${shard} \
        --csv_name test_60_shard_${shard}_aggregated_results.csv \
        --jepa_result_path /tmp/not_exists_jepa.json
done