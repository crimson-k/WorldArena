# 处理数据
## 左右拼接视频，左为GT，右为Gen，summary.json为指令
python /data/liuwenhao/WorldArena/data_process.py --results_dir /data/liuwenhao/WorldArena/video_quality/data/test_60 --model_name test_60

# 普通指标评测
cd video_quality
nohup env CUDA_VISIBLE_DEVICES=3 bash run_evaluation.sh \
    test_60 \
    /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test \
    /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/summary.json \
    "semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim" > ./output_run_evaluation_60.log 2>&1 &

# VLM评测
cd video_quality
nohup env CUDA_VISIBLE_DEVICES=1 bash run_VLM_judge.sh \
    test_60 \
    /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test_vlm \
    /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/summary.json \
    all \
    /data/liuwenhao/WorldArena/video_quality/config/config.yaml > ./output_run_VLM_judge_60.log 2>&1 &

# JEPA评测
## 5090环境不适配
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision
## 
nohup env CUDA_VISIBLE_DEVICES=2 bash video_quality/run_evaluation_JEPA.sh \
    test_60 \
    /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/test_60_test_vlm \
    /data/liuwenhao/WorldArena/video_quality/data/test_60/worldarena_auto_eval/gt_videos > ./video_quality/output_run_evaluation_JEPA_60.log 2>&1 &
## 单独一个文件保存所有视频的结果
video_quality/data/test_60/output_JEDi/results.json

# 结果集成
python /data/liuwenhao/WorldArena/video_quality/csv_results/aggregate_results.py \
    --base_dir /data/liuwenhao/WorldArena/video_quality/data/test_60 \
    --model_name test_60 \
    --vlm_model_dir test_60 \
    --csv_name test_60_aggregated_results.csv \
    --jepa_result_path /tmp/not_exists_jepa.json
