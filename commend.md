# 处理数据
## 左右拼接视频，左为GT，右为Gen，summary.json为指令
python /data/liuwenhao/WorldArena/data_process.py --results_dir /data/liuwenhao/WorldArena/video_quality/data/test_60 --model_name test_60

# 普通指标
cd video_quality
nohup env CUDA_VISIBLE_DEVICES=3 bash run_evaluation.sh \
    test_10 \
    /data/liuwenhao/WorldArena/video_quality/data/test_10/worldarena_auto_eval/test_10_test \
    /data/liuwenhao/WorldArena/video_quality/data/test_10/worldarena_auto_eval/summary.json \
    "semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim" > ./output_run_evaluation_10.log 2>&1 &


# action_following
bash video_quality/run_action_following.sh \
    my_model \
    /data/liuwenhao/WorldArena/data/my_model_test \
    /data/liuwenhao/WorldArena/data/summary.json \
    /data/liuwenhao/WorldArena/video_quality/config/config.yaml

# VLM评测
bash video_quality/run_VLM_judge.sh \
    my_model \
    /data/liuwenhao/WorldArena/data/my_model_test \
    /data/liuwenhao/WorldArena/data/summary.json \
    all \
    /data/liuwenhao/WorldArena/video_quality/config/config.yaml