# RoboTwin 夹爪末端投影

读取 HDF5 的 `endpose/left_endpose`、`endpose/right_endpose`，默认将位姿原点沿姿态局部 +X 偏移 0.12 米，得到 RoboTwin 定义的 TCP（夹爪中心）坐标，再使用相机 `extrinsic_cv`（世界到相机）和 `intrinsic_cv` 投影到图像。四元数按 `[w, x, y, z]` 解析，偏移会随每帧姿态旋转到世界坐标系。可用 `--point-type endpose` 选择未偏移的位姿原点，或用 `--tcp-offset` 改变 TCP 偏移距离。这里对应 RoboTwin 的 TCP 中心定义，不是左右指爪各自的物理尖端。坐标和外参使用同一世界参考系及长度单位。

在 WorldArena 根目录执行：

```bash
pip install -r interaction_mask/requirements.txt
python interaction_mask/project_gripper.py \
  --hdf5 /data1/common_data/RoboTwin2.0/dataset/adjust_bottle/aloha-agilex_clean_50/data/episode37.hdf5 \
  --camera head_camera \
  --output-dir interaction_mask/outputs/adjust_bottle
```

以上命令默认标注 TCP 点。若要标注原始末端位姿原点，可追加 `--point-type endpose`；例如用 `--tcp-offset 0.10` 可将 TCP 偏移改为 10 cm。

将 RoboTwin 坐标投影到 BWM 转换后的 256×192 视频，并为每个 episode 生成坐标 JSON 和叠加标注的视频：

```bash
python interaction_mask/project_gripper.py \
  --converted-dataset /data1/fangxuebin/boundless-world-model/converted_dataset_bwm_6tasks_256x192 \
  --robotwin-dataset /data1/common_data/RoboTwin2.0/dataset \
  --output-dir interaction_mask/outputs/converted_dataset_bwm_6tasks_256x192
```

工具读取 `metadata_train.jsonl` 和 `metadata_test.jsonl`，按视频去重，再使用 `task` 与 `source_episode_index` 找到源 HDF5。它要求视频尺寸为 256×192，逐帧数与 HDF5 相机图像数一致，并将视频和 JSON 写到输出目录的 `videos/` 与 `coordinates/` 子目录。默认选取 `head_camera`；如果转换视频来自其他相机，可用 `--camera` 改选。输出目录中已存在同名结果时会报错，避免覆盖。

先验证一个 episode 时，可在上述命令中加入 `--task adjust_bottle --episode-index 0`；这里的 episode index 对应源 RoboTwin 文件 `episode0.hdf5`。

## 按夹爪点裁剪

`crop_gripper.py` 使用投影坐标 JSON 中每帧的夹爪位置作为裁剪中心。默认输出 160×120，边界超出原画面时补黑，并在裁剪中心画出绿色点。默认裁剪左夹爪；用 `--side right` 选择右夹爪。

```bash
python interaction_mask/crop_gripper.py \
  --video /data1/fangxuebin/boundless-world-model/converted_dataset_bwm_6tasks_256x192/videos/adjust_bottle/aloha-agilex_clean_50/episode_000000.mp4 \
  --coordinates interaction_mask/outputs/converted_dataset_bwm_6tasks_256x192/coordinates/adjust_bottle/aloha-agilex_clean_50/episode_000000.json \
  --side left \
  --output-video interaction_mask/outputs/crops/adjust_bottle/episode_000000_left_160x120.mp4
```

输出视频对应的 JSON 逐帧记录原始投影坐标、裁剪窗口左上角、点在裁剪画面中的坐标和原视频边界补黑情况。视频帧数与输入相同；坐标 JSON 和视频帧数不一致时会报错。

## 用 GT 夹爪轨迹裁剪并评测生成视频

`crop_and_evaluate.py` 会从 generated root 匹配任务与 episode，从 GT manifest 找对应的转换视频，并从 RoboTwin HDF5 读取末端位姿及相机参数。默认使用按每帧姿态旋转后的 TCP 点（局部 +X，0.12 米）投影。它根据两臂末端运动、夹爪开合和 joint action 的变化自动判断活动臂；`place_burger_fries` 按双臂任务同时处理左右臂。GT 裁剪逐帧使用原始 GT 投影点；生成视频帧数不同的时候，生成侧的裁剪中心从整段 GT 二维轨迹按片段进度线性插值。两路裁剪视频各自保留原始帧数和帧率，不对视频帧做重采样。默认输出 160×120，裁剪视频不叠加标记，以免改变图像质量指标。

```bash
python interaction_mask/crop_and_evaluate.py \
  --generated-root /data1/fangxuebin/boundless-world-model/outputs/infer/sft_iter2000_stage2_6tasks \
  --gt-root /data1/fangxuebin/boundless-world-model/converted_dataset_bwm_6tasks_256x192 \
  --robotwin-dataset /data1/common_data/RoboTwin2.0/dataset \
  --output-dir interaction_mask/outputs/sft_iter2000_stage2_6tasks_crop160x120 \
  --width 160 --height 120 \
  --gpus 0,1
```

默认分别调用 `conda run -n WorldArena python -m video_quality.stack_and_evaluate` 评测左右臂裁剪集，由该脚本检查原始帧率和帧数并决定可计算的指标。当前帧率或帧数不匹配时，它跳过 PSNR 和 SSIM，继续评测其余六项；缺失分数在结果中记为 `-`。`place_burger_fries` 的最终 `results.csv` 对每个 episode 的左右臂指标取算术平均；`results_by_arm.csv` 保留两臂原始分数。可传 `--shared-gt-root /path/to/gt_shared`，让多组相同 GT 的基线复用投影坐标和 GT 裁剪视频，模型目录只保存各自的生成视频裁剪和评测结果。可传 `--eval-python /path/to/python` 指定评测解释器，或传 `--summary-only` 只生成裁剪视频和评测输入清单。可用 `--task`、`--episode-index` 先验证样本。输出目录必须是新目录，里面包含活动臂判断依据、投影坐标路径、生成视频裁剪和裁剪元数据、`crop_pairs.json` 及评测结果。坐标采用 GT 的 head camera 逐帧轨迹，按原 256×192 视野投影。

可追加 `--point-type endpose` 使用原始末端位姿原点，或用 `--tcp-offset 0.10` 改变默认的 0.12 米 TCP 偏移。该选择同时用于 GT 和 generated 的配对裁剪。

默认解码 HDF5 中 `observation/<camera>/rgb` 的原始图像，逐帧绘制左 TCP（绿色 L）和右 TCP（橙色 R）。传入 `--point-type endpose` 时改为绘制位姿原点。输出：

- `episode37_head_camera_gripper.mp4`：标注视频，默认播放帧率 30，可用 `--fps` 指定。该默认值不代表采集时间戳。
- `episode37_head_camera_gripper.json`：每帧左右投影点三维坐标 `xyz`、未偏移末端坐标 `endpose_xyz`、相机坐标 `camera_xyz`、输出图像二维坐标 `uv`、是否位于相机前方 `in_front` 和是否处于画面内 `in_frame`，以及点类型和偏移参数。

二维坐标以左上角为原点，u 向右，v 向下。相机后方或无效坐标的 `uv` 写为 `[null, null]`；画外点保留投影数值但不绘制。`in_frame` 不判断遮挡，被物体挡住的末端也可能被标注。

单 episode 模式可以传入 `--video /path/to/episode37.mp4` 标注外部视频。必须与选择的相机具有相同视野和逐帧对应关系；工具检查解码帧数，并按输入和输出分辨率比例同步缩放像素坐标。裁剪、拼接、多视角画面或时间重采样的视频需要先对齐，不能直接使用此参数。外部视频默认沿用其帧率。

若 generated 与 GT 帧数不同，`generated_coordinates` 中的 `gt_frame_position` 记录生成帧在整段 GT 轨迹上的插值位置；它是裁剪中心的坐标来源，不是从 GT 视频抽取的帧号。GT 和 generated 裁剪元数据中的 `source_frame_index` 分别记录各自原视频的帧序号。

可选相机以 HDF5 字段为准，例如 `head_camera`、`front_camera`、`left_camera`、`right_camera`。输出已存在时拒绝覆盖；多个任务应使用不同输出目录。

其他评测代码可以调用 `project_points(xyz, intrinsic, extrinsic)` 仅计算投影，或者调用 `annotate_episode(...)`、`annotate_converted_dataset(...)` 导出标注结果。工具得到的是原始数据轨迹的二维标注，不能代表生成视频中夹爪的实际位置。
