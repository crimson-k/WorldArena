# WorldArena 八指标视频评估

[English](README.md) | [中文详细说明](USAGE_CN.md)

当前评估器只保留：

`PSNR`、`SSIM`、`Aesthetic Quality`、`Image Quality`、`JEPA Similarity`、
`Subject Consistency`、`Trajectory Accuracy` 和 `Depth Accuracy`。

所有命令都从仓库根目录执行。`video_quality/config/config.yaml` 中的模型路径
相对于仓库根目录解析，移动项目目录后无需修改配置。

## 创建两个环境

完整的常规依赖分别写在：

- `video_quality/environment-core.yml`：七个非 JEPA 指标；
- `video_quality/environment-jepa.yml`：JEPA。

创建全新环境：

```bash
conda env create -f video_quality/environment-core.yml
conda run --no-capture-output -n WorldArena \
  python -m pip install --no-deps pyiqa==0.1.14.1

conda env create -f video_quality/environment-jepa.yml
conda run --no-capture-output -n WorldArena_JEPA \
  python -m pip install --no-deps vjepa==0.1.2 videojedi==1.1.0
```

如果同名环境已经存在，将 `env create` 改为：

```bash
conda env update -f video_quality/environment-core.yml --prune
conda env update -f video_quality/environment-jepa.yml --prune
```

两条 `--no-deps` 命令是必要的：

- `pyiqa==0.1.14.1` 的元数据错误固定了
  `transformers==4.37.2`，而 Depth Anything 和本项目使用 4.51.3；
- `vjepa==0.1.2` 的元数据错误限制 `torchvision<0.20`，RTX 5090
  需要支持 Blackwell 的 PyTorch 2.10 / CUDA 12.8。

验证两个环境并实际执行 CUDA kernel：

```bash
conda run --no-capture-output -n WorldArena python -c \
  "import torch, pyiqa; x=torch.ones(1,device='cuda'); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),float(x.sum()))"

conda run --no-capture-output -n WorldArena_JEPA python -c \
  "import torch, videojedi, vjepa; x=torch.ones(1,device='cuda'); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),float(x.sum()))"
```

## 输入 summary

### GT 与生成视频分开

```json
[
  {
    "sample_id": "task__episode_000040",
    "gt_path": "/path/to/gt.mp4",
    "generated_video": "/path/to/generated.mp4"
  }
]
```

分离视频可以省略 `sample_id`，此时使用 GT 文件名 stem。所有 ID 必须全局唯一。

### 上下拼接视频

MP4 上半部分为 GT、下半部分为生成结果时：

```json
[
  {
    "sample_id": "task__episode_000040",
    "stacked_video": "/path/to/stacked.mp4"
  }
]
```

帧高度必须为偶数。程序只解码一次并在 `height / 2` 处裁切，不生成二次压缩的
中间视频。不同任务存在同名 episode 时，应显式填写带任务名的 `sample_id`。

从 `<input-root>/<task>/*.mp4` 自动生成稳定清单：

```bash
conda run --no-capture-output -n WorldArena \
  python video_quality/build_stacked_summary.py \
  --input-root /path/to/input-root \
  --output /path/to/stacked-summary.json
```

## 执行评估

设置常用路径：

```bash
SUMMARY=/path/to/summary.json
EVAL_ROOT=/path/to/evaluation-output
CONFIG=video_quality/config/config.yaml
METRICS=psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,subject_consistency,trajectory_accuracy,depth_accuracy
```

准备 PNG/JPEG 帧：

```bash
conda run --no-capture-output -n WorldArena \
  python -m video_quality.cli prepare \
  --summary "$SUMMARY" \
  --output-dir "$EVAL_ROOT" \
  --metrics "$METRICS"
```

使用四张卡、每卡一个进程执行七个非 JEPA 指标：

```bash
conda run --no-capture-output -n WorldArena \
  python -m video_quality.cli evaluate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --config "$CONFIG" \
  --metrics "$METRICS" \
  --gpus 0,1,2,3 \
  --processes-per-gpu 1
```

`--gpus` 指定物理卡号；`--processes-per-gpu` 指定每张卡共享的模型进程数。
模型指标建议保持每卡一个进程。

获取 JEPA 环境解释器：

```bash
JEPA_PYTHON="$(conda run -n WorldArena_JEPA python -c 'import sys; print(sys.executable)')"
```

上下拼接输入：

```bash
conda run --no-capture-output -n WorldArena \
  python -m video_quality.cli jepa \
  --stacked-summary "$SUMMARY" \
  --output-dir "$EVAL_ROOT" \
  --config "$CONFIG" \
  --jepa-python "$JEPA_PYTHON" \
  --gpu 0
```

分离输入也可以提供两个顶层包含同名 MP4 的目录：

```bash
conda run --no-capture-output -n WorldArena \
  python -m video_quality.cli jepa \
  --real-dir /path/to/real-mp4 \
  --gen-dir /path/to/generated-mp4 \
  --output-dir "$EVAL_ROOT" \
  --config "$CONFIG" \
  --jepa-python "$JEPA_PYTHON" \
  --gpu 0
```

聚合八项结果：

```bash
conda run --no-capture-output -n WorldArena \
  python -m video_quality.cli aggregate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --metrics "$METRICS"
```

## 输出与计时

```text
<evaluation-output>/
├── run_manifest.json
├── logs/
│   ├── metric_timings.log
│   └── metric_timings.json
└── results/
    ├── metrics/<metric>.json
    ├── jepa/results.json
    ├── results.json
    └── results.csv
```

只有分布式 rank 0 写计时日志。日志包含每个阶段的开始时间、结束时间、秒数和
非 JEPA 总墙钟时间，并同步输出到终端。

Trajectory 和 Depth 使用排行榜归一化：

```text
Trajectory = clip(raw_NDTW / 40.8540, 0, 1)
Depth      = 1 - clip((raw_AbsRel - 0.2228) / (4.3711 - 0.2228), 0, 1)
```

JEPA 输出 `exp(-0.4D)`，其中 `D` 为 JEDi 距离。聚合时同一个数据集级 JEPA
分数会写入每条样本。

## 模型文件

`video_quality/config/config.yaml` 默认使用以下仓库相对目录：

- `video_quality/models_downloaded/...`：CLIP、MUSIQ、DINO、RAFT 和
  Depth Anything；
- `sam/sam3.pt` 与 `sam/bpe_simple_vocab_16e6.txt.gz`；
- `video_quality/JEDi/pretrained_models/vith16.pth.tar`；
- `video_quality/JEDi/pretrained_models/ssv2-probe.pth.tar`。

权重、数据集、缓存和评估结果均不会提交到 Git。
