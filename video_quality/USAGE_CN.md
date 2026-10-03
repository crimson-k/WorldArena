# WorldArena 八指标评估详细使用说明

本文档约定：

- 所有命令从仓库根目录执行；
- 核心环境：Conda 环境 `WorldArena`
- JEPA 环境：Conda 环境 `WorldArena_JEPA`
- 模型配置：`video_quality/config/config.yaml`

本文只说明当前保留的八个视频质量指标。`embodied_task`、Track2、训练和提交打包
不在本文范围内。

## 1. 八个指标与环境划分

| 指标 | CLI 名称 | 使用环境 | 结果方向 |
|---|---|---|---|
| PSNR | `psnr` | `WorldArena` | 越高越好 |
| SSIM | `ssim` | `WorldArena` | 越高越好 |
| Aesthetic Quality | `aesthetic_quality` | `WorldArena` | 越高越好 |
| Image Quality | `image_quality` | `WorldArena` | 越高越好 |
| JEPA Similarity | `jepa_similarity` | `WorldArena_JEPA` | 越高越好 |
| Subject Consistency | `subject_consistency` | `WorldArena` | 越高越好 |
| Trajectory Accuracy | `trajectory_accuracy` | `WorldArena` | 越高越好 |
| Depth Accuracy | `depth_accuracy` | `WorldArena` | 越高越好 |

两个环境都使用支持 RTX 5090 的 PyTorch 2.10.0/CUDA 12.8。JEPA 单独建环境，
是为了隔离 `vjepa/videojedi` 依赖。

## 2. 当前环境和模型位置

### 2.1 Python 环境

项目提供两份完整的 Conda YAML：

```text
video_quality/environment-core.yml
video_quality/environment-jepa.yml
```

创建全新环境：

```bash
conda env create -f video_quality/environment-core.yml
conda run --no-capture-output -n WorldArena \
  python -m pip install --no-deps pyiqa==0.1.14.1

conda env create -f video_quality/environment-jepa.yml
conda run --no-capture-output -n WorldArena_JEPA \
  python -m pip install --no-deps vjepa==0.1.2 videojedi==1.1.0
```

已存在同名环境时：

```bash
conda env update -f video_quality/environment-core.yml --prune
conda env update -f video_quality/environment-jepa.yml --prune
```

`pyiqa` 和 `vjepa` 的包元数据分别包含错误的 Transformers/torchvision 版本
上限，因此只对这三个包本体使用 `--no-deps`；它们的实际运行依赖已经完整写入
对应 YAML。

### 2.2 模型文件

当前模型已下载并写入 `video_quality/config/config.yaml`：

| 用途 | 本地位置 |
|---|---|
| CLIP ViT-L/14 | `video_quality/models_downloaded/aesthetic_quality/ViT-L-14.pt` |
| Aesthetic Head | `video_quality/models_downloaded/aesthetic_quality/sa_0_4_vit_l_14_linear.pth` |
| MUSIQ SPAQ | `video_quality/models_downloaded/image_quality/musiq_spaq_ckpt-358bb6af.pth` |
| Facebook DINO 源码 | `video_quality/models_downloaded/subject_consistency/facebookresearch_dino_main` |
| DINO ViT-B/16 | `video_quality/models_downloaded/subject_consistency/dino_vitbase16_pretrain.pth` |
| RAFT Things | `video_quality/models_downloaded/subject_consistency/raft-things.pth` |
| Depth Anything V2 Small | `video_quality/models_downloaded/depth_accuracy/Depth-Anything-V2-Small-hf` |
| SAM3 | `sam/` |
| JEPA ViT-H/16 | `video_quality/JEDi/pretrained_models/vith16.pth.tar` |
| JEPA SSV2 Probe | `video_quality/JEDi/pretrained_models/ssv2-probe.pth.tar` |

配置文件中的模型路径均为相对路径，并统一相对于项目根目录解析。命令行中的
summary、输出目录、JEPA 输入目录等运行数据路径仍建议使用绝对路径。

## 3. 运行前检查

进入项目目录，例如：

```bash
cd /path/to/WorldArena
```

检查两个环境：

```bash
conda run -n WorldArena python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

conda run -n WorldArena_JEPA python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

预期的软件版本分别包含：

```text
WorldArena:      torch 2.10.0+cu128
WorldArena_JEPA: torch 2.10.0+cu128
```

检查 GPU：

```bash
nvidia-smi
```

完整八指标评估需要 NVIDIA GPU。若 `nvidia-smi` 失败，或
`torch.cuda.is_available()` 为 `False`，应先解决宿主机驱动、GPU 分配或容器
设备映射问题。Python 环境中包含 CUDA 运行库，但它不能替代宿主机 NVIDIA
驱动。

检查 CLI：

```bash
conda run -n WorldArena python -m video_quality.cli --help
```

## 4. 准备输入

### 4.1 Summary JSON

评估入口是一个非空 JSON 列表。每个元素表示一对 GT/生成视频：

```json
[
  {
    "gt_path": "/absolute/path/to/gt/episode0.mp4",
    "generated_video": "/absolute/path/to/generated/episode0.mp4"
  },
  {
    "gt_path": "/absolute/path/to/gt/episode1.mp4",
    "generated_video": "/absolute/path/to/generated/episode1.mp4"
  }
]
```

必须满足：

1. `gt_path` 和 `generated_video` 都必须存在，并使用视频文件路径。
2. 推荐使用绝对路径。
3. 文件必须存在且能被 OpenCV 解码。
4. 可以显式提供非空 `sample_id`；省略时取 GT 文件名 stem。
5. 所有 `sample_id` 必须全局唯一。
6. 普通指标允许 GT 和生成视频使用不同文件名；JEPA 的规则不同，见下一节。

例如：

```text
gt_path=/dataset/task_a/episode40.mp4
sample_id=episode40
```

如果两个 GT 路径都叫 `episode40.mp4`，程序会报：

```text
Duplicate sample_id derived from gt_path: episode40
```

这种情况下直接为每条记录填写带任务名的唯一 `sample_id`。

### 4.2 上下拼接视频

如果一个 MP4 的上半部分是 GT、下半部分是生成结果，可以直接使用：

```json
[
  {
    "sample_id": "task_a__episode40",
    "stacked_video": "/absolute/path/to/task_a/episode40.mp4"
  }
]
```

帧高度必须为偶数。程序只解码一次，并在 `height / 2` 处裁切上下半帧，不会
生成二次压缩的中间 MP4。不同任务下存在同名 episode 时必须显式填写唯一 ID。

### 4.3 视频建议

评估代码只要求视频可解码，但为了与 WorldArena 数据一致，建议：

- 分辨率：`640×480` 或更高；
- 帧数：121 帧；
- 帧率：24 FPS；
- GT 和生成视频尽量使用一致的分辨率、帧率和帧数。

PSNR/SSIM 对逐帧对应关系敏感。帧数、空间尺寸或时间对齐方式不同会影响结果，
甚至导致评估失败。

### 4.4 JEPA 输入

上下拼接输入直接复用含 `stacked_video` 的 summary。

分离输入则读取两个非递归目录，并按 MP4 文件名 stem 取交集：

```text
/data/eval/jepa_real/
├── episode0.mp4
└── episode1.mp4

/data/eval/jepa_generated/
├── episode0.mp4
└── episode1.mp4
```

注意：

- 只扫描目录顶层的 `*.mp4`，不递归扫描子目录。
- 两侧需要使用相同文件名或至少相同 stem。
- 不在交集中的视频不会参与 JEPA。
- 无法解码的视频对会被跳过。
- 如果交集为空，会报 `No intersected mp4 filenames`。

可以用软链接构造 JEPA 目录，不需要重复复制大视频：

```bash
mkdir -p /data/eval/jepa_real /data/eval/jepa_generated

ln -s /absolute/path/to/gt/episode0.mp4 \
  /data/eval/jepa_real/episode0.mp4

ln -s /absolute/path/to/generated/model_output.mp4 \
  /data/eval/jepa_generated/episode0.mp4
```

## 5. 推荐运行方式：分阶段执行

分阶段执行更容易定位失败指标，也便于单独重跑。

以下变量仅作为示例，请替换四个数据路径：

```bash
PROJECT_ROOT="$(pwd)"
SUMMARY_JSON=/absolute/path/to/summary.json
EVAL_ROOT=/absolute/path/to/evaluation_output
JEPA_REAL=/absolute/path/to/jepa_real
JEPA_GENERATED=/absolute/path/to/jepa_generated
CORE_PYTHON=$(conda run -n WorldArena python -c 'import sys; print(sys.executable)')
JEPA_PYTHON=$(conda run -n WorldArena_JEPA python -c 'import sys; print(sys.executable)')

cd "$PROJECT_ROOT"
```

### 5.1 准备帧和 manifest

```bash
"$CORE_PYTHON" -m video_quality.cli prepare \
  --summary "$SUMMARY_JSON" \
  --output-dir "$EVAL_ROOT"
```

默认选择全部八个指标。`prepare` 会：

- 为 PSNR/SSIM 抽取无损 PNG；
- 为五个基于图像帧的模型指标抽取 JPEG；
- 写入 `$EVAL_ROOT/run_manifest.json`；
- 不为 JEPA 抽帧，因为 JEPA 直接读取 MP4。

注意：同时生成 PNG 和 JPEG 会占用较多磁盘空间。视频数量很多时，应先确认输出
分区空间。

### 5.2 执行七个非 JEPA 指标

单进程执行：

```bash
"$CORE_PYTHON" -m video_quality.cli evaluate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --config video_quality/config/config.yaml
```

例如使用物理 GPU 0、1、2、3，将每个指标的数据分别切成四片：

```bash
"$CORE_PYTHON" -m video_quality.cli evaluate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --config video_quality/config/config.yaml \
  --gpus 0,1,2,3 \
  --processes-per-gpu 1
```

指标按顺序执行并分别写结果。某个指标失败时，已经成功写入的指标 JSON 会保留，
可以修复问题后只重跑失败指标。

### 5.3 执行 JEPA

上下拼接 summary：

```bash
"$CORE_PYTHON" -m video_quality.cli jepa \
  --stacked-summary "$SUMMARY_JSON" \
  --output-dir "$EVAL_ROOT" \
  --config video_quality/config/config.yaml \
  --jepa-python "$JEPA_PYTHON" \
  --gpu 0
```

分离目录：

```bash
"$CORE_PYTHON" -m video_quality.cli jepa \
  --real-dir "$JEPA_REAL" \
  --gen-dir "$JEPA_GENERATED" \
  --output-dir "$EVAL_ROOT" \
  --config video_quality/config/config.yaml \
  --jepa-python "$JEPA_PYTHON"
```

外层 CLI 使用核心环境，JEPA 计算由 `--jepa-python` 指定的独立环境执行。

JEPA 会：

- 每个视频均匀采样 16 帧；
- 将帧缩放到 `224×224`；
- 计算 dataset-level JEDi 距离；
- 使用 `exp(-0.4D)` 转为相似度；
- 把结果保存到 `$EVAL_ROOT/results/jepa/results.json`。

### 5.4 聚合

只有全部目标指标都已经产生结果文件后，再执行聚合：

```bash
"$CORE_PYTHON" -m video_quality.cli aggregate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT"
```

最终主要结果：

```text
$EVAL_ROOT/results/results.csv
$EVAL_ROOT/results/results.json
```

## 6. 一条命令执行全部八指标

下面的命令显式指定：

- 使用物理 GPU 0、1、2、3；
- 运行完整八指标；
- 七个非 JEPA 指标逐指标进行四卡数据分片；
- JEPA 不分片，只使用 `--gpus` 中的第一张卡，即物理 GPU 0。

```bash
PROJECT_ROOT="$(pwd)"
CORE_PYTHON=$(conda run -n WorldArena python -c 'import sys; print(sys.executable)')
JEPA_PYTHON=$(conda run -n WorldArena_JEPA python -c 'import sys; print(sys.executable)')

cd "$PROJECT_ROOT"

"$CORE_PYTHON" -m video_quality.cli all \
  --summary /absolute/path/to/summary.json \
  --output-dir /absolute/path/to/evaluation_output \
  --config video_quality/config/config.yaml \
  --metrics psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,subject_consistency,trajectory_accuracy,depth_accuracy \
  --jepa-real-dir /absolute/path/to/jepa_real \
  --jepa-gen-dir /absolute/path/to/jepa_generated \
  --jepa-python "$JEPA_PYTHON" \
  --gpus 0,1,2,3 \
  --processes-per-gpu 1
```

这里显式写出了八个 `--metrics`。如果完全省略 `--metrics`，CLI 的默认值也
是这八个指标，效果相同。只要所选指标中包含 `jepa_similarity`，应提供
`--jepa-stacked-summary`，或同时提供 `--jepa-real-dir` 和
`--jepa-gen-dir`。

### 6.1 多卡说明

`--gpus` 接收逗号分隔的宿主机物理卡号；`--processes-per-gpu` 指定每张卡
启动多少个非 JEPA worker，默认值为 1。多卡执行流程是：

1. 主进程执行一次 `prepare`，避免多个进程同时抽帧；
2. 主进程自动启动 `torchrun`，worker 总数为 GPU 数乘以每卡进程数；
3. 七个非 JEPA 指标仍然按指标顺序执行；
4. 对于每一个指标，样本按 `samples[rank::world_size]` 分片到所有 GPU；
5. 所有 rank 汇总逐样本结果，只有 rank 0 写指标 JSON；
6. 分布式进程退出后，JEPA 在第一张所选 GPU 上单进程执行；
7. 主进程最后执行一次结果聚合。

因此，这不是“每张卡负责一个不同指标”，而是“同一个指标同时使用多张卡处理
不同样本”。每个进程都会加载当前指标的一份模型权重。

PSNR 和 SSIM 的实现是 CPU 计算。它们也按 rank 分片，但多卡本身不会加速计算；
实际加速来自多个 worker 同时使用 CPU 和读取不同的图片。PSNR 与 SSIM 同时选择
时，同一对图片只读取一次。

注意：

- worker 总数（GPU 数 × 每卡进程数）不能超过 manifest 中的样本数量；
- 不要再在 `all` 外层手工使用 `torchrun`，直接传 `--gpus`；
- 不传 `--gpus` 时保持原来的单进程行为；
- 分片粒度是完整视频样本，不会把同一段视频的帧拆给不同 GPU；
- 某些视频明显更长时，各 rank 的完成时间可能不完全一致。

模型指标通常建议保持 `--processes-per-gpu 1`。提高该值会让多个模型副本共享
同一张 GPU，可能更充分地利用轻量指标，但也会成倍增加显存占用。PSNR/SSIM 是
CPU 指标，可以较安全地增加该值。

例如，使用物理 GPU 0 和 2、每张卡 3 个进程，总共会启动 6 个 worker：

```bash
"$CORE_PYTHON" -m video_quality.cli evaluate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --config video_quality/config/config.yaml \
  --metrics psnr,ssim \
  --gpus 0,2 \
  --processes-per-gpu 3
```

如果只需要运行部分非 JEPA 指标，同样可以使用 `evaluate --gpus`：

```bash
"$CORE_PYTHON" -m video_quality.cli evaluate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --config video_quality/config/config.yaml \
  --metrics aesthetic_quality,subject_consistency,depth_accuracy \
  --gpus 0,1,2,3 \
  --processes-per-gpu 1
```

第一次评估或排查问题时，建议使用上一节的分阶段方式。

## 7. 只运行部分指标

可用名称：

```text
psnr
ssim
aesthetic_quality
image_quality
jepa_similarity
subject_consistency
trajectory_accuracy
depth_accuracy
```

`--metrics` 接收逗号分隔的名称，不接受字面值 `all`。省略 `--metrics` 才表示
选择全部八个指标。

### 7.1 快速检查 PSNR 和 SSIM

```bash
CORE_PYTHON=$(conda run -n WorldArena python -c 'import sys; print(sys.executable)')
EVAL_ROOT=/absolute/path/to/psnr_ssim_output

"$CORE_PYTHON" -m video_quality.cli prepare \
  --summary /absolute/path/to/summary.json \
  --output-dir "$EVAL_ROOT" \
  --metrics psnr,ssim

"$CORE_PYTHON" -m video_quality.cli evaluate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --config video_quality/config/config.yaml \
  --metrics psnr,ssim

"$CORE_PYTHON" -m video_quality.cli aggregate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --metrics psnr,ssim
```

### 7.2 只运行一个模型指标

以 Image Quality 为例：

```bash
CORE_PYTHON=$(conda run -n WorldArena python -c 'import sys; print(sys.executable)')
EVAL_ROOT=/absolute/path/to/image_quality_output

"$CORE_PYTHON" -m video_quality.cli prepare \
  --summary /absolute/path/to/summary.json \
  --output-dir "$EVAL_ROOT" \
  --metrics image_quality

"$CORE_PYTHON" -m video_quality.cli evaluate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --config video_quality/config/config.yaml \
  --metrics image_quality

"$CORE_PYTHON" -m video_quality.cli aggregate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --metrics image_quality
```

三个阶段的 `--metrics` 应保持一致，或保证后续选择是 manifest 中已声明指标的
子集。不能在 `prepare` 只声明 PSNR 后，直接要求 `evaluate` 运行 SSIM。

## 8. 输出目录说明

完整评估输出结构：

```text
evaluation_output/
├── run_manifest.json
├── metric_inputs.json
├── logs/
│   ├── metric_timings.log
│   └── metric_timings.json
├── cache/
│   ├── png_frames/
│   │   ├── gt_dataset/
│   │   └── generated_dataset/
│   └── jpeg_frames/
│       ├── gt_dataset/
│       └── generated_dataset/
└── results/
    ├── metrics/
    │   ├── psnr.json
    │   ├── ssim.json
    │   ├── aesthetic_quality.json
    │   ├── image_quality.json
    │   ├── subject_consistency.json
    │   ├── trajectory_accuracy.json
    │   └── depth_accuracy.json
    ├── jepa/
    │   ├── intersection_names.json
    │   ├── train.npy
    │   ├── test.npy
    │   └── results.json
    ├── results.json
    └── results.csv
```

说明：

- `run_manifest.json` 保存本次声明的指标和准备后的路径。
- `results/metrics/*.json` 是每个非 JEPA 指标的独立结果。
- JEPA 是数据集级指标，因此只有一个数据集分数。
- 聚合时，同一个 JEPA 分数会写入每个样本行。
- `results.csv` 最后一行是 `AVERAGE`。
- `logs/metric_timings.*` 记录各非 JEPA 阶段和总墙钟耗时，只由 rank 0 写入。
- 完全相同的视频帧可能产生 `PSNR=inf`，这是保留的原始行为。

Trajectory 和 Depth 在输出前会使用 leaderboard 归一化：

```text
Trajectory Accuracy = clip(raw_NDTW / 40.8540, 0, 1)

Depth Accuracy = 1 - clip(
    (raw_AbsRel - 0.2228) / (4.3711 - 0.2228),
    0,
    1
)
```

## 9. Trajectory Accuracy 和 SAM3

Trajectory Accuracy 需要 GT 与生成视频各自的轨迹：

```text
GT episode/traj/traj.npy
generated episode/1/traj/traj.npy
```

在当前评估输出中，对应位置位于 JPEG cache 的 episode 目录下。

规则：

- 两侧轨迹都存在：直接复用。
- 只有一侧缺失：只对缺失侧运行 SAM3。
- 两侧都缺失：分别对两侧运行 SAM3。

SAM3 模型目录：

```text
sam/
├── sam3.pt
└── bpe_simple_vocab_16e6.txt.gz
```

程序不会记录轨迹对应的视频哈希。如果替换了视频或 SAM3 权重，需要手工删除
对应的 `traj/traj.npy`，否则旧轨迹仍会被复用。

## 10. 缓存和重跑规则

### 10.1 最稳妥的方式

每次数据、模型或参数发生变化时，使用新的输出目录：

```text
/data/evaluations/model_a_run_001
/data/evaluations/model_a_run_002
```

### 10.2 `prepare` 的行为

`prepare` 会重建本次所选指标需要的 PNG/JPEG `video` 目录，并重写
`run_manifest.json`。它不会自动清理旧指标结果；之后执行 `aggregate` 时，
已有的指标列会与本次新指标合并到同一个结果表中。合并要求两次评测的
`sample_id` 集合一致。

### 10.3 JEPA 缓存

JEPA 会优先复用：

```text
$EVAL_ROOT/results/jepa/train.npy
$EVAL_ROOT/results/jepa/test.npy
```

如果更换了真实视频或生成视频，必须：

1. 使用新的 `EVAL_ROOT`；或
2. 删除旧的 `train.npy`、`test.npy` 和 `results.json` 后重跑。

否则可能得到基于旧视频特征的结果。

### 10.4 单独重跑失败指标

例如只重跑 Depth Accuracy：

```bash
conda run -n WorldArena python -m video_quality.cli evaluate \
  --manifest /absolute/path/to/evaluation_output/run_manifest.json \
  --output-dir /absolute/path/to/evaluation_output \
  --config video_quality/config/config.yaml \
  --metrics depth_accuracy
```

成功后，再用与目标结果一致的指标列表执行 `aggregate`。

## 11. 常见问题

### 11.1 `torch.cuda.is_available()` 为 `False`

优先检查：

```bash
nvidia-smi
```

这通常是 GPU 未分配、宿主机驱动不可见或容器未启用 NVIDIA 设备，不是重新
安装 PyTorch 就能解决的问题。

### 11.2 `Checkpoint does not exist`

检查当前项目是否仍位于：

```text
<project-root>
```

如果项目被移动，只需要继续从新的项目根目录运行；配置中的相对模型路径无需修改。
如果某个模型单独放在项目目录外，可以只为该模型填写绝对路径。

### 11.3 `Summary JSON must be a non-empty list`

summary 顶层必须是 JSON 列表，而不是单个对象：

```json
[
  {"gt_path": "...", "generated_video": "..."}
]
```

### 11.4 `Duplicate sample_id`

不同 GT 视频使用了相同文件名。给它们建立唯一名称的副本或软链接，然后更新
summary。

### 11.5 `Video contains no decodable frames`

先用 FFmpeg 检查：

```bash
ffprobe /absolute/path/to/video.mp4
```

必要时重新编码为常见的 H.264 MP4。

### 11.6 JEPA 找不到配对

检查两个目录顶层的文件：

```bash
find /absolute/path/to/jepa_real -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' | sort
find /absolute/path/to/jepa_generated -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' | sort
```

两侧文件 stem 必须有交集。

### 11.7 JEPA 结果没有随输入变化

删除旧缓存，或换一个输出目录：

```bash
rm /absolute/path/to/evaluation_output/results/jepa/train.npy
rm /absolute/path/to/evaluation_output/results/jepa/test.npy
rm /absolute/path/to/evaluation_output/results/jepa/results.json
```

执行删除前应确认路径确实属于当前评估输出。

### 11.8 GPU 显存不足

建议：

1. 分指标执行，避免一个长流程中连续持有模型和缓存。
2. 先用少量视频完成端到端验证。
3. JEPA 可在 `video_quality/JEDi/configs/vith16_ssv2_16x2x3.yaml` 中降低
   `optimization.batch_size`，默认是 4。
4. 确认 GPU 上没有无关进程。

### 11.9 `pip check` 报 pyiqa/Transformers 冲突

当前部署保留项目要求的：

```text
pyiqa==0.1.14.1
transformers==4.51.3
```

`pyiqa` 的包元数据固定声明 `transformers==4.37.2`，但当前项目明确使用
4.51.3，因此安装时只对 `pyiqa` 本体使用了无依赖安装。相关模块已经过导入和
模型构造验证。不要仅为了让 `pip check` 无警告而直接降级 Transformers。

### 11.10 `pip check` 提示 decord 不支持平台

当前安装的 `decord==0.6.0` 已经通过实际导入验证。若评估时确实出现视频解码
错误，再根据具体视频和动态库报错处理，不必仅因 `pip check` 提示重复安装。

## 12. 验证项目

运行不需要数据集的单元测试：

```bash
cd /path/to/WorldArena

conda run -n WorldArena python -m pytest -q \
  video_quality/tests
```

当前预期：

```text
15 passed, 1 skipped
```

PSNR 测试可能产生 `divide by zero` 警告，因为完全相同帧的 PSNR 是正无穷；
这不是测试失败。

## 13. 首次正式评估建议

建议按以下顺序：

1. 先用 1～2 对视频运行 `psnr,ssim`。
2. 再对同一小样本分别验证五个核心模型指标。
3. 构造同名 JEPA 目录并单独运行 JEPA。
4. 检查每个独立 JSON。
5. 最后对完整数据集运行八指标并聚合。

这样可以尽早发现视频解码、命名、GPU、显存和轨迹缓存问题，避免完整任务运行
很久后才失败。
