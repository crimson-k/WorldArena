# WorldArena 八指标评估

[English](README.md)

本评估工具只保留以下 8 个指标：

1. PSNR
2. SSIM
3. Aesthetic Quality（美学质量/高保真）
4. Image Quality（图像质量/高保真）
5. JEPA Similarity（全局一致性）
6. Subject Consistency（主体一致性/物体幻觉）
7. Trajectory Accuracy（轨迹准确性/动作跟随）
8. Depth Accuracy（深度准确性）

Track2 和其他视频质量指标均不属于当前评估工具。

## 源码边界

WorldArena 参考源码固定为 GitHub 提交
[`a918b93f8533a4e9452a224c0ec54d27e527c4bb`](https://github.com/tsinghua-fib-lab/WorldArena/commit/a918b93f8533a4e9452a224c0ec54d27e527c4bb)。

| 指标 | 当前使用的实现 |
|---|---|
| PSNR / SSIM | 固定提交中的 `WorldArena/basic_metrics.py`，未修改 |
| Aesthetic Quality | 保留现有本地 WorldArena 实现；评分逻辑未修改，只禁用了隐式权重下载，并为 `torch.load` 增加 CPU 映射 |
| Image Quality | 保留现有本地 WorldArena MUSIQ 实现；与固定提交相比只移除了未使用的 import |
| JEPA Similarity | 固定提交中的 `JEDi/batch.py`；仅将 `model_dir`、`config_path` 和 `output_root` 传入 `JEDiMetric` |
| Subject Consistency | 固定提交中的 `subject_consistency.py` 和 `dynamic_degree.py`，未修改 |
| Trajectory Accuracy | 固定提交中的 `trajectory_accuracy.py`，未修改 |
| Depth Accuracy | 固定提交中的 `depth_accuracy.py`，未修改 |

上游 `subject_consistency.py` 会导入一个未定义且未使用的 `CACHE_DIR`。
当前只在 `WorldArena/utils.py` 中补充了这个兼容变量，没有修改指标文件。
外层调用代码只负责准备输入、调用一个指标并保存该指标的结果。

## 输入格式

输入 summary 是一个非空 JSON 列表：

```json
[
  {
    "gt_path": "/absolute/path/to/episode0.mp4",
    "generated_video": "/absolute/path/to/generated_episode0.mp4"
  }
]
```

GT 视频文件名（不含扩展名）会作为 `sample_id`，因此所有 GT 视频文件名必须唯一。

`prepare` 只重建所选指标需要的帧目录：

- PSNR/SSIM：GT 和生成视频都抽取为无损 PNG。固定版本的
  `basic_metrics.py` 要求 GT 文件名符合 `frame_*.png`。
- Aesthetic、Image、Subject、Trajectory、Depth：抽取为 JPEG，使用
  OpenCV 默认 JPEG quality=95。
- JEPA：直接读取 MP4，不使用抽帧目录。

manifest 只保存路径，不保存视频哈希、模型签名、缓存元数据、分片信息，
也不会自动进行复杂的文件匹配。

## 命令

### 1. 准备输入

为全部 8 个指标准备输入：

```bash
python -m video_quality.cli prepare \
  --summary /path/to/summary.json \
  --output-dir /path/to/evaluation
```

### 2. 执行七个非 JEPA 指标

```bash
python -m video_quality.cli evaluate \
  --manifest /path/to/evaluation/run_manifest.json \
  --output-dir /path/to/evaluation \
  --config video_quality/config/config.yaml
```

### 3. 执行 JEPA Similarity

JEPA 是独立的 dataset-level 指标。需要手工准备两个非递归目录，两个目录中
参与评估的 MP4 应使用相同文件名：

```text
real_mp4/
├── episode0.mp4
└── episode1.mp4

generated_mp4/
├── episode0.mp4
└── episode1.mp4
```

然后执行：

```bash
/path/to/WorldArena_JEPA/bin/python -m video_quality.cli jepa \
  --real-dir /path/to/real_mp4 \
  --gen-dir /path/to/generated_mp4 \
  --output-dir /path/to/evaluation \
  --config video_quality/config/config.yaml \
  --jepa-python /path/to/WorldArena_JEPA/bin/python
```

固定版本的 JEPA 源码会使用两个目录中文件名的交集，并跳过无效视频对。
每个视频均匀采样 16 帧，缩放到 224×224，然后用以下公式将 JEDi 距离
`D` 转换为相似度：

```text
JEPA Similarity = exp(-0.4D)
```

`videojedi` 会在 JEPA 输出目录下写入 `train.npy` 和 `test.npy`。再次使用
同一个输出目录时会复用这两个文件。如果输入视频发生变化，需要删除这两个
文件，或改用新的输出目录。

### 4. 聚合结果

```bash
python -m video_quality.cli aggregate \
  --manifest /path/to/evaluation/run_manifest.json \
  --output-dir /path/to/evaluation
```

`prepare`、`evaluate` 和 `aggregate` 都支持用 `--metrics` 指定逗号分隔的
指标子集。可用名称为：

```text
psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,
subject_consistency,trajectory_accuracy,depth_accuracy
```

例如，只执行 PSNR 和 SSIM：

```bash
python -m video_quality.cli prepare \
  --summary /path/to/summary.json \
  --output-dir /path/to/evaluation \
  --metrics psnr,ssim

python -m video_quality.cli evaluate \
  --manifest /path/to/evaluation/run_manifest.json \
  --output-dir /path/to/evaluation \
  --config video_quality/config/config.yaml \
  --metrics psnr,ssim

python -m video_quality.cli aggregate \
  --manifest /path/to/evaluation/run_manifest.json \
  --output-dir /path/to/evaluation \
  --metrics psnr,ssim
```

也可以使用 `all` 一次执行 prepare、非 JEPA 指标、JEPA 和 aggregate。
选择 JEPA 时必须同时提供 `--jepa-real-dir` 和 `--jepa-gen-dir`：

```bash
python -m video_quality.cli all \
  --summary /path/to/summary.json \
  --output-dir /path/to/evaluation \
  --config video_quality/config/config.yaml \
  --jepa-real-dir /path/to/real_mp4 \
  --jepa-gen-dir /path/to/generated_mp4 \
  --jepa-python /path/to/WorldArena_JEPA/bin/python
```

## 指标输出

每个指标单独保存，不共用 partial result：

```text
evaluation/
├── run_manifest.json
├── metric_inputs.json
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
    │   └── results.json
    ├── results.json
    └── results.csv
```

聚合结果包含每个视频一行，以及一行等权重的 `AVERAGE`。JEPA 是数据集级
指标，因此同一个 JEPA 分数会复制到每个视频行中。

PSNR 和 SSIM 保留固定版本 scikit-image 的行为。完全相同的帧，其 PSNR
为正无穷 `inf`，不会人为截断为 100。

Trajectory 和 Depth 输出 WorldArena leaderboard 归一化结果，而不是原始值：

```text
Trajectory Accuracy = clip(raw_NDTW / 40.8540, 0, 1)

Depth Accuracy = 1 - clip(
    (raw_AbsRel - 0.2228) / (4.3711 - 0.2228),
    0,
    1
)
```

其余指标保持源码输出。

## Trajectory 文件与 SAM3

每个样本预期使用以下轨迹文件：

```text
GT episode/traj/traj.npy
generated episode/1/traj/traj.npy
```

处理规则如下：

- 两侧 `traj.npy` 都存在：直接复用，不运行 SAM3。
- 只有 GT 缺失：只为 GT 运行 SAM3。
- 只有生成侧缺失：只为生成侧运行 SAM3。
- 两侧都缺失：分别为两侧运行 SAM3。

程序不检查视频哈希、SAM3 模型签名或轨迹元数据。如果视频或模型发生变化，
需要手工删除对应的 `traj.npy`，才能重新生成该侧轨迹。

SAM3 配置目录顶层必须包含：

```text
sam3.pt
bpe_simple_vocab_16e6.txt.gz
```

当前配置路径为：

```text
/data1/liuwenhao/Projects/WorldArena/sam
```

## 模型权重

权重路径在 `config/config.yaml` 中配置。只有实际调用某个指标时，程序才会
读取该指标对应的路径。

### Aesthetic Quality

需要 OpenAI CLIP ViT-L/14 和 LAION aesthetic linear head：

```bash
mkdir -p video_quality/models_downloaded/aesthetic_quality

wget -O video_quality/models_downloaded/aesthetic_quality/ViT-L-14.pt \
  https://huggingface.co/jinaai/clip-models/resolve/main/ViT-L-14.pt

wget -O video_quality/models_downloaded/aesthetic_quality/sa_0_4_vit_l_14_linear.pth \
  'https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true'
```

对应配置：

```yaml
ckpt:
  aesthetic_quality:
    clip: /path/to/ViT-L-14.pt
    aesthetic_head: /path/to/sa_0_4_vit_l_14_linear.pth
```

### Image Quality

使用 IQA-PyTorch 的 MUSIQ SPAQ 权重：

```bash
mkdir -p video_quality/models_downloaded/image_quality

wget -O video_quality/models_downloaded/image_quality/musiq_spaq_ckpt-358bb6af.pth \
  https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_spaq_ckpt-358bb6af.pth
```

### Subject Consistency

需要本地 Facebook DINO 源码、DINO ViT-B/16 权重和 RAFT 权重：

```bash
mkdir -p video_quality/models_downloaded/subject_consistency

git clone https://github.com/facebookresearch/dino.git \
  video_quality/models_downloaded/subject_consistency/facebookresearch_dino_main

wget -O video_quality/models_downloaded/subject_consistency/dino_vitbase16_pretrain.pth \
  https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth

cd video_quality/WorldArena/third_party/RAFT
./download_models.sh
cd ../../../..
```

将 RAFT 的 `raft-things.pth`、DINO 仓库和 DINO 权重路径写入配置。

### Depth Accuracy

需要完整的 Depth Anything V2 Small Hugging Face 目录，不能只下载单个权重文件：

```bash
mkdir -p video_quality/models_downloaded/depth_accuracy

hf download depth-anything/Depth-Anything-V2-Small-hf \
  --local-dir video_quality/models_downloaded/depth_accuracy/Depth-Anything-V2-Small-hf
```

### JEPA Similarity

WorldArena README 给出的下载命令为：

```bash
mkdir -p video_quality/JEDi/pretrained_models

wget -O video_quality/JEDi/pretrained_models/vith16.pth.tar \
  https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar

wget -O video_quality/JEDi/pretrained_models/ssv2-probe.pth.tar \
  https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar
```

两个文件名必须分别为：

```text
vith16.pth.tar
ssv2-probe.pth.tar
```

## 环境

核心指标和 JEPA 使用两个独立的 Python 环境。必须从 `video_quality/` 目录
执行以下命令，这样 YAML 中相对路径形式的 requirements 文件才能被正确解析。

### Core 环境

该环境用于 PSNR、SSIM、Aesthetic、Image、Subject、Trajectory 和 Depth：

```bash
cd /path/to/WorldArena/video_quality
conda env create -f environment-core.yml
conda activate WorldArena
```

如果 `WorldArena` 环境已经存在，使用以下命令更新：

```bash
cd /path/to/WorldArena/video_quality
conda env update -n WorldArena -f environment-core.yml --prune
```

### JEPA 环境

JEPA 使用独立环境：

```bash
cd /path/to/WorldArena/video_quality
conda env create -f environment-jepa.yml
conda activate WorldArena_JEPA
```

如果 `WorldArena_JEPA` 已经存在，使用以下命令更新：

```bash
cd /path/to/WorldArena/video_quality
conda env update -n WorldArena_JEPA -f environment-jepa.yml --prune
```

### 验证安装

在 Core 环境中检查 CLI：

```bash
conda activate WorldArena
cd /path/to/WorldArena
python -m video_quality.cli --help
```

已经删除的 VLM 指标不属于当前评估工具，因此不再需要
`WorldArena_VLM` 环境。

### Apptainer/Singularity 容器

容器定义会同时创建 Core 和 JEPA 两个 Conda 环境：

```bash
cd /path/to/WorldArena/video_quality
apptainer build containers/WorldArena.sif containers/WorldArena.def
```

如果集群环境要求管理员权限，可以使用 `sudo apptainer build`；使用旧版
Singularity 时可将命令换成 `singularity build`。

容器定义位于 `containers/WorldArena.def`，Slurm 示例位于
`slurm/run_eight_metrics.sbatch`。
