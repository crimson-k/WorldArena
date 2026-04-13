# WorldArena 评测命令说明与数据目录结构（对接版）

本文用于团队内外对接，统一说明：
1. 各脚本参数含义。
2. `video_quality/data/` 下每个模型目录的标准结构。
3. 分片评测与最终聚合关系。

## 0. 路径变量约定

| 变量 | 含义 |
|---|---|
| `MODEL_NAME` | 模型名，例如 `RoboTwin_Clean_50_multi_node_4_202604111721_step_17500` |
| `ROOT` | 仓库根目录：`/data/liuwenhao/WorldArena` |
| `VQ_ROOT` | 视频评测目录：`/data/liuwenhao/WorldArena/video_quality` |
| `MODEL_ROOT` | 模型数据目录：`$VQ_ROOT/data/$MODEL_NAME` |
| `GEN_TEST_DIR` | 生成视频目录：`$MODEL_ROOT/worldarena_auto_eval/${MODEL_NAME}_test` |
| `GEN_TEST_VLM_DIR` | VLM 生成视频目录：`$MODEL_ROOT/worldarena_auto_eval/${MODEL_NAME}_test_vlm` |
| `SUMMARY_JSON` | 指令与样本映射：`$MODEL_ROOT/worldarena_auto_eval/summary.json` |
| `GT_VIDEOS_DIR` | GT 视频目录：`$MODEL_ROOT/worldarena_auto_eval/gt_videos` |
| `CONFIG_PATH` | 配置：`$VQ_ROOT/config/config.yaml` |

---

## 1. 数据准备脚本

### 1.1 `data_process.py`

命令：

```bash
python /data/liuwenhao/WorldArena/data_process.py \
  --results_dir <RESULTS_DIR> \
  --model_name <MODEL_NAME> \
  --num_shards <N>
```

参数含义：

| 参数 | 必填 | 含义 |
|---|---|---|
| `--results_dir` | 是 | 某个模型的结果目录，通常为 `video_quality/data/<MODEL_NAME>` |
| `--model_name` | 是 | 模型名，用于生成 `<MODEL_NAME>_test` 与 `<MODEL_NAME>_test_vlm` |
| `--num_shards` | 否 | 分片数量。大于 1 时会生成 `worldarena_auto_eval/shards/shard_xx/` |

输出关键目录：
- `worldarena_auto_eval/summary.json`
- `worldarena_auto_eval/gt_videos/`
- `worldarena_auto_eval/gt_first_frames/`
- `worldarena_auto_eval/<MODEL_NAME>_test/`
- `worldarena_auto_eval/<MODEL_NAME>_test_vlm/`
- `worldarena_auto_eval/shards/shard_xx/...`（可选）

---

## 2. 普通指标评测脚本

### 2.1 `run_evaluation.sh`

命令：

```bash
bash /data/liuwenhao/WorldArena/video_quality/run_evaluation.sh \
  <MODEL_NAME> \
  <GEN_VIDEO_DIR> \
  <SUMMARY_JSON> \
  <METRIC_LIST> \
  [CONFIG_PATH] \
  [SHARD_INDEX] [NUM_SHARDS]
```

参数含义：

| 位置参数 | 必填 | 含义 |
|---|---|---|
| `MODEL_NAME` | 是 | 基础模型名，分片模式下会自动扩展为 `MODEL_NAME_shard_xx` |
| `GEN_VIDEO_DIR` | 是 | 生成视频目录（通常是 `.../<MODEL_NAME>_test`） |
| `SUMMARY_JSON` | 是 | 样本清单（通常是 `.../summary.json`） |
| `METRIC_LIST` | 是 | 逗号分隔的指标字符串 |
| `CONFIG_PATH` | 否 | 配置路径，默认 `./config/config.yaml` |
| `SHARD_INDEX` | 否 | 分片编号（从 0 开始） |
| `NUM_SHARDS` | 否 | 分片总数，需和 `SHARD_INDEX` 同时传入 |

`METRIC_LIST` 支持示例：
- 传统指标：`semantic_alignment,aesthetic_quality,...,trajectory_accuracy,psnr,ssim`
- 新增指标：`mse,lpips,fid,fvd`

说明：
- `action_following` 不在该脚本内执行，需要单独跑 `run_action_following.sh`。
- 分片模式下，脚本会自动读取 `summary.json` 所在目录下的 `shards/shard_xx/`。

### 2.2 多 GPU 分片模板（推荐，GPU 与 shard 解耦）

常见错误：把 `CUDA_VISIBLE_DEVICES` 和 `SHARD_INDEX` 绑定成同一个变量，导致 `for shard in 1 2` 时传入非法分片索引。

正确方式：`shard_idx` 固定走 `0..N-1`，`gpu` 由你自定义。

```bash
cd /data/liuwenhao/WorldArena/video_quality

MODEL_NAME=RoboTwin_Clean_50_multi_node_4_202604111721_step_17500
GEN_VIDEO_DIR=/data/liuwenhao/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/${MODEL_NAME}_test
SUMMARY_JSON=/data/liuwenhao/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/summary.json
CONFIG_PATH=/data/liuwenhao/WorldArena/video_quality/config/config.yaml
METRICS="semantic_alignment,aesthetic_quality,background_consistency,dynamic_degree,flow_score,photometric_smoothness,motion_smoothness,subject_consistency,image_quality,depth_accuracy,trajectory_accuracy,psnr,ssim,mse,lpips,fid,fvd"

GPUS=(1 2)                  # 可任意指定，例如 (3 5)
NUM_SHARDS=${#GPUS[@]}      # 分片数必须等于使用 GPU 数量

for shard_idx in "${!GPUS[@]}"; do
  gpu=${GPUS[$shard_idx]}
  nohup env CUDA_VISIBLE_DEVICES=$gpu bash run_evaluation.sh \
    "$MODEL_NAME" \
    "$GEN_VIDEO_DIR" \
    "$SUMMARY_JSON" \
    "$METRICS" \
    "$CONFIG_PATH" \
    "$shard_idx" "$NUM_SHARDS" \
    > "./nohup_log/output_run_evaluation_${MODEL_NAME}_shard${shard_idx}_gpu${gpu}.log" 2>&1 &
done
```

规则：
- `SHARD_INDEX` 只能是 `0..NUM_SHARDS-1`。
- `NUM_SHARDS` 要和 `worldarena_auto_eval/shards/` 的实际分片数量一致。
- GPU 编号可以是任意可用卡，不要求连续。

---

## 3. Action Following 评测脚本

### 3.1 `run_action_following.sh`

命令：

```bash
bash /data/liuwenhao/WorldArena/video_quality/run_action_following.sh \
  <MODEL_NAME> \
  <GEN_VIDEO_DIR> \
  <SUMMARY_JSON> \
  [CONFIG_PATH]
```

参数含义：

| 位置参数 | 必填 | 含义 |
|---|---|---|
| `MODEL_NAME` | 是 | 模型名 |
| `GEN_VIDEO_DIR` | 是 | 生成视频目录（`_test`） |
| `SUMMARY_JSON` | 是 | 样本清单 |
| `CONFIG_PATH` | 否 | 配置文件路径 |

输出目录：
- `video_quality/data_action_following/<MODEL_NAME>/generated_dataset/...`
- `video_quality/data_action_following/<MODEL_NAME>/gt_dataset/...`

---

## 4. VLM 评测脚本

### 4.1 `run_VLM_judge.sh`

命令：

```bash
bash /data/liuwenhao/WorldArena/video_quality/run_VLM_judge.sh \
  <MODEL_NAME> \
  <VIDEO_DIR> \
  <SUMMARY_JSON> \
  [METRICS] \
  [CONFIG_PATH] \
  [SHARD_INDEX] [NUM_SHARDS]
```

参数含义：

| 位置参数 | 必填 | 含义 |
|---|---|---|
| `MODEL_NAME` | 是 | 基础模型名 |
| `VIDEO_DIR` | 是 | VLM 输入视频目录（通常 `.../<MODEL_NAME>_test_vlm`） |
| `SUMMARY_JSON` | 是 | 样本清单 |
| `METRICS` | 否 | `all` 或指定 VLM 维度 |
| `CONFIG_PATH` | 否 | 配置文件路径 |
| `SHARD_INDEX` | 否 | 分片编号（从 0 开始） |
| `NUM_SHARDS` | 否 | 分片总数 |

输出目录：
- `video_quality/data/<EFFECTIVE_MODEL_NAME>/output_VLM/<EFFECTIVE_MODEL_NAME>/*.json`
- `video_quality/data/<EFFECTIVE_MODEL_NAME>/tmp_VLM/...`

### 4.2 VLM 多 GPU 分片模板（同样解耦）

```bash
cd /data/liuwenhao/WorldArena/video_quality

MODEL_NAME=RoboTwin_Clean_50_multi_node_4_202604111721_step_17500
VIDEO_DIR=/data/liuwenhao/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/${MODEL_NAME}_test_vlm
SUMMARY_JSON=/data/liuwenhao/WorldArena/video_quality/data/${MODEL_NAME}/worldarena_auto_eval/summary.json
CONFIG_PATH=/data/liuwenhao/WorldArena/video_quality/config/config.yaml

GPUS=(1 2)
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
done
```

---

## 5. JEPA 评测脚本

### 5.1 `run_evaluation_JEPA.sh`

命令：

```bash
bash /data/liuwenhao/WorldArena/video_quality/run_evaluation_JEPA.sh \
  <MODEL_NAME> \
  <GEN_VIDEO_DIR> \
  <GT_VIDEO_DIR>
```

参数含义：

| 位置参数 | 必填 | 含义 |
|---|---|---|
| `MODEL_NAME` | 是 | 模型名 |
| `GEN_VIDEO_DIR` | 是 | 生成视频目录（通常 `_test_vlm`） |
| `GT_VIDEO_DIR` | 是 | GT 视频目录（通常 `worldarena_auto_eval/gt_videos`） |

输出目录：
- `video_quality/data/<MODEL_NAME>/output_JEDi/results.json`

---

## 6. 结果聚合脚本

### 6.1 `csv_results/aggregate_results.py`

单模型聚合命令：

```bash
python /data/liuwenhao/WorldArena/video_quality/csv_results/aggregate_results.py \
  --base_dir /data/liuwenhao/WorldArena/video_quality/data/<MODEL_NAME> \
  --model_name <MODEL_NAME> \
  --csv_name <MODEL_NAME>_aggregated_results.csv \
  --jepa_result_path /data/liuwenhao/WorldArena/video_quality/data/<MODEL_NAME>/output_JEDi/generated_results.json
```

分片总聚合推荐命令：

```bash
python /data/liuwenhao/WorldArena/video_quality/csv_results/aggregate_results.py \
  --model_name <MODEL_NAME> \
  --video_quality_root /data/liuwenhao/WorldArena/video_quality \
  --num_shards <N> \
  --shard_ids 00,01,... \
  --csv_name <MODEL_NAME>.csv \
  --jepa_result_path /data/liuwenhao/WorldArena/video_quality/data/<MODEL_NAME>/output_JEDi/results.json
```

参数含义：

| 参数 | 必填 | 含义 |
|---|---|---|
| `--base_dir` | 单模型模式必填 | 单模型目录（如 `.../data/<MODEL_NAME>`） |
| `--model_name` | 是 | 基础模型名 |
| `--vlm_model_dir` | 单模型模式可选 | `output_VLM` 下子目录名，默认同 `model_name` |
| `--video_quality_root` | 分片模式建议 | `video_quality` 根目录 |
| `--num_shards` | 分片模式可选 | 分片数 |
| `--shard_ids` | 分片模式建议 | 明确分片后缀，如 `00,01` |
| `--global_vlm_model_dir` | 分片模式可选 | 分片总表回填 VLM 时使用的子目录名 |
| `--csv_name` | 否 | 输出 CSV 文件名 |
| `--jepa_result_path` | 否 | JEPA 结果 JSON 路径，支持逐条或全局 score |

说明：
- 当前聚合列已包含 `PSNR, SSIM, MSE, LPIPS, FID, FVD`。
- 最后一行会自动追加 `AVERAGE` 平均值。
- 列顺序规则：普通指标按名称顺序，`PSNR, SSIM, MSE, LPIPS, FID, FVD` 固定放在最后六列。
- JEPA 支持两种输入：
  - 逐条：`output_JEDi/generated_results.json`（优先）
  - 全局：`output_JEDi/results.json`（无逐条时回填全局分）

### 6.2 `csv_results/build_ewm_leaderboard.py`

用途：
- 基于聚合后的模型 CSV（二次计算）得到：
  - 六大维度分数
  - `EWMScore`
  - 与现有榜单合并后的新榜单
- `Action Following` 缺失时按 `0` 处理（可通过参数改）。

推荐命令：

```bash
python /data/liuwenhao/WorldArena/video_quality/csv_results/build_ewm_leaderboard.py \
  --aggregated_csv /data/liuwenhao/WorldArena/video_quality/data/<MODEL_NAME>/csv_results/<MODEL_NAME>.csv \
  --leaderboard_csv /data/liuwenhao/WorldArena/worldarena_leaderboard.csv \
  --my_model_name <MODEL_NAME> \
  --my_open_source Open-source \
  --my_year 2026 \
  --output_csv /data/liuwenhao/WorldArena/worldarena_leaderboard_with_my_model.csv \
  --output_md /data/liuwenhao/WorldArena/worldarena_leaderboard_with_my_model.md
```

关键参数：
- `--my_model_name`：最终汇总表中的 `Model` 名称（你可自定义）。
- `--scale_mode`：默认 `auto`，当聚合 CSV 分数在 `[0,1]` 区间时自动乘 `100`。
- `--action_following_default`：默认 `0.0`。

输出说明：
- `output_csv`：数值型合并榜单。
- `output_md`：Markdown 榜单；`EWMScore`、六维以及 16 个基础指标都会自动标注 `(MAX)`；16 个基础指标按单词字母序展示。

---

## 7. `video_quality/data/` 下模型目录标准结构

> 下述是“对接时应遵守”的结构模板。具体模型可能缺少部分目录（取决于是否执行了对应评测）。

### 7.1 非分片模型目录（`data/<MODEL_NAME>/`）

```text
video_quality/data/<MODEL_NAME>/
├── worldarena_auto_eval/
│   ├── summary.json
│   ├── gt_videos/
│   ├── gt_first_frames/
│   ├── <MODEL_NAME>_test/
│   ├── <MODEL_NAME>_test_vlm/
│   └── shards/                       # 可选，只有做分片时才有
│       ├── shard_00/
│       │   ├── summary.json
│       │   ├── gt_videos/
│       │   ├── gt_first_frames/
│       │   ├── <MODEL_NAME>_test/
│       │   └── <MODEL_NAME>_test_vlm/
│       └── shard_01/
├── generated_dataset/                # 普通指标预处理后产物（可选）
├── gt_dataset/                       # 普通指标预处理后产物（可选）
├── output/                           # 普通指标结果 generated_results.json
├── output_VLM/
│   └── <MODEL_NAME>/                 # VLM 评测输出 json
├── tmp_VLM/
├── output_JEDi/
│   ├── results.json
│   └── features_cache/
├── csv_results/
└── videos/                           # 原始拼接视频（可选）
```

### 7.2 分片模型目录（`data/<MODEL_NAME>_shard_xx/`）

```text
video_quality/data/<MODEL_NAME>_shard_xx/
├── generated_dataset/
├── gt_dataset/
├── output/                           # 普通指标该 shard 的结果
└── csv_results/                      # 该 shard 的中间聚合结果
```

说明：
- VLM 与 JEPA 通常先在主目录 `data/<MODEL_NAME>/` 统一产出，再在最终聚合时按 `Video_ID` 回填到分片总表。

### 7.3 Action Following 专用目录（`data_action_following/<MODEL_NAME>/`）

```text
video_quality/data_action_following/<MODEL_NAME>/
├── generated_dataset/
│   └── <task>/<episode>/{1,2,3}/video/frame_*.jpg
└── gt_dataset/
    └── <task>/<episode>/
        ├── prompt/
        │   ├── init_frame.png
        │   └── prompt.txt
        └── video/frame_*.jpg
```

---

## 8. 当前目录已存在模型及结构类型（便于对接）

| 模型目录 | 结构类型 |
|---|---|
| `test_10` | 非分片完整结构（含 `output/output_VLM/output_JEDi/csv_results`） |
| `test_60` | 非分片完整结构（含 `output/output_VLM/output_JEDi/csv_results`） |
| `RoboTwin_Clean_50_multi_node_4_202604111721` | 非分片结构（含 `generated_dataset/gt_dataset/output_VLM/output_JEDi`） |
| `RoboTwin_Clean_50_multi_node_4_202604111721_step_17500` | 主目录 + `worldarena_auto_eval/shards/` + `output_VLM/output_JEDi/csv_results` |
| `RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_00` | 分片目录（`generated_dataset/gt_dataset/output/csv_results`） |
| `RoboTwin_Clean_50_multi_node_4_202604111721_step_17500_shard_01` | 分片目录（`generated_dataset/gt_dataset/output/csv_results`） |
| `RoboTwin_Clean_50_multi_node_4_202604111721_step_6000` | 仅原始 `videos/`，未完成评测产物 |

---

## 9. 指标与模型依赖（需要提前下载）

| 指标 | 是否需要额外模型 | config 键 | 典型文件 |
|---|---|---|---|
| `psnr` | 否 | 无 | 无 |
| `ssim` | 否 | 无 | 无 |
| `mse` | 否 | 无 | 无 |
| `lpips` | 是 | `ckpt.lpips.alexnet` | `alexnet-owt-7be5be79.pth` |
| `fid` | 是 | `ckpt.fid.inception` | `pt_inception-2015-12-05-6726825d.pth` |
| `fvd` | 是 | `ckpt.fvd.i3d` | `i3d_torchscript.pt` |
| `semantic_alignment` | 是 | `ckpt.semantic_alignment.caption` / `CLIP` | Qwen2.5-VL + CLIP |
| `subject_consistency` | 是 | `ckpt.subject_consistency.*` | DINO repo + weight + RAFT |
| `flow_score` / `dynamic_degree` / `background_consistency` | 是 | `ckpt.<metric>.raft` | `raft-things.pth` |
| `aesthetic_quality` | 是 | `ckpt.aesthetic_quality.*` | ViT-L-14 + head |
| `image_quality` | 是 | `ckpt.image_quality.musiq` | musiq checkpoint |
| `photometric_smoothness` | 是 | `ckpt.photometric_smoothness.*` | SEA-RAFT cfg + weight |
| `motion_smoothness` | 是 | `ckpt.motion_smoothness.model` | VFIMamba weight |
| `depth_accuracy` | 是 | `ckpt.depth_accuracy` | Depth Anything V2 |

---

## 10. 六维度与 EWMScore 说明

- 当前仓库代码中**没有内置**“六个维度综合分”或 `EWMScore` 的计算逻辑。
- 现在的 `aggregate_results.py` 只做逐视频指标汇总 + `AVERAGE` 行，不会自动生成综合总分列。
- 如果你们要对外统一“六维 + EWMScore”，需要单独约定：
  1. 哪六个维度参与。
  2. 每个维度正向/反向归一化方式。
  3. 权重来源（固定权重或熵权法）。
  4. 最终公式与保留小数位。

---

## 11. 推荐执行顺序（分片）

1. 准备数据并分片：`data_process.py --num_shards N`
2. 并行跑普通指标：`run_evaluation.sh ... SHARD_INDEX NUM_SHARDS`
3. 跑 VLM：`run_VLM_judge.sh ... SHARD_INDEX NUM_SHARDS` 或主目录一次性跑
4. 跑 JEPA：`run_evaluation_JEPA.sh`（通常主目录一次）
5. 最终聚合：`aggregate_results.py --model_name ... --num_shards ... --shard_ids ...`
