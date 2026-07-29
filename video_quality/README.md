# WorldArena eight-metric video evaluation

[中文说明](README_CN.md) | [Detailed Chinese guide](USAGE_CN.md)

This focused evaluator keeps:

`PSNR`, `SSIM`, `Aesthetic Quality`, `Image Quality`, `JEPA Similarity`,
`Subject Consistency`, `Trajectory Accuracy`, and `Depth Accuracy`.

Run every command from the repository root. Model paths in
`video_quality/config/config.yaml` are repository-relative, so moving the clone
does not require editing the config.

## Create the two environments

The complete normal dependencies are declared in two Conda YAML files:

- `video_quality/environment-core.yml`: seven non-JEPA metrics.
- `video_quality/environment-jepa.yml`: JEPA.

Create clean environments:

```bash
conda env create -f video_quality/environment-core.yml
conda run --no-capture-output -n WorldArena \
  python -m pip install --no-deps pyiqa==0.1.14.1

conda env create -f video_quality/environment-jepa.yml
conda run --no-capture-output -n WorldArena_JEPA \
  python -m pip install --no-deps vjepa==0.1.2 videojedi==1.1.0
```

For environments that already exist, replace `env create` with:

```bash
conda env update -f video_quality/environment-core.yml --prune
conda env update -f video_quality/environment-jepa.yml --prune
```

The two explicit `--no-deps` commands are intentional:

- `pyiqa==0.1.14.1` incorrectly pins `transformers==4.37.2`, while Depth
  Anything and this evaluator use `transformers==4.51.3`.
- `vjepa==0.1.2` incorrectly restricts `torchvision<0.20`; RTX 5090 requires
  the Blackwell-compatible PyTorch 2.10 / CUDA 12.8 stack.

Verify both environments, including a real CUDA kernel:

```bash
conda run --no-capture-output -n WorldArena python -c \
  "import torch, pyiqa; x=torch.ones(1,device='cuda'); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),float(x.sum()))"

conda run --no-capture-output -n WorldArena_JEPA python -c \
  "import torch, videojedi, vjepa; x=torch.ones(1,device='cuda'); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),float(x.sum()))"
```

## Input summaries

### Separate GT and generated videos

```json
[
  {
    "sample_id": "task__episode_000040",
    "gt_path": "/path/to/gt.mp4",
    "generated_video": "/path/to/generated.mp4"
  }
]
```

`sample_id` is optional for separate videos; otherwise the GT filename stem is
used. IDs must be globally unique.

### Vertically stacked videos

For an MP4 whose top half is GT and bottom half is generated:

```json
[
  {
    "sample_id": "task__episode_000040",
    "stacked_video": "/path/to/stacked.mp4"
  }
]
```

The frame height must be even. The evaluator decodes each MP4 once and crops at
`height / 2`, avoiding intermediate video re-encoding. `sample_id` should be
explicit when identical episode filenames occur under different task folders.

To build a deterministic summary from `<input-root>/<task>/*.mp4`:

```bash
conda run --no-capture-output -n WorldArena \
  python video_quality/build_stacked_summary.py \
  --input-root /path/to/input-root \
  --output /path/to/stacked-summary.json
```

## Run the evaluation

Set reusable paths:

```bash
SUMMARY=/path/to/summary.json
EVAL_ROOT=/path/to/evaluation-output
CONFIG=video_quality/config/config.yaml
METRICS=psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,subject_consistency,trajectory_accuracy,depth_accuracy
```

Prepare PNG/JPEG frame layouts:

```bash
conda run --no-capture-output -n WorldArena \
  python -m video_quality.cli prepare \
  --summary "$SUMMARY" \
  --output-dir "$EVAL_ROOT" \
  --metrics "$METRICS"
```

Run the seven non-JEPA metrics on four GPUs, one worker per GPU:

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

`--gpus` selects physical GPU IDs. `--processes-per-gpu` controls how many model
processes share each selected GPU; one is recommended for model metrics.

Resolve the JEPA interpreter:

```bash
JEPA_PYTHON="$(conda run -n WorldArena_JEPA python -c 'import sys; print(sys.executable)')"
```

For a stacked summary:

```bash
conda run --no-capture-output -n WorldArena \
  python -m video_quality.cli jepa \
  --stacked-summary "$SUMMARY" \
  --output-dir "$EVAL_ROOT" \
  --config "$CONFIG" \
  --jepa-python "$JEPA_PYTHON" \
  --gpu 0
```

For separate inputs, JEPA accepts two non-recursive directories containing
same-stem MP4 files:

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

Aggregate:

```bash
conda run --no-capture-output -n WorldArena \
  python -m video_quality.cli aggregate \
  --manifest "$EVAL_ROOT/run_manifest.json" \
  --output-dir "$EVAL_ROOT" \
  --metrics "$METRICS"
```

## Outputs and timing

Important files:

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

Only distributed rank 0 writes timing logs. Each stage records start time, end
time, elapsed seconds, and total non-JEPA wall time. Timing lines are also
printed to stdout.

Trajectory and Depth are leaderboard-normalized:

```text
Trajectory = clip(raw_NDTW / 40.8540, 0, 1)
Depth      = 1 - clip((raw_AbsRel - 0.2228) / (4.3711 - 0.2228), 0, 1)
```

JEPA reports `exp(-0.4D)`, where `D` is JEDi distance. Dataset-level JEPA is
copied to every sample row during aggregation.

## Model files

Place models at the paths below. Every path is relative to the WorldArena
repository root:

| Model | Path |
| --- | --- |
| CLIP ViT-L/14 | `video_quality/models_downloaded/aesthetic_quality/ViT-L-14.pt` |
| Aesthetic Head | `video_quality/models_downloaded/aesthetic_quality/sa_0_4_vit_l_14_linear.pth` |
| MUSIQ SPAQ | `video_quality/models_downloaded/image_quality/musiq_spaq_ckpt-358bb6af.pth` |
| Facebook DINO source | `video_quality/models_downloaded/subject_consistency/facebookresearch_dino_main/` |
| DINO ViT-B/16 | `video_quality/models_downloaded/subject_consistency/dino_vitbase16_pretrain.pth` |
| RAFT Things | `video_quality/models_downloaded/subject_consistency/raft-things.pth` |
| Depth Anything V2 Small | `video_quality/models_downloaded/depth_accuracy/Depth-Anything-V2-Small-hf/` |
| SAM3 checkpoint | `sam/sam3.pt` |
| SAM3 tokenizer | `sam/bpe_simple_vocab_16e6.txt.gz` |
| JEPA ViT-H/16 | `video_quality/JEDi/pretrained_models/vith16.pth.tar` |
| JEPA SSV2 Probe | `video_quality/JEDi/pretrained_models/ssv2-probe.pth.tar` |

These defaults are also recorded in `video_quality/config/config.yaml`. If a
model is stored elsewhere, update that configuration; relative paths are still
resolved from the repository root.

Weights, datasets, caches, and evaluation outputs are intentionally excluded
from Git.
