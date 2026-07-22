# WorldArena eight-metric evaluation

[中文说明](README_CN.md)

This evaluator keeps only:

1. PSNR
2. SSIM
3. Aesthetic Quality
4. Image Quality
5. JEPA Similarity
6. Subject Consistency
7. Trajectory Accuracy
8. Depth Accuracy

Track2 and all other video-quality metrics are outside this evaluator.

## Source boundary

The WorldArena reference is pinned to GitHub commit
[`a918b93f8533a4e9452a224c0ec54d27e527c4bb`](https://github.com/tsinghua-fib-lab/WorldArena/commit/a918b93f8533a4e9452a224c0ec54d27e527c4bb).

| Metric | Implementation used here |
|---|---|
| PSNR / SSIM | Pinned `WorldArena/basic_metrics.py`, unchanged |
| Aesthetic Quality | Existing local WorldArena implementation; scoring is unchanged, implicit checkpoint download is disabled and `torch.load` uses CPU mapping |
| Image Quality | Existing local WorldArena MUSIQ implementation; only unused imports differ |
| JEPA Similarity | Pinned `JEDi/batch.py`; `model_dir`, `config_path`, and `output_root` are passed to `JEDiMetric` |
| Subject Consistency | Pinned `subject_consistency.py` and `dynamic_degree.py`, unchanged |
| Trajectory Accuracy | Pinned `trajectory_accuracy.py`, unchanged |
| Depth Accuracy | Pinned `depth_accuracy.py`, unchanged |

`subject_consistency.py` imports an undefined and unused upstream `CACHE_DIR` name.
The compatibility name is supplied by `WorldArena/utils.py`; the metric file is
not modified. The outer runner only prepares inputs, calls one metric, and saves
that metric's result.

## Input

The summary is a non-empty JSON list:

```json
[
  {
    "gt_path": "/absolute/path/to/episode0.mp4",
    "generated_video": "/absolute/path/to/generated_episode0.mp4"
  }
]
```

The GT filename stem is the sample ID and must be unique.

`prepare` rebuilds only the frame layouts needed by the selected metrics:

- PSNR/SSIM: GT and generated frames are both lossless PNG, because the pinned
  `basic_metrics.py` requires GT `frame_*.png` files.
- Aesthetic, Image, Subject, Trajectory, and Depth: frames are JPEG written with
  OpenCV's default JPEG quality, which is 95.
- JEPA reads MP4 files directly and does not use prepared frame directories.

The manifest contains paths only. It does not contain video hashes, checkpoint
signatures, cache metadata, shards, or automatic file matching.

## Commands

Prepare all eight metrics:

```bash
python -m video_quality.cli prepare \
  --summary /path/to/summary.json \
  --output-dir /path/to/evaluation
```

Run the seven non-JEPA metrics:

```bash
python -m video_quality.cli evaluate \
  --manifest /path/to/evaluation/run_manifest.json \
  --output-dir /path/to/evaluation \
  --config video_quality/config/config.yaml
```

JEPA is an independent dataset-level call. Manually prepare two non-recursive
directories containing same-name MP4 files, then run:

```bash
/path/to/WorldArena_JEPA/bin/python -m video_quality.cli jepa \
  --real-dir /path/to/real_mp4 \
  --gen-dir /path/to/generated_mp4 \
  --output-dir /path/to/evaluation \
  --config video_quality/config/config.yaml \
  --jepa-python /path/to/WorldArena_JEPA/bin/python
```

The pinned JEPA source uses the intersection of the two filename sets and skips
invalid pairs. It samples 16 frames uniformly, resizes them to 224×224, and
converts JEDi distance `D` to similarity with `exp(-0.4D)`. `videojedi` writes
`train.npy` and `test.npy` under the JEPA output directory and reuses them when
the same output directory is run again.

Aggregate the selected metric files:

```bash
python -m video_quality.cli aggregate \
  --manifest /path/to/evaluation/run_manifest.json \
  --output-dir /path/to/evaluation
```

Use `--metrics` with a comma-separated subset on `prepare`, `evaluate`, and
`aggregate`. The accepted names are:

```text
psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,
subject_consistency,trajectory_accuracy,depth_accuracy
```

The `all` command additionally requires `--jepa-real-dir` and `--jepa-gen-dir`
when JEPA is selected.

## Metric outputs

Each metric is saved independently:

```text
evaluation/results/metrics/psnr.json
evaluation/results/metrics/ssim.json
evaluation/results/metrics/aesthetic_quality.json
...
evaluation/results/jepa/results.json
```

Aggregation produces `results/results.json` and `results/results.csv`, including
one row per video and an equally weighted `AVERAGE` row. Dataset-level JEPA is
copied into every video row.

PSNR and SSIM retain the pinned scikit-image behavior. In particular, identical
frames have infinite PSNR. Trajectory and Depth are reported with WorldArena's
leaderboard normalization, rather than their raw values:

```text
Trajectory = clip(raw_NDTW / 40.8540, 0, 1)
Depth      = 1 - clip((raw_AbsRel - 0.2228) / (4.3711 - 0.2228), 0, 1)
```

The remaining metrics keep their source outputs.

## Trajectory files

For each sample the runner expects:

```text
GT episode/traj/traj.npy
generated episode/1/traj/traj.npy
```

An existing file is used directly. If only one side is missing, SAM3 runs only
for that side; if both are missing, it runs for both. There is no hash or model
signature check. Delete a `traj.npy` yourself when you want that side regenerated.

The configured SAM3 directory must contain:

```text
sam3.pt
bpe_simple_vocab_16e6.txt.gz
```

The current configuration points to `/data1/liuwenhao/Projects/WorldArena/sam`.

## Model weights

Edit `config/config.yaml`. Paths are checked only when their metric is called.

- Aesthetic: OpenAI CLIP ViT-L/14 plus LAION
  `sa_0_4_vit_l_14_linear.pth`.
- Image: IQA-PyTorch `musiq_spaq_ckpt-358bb6af.pth`.
- Subject: local Facebook DINO repository,
  `dino_vitbase16_pretrain.pth`, and RAFT `raft-things.pth`.
- Depth: the complete `depth-anything/Depth-Anything-V2-Small-hf` directory.
- JEPA: `video_quality/JEDi/pretrained_models/vith16.pth.tar` and
  `ssv2-probe.pth.tar`.

The JEPA files can be downloaded with the commands published in WorldArena:

```bash
mkdir -p video_quality/JEDi/pretrained_models
wget -O video_quality/JEDi/pretrained_models/vith16.pth.tar \
  https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar
wget -O video_quality/JEDi/pretrained_models/ssv2-probe.pth.tar \
  https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar
```

## Environment setup

Core metrics and JEPA use separate Python environments. Run the commands from
`video_quality/` so the relative requirements paths in the YAML files resolve
correctly.

Create the core environment for PSNR, SSIM, Aesthetic, Image, Subject,
Trajectory, and Depth:

```bash
cd /path/to/WorldArena/video_quality
conda env create -f environment-core.yml
conda activate WorldArena
```

If the environment already exists, update it instead:

```bash
cd /path/to/WorldArena/video_quality
conda env update -n WorldArena -f environment-core.yml --prune
```

Create the independent JEPA environment:

```bash
cd /path/to/WorldArena/video_quality
conda env create -f environment-jepa.yml
conda activate WorldArena_JEPA
```

To update an existing JEPA environment:

```bash
cd /path/to/WorldArena/video_quality
conda env update -n WorldArena_JEPA -f environment-jepa.yml --prune
```

Verify the command-line entry point in the core environment:

```bash
conda activate WorldArena
cd /path/to/WorldArena
python -m video_quality.cli --help
```

The removed VLM metrics are not part of this evaluator, so a
`WorldArena_VLM` environment is no longer required.

### Apptainer/Singularity

The container definition creates both Conda environments:

```bash
cd /path/to/WorldArena/video_quality
apptainer build containers/WorldArena.sif containers/WorldArena.def
```

Use `sudo apptainer build` or `singularity build` if required by the local
cluster configuration. The Slurm example is `slurm/run_eight_metrics.sbatch`.
