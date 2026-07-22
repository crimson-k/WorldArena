"""Thin calls into the eight retained metric implementations."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Iterable

from .constants import normalize_metrics, normalize_worldarena
from .manifest import Sample, atomic_write_json, load_manifest


MODEL_METRICS = {
    "aesthetic_quality",
    "image_quality",
    "subject_consistency",
    "trajectory_accuracy",
    "depth_accuracy",
}


def load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load the evaluation config") from exc
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    return config


def _checkpoint(config: dict, *keys: str) -> str:
    value: object = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f"Missing config field: {'.'.join(keys)}")
        value = value[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Config field must be a path: {'.'.join(keys)}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist ({'.'.join(keys)}): {path}")
    return str(path)


def _write_metric(output_dir: Path, metric: str, values: dict[str, float]) -> Path:
    path = output_dir / "results" / "metrics" / f"{metric}.json"
    atomic_write_json(
        path,
        {
            "metric": metric,
            "values": {sample_id: float(value) for sample_id, value in values.items()},
        },
    )
    return path


def _frame_samples(samples: Iterable[Sample]) -> list[Sample]:
    rows = list(samples)
    for sample in rows:
        if not sample.gt_frames or not sample.generated_frames:
            raise ValueError(f"JPEG frames were not prepared for {sample.sample_id}")
    return rows


def _write_full_info(output_dir: Path, samples: list[Sample], metrics: list[str]) -> Path:
    path = output_dir / "metric_inputs.json"
    atomic_write_json(
        path,
        [
            {"dimension": metrics, "video_list": [sample.generated_frames]}
            for sample in samples
        ],
    )
    return path


def _detail_values(result: object, samples: list[Sample]) -> dict[str, float]:
    details = result[1]
    sample_by_path = {
        str(Path(sample.generated_frames).resolve()): sample.sample_id for sample in samples
    }
    values = {}
    for item in details:
        sample_id = sample_by_path.get(str(Path(item["video_path"]).resolve()))
        if sample_id is not None:
            values[sample_id] = float(item["video_results"])
    return values


def run_basic_metrics(
    samples: list[Sample], output_dir: Path, metrics: list[str]
) -> None:
    selected = [metric for metric in metrics if metric in {"psnr", "ssim"}]
    if not selected:
        return
    first = samples[0]
    if not first.gt_png_frames or not first.generated_png_frames:
        raise ValueError("PNG frames were not prepared for PSNR/SSIM")

    from .WorldArena.basic_metrics import compute_basic_metrics

    gt_root = Path(first.gt_png_frames).parents[2]
    generated_root = Path(first.generated_png_frames).parents[3]
    result = compute_basic_metrics(str(gt_root), str(generated_root), selected)
    for metric in selected:
        values = {
            sample.sample_id: float(result[metric]["default"][sample.sample_id]["1"])
            for sample in samples
        }
        _write_metric(output_dir, metric, values)


def _trajectory_paths(sample: Sample) -> tuple[Path, Path]:
    gt_episode = Path(sample.gt_frames).parent
    generated_episode = Path(sample.generated_frames).parents[1]
    return (
        gt_episode / "traj" / "traj.npy",
        generated_episode / "1" / "traj" / "traj.npy",
    )


def _prepare_missing_trajectories(samples: list[Sample], config: dict) -> None:
    missing = []
    for sample in samples:
        gt_traj, generated_traj = _trajectory_paths(sample)
        if not gt_traj.is_file():
            missing.append((sample, "gt"))
        if not generated_traj.is_file():
            missing.append((sample, "generated"))
    if not missing:
        return

    from .processing.detection_tracking import GripperDetector, process_video_with_tracking

    detector = GripperDetector(model_path=_checkpoint(config, "ckpt", "sam3_model_ckpt"))
    for sample, side in missing:
        if side == "gt":
            ok = process_video_with_tracking(
                input_path=sample.gt_frames,
                output_path=str(Path(sample.gt_frames).parent),
                detector=detector,
                gid=None,
                data_type="gt",
            )
        else:
            ok = process_video_with_tracking(
                input_path=sample.generated_frames,
                output_path=str(Path(sample.generated_frames).parents[1]),
                detector=detector,
                gid="1",
                data_type="val",
            )
        if not ok:
            raise RuntimeError(
                f"Trajectory extraction failed for {sample.sample_id} ({side})"
            )


def run_model_metrics(
    samples: list[Sample], output_dir: Path, metrics: list[str], config: dict
) -> None:
    selected = [metric for metric in metrics if metric in MODEL_METRICS]
    if not selected:
        return
    rows = _frame_samples(samples)
    info_path = _write_full_info(output_dir, rows, selected)

    if "aesthetic_quality" in selected:
        from .WorldArena.aesthetic_quality import compute_aesthetic_quality

        result = compute_aesthetic_quality(
            str(info_path),
            {
                "clip_model": _checkpoint(config, "ckpt", "aesthetic_quality", "clip"),
                "aesthetic_head": _checkpoint(
                    config, "ckpt", "aesthetic_quality", "aesthetic_head"
                ),
            },
        )
        _write_metric(
            output_dir, "aesthetic_quality", _detail_values(result, rows)
        )

    if "image_quality" in selected:
        from .WorldArena.imaging_quality import compute_imaging_quality

        result = compute_imaging_quality(
            str(info_path),
            {"model_path": _checkpoint(config, "ckpt", "image_quality", "musiq")},
        )
        _write_metric(output_dir, "image_quality", _detail_values(result, rows))

    if "subject_consistency" in selected:
        from .WorldArena.subject_consistency import compute_subject_consistency

        subject = config.get("ckpt", {}).get("subject_consistency", {})
        result = compute_subject_consistency(
            str(info_path),
            {
                "repo_or_dir": _checkpoint(config, "ckpt", "subject_consistency", "repo"),
                "path": _checkpoint(config, "ckpt", "subject_consistency", "weight"),
                "model": subject.get("model", "dino_vitb16"),
                "source": "local",
                "read_frame": False,
                "raft_model": _checkpoint(
                    config, "ckpt", "subject_consistency", "raft"
                ),
            },
        )
        _write_metric(
            output_dir, "subject_consistency", _detail_values(result, rows)
        )

    if "trajectory_accuracy" in selected:
        from .WorldArena.trajectory_accuracy import eval_traj

        _prepare_missing_trajectories(rows, config)
        values = {}
        for sample in rows:
            gt_traj, generated_traj = _trajectory_paths(sample)
            raw = float(eval_traj(str(generated_traj), str(gt_traj))["ndtw"])
            values[sample.sample_id] = normalize_worldarena(
                "trajectory_accuracy", raw
            )
        _write_metric(output_dir, "trajectory_accuracy", values)

    if "depth_accuracy" in selected:
        from .WorldArena.depth_accuracy import compute_depth_accuracy

        gt_root = Path(rows[0].gt_frames).parents[2]
        result = compute_depth_accuracy(
            str(info_path),
            {"model": _checkpoint(config, "ckpt", "depth_accuracy")},
            gt_path=str(gt_root),
        )
        raw_values = _detail_values(result, rows)
        values = {
            sample_id: normalize_worldarena("depth_accuracy", value)
            for sample_id, value in raw_values.items()
        }
        _write_metric(output_dir, "depth_accuracy", values)


def run_metric(
    metric: str,
    samples: list[Sample],
    output_dir: str | Path,
    config: dict,
) -> None:
    """Call one retained metric and write its independent result file."""
    metric_name = normalize_metrics([metric])[0]
    if metric_name == "jepa_similarity":
        raise ValueError("JEPA uses run_jepa with explicit real/gen directories")
    output_root = Path(output_dir).expanduser().resolve()
    if metric_name in {"psnr", "ssim"}:
        run_basic_metrics(samples, output_root, [metric_name])
    else:
        run_model_metrics(samples, output_root, [metric_name], config)


def run_evaluate(
    manifest: str | Path,
    output_dir: str | Path,
    metrics: Iterable[str],
    config_path: str | Path,
) -> None:
    declared_metrics, samples = load_manifest(manifest)
    metric_list = normalize_metrics(metrics)
    undeclared = sorted(set(metric_list) - set(declared_metrics))
    if undeclared:
        raise ValueError(f"Metrics were not prepared: {', '.join(undeclared)}")
    config = load_config(config_path)
    for metric in metric_list:
        run_metric(metric, samples, output_dir, config)


def run_jepa(
    real_dir: str | Path,
    generated_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    python_executable: str = sys.executable,
) -> None:
    real_root = Path(real_dir).expanduser().resolve()
    generated_root = Path(generated_dir).expanduser().resolve()
    if not real_root.is_dir() or not generated_root.is_dir():
        raise NotADirectoryError("JEPA real/gen inputs must both be directories")

    config = load_config(config_path)
    model_dir = _checkpoint(config, "ckpt", "jepa_similarity", "model_dir")
    jepa_config = _checkpoint(config, "ckpt", "jepa_similarity", "config")
    result_dir = Path(output_dir).expanduser().resolve() / "results" / "jepa"
    result_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "JEDi" / "batch.py"
    subprocess.run(
        [
            python_executable,
            str(script),
            "--real_dir",
            str(real_root),
            "--gen_dir",
            str(generated_root),
            "--model_dir",
            model_dir,
            "--config_path",
            jepa_config,
            "--output_root",
            str(result_dir),
            "--save_intersection",
            str(result_dir / "intersection_names.json"),
        ],
        check=True,
    )
