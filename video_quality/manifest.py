"""Prepare the frame layouts consumed by the retained metric sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
from typing import Iterable

from .constants import JPEG_METRICS, normalize_metrics


MANIFEST_VERSION = 2
BASIC_METRICS = frozenset({"psnr", "ssim"})


@dataclass(frozen=True)
class Sample:
    sample_id: str
    gt_path: str
    generated_video: str
    gt_frames: str | None = None
    generated_frames: str | None = None
    gt_png_frames: str | None = None
    generated_png_frames: str | None = None
    layout: str = "separate"


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _video_path(value: object, field: str, index: int) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"summary[{index}].{field} must be a non-empty path string")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"summary[{index}].{field} does not exist: {path}")
    return path


def load_summary(path: str | Path) -> list[Sample]:
    summary_path = Path(path).expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Summary JSON must be a non-empty list")

    samples = []
    seen = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"summary[{index}] must be an object")
        stacked_value = item.get("stacked_video")
        if stacked_value is not None:
            stacked_video = _video_path(stacked_value, "stacked_video", index)
            gt_path = stacked_video
            generated_video = stacked_video
            layout = "vertical_gt_top"
        else:
            gt_path = _video_path(item.get("gt_path"), "gt_path", index)
            generated_video = _video_path(
                item.get("generated_video"), "generated_video", index
            )
            layout = "separate"

        explicit_id = item.get("sample_id")
        if explicit_id is None:
            sample_id = gt_path.stem
        elif not isinstance(explicit_id, str) or not explicit_id.strip():
            raise ValueError(f"summary[{index}].sample_id must be a non-empty string")
        else:
            sample_id = explicit_id.strip()
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        samples.append(
            Sample(
                sample_id=sample_id,
                gt_path=str(gt_path),
                generated_video=str(generated_video),
                layout=layout,
            )
        )
    return samples


def _extract_frames(video: Path, output_dir: Path, suffix: str) -> None:
    import cv2

    temporary = output_dir.with_name(output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            target = temporary / f"frame_{count:05d}.{suffix}"
            if not cv2.imwrite(str(target), frame):
                raise RuntimeError(f"Failed to write extracted frame: {target}")
            count += 1
    finally:
        capture.release()

    if count == 0:
        shutil.rmtree(temporary)
        raise ValueError(f"Video contains no decodable frames: {video}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    temporary.replace(output_dir)


def _extract_stacked_frames(
    video: Path,
    outputs: list[tuple[Path, Path, str]],
) -> None:
    """Decode once and write GT-top/generated-bottom crops to all requested layouts."""
    import cv2

    temporary_outputs = []
    for gt_output, generated_output, suffix in outputs:
        gt_temporary = gt_output.with_name(gt_output.name + ".tmp")
        generated_temporary = generated_output.with_name(
            generated_output.name + ".tmp"
        )
        for temporary in (gt_temporary, generated_temporary):
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
        temporary_outputs.append(
            (gt_output, generated_output, gt_temporary, generated_temporary, suffix)
        )

    capture = cv2.VideoCapture(str(video))
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height = frame.shape[0]
            if height % 2:
                raise ValueError(
                    f"Stacked video frame height must be even, got {height}: {video}"
                )
            midpoint = height // 2
            gt_frame = frame[:midpoint]
            generated_frame = frame[midpoint:]
            for _, _, gt_temporary, generated_temporary, suffix in temporary_outputs:
                gt_target = gt_temporary / f"frame_{count:05d}.{suffix}"
                generated_target = (
                    generated_temporary / f"frame_{count:05d}.{suffix}"
                )
                if not cv2.imwrite(str(gt_target), gt_frame):
                    raise RuntimeError(f"Failed to write extracted frame: {gt_target}")
                if not cv2.imwrite(str(generated_target), generated_frame):
                    raise RuntimeError(
                        f"Failed to write extracted frame: {generated_target}"
                    )
            count += 1
    except Exception:
        for _, _, gt_temporary, generated_temporary, _ in temporary_outputs:
            shutil.rmtree(gt_temporary, ignore_errors=True)
            shutil.rmtree(generated_temporary, ignore_errors=True)
        raise
    finally:
        capture.release()

    if count == 0:
        for _, _, gt_temporary, generated_temporary, _ in temporary_outputs:
            shutil.rmtree(gt_temporary, ignore_errors=True)
            shutil.rmtree(generated_temporary, ignore_errors=True)
        raise ValueError(f"Video contains no decodable frames: {video}")

    for gt_output, generated_output, gt_temporary, generated_temporary, _ in (
        temporary_outputs
    ):
        if gt_output.exists():
            shutil.rmtree(gt_output)
        if generated_output.exists():
            shutil.rmtree(generated_output)
        gt_temporary.replace(gt_output)
        generated_temporary.replace(generated_output)


def prepare(
    summary_json: str | Path,
    output_dir: str | Path,
    metrics: Iterable[str],
) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    metric_list = normalize_metrics(metrics)
    samples = load_summary(summary_json)
    prepared = []
    need_png = bool(set(metric_list) & BASIC_METRICS)
    need_jpeg = bool(set(metric_list) & JPEG_METRICS)

    for index, sample in enumerate(samples, start=1):
        png_outputs = None
        if need_png:
            gt_png = (
                output_root
                / "cache"
                / "png_frames"
                / "gt_dataset"
                / "default"
                / sample.sample_id
                / "video"
            )
            generated_png = (
                output_root
                / "cache"
                / "png_frames"
                / "generated_dataset"
                / "default"
                / sample.sample_id
                / "1"
                / "video"
            )
            png_outputs = (gt_png, generated_png, "png")
            sample = replace(
                sample,
                gt_png_frames=str(gt_png),
                generated_png_frames=str(generated_png),
            )

        jpeg_outputs = None
        if need_jpeg:
            gt_frames = (
                output_root
                / "cache"
                / "jpeg_frames"
                / "gt_dataset"
                / "default"
                / sample.sample_id
                / "video"
            )
            generated_frames = (
                output_root
                / "cache"
                / "jpeg_frames"
                / "generated_dataset"
                / "default"
                / sample.sample_id
                / "1"
                / "video"
            )
            jpeg_outputs = (gt_frames, generated_frames, "jpg")
            sample = replace(
                sample,
                gt_frames=str(gt_frames),
                generated_frames=str(generated_frames),
            )

        if sample.layout == "vertical_gt_top":
            outputs = [
                output for output in (png_outputs, jpeg_outputs) if output is not None
            ]
            if outputs:
                _extract_stacked_frames(Path(sample.gt_path), outputs)
        else:
            if png_outputs is not None:
                _extract_frames(Path(sample.gt_path), png_outputs[0], "png")
                _extract_frames(
                    Path(sample.generated_video), png_outputs[1], "png"
                )
            if jpeg_outputs is not None:
                # No explicit quality argument: OpenCV's default JPEG quality is 95.
                _extract_frames(Path(sample.gt_path), jpeg_outputs[0], "jpg")
                _extract_frames(
                    Path(sample.generated_video), jpeg_outputs[1], "jpg"
                )

        prepared.append(sample)
        print(f"[prepare] {index}/{len(samples)} {sample.sample_id}", flush=True)

    manifest_path = output_root / "run_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "version": MANIFEST_VERSION,
            "summary_json": str(Path(summary_json).expanduser().resolve()),
            "metrics": metric_list,
            "samples": [asdict(sample) for sample in prepared],
        },
    )
    return manifest_path


def load_manifest(path: str | Path) -> tuple[list[str], list[Sample]]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported manifest version in {manifest_path}")
    metrics = payload.get("metrics")
    rows = payload.get("samples")
    if not isinstance(metrics, list) or not isinstance(rows, list) or not rows:
        raise ValueError(f"Invalid run manifest: {manifest_path}")
    return metrics, [Sample(**row) for row in rows]
