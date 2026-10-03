"""Match GT and generated MP4s and evaluate them directly."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS = "psnr,ssim,aesthetic_quality,image_quality,jepa_similarity,depth_accuracy,subject_consistency,trajectory_accuracy"


def index_videos(root: Path, task_name: str | None = None) -> dict:
    """Task is the first subdirectory; deeper camera/config directories are allowed."""
    if not root.is_dir():
        raise NotADirectoryError(root)
    indexed = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".mp4":
            continue
        match = re.fullmatch(r"(?:(.+)__)?episode[_-]?(\d+)", path.stem)
        if not match:
            raise ValueError(f"Unrecognized episode filename: {path}")
        relative = path.relative_to(root)
        task = match[1] or (relative.parts[0] if len(relative.parts) > 1 else task_name)
        if not task:
            raise ValueError(f"Flat input needs --task-name or task__episodeN filenames: {path}")
        if task in {".", ".."} or "/" in task or "\\" in task:
            raise ValueError(f"Invalid task name: {task!r}")
        key = (task, int(match[2]))
        if key in indexed:
            raise ValueError(f"Ambiguous task/episode {key}: {indexed[key]} and {path}")
        indexed[key] = path
    if not indexed:
        raise ValueError(f"No MP4 videos found in {root}")
    return indexed


def video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        info = dict(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
        if any(not math.isfinite(v) or v <= 0 for v in info.values()):
            raise ValueError(f"Invalid video metadata: {path}: {info}")
        return info
    finally:
        capture.release()


def plan_pairs(generated: dict, gt: dict, trim_gt: bool) -> list:
    missing = sorted(set(generated) - set(gt))
    if missing:
        raise ValueError(f"Missing GT task/episode matches: {missing}")
    rows = []
    for task, episode in sorted(generated):
        key = (task, episode)
        source, prediction = gt[key], generated[key]
        a, b = video_info(source), video_info(prediction)
        same_format = (a["width"], a["height"]) == (b["width"], b["height"])
        same_format &= math.isclose(a["fps"], b["fps"], rel_tol=1e-4)
        trim_frames = b["frames"] if trim_gt and same_format and a["frames"] > b["frames"] else None
        aligned = same_format and (a["frames"] == b["frames"] or trim_frames is not None)
        rows.append(dict(sample_id=f"{task}__episode{episode}", gt_path=str(source),
                         generated_video=str(prediction), gt_info=a, generated_info=b,
                         aligned=bool(aligned), trim_frames=trim_frames))
    return rows


def trim_gt_tail(row: dict, output: Path, ffmpeg: str) -> None:
    """Keep the first generated-frame-count decoded GT frames without pixel loss."""
    temporary = output.with_suffix(".tmp.mp4")
    capture = cv2.VideoCapture(row["gt_path"])
    encoder = None
    try:
        info = row["gt_info"]
        encoder = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
             f"{info['width']}x{info['height']}", "-r", str(info["fps"]),
             "-i", "pipe:0", "-an", "-c:v", "libx264rgb", "-crf", "0",
             "-preset", "fast", "-threads", "2", str(temporary)],
            stdin=subprocess.PIPE,
        )
        for index in range(row["trim_frames"]):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode GT frame {index}: {row['sample_id']}")
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg failed while trimming GT: {row['sample_id']}")
        temporary.replace(output)
    finally:
        capture.release()
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        if encoder is not None and not encoder.stdin.closed:
            encoder.stdin.close()
        temporary.unlink(missing_ok=True)


def evaluate(rows: list, metrics: list, output: Path, args, summary: Path | None = None) -> None:
    if summary is None:
        summary = output / "video_summary.json"
    output.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps([
        {key: row[key] for key in ("sample_id", "generated_video", "gt_path") if key in row}
        for row in rows
    ], indent=2) + "\n")
    command = [sys.executable, "-m", "video_quality.cli", "all", "--summary", str(summary),
               "--output-dir", str(output), "--config", str(args.config.expanduser().resolve()),
               "--metrics", ",".join(metrics), "--processes-per-gpu", str(args.processes_per_gpu)]
    if args.gpus:
        command += ["--gpus", args.gpus]
    if "jepa_similarity" in metrics:
        jepa_python = args.jepa_python or subprocess.check_output(
            ["conda", "run", "-n", "WorldArena_JEPA", "python", "-c", "import sys; print(sys.executable)"],
            text=True,
        ).strip()
        jepa_root = output / "cache" / "jepa_pairs"
        for kind, field in (("gt", "gt_path"), ("generated", "generated_video")):
            directory = jepa_root / kind
            directory.mkdir(parents=True, exist_ok=True)
            for old_link in directory.glob("*.mp4"):
                if old_link.is_symlink():
                    old_link.unlink()
            for row in rows:
                link = directory / f"{row['sample_id']}.mp4"
                link.symlink_to(row[field])
        command += ["--jepa-real-dir", str(jepa_root / "gt"),
                    "--jepa-gen-dir", str(jepa_root / "generated"),
                    "--jepa-python", jepa_python]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def merge_results(rows: list, metrics: list, output: Path, other_dir: Path | None,
                  basic_dir: Path | None) -> None:
    from video_quality.constants import METRIC_COLUMNS

    def load(directory):
        return {row["sample_id"]: row for row in json.loads(
            (directory / "results/results.json").read_text())["rows"]}

    other = load(other_dir) if other_dir else {}
    basic = load(basic_dir) if basic_dir else {}
    result_rows = []
    for source in rows:
        sample_id = source["sample_id"]
        result = {"sample_id": sample_id}
        for metric in metrics:
            column = METRIC_COLUMNS[metric]
            if metric in {"psnr", "ssim"}:
                result[column] = basic[sample_id][column] if source.get("aligned") else "-"
            else:
                result[column] = other[sample_id][column]
        result_rows.append(result)
    average = {"sample_id": "AVERAGE"}
    for metric in metrics:
        column = METRIC_COLUMNS[metric]
        values = [float(row[column]) for row in result_rows
                  if row[column] != "-" and math.isfinite(float(row[column]))]
        average[column] = float(np.mean(values)) if values else "-"
    result_dir = output / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    if basic_dir is not None and basic_dir != output:
        for metric in metrics:
            source = basic_dir / "results" / "metrics" / f"{metric}.json"
            if source.is_file():
                target = result_dir / "metrics" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    (result_dir / "results.json").write_text(json.dumps(
        {"rows": result_rows, "average": average}, indent=2) + "\n")
    with (result_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", *[METRIC_COLUMNS[m] for m in metrics]])
        writer.writeheader()
        for row in [*result_rows, average]:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value
                             for key, value in row.items()})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", required=True, type=Path)
    parser.add_argument("--gt-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New run directory")
    parser.add_argument("--task-name", help="Task for videos directly inside either input root")
    parser.add_argument("--trim-gt-to-generated", action="store_true",
                        help="Trim only extra GT frames at the end when FPS and size match")
    parser.add_argument("--summary-only", action="store_true", help="Write the input summary without evaluation")
    parser.add_argument(
        "--skip-gt-metrics",
        action="store_true",
        help="Skip PSNR, SSIM, JEPA, trajectory, and depth metrics; evaluate generated videos only",
    )
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "video_quality/config/config.yaml")
    parser.add_argument("--gpus", help="Forward GPU IDs, e.g. 0,1,2,3; omitted uses one worker")
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    parser.add_argument("--jepa-python", help="Defaults to Python from conda environment WorldArena_JEPA")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg used only when trimming GT tails")
    args = parser.parse_args(argv)
    from video_quality.constants import normalize_metrics

    metrics = normalize_metrics(args.metrics.split(","))
    from video_quality.constants import GT_REQUIRED_METRICS

    if args.skip_gt_metrics:
        skipped = [metric for metric in metrics if metric in GT_REQUIRED_METRICS]
        metrics = [metric for metric in metrics if metric not in GT_REQUIRED_METRICS]
        if not metrics:
            parser.error(
                "--skip-gt-metrics removed every selected metric; select at least one "
                "of aesthetic_quality, image_quality, or subject_consistency"
            )
        if skipped:
            print(f"Skipping GT-dependent metrics: {', '.join(skipped)}", flush=True)
    elif args.gt_root is None:
        parser.error("--gt-root is required unless --skip-gt-metrics is set")

    output = args.output_dir.expanduser().resolve()
    generated_root = args.generated_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve() if args.gt_root is not None else None
    if output.exists():
        print(f"Warning: --output-dir already exists; files may be overwritten: {output}", flush=True)
    for root in (generated_root, gt_root):
        if root is None:
            continue
        if output.is_relative_to(root):
            parser.error("--output-dir must be outside both input roots")
    output.mkdir(parents=True, exist_ok=True)
    generated = index_videos(generated_root, args.task_name)
    if args.skip_gt_metrics:
        rows = [
            {
                "sample_id": f"{task}__episode{episode}",
                "generated_video": str(generated[(task, episode)]),
            }
            for task, episode in sorted(generated)
        ]
        print(f"Found {len(rows)} generated videos; GT-dependent metrics skipped", flush=True)
    else:
        gt = index_videos(gt_root, args.task_name)
        rows = plan_pairs(generated, gt, args.trim_gt_to_generated)
        print(f"Matched {len(rows)} videos; {len(set(gt) - set(generated))} extra GT videos unused", flush=True)
        to_trim = [row for row in rows if row["trim_frames"] is not None]
        if to_trim:
            ffmpeg = shutil.which(args.ffmpeg)
            if not ffmpeg:
                parser.error("FFmpeg with libx264rgb support is required to trim GT tails")
            trimmed_root = output / "cache" / "trimmed_gt_videos"
            trimmed_root.mkdir(parents=True, exist_ok=True)
            for row in to_trim:
                target = trimmed_root / f"{row['sample_id']}.mp4"
                trim_gt_tail(row, target, ffmpeg)
                row["original_gt_path"] = row["gt_path"]
                row["gt_path"] = str(target)
                print(f"[trim] {row['sample_id']}: {row['gt_info']['frames']} -> {row['trim_frames']} GT frames", flush=True)
    summary_rows = [
        {key: row[key] for key in ("sample_id", "generated_video", "gt_path") if key in row}
        for row in rows
    ]
    summary = output / ("generated_summary.json" if args.skip_gt_metrics else "video_summary.json")
    summary.write_text(json.dumps(summary_rows, indent=2) + "\n")
    (output / "video_pairs.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not args.summary_only:
        mismatched = [row for row in rows if not row.get("aligned", True)]
        basic_metrics = [metric for metric in metrics if metric in {"psnr", "ssim"}]
        if not mismatched or not basic_metrics:
            evaluate(rows, metrics, output, args, summary)
        else:
            other_metrics = [metric for metric in metrics if metric not in {"psnr", "ssim"}]
            aligned = [row for row in rows if row["aligned"]]
            for row in mismatched:
                print(f"Skipping PSNR/SSIM for {row['sample_id']}: GT={row['gt_info']}, generated={row['generated_info']}", flush=True)
            other_dir = output if other_metrics else None
            basic_dir = None
            if aligned:
                basic_dir = output / "cache" / "basic_metrics" if other_metrics else output
            if other_dir:
                evaluate(rows, other_metrics, other_dir, args, summary)
            if basic_dir:
                evaluate(aligned, basic_metrics, basic_dir, args,
                         basic_dir / "video_summary.json" if basic_dir != output
                         else output / "cache" / "aligned_summary.json")
            merge_results(rows, metrics, output, other_dir, basic_dir)
    print(f"Summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
