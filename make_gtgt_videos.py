#!/usr/bin/env python3
"""Build GTGT videos by duplicating each GT video side-by-side."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_INPUT_DIR = (
    "video_quality/data/"
    "RoboTwin_Clean_50_multi_node_4_202604111721_step_17500/"
    "worldarena_auto_eval/gt_videos"
)
DEFAULT_OUTPUT_DIR = "video_quality/data/GTGT/videos"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def run_ffmpeg(input_file: Path, output_file: Path, overwrite: bool) -> tuple[bool, str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False, "ffmpeg not found in PATH"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_file),
        "-filter_complex",
        "[0:v]split=2[left][right];[left][right]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "copy",
        str(output_file),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""

    err = proc.stderr.strip() or proc.stdout.strip() or f"ffmpeg exit code {proc.returncode}"
    return False, err


def list_videos(input_dir: Path) -> list[Path]:
    files = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            files.append(p)
    return sorted(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Duplicate each GT video and stitch side-by-side (left GT + right GT). "
            "Outputs to video_quality/data/GTGT/videos by default."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel ffmpeg workers (default: 8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        return 1
    if not input_dir.is_dir():
        print(f"[ERROR] Input path is not a directory: {input_dir}")
        return 1

    videos = list_videos(input_dir)
    if not videos:
        print(f"[ERROR] No videos found in: {input_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    futures = {}
    success = 0
    failed = 0

    print(f"[INFO] Found {len(videos)} videos")
    print(f"[INFO] Input : {input_dir}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Workers: {args.workers}")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for video in videos:
            out = output_dir / video.name
            if out.exists() and not args.overwrite:
                print(f"[SKIP] {video.name} (exists)")
                continue
            fut = ex.submit(run_ffmpeg, video, out, args.overwrite)
            futures[fut] = video.name

        total_jobs = len(futures)
        done_jobs = 0
        for fut in as_completed(futures):
            name = futures[fut]
            ok, err = fut.result()
            done_jobs += 1
            if ok:
                success += 1
                if done_jobs % 25 == 0 or done_jobs == total_jobs:
                    print(f"[OK] {done_jobs}/{total_jobs}")
            else:
                failed += 1
                print(f"[FAIL] {name}: {err}")

    print(f"[DONE] success={success}, failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
