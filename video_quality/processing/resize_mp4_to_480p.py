#!/usr/bin/env python3
"""Batch resize all MP4 videos in a folder to 480p.

Usage:
  python resize_mp4_to_480p.py /path/to/input_folder

Behavior:
- Recursively finds *.mp4 under input_folder.
- Writes outputs to sibling folder: <input_folder>_480p
- Keeps relative subdirectory structure.
- Keeps aspect ratio by forcing height=480 and width auto-adjusted to even value.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch resize MP4 videos to 480p")
    parser.add_argument("input_dir", type=Path, help="Folder that contains MP4 files")
    return parser.parse_args()


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found in PATH. Please install ffmpeg first."
        )


def collect_mp4_files(input_dir: Path) -> list[Path]:
    files = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"
    )
    return files


def resize_one(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        "scale=-2:480",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(dst),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] input_dir not found or not a directory: {input_dir}", file=sys.stderr)
        return 1

    output_dir = input_dir.parent / f"{input_dir.name}_480p"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        require_ffmpeg()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    mp4_files = collect_mp4_files(input_dir)
    if not mp4_files:
        print(f"[ERROR] No mp4 files found under: {input_dir}", file=sys.stderr)
        return 1

    print(f"[INFO] input_dir : {input_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] total_mp4 : {len(mp4_files)}")

    ok = 0
    fail = 0

    for idx, src in enumerate(mp4_files, start=1):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel

        try:
            resize_one(src, dst)
            ok += 1
            print(f"[{idx}/{len(mp4_files)}] OK   {src} -> {dst}")
        except Exception as exc:
            fail += 1
            print(f"[{idx}/{len(mp4_files)}] FAIL {src} | {exc}", file=sys.stderr)

    print("\n[DONE]")
    print(f"  success: {ok}")
    print(f"  failed : {fail}")
    print(f"  output : {output_dir}")

    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
