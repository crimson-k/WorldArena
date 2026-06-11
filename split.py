#!/usr/bin/env python3
"""Split a run folder into N sibling folders by videos and sample_records.jsonl.

Example:
  python split_run_folder.py /data/pingce/run_xxx
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split folder videos and sample_records.jsonl into sibling shards."
    )
    parser.add_argument("run_dir", type=Path, help="Path to source run folder.")
    parser.add_argument(
        "--num-splits",
        type=int,
        default=4,
        help="Number of output splits. Default: 4",
    )
    parser.add_argument(
        "--transfer",
        choices=("hardlink", "copy", "move", "symlink"),
        default="hardlink",
        help=(
            "How to place video files into split folders. "
            "Default: hardlink (fallback to copy on failure)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview split plan without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output split folders if they exist.",
    )
    return parser.parse_args()


def validate_inputs(run_dir: Path, num_splits: int) -> Tuple[Path, Path]:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist or is not a directory: {run_dir}")
    if num_splits <= 0:
        raise ValueError(f"--num-splits must be > 0, got {num_splits}")

    videos_dir = run_dir / "videos"
    jsonl_path = run_dir / "sample_records.jsonl"
    if not videos_dir.exists() or not videos_dir.is_dir():
        raise FileNotFoundError(f"videos directory not found: {videos_dir}")
    if not jsonl_path.exists() or not jsonl_path.is_file():
        raise FileNotFoundError(f"sample_records.jsonl not found: {jsonl_path}")

    return videos_dir, jsonl_path


def list_video_files(videos_dir: Path) -> List[Path]:
    files = [
        p
        for p in videos_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    files.sort(key=lambda p: p.name)
    return files


def assign_splits(video_files: List[Path], num_splits: int) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for i, p in enumerate(video_files):
        split_id = i % num_splits
        mapping[p.name] = split_id
    return mapping


def prepare_output_dirs(
    run_dir: Path, num_splits: int, overwrite: bool, dry_run: bool
) -> List[Path]:
    parent = run_dir.parent
    base = run_dir.name
    out_dirs = [parent / f"{base}_{i}" for i in range(num_splits)]

    for out_dir in out_dirs:
        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Output folder already exists: {out_dir}. "
                    "Use --overwrite to remove it."
                )
            if not dry_run:
                shutil.rmtree(out_dir)
        if not dry_run:
            (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    return out_dirs


def transfer_video(src: Path, dst: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unsupported transfer mode: {mode}")


def split_records(
    jsonl_path: Path,
    out_dirs: List[Path],
    video_to_split: Dict[str, int],
    dry_run: bool,
) -> Tuple[List[int], int]:
    split_records: List[List[dict]] = [[] for _ in out_dirs]
    unmatched = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {jsonl_path}: {exc}") from exc

            video_path = record.get("video_path")
            if not isinstance(video_path, str):
                unmatched += 1
                continue

            video_name = Path(video_path).name
            split_id = video_to_split.get(video_name)
            if split_id is None:
                unmatched += 1
                continue

            new_video_path = str((out_dirs[split_id] / "videos" / video_name).resolve())
            record["video_path"] = new_video_path
            split_records[split_id].append(record)

    counts = [len(items) for items in split_records]
    if not dry_run:
        for i, items in enumerate(split_records):
            out_jsonl = out_dirs[i] / "sample_records.jsonl"
            with out_jsonl.open("w", encoding="utf-8") as fw:
                for rec in items:
                    fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return counts, unmatched


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()

    videos_dir, jsonl_path = validate_inputs(run_dir, args.num_splits)
    video_files = list_video_files(videos_dir)
    if not video_files:
        raise RuntimeError(f"No video files found in {videos_dir}")
    if args.num_splits > len(video_files):
        raise ValueError(
            f"--num-splits ({args.num_splits}) > number of videos ({len(video_files)})"
        )

    video_to_split = assign_splits(video_files, args.num_splits)
    out_dirs = prepare_output_dirs(run_dir, args.num_splits, args.overwrite, args.dry_run)

    video_counts = [0] * args.num_splits
    for video in video_files:
        split_id = video_to_split[video.name]
        video_counts[split_id] += 1
        if not args.dry_run:
            dst = out_dirs[split_id] / "videos" / video.name
            transfer_video(video, dst, args.transfer)

    record_counts, unmatched = split_records(
        jsonl_path=jsonl_path,
        out_dirs=out_dirs,
        video_to_split=video_to_split,
        dry_run=args.dry_run,
    )

    total_videos = len(video_files)
    expected = math.ceil(total_videos / args.num_splits)
    print(f"Source: {run_dir}")
    print(f"Videos found: {total_videos}")
    print(f"Num splits: {args.num_splits}")
    print(f"Transfer mode: {args.transfer}")
    print(f"Dry run: {args.dry_run}")
    print(f"Per split videos: {video_counts} (max around {expected})")
    print(f"Per split sample_records: {record_counts}")
    print(f"Unmatched jsonl records skipped: {unmatched}")
    print("Output dirs:")
    for p in out_dirs:
        print(f"- {p}")


if __name__ == "__main__":
    main()
