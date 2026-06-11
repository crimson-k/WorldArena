#!/usr/bin/env python3
"""Copy 500 GTGT 480p videos and build sample_records.jsonl.

Input jsonl format example (source):
  {
    "episode_index": 0,
    "video": "dataset/adjust_bottle/videos/chunk-000/episode_000040_480p.mp4",
    "prompt": "...",
    "task": "adjust_bottle",
    ...
  }

Output jsonl format (same fields as video_quality/data/500_RoboTwin/sample_records.jsonl):
  {
    "video_path": "/abs/path/to/copied/video.mp4",
    "prompt": "...",
    "episode_index": 0,
    "sample_index": 0,
    "rank": 0
  }
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Iterator, Optional


DEFAULT_INPUT_JSONL = (
    "/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/500_org_resize_GTGT/"
    "episodes_val_cam_high_org.jsonl"
)
DEFAULT_SOURCE_ROOT = "/ssdfs/datahome/usersht/datasets/RoboTwin2.0_lerobot"
DEFAULT_TARGET_VIDEOS_DIR = (
    "/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/500_org_resize_GTGT/videos"
)
DEFAULT_OUTPUT_JSONL = (
    "/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/500_org_resize_GTGT/sample_records.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy videos referenced by episodes jsonl and build sample_records.jsonl"
    )
    parser.add_argument("--input_jsonl", type=Path, default=Path(DEFAULT_INPUT_JSONL))
    parser.add_argument("--source_root", type=Path, default=Path(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--target_videos_dir", type=Path, default=Path(DEFAULT_TARGET_VIDEOS_DIR))
    parser.add_argument("--output_jsonl", type=Path, default=Path(DEFAULT_OUTPUT_JSONL))
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default="",
        help="Optional prefix added before prompt, e.g. '[Embodiment: Aloha-Agilex] '",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files with same destination name.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned actions; do not copy or write output.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} in {path} is not a JSON object")
            yield obj


def extract_episode_num(video_rel: str) -> Optional[int]:
    """Extract episode number from file names like episode_000040_480p.mp4."""
    name = Path(video_rel).name
    match = re.search(r"episode[_-]?(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def safe_filename(task: str, episode_num: Optional[int], sample_index: int) -> str:
    task_clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", task or "unknown_task")
    if episode_num is None:
        return f"{task_clean}_sample{sample_index}.mp4"
    return f"{task_clean}_episode{episode_num}.mp4"


def main() -> None:
    args = parse_args()

    input_jsonl = args.input_jsonl.resolve()
    source_root = args.source_root.resolve()
    target_videos_dir = args.target_videos_dir.resolve()
    output_jsonl = args.output_jsonl.resolve()

    if not input_jsonl.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_jsonl}")
    if not source_root.exists():
        raise FileNotFoundError(f"source_root not found: {source_root}")

    if not args.dry_run:
        target_videos_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    missing = 0
    skipped_existing = 0
    renamed_due_to_collision = 0

    output_records = []
    stem_counts: Dict[str, int] = {}

    for sample_index, rec in enumerate(iter_jsonl(input_jsonl)):
        total += 1

        video_rel = str(rec.get("video", "")).strip()
        if not video_rel:
            missing += 1
            continue

        src = (source_root / video_rel).resolve()
        if not src.exists():
            missing += 1
            print(f"[MISSING] {src}")
            continue

        task = str(rec.get("task", "")).strip()
        ep_num = extract_episode_num(video_rel)
        dst_name = safe_filename(task=task, episode_num=ep_num, sample_index=sample_index)

        dst_stem = Path(dst_name).stem
        used_count = stem_counts.get(dst_stem, 0)
        stem_counts[dst_stem] = used_count + 1
        if used_count > 0:
            renamed_due_to_collision += 1
            dst_name = f"{dst_stem}_dup{used_count}.mp4"

        dst = target_videos_dir / dst_name

        if dst.exists() and not args.overwrite:
            skipped_existing += 1
        else:
            if not args.dry_run:
                shutil.copy2(src, dst)
            copied += 1

        prompt = str(rec.get("prompt", ""))
        if args.prompt_prefix:
            prompt = f"{args.prompt_prefix}{prompt}"

        out = {
            "video_path": str(dst),
            "prompt": prompt,
            "episode_index": rec.get("episode_index"),
            "sample_index": 0,
            "rank": 0,
        }
        output_records.append(out)

    if not args.dry_run:
        with output_jsonl.open("w", encoding="utf-8") as f:
            for row in output_records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("[DONE] copy_and_build_500_org_resize_GTGT")
    print(f"  total_records: {total}")
    print(f"  copied: {copied}")
    print(f"  missing_source: {missing}")
    print(f"  skipped_existing: {skipped_existing}")
    print(f"  renamed_due_to_collision: {renamed_due_to_collision}")
    print(f"  target_videos_dir: {target_videos_dir}")
    print(f"  output_jsonl: {output_jsonl}")


if __name__ == "__main__":
    main()
