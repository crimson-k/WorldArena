#!/usr/bin/env python3
"""Prepare clean_50_gt_extract into WorldArena GT-vs-GT evaluable structure.

This script does two steps:
1) Rebuild a local `sample_records.jsonl` from raw clean_50 jsonl.
2) Call `prepare_gtgt_data` to generate:
   - gt_videos/
   - <model_name>_test/
   - <model_name>_test_vlm/
   - gt_first_frames/
   - summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from data_process_gtgt import prepare_gtgt_data


DEFAULT_INPUT_ROOT = (
    "/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/500_480p_GTGT"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert clean_50_gt_extract into GT-vs-GT evaluation layout."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path(DEFAULT_INPUT_ROOT),
        help="Root directory of clean_50_gt_extract.",
    )
    parser.add_argument(
        "--raw_jsonl",
        type=Path,
        default='/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/500_480p_GTGT/episodes_val_cam_high_480p.jsonl',
        help="Raw source jsonl (defaults to auto-discovery under input_root).",
    )
    parser.add_argument(
        "--records_out",
        type=Path,
        default=None,
        help="Output normalized sample_records.jsonl path. Default: <input_root>/sample_records.jsonl",
    )
    parser.add_argument(
        "--videos_dir",
        type=Path,
        default='/ssdfs/datahome/usersht/dev/lwh/WorldArena/video_quality/data/500_480p_GTGT/videos',
        help="Videos root for GTGT prepare step. Default: <input_root>/dataset",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Output work dir. Default: <input_root>/worldarena_auto_eval",
    )
    parser.add_argument("--model_name", type=str, default="GTGT")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--only_build_records", action="store_true")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--copy_only", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}") from exc
            if isinstance(obj, dict):
                yield obj


def auto_find_raw_jsonl(input_root: Path) -> Path:
    candidates = sorted(input_root.glob("*_episodes_clipped_val_*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            "Multiple raw jsonl files found. Please pass --raw_jsonl explicitly:\n"
            + "\n".join(str(p) for p in candidates)
        )
    fallback = input_root / "aloha-agilex_clean_50_episodes_clipped_val_aloha-agilex.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find raw jsonl in {input_root}. Please provide --raw_jsonl."
    )


def parse_episode_index(raw_episode_index: object, video_rel: str) -> Optional[int]:
    if raw_episode_index is not None:
        try:
            return int(raw_episode_index)
        except (TypeError, ValueError):
            pass

    match = re.search(r"episode(\d+)", Path(video_rel).name)
    if match:
        return int(match.group(1))
    return None


def rebuild_sample_records(
    *,
    input_root: Path,
    raw_jsonl: Path,
    records_out: Path,
    max_records: Optional[int],
) -> Tuple[int, int, int]:
    total = 0
    kept = 0
    missing = 0
    outputs: List[Dict] = []

    for record in iter_jsonl(raw_jsonl):
        total += 1
        if max_records is not None and kept >= max_records:
            break

        video_rel = str(record.get("video", "")).strip()
        if not video_rel:
            missing += 1
            continue

        video_abs = (input_root / video_rel).resolve()
        if not video_abs.exists():
            missing += 1
            continue

        episode_index = parse_episode_index(record.get("episode_index"), video_rel)
        sample_index = record.get("sample_index")
        if sample_index is None:
            sample_index = total - 1

        out = {
            "video_path": str(video_abs),
            "prompt": record.get("prompt", ""),
            "episode_index": episode_index,
            "sample_index": sample_index,
            "task": record.get("task", ""),
        }
        outputs.append(out)
        kept += 1

    # records_out.parent.mkdir(parents=True, exist_ok=True)
    # with records_out.open("w", encoding="utf-8") as f:
    #     for row in outputs:
    #         f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return total, kept, missing


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    raw_jsonl = args.raw_jsonl.resolve() if args.raw_jsonl else auto_find_raw_jsonl(input_root)
    records_out = (
        args.records_out.resolve()
        if args.records_out is not None
        else (input_root / "sample_records.jsonl").resolve()
    )
    videos_dir = (
        args.videos_dir.resolve()
        if args.videos_dir is not None
        else (input_root / "dataset").resolve()
    )
    work_dir = (
        args.work_dir.resolve()
        if args.work_dir is not None
        else (input_root / "worldarena_auto_eval").resolve()
    )

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")
    if not raw_jsonl.exists():
        raise FileNotFoundError(f"raw_jsonl not found: {raw_jsonl}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"videos_dir not found: {videos_dir}")

    total, kept, missing = rebuild_sample_records(
        input_root=input_root,
        raw_jsonl=raw_jsonl,
        records_out=records_out,
        max_records=args.max_records,
    )
    print(
        "[prepare-clean50-gtgt] Rebuilt sample records: "
        f"total={total}, kept={kept}, missing_video={missing}, output={records_out}"
    )

    if args.only_build_records:
        return

    prepare_gtgt_data(
        results_dir=input_root,
        model_name=args.model_name,
        force_rebuild=args.force_rebuild,
        num_shards=args.num_shards,
        work_dir=work_dir,
        records_jsonl=records_out,
        videos_dir=videos_dir,
        copy_only=args.copy_only,
    )
    print(
        "[prepare-clean50-gtgt] Done. "
        f"work_dir={work_dir}, model_name={args.model_name}"
    )


if __name__ == "__main__":
    main()
