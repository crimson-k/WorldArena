#!/usr/bin/env python3
"""Prepare WorldArena eval inputs for GT-vs-GT evaluation.

Compared with `data_process.py`, this script skips stitched-video splitting:
- each input GT video is used directly as GT output
- the same GT video is also used as generated output
- first frame is extracted for summary/image

Output layout (same convention as `data_process.py`):
  <work_dir>/
    gt_videos/
    <model_name>_test/
    <model_name>_test_vlm/
    gt_first_frames/
    summary.json
    shards/shard_xx/... (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2


Logger = Callable[[str], None]


@dataclass
class PreparedSample:
    video_id: str
    gt_video: Path
    gen_video: Path
    vlm_video: Path
    first_frame: Path
    prompt: str


def _default_log(msg: str) -> None:
    print(f"[data-process-gtgt] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_only: bool) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()

    if copy_only:
        shutil.copy2(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                _default_log(f"[WARN] Skip invalid JSON line {line_no}: {path}")
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def resolve_inputs(
    results_dir: Path,
    videos_dir: Optional[Path],
    records_jsonl: Optional[Path],
) -> Tuple[Path, Path, Path]:
    resolved_results_dir = results_dir.resolve()
    if videos_dir is not None:
        resolved_videos_dir = videos_dir.resolve()
    else:
        preferred = resolved_results_dir / "videos_GT"
        fallback = resolved_results_dir / "videos"
        resolved_videos_dir = preferred if preferred.exists() else fallback

    resolved_records_path = (
        records_jsonl.resolve()
        if records_jsonl is not None
        else (resolved_results_dir / "sample_records.jsonl")
    )
    return resolved_results_dir, resolved_videos_dir, resolved_records_path


def resolve_input_video(record_video_path: str, local_videos_dir: Path) -> Optional[Path]:
    rec_path = Path(record_video_path)
    by_name = local_videos_dir / rec_path.name
    if by_name.exists():
        return by_name
    if rec_path.exists():
        return rec_path
    return None


def parse_episode_base(record: Dict, video_name: str) -> str:
    ep_idx = record.get("episode_index")
    if ep_idx is not None:
        ep_str = str(ep_idx)
        if ep_str.startswith("episode"):
            return ep_str
        return f"episode{ep_str}"

    match = re.search(r"ep(\d+)", video_name)
    if match:
        return f"episode{match.group(1)}"
    return "episode_unknown"


def make_video_id(record: Dict, video_name: str, fallback_index: int) -> str:
    ep = parse_episode_base(record, video_name)
    sample_idx = record.get("sample_index", fallback_index)
    try:
        sample_idx_int = int(sample_idx)
    except (TypeError, ValueError):
        sample_idx_int = fallback_index
    return f"{ep}_s{sample_idx_int:06d}"


def normalize_prompt(prompt_raw: object) -> str:
    if isinstance(prompt_raw, list) and prompt_raw:
        return str(prompt_raw[0]).strip()
    if isinstance(prompt_raw, str):
        return prompt_raw.strip()
    return ""


def extract_first_frame(video_path: Path, image_out: Path) -> None:
    ensure_dir(image_out.parent)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for first frame: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to read first frame: {video_path}")
    ok = cv2.imwrite(str(image_out), frame)
    if not ok:
        raise RuntimeError(f"Failed to write first frame image: {image_out}")


def build_preprocess_compatible_gen_name(gt_video: Path, video_id: str) -> str:
    """Match `video_quality/preprocess_datasets.py` lookup rule."""
    gt_parts = gt_video.resolve().parts
    prefix = gt_parts[-5] if len(gt_parts) >= 5 else "data"
    return f"{prefix}_{video_id}.mp4"


def format_shard_name(index: int, num_shards: int) -> str:
    width = max(2, len(str(max(0, num_shards - 1))))
    return f"shard_{index:0{width}d}"


def split_prepared_evenly(prepared: List[PreparedSample], num_shards: int) -> List[List[PreparedSample]]:
    buckets: List[List[PreparedSample]] = [[] for _ in range(num_shards)]
    for idx, sample in enumerate(prepared):
        buckets[idx % num_shards].append(sample)
    return buckets


def build_shard_inputs(
    *,
    prepared: List[PreparedSample],
    work_dir: Path,
    model_name: str,
    num_shards: int,
    copy_only: bool,
    logger: Logger,
) -> None:
    if num_shards <= 1:
        return

    total = len(prepared)
    effective_shards = min(num_shards, total)
    if effective_shards < num_shards:
        logger(
            f"[WARN] num_shards={num_shards} > prepared={total}; "
            f"reduced to {effective_shards}."
        )

    shard_root = work_dir / "shards"
    ensure_dir(shard_root)
    buckets = split_prepared_evenly(prepared, effective_shards)
    manifest: List[Dict] = []

    for shard_index, shard_samples in enumerate(buckets):
        shard_name = format_shard_name(shard_index, effective_shards)
        shard_dir = shard_root / shard_name
        shard_gt_dir = shard_dir / "gt_videos"
        shard_gen_dir = shard_dir / f"{model_name}_test"
        shard_vlm_dir = shard_dir / f"{model_name}_test_vlm"
        shard_first_frames_dir = shard_dir / "gt_first_frames"
        for d in [shard_gt_dir, shard_gen_dir, shard_vlm_dir, shard_first_frames_dir]:
            ensure_dir(d)

        shard_summary: List[Dict] = []
        for sample in shard_samples:
            shard_gt_video = shard_gt_dir / f"{sample.video_id}.mp4"
            shard_first_frame = shard_first_frames_dir / f"{sample.video_id}.png"
            shard_vlm_video = shard_vlm_dir / f"{sample.video_id}.mp4"
            shard_gen_video = shard_gen_dir / build_preprocess_compatible_gen_name(
                shard_gt_video, sample.video_id
            )

            link_or_copy(sample.gt_video, shard_gt_video, copy_only)
            link_or_copy(sample.first_frame, shard_first_frame, copy_only)
            link_or_copy(sample.vlm_video, shard_vlm_video, copy_only)
            link_or_copy(sample.gen_video, shard_gen_video, copy_only)

            shard_summary.append(
                {
                    "gt_path": str(shard_gt_video.resolve()),
                    "image": str(shard_first_frame.resolve()),
                    "prompt": [sample.prompt],
                }
            )

        shard_summary_path = shard_dir / "summary.json"
        with shard_summary_path.open("w", encoding="utf-8") as f:
            json.dump(shard_summary, f, ensure_ascii=False, indent=2)

        shard_meta = {
            "shard_index": shard_index,
            "num_shards": effective_shards,
            "num_samples": len(shard_samples),
            "summary_json": str(shard_summary_path.resolve()),
            "gen_video_dir": str(shard_gen_dir.resolve()),
            "vlm_video_dir": str(shard_vlm_dir.resolve()),
            "model_name": model_name,
        }
        shard_meta_path = shard_dir / "meta.json"
        with shard_meta_path.open("w", encoding="utf-8") as f:
            json.dump(shard_meta, f, ensure_ascii=False, indent=2)
        manifest.append(shard_meta)

    manifest_path = shard_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger(f"Shard inputs generated under: {shard_root}")


def prepare_gtgt_data(
    *,
    results_dir: Path,
    model_name: str,
    force_rebuild: bool = False,
    num_shards: int = 1,
    work_dir: Optional[Path] = None,
    records_jsonl: Optional[Path] = None,
    videos_dir: Optional[Path] = None,
    copy_only: bool = False,
    logger: Optional[Logger] = None,
) -> None:
    log = logger or _default_log
    resolved_results_dir, resolved_videos_dir, resolved_records_path = resolve_inputs(
        results_dir=results_dir,
        videos_dir=videos_dir,
        records_jsonl=records_jsonl,
    )

    if not resolved_videos_dir.exists() or not resolved_records_path.exists():
        raise FileNotFoundError(
            f"`{resolved_results_dir}` must contain `videos_GT/` (or provided videos_dir) "
            f"and `sample_records.jsonl` (or provided records_jsonl)."
        )

    resolved_work_dir = (
        work_dir.resolve() if work_dir else (resolved_results_dir / "worldarena_auto_eval").resolve()
    )
    ensure_dir(resolved_work_dir)

    gt_videos_dir = resolved_work_dir / "gt_videos"
    gt_first_frames_dir = resolved_work_dir / "gt_first_frames"
    gen_videos_dir = resolved_work_dir / f"{model_name}_test"
    vlm_videos_dir = resolved_work_dir / f"{model_name}_test_vlm"
    for d in [gt_videos_dir, gt_first_frames_dir, gen_videos_dir, vlm_videos_dir]:
        ensure_dir(d)

    records = load_jsonl(resolved_records_path)
    if not records:
        raise RuntimeError(f"No valid records found in {resolved_records_path}")
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    log(f"results_dir: {resolved_results_dir}")
    log(f"videos_dir: {resolved_videos_dir}")
    log(f"records: {len(records)}")
    log(f"work_dir: {resolved_work_dir}")

    prepared: List[PreparedSample] = []
    used_ids: set[str] = set()
    skipped_missing = 0

    for idx, rec in enumerate(records):
        rec_video = str(rec.get("video_path", ""))
        input_video = resolve_input_video(rec_video, resolved_videos_dir)
        if input_video is None:
            skipped_missing += 1
            log(f"[WARN] Skip record {idx}, video not found: {rec_video}")
            continue

        video_id = make_video_id(rec, input_video.name, idx)
        while video_id in used_ids:
            video_id = f"{video_id}_dup"
        used_ids.add(video_id)

        gt_video = gt_videos_dir / f"{video_id}.mp4"
        gen_video = gen_videos_dir / build_preprocess_compatible_gen_name(gt_video, video_id)
        vlm_video = vlm_videos_dir / f"{video_id}.mp4"
        first_frame = gt_first_frames_dir / f"{video_id}.png"

        need_rebuild = force_rebuild or not (
            gt_video.exists() and gen_video.exists() and vlm_video.exists() and first_frame.exists()
        )
        if need_rebuild:
            link_or_copy(input_video, gt_video, copy_only)
            link_or_copy(input_video, gen_video, copy_only)
            link_or_copy(input_video, vlm_video, copy_only)
            extract_first_frame(input_video, first_frame)

        prompt = normalize_prompt(rec.get("prompt", ""))
        prepared.append(
            PreparedSample(
                video_id=video_id,
                gt_video=gt_video,
                gen_video=gen_video,
                vlm_video=vlm_video,
                first_frame=first_frame,
                prompt=prompt,
            )
        )

    if not prepared:
        raise RuntimeError("No samples prepared. Check inputs.")

    summary_data = [
        {
            "gt_path": str(item.gt_video.resolve()),
            "image": str(item.first_frame.resolve()),
            "prompt": [item.prompt],
        }
        for item in prepared
    ]
    summary_json = resolved_work_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    build_shard_inputs(
        prepared=prepared,
        work_dir=resolved_work_dir,
        model_name=model_name,
        num_shards=num_shards,
        copy_only=copy_only,
        logger=log,
    )

    log(
        "Done. "
        f"prepared={len(prepared)}, skipped_missing={skipped_missing}, "
        f"summary_json={summary_json}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare WorldArena eval data for GT-vs-GT (no split)."
    )
    parser.add_argument("--results_dir", type=Path, required=True, help="Path to results root directory.")
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Output work dir. Default: <results_dir>/worldarena_auto_eval",
    )
    parser.add_argument("--model_name", type=str, default="GTGT")
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild links/copies and first frames.",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Evenly split prepared data into this many shard folders.",
    )
    parser.add_argument("--records_jsonl", type=Path, default=None, help="Optional records jsonl path.")
    parser.add_argument("--videos_dir", type=Path, default=None, help="Optional videos directory path.")
    parser.add_argument(
        "--copy_only",
        action="store_true",
        help="Use copy instead of hard-link (hard-link fallback is already automatic).",
    )
    return parser


def main() -> None:
    start_time = time.time()
    args = build_arg_parser().parse_args()
    prepare_gtgt_data(
        results_dir=args.results_dir,
        model_name=args.model_name,
        force_rebuild=args.force_rebuild,
        num_shards=args.num_shards,
        work_dir=args.work_dir,
        records_jsonl=args.records_jsonl,
        videos_dir=args.videos_dir,
        copy_only=args.copy_only,
    )
    elapsed_seconds = time.time() - start_time
    _default_log(f"Elapsed_s={elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
