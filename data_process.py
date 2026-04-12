#!/usr/bin/env python3
"""Prepare minimal WorldArena eval inputs from stitched result videos.

This script keeps only the pieces required by the downstream README format:
- split each stitched video into left-half GT and right-half generated video
- save GT first frame
- write a minimal summary.json with gt_path/image/prompt
- optionally split prepared data into evenly balanced shard folders

Output layout:
  <work_dir>/
    gt_videos/
    <model_name>_test/        # generated videos for action_following preprocess
    <model_name>_test_vlm/    # generated videos for run_VLM_judge.sh
    gt_first_frames/
    summary.json
    shards/
      shard_00/
        gt_videos/
        <model_name>_test/
        <model_name>_test_vlm/
        gt_first_frames/
        summary.json
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


@dataclass
class PreparedData:
    results_dir: Path
    videos_dir: Path
    records_path: Path
    work_dir: Path
    summary_json: Path
    gt_videos_dir: Path
    gt_first_frames_dir: Path
    gen_videos_dir: Path
    vlm_videos_dir: Path
    prepared: List[PreparedSample]


def _default_log(msg: str) -> None:
    print(f"[data-process] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()
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
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                _default_log(f"[WARN] Skip invalid JSON line {line_no}: {path}")
    return records


def resolve_inputs(
    results_dir: Path,
    videos_dir: Optional[Path] = None,
    records_jsonl: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    resolved_results_dir = results_dir.resolve()
    resolved_videos_dir = videos_dir.resolve() if videos_dir else (resolved_results_dir / "videos")
    resolved_records_path = records_jsonl.resolve() if records_jsonl else (resolved_results_dir / "sample_records.jsonl")
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


def row_bounds(height: int, row_mode: str) -> Tuple[int, int]:
    if row_mode == "full":
        return 0, height

    idx_map = {"top": 0, "middle": 1, "bottom": 2}
    row_idx = idx_map[row_mode]
    row_h = max(1, height // 3)
    y0 = row_idx * row_h
    y1 = height if row_idx == 2 else min(height, (row_idx + 1) * row_h)
    return y0, y1


def split_stitched_video(
    src_video: Path,
    gt_video_out: Path,
    gen_video_out: Path,
    first_frame_out: Path,
    row_mode: str,
    vlm_video_out: Optional[Path] = None,
) -> int:
    ensure_dir(gt_video_out.parent)
    ensure_dir(gen_video_out.parent)
    ensure_dir(first_frame_out.parent)
    if vlm_video_out is not None:
        ensure_dir(vlm_video_out.parent)

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 2 or height < 2:
        cap.release()
        raise RuntimeError(f"Invalid video shape ({width}x{height}): {src_video}")

    split_x = width // 2
    y0, y1 = row_bounds(height, row_mode)

    out_w = split_x
    out_h = max(1, y1 - y0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    gt_writer = cv2.VideoWriter(str(gt_video_out), fourcc, fps, (out_w, out_h))
    gen_writer = cv2.VideoWriter(str(gen_video_out), fourcc, fps, (out_w, out_h))
    vlm_writer = cv2.VideoWriter(str(vlm_video_out), fourcc, fps, (out_w, out_h)) if vlm_video_out else None

    n_frames = 0
    first_saved = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue

        gt_frame = frame[y0:y1, :split_x]
        gen_frame = frame[y0:y1, split_x:width]
        if gt_frame.size == 0 or gen_frame.size == 0:
            continue

        gt_writer.write(gt_frame)
        gen_writer.write(gen_frame)
        if vlm_writer is not None:
            vlm_writer.write(gen_frame)

        if not first_saved:
            cv2.imwrite(str(first_frame_out), gt_frame)
            first_saved = True
        n_frames += 1

    cap.release()
    gt_writer.release()
    gen_writer.release()
    if vlm_writer is not None:
        vlm_writer.release()

    if n_frames == 0:
        raise RuntimeError(f"No frames written after split: {src_video}")
    return n_frames


def build_preprocess_compatible_gen_name(gt_video: Path, video_id: str) -> str:
    """Match the lookup rule used by video_quality/preprocess_datasets.py.

    That script looks up generated videos as:
      {gt_path.parts[-5]}_{gt_video_stem}.mp4
    """
    gt_parts = gt_video.resolve().parts
    if len(gt_parts) >= 5:
        prefix = gt_parts[-5]
    else:
        prefix = "data"
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
    logger: Logger,
) -> None:
    if num_shards <= 1:
        return

    effective_shards = num_shards
    total = len(prepared)
    if effective_shards > total:
        effective_shards = total
        logger(
            f"[WARN] num_shards={num_shards} is larger than prepared samples={total}; "
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

            link_or_copy(sample.gt_video, shard_gt_video)
            link_or_copy(sample.first_frame, shard_first_frame)
            link_or_copy(sample.vlm_video, shard_vlm_video)
            link_or_copy(sample.gen_video, shard_gen_video)

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

    logger(
        f"Shard inputs generated under: {shard_root} "
        f"(effective_num_shards={effective_shards})."
    )


def prepare_results_data(
    *,
    results_dir: Path,
    model_name: str,
    task_name: str,
    row_mode: str,
    force_rebuild: bool = False,
    num_shards: int = 1,
    work_dir: Optional[Path] = None,
    records_jsonl: Optional[Path] = None,
    videos_dir: Optional[Path] = None,
    logger: Optional[Logger] = None,
) -> PreparedData:
    log = logger or _default_log
    resolved_results_dir, resolved_videos_dir, resolved_records_path = resolve_inputs(
        results_dir=results_dir,
        videos_dir=videos_dir,
        records_jsonl=records_jsonl,
    )

    if not resolved_videos_dir.exists() or not resolved_records_path.exists():
        raise FileNotFoundError(
            f"`{resolved_results_dir}` must contain `videos/` and `sample_records.jsonl` "
            f"(or provide explicit videos_dir/records_jsonl)."
        )

    resolved_work_dir = work_dir.resolve() if work_dir else (resolved_results_dir / "worldarena_auto_eval").resolve()
    ensure_dir(resolved_work_dir)

    log(f"results_dir: {resolved_results_dir}")
    log(f"work_dir: {resolved_work_dir}")

    gt_videos_dir = resolved_work_dir / "gt_videos"
    gt_first_frames_dir = resolved_work_dir / "gt_first_frames"
    gen_videos_dir = resolved_work_dir / f"{model_name}_test"
    vlm_videos_dir = resolved_work_dir / f"{model_name}_test_vlm"

    for d in [gt_videos_dir, gt_first_frames_dir, gen_videos_dir, vlm_videos_dir]:
        ensure_dir(d)

    records = load_jsonl(resolved_records_path)
    if not records:
        raise RuntimeError(f"No valid records found in {resolved_records_path}")
    log(f"Loaded {len(records)} records.")
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")

    used_ids: set[str] = set()
    prepared: List[PreparedSample] = []

    for idx, rec in enumerate(records):
        rec_video = str(rec.get("video_path", ""))
        input_video = resolve_input_video(rec_video, resolved_videos_dir)
        if input_video is None:
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

        need_rebuild = force_rebuild or not (gt_video.exists() and gen_video.exists() and first_frame.exists())
        if need_rebuild:
            n_frames = split_stitched_video(
                src_video=input_video,
                gt_video_out=gt_video,
                gen_video_out=gen_video,
                first_frame_out=first_frame,
                row_mode=row_mode,
                vlm_video_out=vlm_video,
            )
            log(f"Prepared {input_video.name} -> frames={n_frames}, id={video_id}")
        elif not vlm_video.exists():
            # Backfill VLM-friendly filename for existing processed outputs.
            try:
                os.link(gen_video, vlm_video)
            except Exception:
                shutil.copy2(gen_video, vlm_video)

        prompt = str(rec.get("prompt", "")).strip()
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
    log(f"summary.json written: {summary_json}")

    build_shard_inputs(
        prepared=prepared,
        work_dir=resolved_work_dir,
        model_name=model_name,
        num_shards=num_shards,
        logger=log,
    )

    return PreparedData(
        results_dir=resolved_results_dir,
        videos_dir=resolved_videos_dir,
        records_path=resolved_records_path,
        work_dir=resolved_work_dir,
        summary_json=summary_json,
        gt_videos_dir=gt_videos_dir,
        gt_first_frames_dir=gt_first_frames_dir,
        gen_videos_dir=gen_videos_dir,
        vlm_videos_dir=vlm_videos_dir,
        prepared=prepared,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare WorldArena eval data from results_inference directory."
    )
    parser.add_argument("--results_dir", type=Path, required=True, help="Path to results_inference directory.")
    parser.add_argument("--work_dir", type=Path, default=None, help="Output work dir. Default: <results_dir>/worldarena_auto_eval")
    parser.add_argument("--model_name", type=str, default="results_inference_auto")
    parser.add_argument("--task_name", type=str, default="task_auto")
    parser.add_argument("--row_mode", type=str, default="full", choices=["full", "top", "middle", "bottom"])
    parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild split videos and first frames.")
    parser.add_argument("--num_shards", type=int, default=1, help="Evenly split prepared data into this many shard folders.")
    parser.add_argument("--records_jsonl", type=Path, default=None, help="Optional records jsonl path.")
    parser.add_argument("--videos_dir", type=Path, default=None, help="Optional videos directory path.")
    return parser


def main() -> None:
    start_time = time.time()
    args = build_arg_parser().parse_args()
    prepared = prepare_results_data(
        results_dir=args.results_dir,
        model_name=args.model_name,
        task_name=args.task_name,
        row_mode=args.row_mode,
        force_rebuild=args.force_rebuild,
        num_shards=args.num_shards,
        work_dir=args.work_dir,
        records_jsonl=args.records_jsonl,
        videos_dir=args.videos_dir,
    )
    elapsed_seconds = time.time() - start_time
    _default_log(
        f"Done. prepared={len(prepared.prepared)}, summary_json={prepared.summary_json}, "
        f"work_dir={prepared.work_dir}, elapsed_s={elapsed_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
