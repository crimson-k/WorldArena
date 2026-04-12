#!/usr/bin/env python3
"""Aggregate sharded WorldArena evaluation outputs into one CSV.

This script does two steps:
1) For each shard directory (e.g. data/test_60_shard_00), call per-shard aggregation.
2) Merge all shard CSVs into one final CSV under data/<model_name>/csv_results/.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path
from typing import Dict, List, Optional

from csv_results.aggregate_results import aggregate_results


def _default_width(num_shards: int) -> int:
    if num_shards <= 1:
        return 2
    return max(2, len(str(num_shards - 1)))


def _parse_shard_ids(
    data_root: Path,
    model_name: str,
    num_shards: Optional[int],
    shard_ids: Optional[str],
) -> List[str]:
    if shard_ids:
        ids = [s.strip() for s in shard_ids.split(",") if s.strip()]
        if not ids:
            raise ValueError("--shard_ids is set but empty after parsing.")
        return ids

    if num_shards is not None:
        if num_shards < 1:
            raise ValueError("--num_shards must be >= 1.")
        width = _default_width(num_shards)
        return [f"{idx:0{width}d}" for idx in range(num_shards)]

    pattern = str(data_root / f"{model_name}_shard_*")
    shard_dirs = sorted(glob.glob(pattern))
    if not shard_dirs:
        raise FileNotFoundError(
            f"No shard directories found with pattern: {pattern}. "
            "Please provide --num_shards or --shard_ids explicitly."
        )

    suffixes: List[str] = []
    prefix = f"{model_name}_shard_"
    for d in shard_dirs:
        name = Path(d).name
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix:
            suffixes.append(suffix)
    if not suffixes:
        raise RuntimeError(f"Failed to parse shard suffixes from: {pattern}")

    try:
        return sorted(suffixes, key=lambda s: int(s))
    except ValueError:
        return sorted(suffixes)


def _merge_csvs(
    csv_paths: List[Path],
    unified_model_name: str,
    output_csv: Path,
) -> Dict[str, int]:
    if not csv_paths:
        raise ValueError("No shard CSV files to merge.")

    rows: List[Dict[str, str]] = []
    seen_video_ids = set()
    header: Optional[List[str]] = None
    duplicate_count = 0

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            if not header:
                raise RuntimeError(f"Empty CSV header in {csv_path}")

            for row in reader:
                video_id = (row.get("Video_ID") or "").strip()
                if not video_id:
                    continue
                if video_id in seen_video_ids:
                    duplicate_count += 1
                    continue
                seen_video_ids.add(video_id)
                row["Model_Name"] = unified_model_name
                rows.append(row)

    if not header:
        raise RuntimeError("Failed to determine CSV header while merging shard CSVs.")

    rows.sort(key=lambda r: (r.get("Video_ID") or ""))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "rows_written": len(rows),
        "duplicates_skipped": duplicate_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate sharded evaluation results into one final CSV."
    )
    parser.add_argument("--model_name", required=True, help="Base model name, e.g. test_60")
    parser.add_argument(
        "--video_quality_root",
        default=str(Path(__file__).resolve().parent),
        help="Path to video_quality directory (default: current script directory)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=None,
        help="Number of shards. If omitted, auto-discover data/<model>_shard_*.",
    )
    parser.add_argument(
        "--shard_ids",
        default=None,
        help="Comma-separated shard suffixes, e.g. 00,01,02,03. Overrides --num_shards.",
    )
    parser.add_argument(
        "--jepa_result_path",
        default=None,
        help="Path to JEPA result JSON. Default: data/<model_name>/output_JEDi/results.json if exists.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Final merged CSV path. Default: data/<model_name>/csv_results/<model_name>_aggregated_results_all_shards.csv",
    )
    args = parser.parse_args()

    root = Path(args.video_quality_root).resolve()
    data_root = root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"data directory not found: {data_root}")

    shard_suffixes = _parse_shard_ids(
        data_root=data_root,
        model_name=args.model_name,
        num_shards=args.num_shards,
        shard_ids=args.shard_ids,
    )

    default_jepa = data_root / args.model_name / "output_JEDi" / "results.json"
    jepa_path: Optional[str]
    if args.jepa_result_path:
        jepa_path = args.jepa_result_path
    elif default_jepa.exists():
        jepa_path = str(default_jepa)
    else:
        jepa_path = None

    shard_csv_paths: List[Path] = []
    for suffix in shard_suffixes:
        shard_model_name = f"{args.model_name}_shard_{suffix}"
        shard_base_dir = data_root / shard_model_name
        if not shard_base_dir.exists():
            raise FileNotFoundError(f"Shard base_dir not found: {shard_base_dir}")

        vlm_dir = shard_base_dir / "output_VLM" / shard_model_name
        vlm_jsons = sorted(vlm_dir.glob("*.json")) if vlm_dir.exists() else []
        if not vlm_jsons:
            print(
                f"[WARN] No VLM json found for {shard_model_name}. "
                f"Expected under: {vlm_dir}"
            )
        else:
            print(f"[info] {shard_model_name} VLM json files: {len(vlm_jsons)}")

        shard_csv_name = f"{shard_model_name}_aggregated_results.csv"
        shard_csv_path = Path(
            aggregate_results(
                base_dir=str(shard_base_dir),
                model_name=shard_model_name,
                csv_name=shard_csv_name,
                vlm_model_dir=shard_model_name,
                jepa_result_path=jepa_path,
            )
        )
        shard_csv_paths.append(shard_csv_path)
        print(f"[per-shard] {shard_model_name} -> {shard_csv_path}")

    if args.output_csv:
        merged_csv = Path(args.output_csv).resolve()
    else:
        merged_csv = (
            data_root
            / args.model_name
            / "csv_results"
            / f"{args.model_name}_aggregated_results_all_shards.csv"
        )

    stats = _merge_csvs(
        csv_paths=shard_csv_paths,
        unified_model_name=args.model_name,
        output_csv=merged_csv,
    )
    print(f"[merged] output={merged_csv}")
    print(
        f"[merged] shards={len(shard_csv_paths)} rows={stats['rows_written']} "
        f"duplicates_skipped={stats['duplicates_skipped']}"
    )


if __name__ == "__main__":
    main()
