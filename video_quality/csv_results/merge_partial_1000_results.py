#!/usr/bin/env python3
"""Merge partial 1000-sample overrides into a complete 1000-sample result set."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from aggregate_and_build_leaderboard_partial_1000 import (
    TASK_LENGTH_COLUMN,
    _build_average_row,
    _extract_episode_index,
    write_task_ewmscore_csv,
)


SUMMARY_VIDEO_IDS = {"AVERAGE", "TASK_AVERAGE"}


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_summary_row(row: Dict[str, str]) -> bool:
    return (row.get("Video_ID") or "").strip().upper() in SUMMARY_VIDEO_IDS


def episode_index_from_row(row: Dict[str, str], source: Path) -> int:
    episode_index = _extract_episode_index(row.get("Video_ID", ""))
    if episode_index is None:
        raise ValueError(f"Cannot extract episode index from Video_ID={row.get('Video_ID')!r} in {source}")
    return episode_index


def sample_row_map(
    rows: List[Dict[str, str]],
    source: Path,
) -> Dict[int, Dict[str, str]]:
    result: Dict[int, Dict[str, str]] = {}
    for row in rows:
        if is_summary_row(row):
            continue
        episode_index = episode_index_from_row(row, source)
        if episode_index in result:
            raise ValueError(
                f"Duplicate episode_index={episode_index} in {source}. "
                "This merge script expects already-merged per-sample CSVs."
            )
        result[episode_index] = row
    return result


def length_map_from_task_csv(
    rows: List[Dict[str, str]],
    source: Path,
) -> Dict[int, str]:
    if rows and TASK_LENGTH_COLUMN not in rows[0]:
        raise KeyError(f"Task EWMScore CSV is missing column '{TASK_LENGTH_COLUMN}': {source}")

    result: Dict[int, str] = {}
    for row in rows:
        if is_summary_row(row):
            continue
        episode_index = episode_index_from_row(row, source)
        length = row.get(TASK_LENGTH_COLUMN, "")
        if length == "":
            raise ValueError(f"Missing length for episode_index={episode_index} in {source}")
        result[episode_index] = length
    return result


def sort_key(row: Dict[str, str]) -> Tuple[str, int]:
    episode_index = _extract_episode_index(row.get("Video_ID", ""))
    if episode_index is None:
        episode_index = 10**12
    return ((row.get("task") or "").casefold(), episode_index)


def default_output_aggregated_path(full_aggregated_csv: Path) -> Path:
    return full_aggregated_csv.with_name(
        f"{full_aggregated_csv.stem}_partial_override{full_aggregated_csv.suffix}"
    )


def default_output_task_path(output_aggregated_csv: Path) -> Path:
    return output_aggregated_csv.with_name(
        f"{output_aggregated_csv.stem}_task_ewmscore{output_aggregated_csv.suffix}"
    )


def merge_partial_results(
    full_aggregated_csv: Path,
    full_task_ewmscore_csv: Path,
    partial_aggregated_csv: Path,
    partial_task_ewmscore_csv: Path,
    output_aggregated_csv: Optional[Path] = None,
    output_task_ewmscore_csv: Optional[Path] = None,
) -> Tuple[Path, Path, Path, Dict[str, int]]:
    full_fieldnames, full_rows = read_csv_rows(full_aggregated_csv)
    partial_fieldnames, partial_rows = read_csv_rows(partial_aggregated_csv)
    _, full_task_rows = read_csv_rows(full_task_ewmscore_csv)
    _, partial_task_rows = read_csv_rows(partial_task_ewmscore_csv)

    if "task" not in full_fieldnames:
        raise KeyError(f"Full aggregated CSV is missing column 'task': {full_aggregated_csv}")
    if "task" not in partial_fieldnames:
        raise KeyError(f"Partial aggregated CSV is missing column 'task': {partial_aggregated_csv}")

    output_fieldnames = [
        fieldname for fieldname in full_fieldnames if fieldname != TASK_LENGTH_COLUMN
    ]

    full_by_episode = sample_row_map(full_rows, full_aggregated_csv)
    partial_by_episode = sample_row_map(partial_rows, partial_aggregated_csv)
    full_lengths = length_map_from_task_csv(full_task_rows, full_task_ewmscore_csv)
    partial_lengths = length_map_from_task_csv(partial_task_rows, partial_task_ewmscore_csv)

    missing_full_lengths = sorted(set(full_by_episode) - set(full_lengths))
    if missing_full_lengths:
        raise ValueError(f"Full task CSV is missing lengths for episodes: {missing_full_lengths[:10]}")

    missing_partial_lengths = sorted(set(partial_by_episode) - set(partial_lengths))
    if missing_partial_lengths:
        raise ValueError(f"Partial task CSV is missing lengths for episodes: {missing_partial_lengths[:10]}")

    merged_by_episode = {episode: dict(row) for episode, row in full_by_episode.items()}
    overwritten = 0
    added = 0
    for episode_index, partial_row in partial_by_episode.items():
        if episode_index in merged_by_episode:
            overwritten += 1
        else:
            added += 1
        merged_by_episode[episode_index] = dict(partial_row)

    merged_rows = sorted(merged_by_episode.values(), key=sort_key)

    average_model_name = ""
    for row in full_rows:
        if (row.get("Video_ID") or "").strip().upper() == "AVERAGE":
            average_model_name = row.get("Model_Name", "")
            break
    if not average_model_name and merged_rows:
        average_model_name = merged_rows[0].get("Model_Name", "")

    output_rows = list(merged_rows)
    output_rows.append(
        _build_average_row(
            csv_rows=merged_rows,
            model_name=average_model_name,
            fieldnames=output_fieldnames,
        )
    )

    output_aggregated_csv = output_aggregated_csv or default_output_aggregated_path(full_aggregated_csv)
    output_task_ewmscore_csv = output_task_ewmscore_csv or default_output_task_path(output_aggregated_csv)

    write_csv(output_aggregated_csv, output_fieldnames, output_rows)

    task_score_rows: List[Dict[str, str]] = []
    for episode_index, row in sorted(merged_by_episode.items(), key=lambda item: sort_key(item[1])):
        task_row = dict(row)
        if episode_index in partial_lengths:
            task_row[TASK_LENGTH_COLUMN] = partial_lengths[episode_index]
        else:
            task_row[TASK_LENGTH_COLUMN] = full_lengths[episode_index]
        task_score_rows.append(task_row)

    task_rows_written, output_task_ewmscore_xlsx = write_task_ewmscore_csv(
        output_csv=output_task_ewmscore_csv,
        sample_rows=task_score_rows,
        fieldnames=output_fieldnames,
    )

    stats = {
        "full_samples": len(full_by_episode),
        "partial_samples": len(partial_by_episode),
        "overwritten_samples": overwritten,
        "added_samples": added,
        "merged_samples": len(merged_by_episode),
        "aggregated_rows_written": len(output_rows),
        "task_score_rows_written": task_rows_written,
    }
    return output_aggregated_csv, output_task_ewmscore_csv, output_task_ewmscore_xlsx, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge partial 1000-sample aggregated/task_ewmscore results into a complete "
            "1000-sample result set, overriding by numeric episode index."
        )
    )
    parser.add_argument("--full_aggregated_csv", type=Path, required=True)
    parser.add_argument("--full_task_ewmscore_csv", type=Path, required=True)
    parser.add_argument("--partial_aggregated_csv", type=Path, required=True)
    parser.add_argument("--partial_task_ewmscore_csv", type=Path, required=True)
    parser.add_argument(
        "--output_aggregated_csv",
        type=Path,
        default=None,
        help="Default: <full_aggregated_csv_stem>_partial_override.csv",
    )
    parser.add_argument(
        "--output_task_ewmscore_csv",
        type=Path,
        default=None,
        help="Default: <output_aggregated_csv_stem>_task_ewmscore.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_aggregated_csv, output_task_csv, output_task_xlsx, stats = merge_partial_results(
        full_aggregated_csv=args.full_aggregated_csv.resolve(),
        full_task_ewmscore_csv=args.full_task_ewmscore_csv.resolve(),
        partial_aggregated_csv=args.partial_aggregated_csv.resolve(),
        partial_task_ewmscore_csv=args.partial_task_ewmscore_csv.resolve(),
        output_aggregated_csv=args.output_aggregated_csv.resolve() if args.output_aggregated_csv else None,
        output_task_ewmscore_csv=(
            args.output_task_ewmscore_csv.resolve() if args.output_task_ewmscore_csv else None
        ),
    )

    for key, value in stats.items():
        print(f"[MERGE] {key}={value}")
    print(f"[OK] output_aggregated_csv={output_aggregated_csv}")
    print(f"[OK] output_task_ewmscore_csv={output_task_csv}")
    print(f"[OK] output_task_ewmscore_xlsx={output_task_xlsx}")


if __name__ == "__main__":
    main()
