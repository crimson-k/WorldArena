"""Build one comparison table from completed checkpoint evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from video_quality.constants import METRIC_COLUMNS, SUPPORTED_METRICS
from video_quality.manifest import atomic_write_json


CHECKPOINTS = (8000, 12000, 20000, 22000, 24000, 26000)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", required=True, type=Path)
    args = parser.parse_args()

    output_base = args.output_base.expanduser().resolve()
    columns = [METRIC_COLUMNS[metric] for metric in SUPPORTED_METRICS]
    rows: list[dict[str, object]] = []
    for checkpoint in CHECKPOINTS:
        result_path = (
            output_base
            / f"checkpoint-{checkpoint}"
            / "full-validation-500"
            / "results"
            / "results.json"
        )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        average = payload.get("average")
        if not isinstance(average, dict):
            raise ValueError(f"Missing average result: {result_path}")
        row: dict[str, object] = {"checkpoint": checkpoint}
        for column in columns:
            row[column] = float(average[column])
        rows.append(row)

    atomic_write_json(output_base / "checkpoint_summary.json", rows)
    csv_path = output_base / "checkpoint_summary.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["checkpoint", *columns])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    temporary.replace(csv_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
