"""Aggregate independent metric result files into the public CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import METRIC_COLUMNS, SUPPORTED_METRICS, normalize_metrics
from .manifest import atomic_write_json, load_manifest


_COLUMN_TO_METRIC = {column: metric for metric, column in METRIC_COLUMNS.items()}


def _load_metric(output_dir: Path, metric: str) -> dict[str, float]:
    path = output_dir / "results" / "metrics" / f"{metric}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("metric") != metric or not isinstance(payload.get("values"), dict):
        raise ValueError(f"Invalid metric result: {path}")
    return {sample_id: float(value) for sample_id, value in payload["values"].items()}


def _load_previous_results(
    output_dir: Path, samples
) -> dict[str, dict[str, float]]:
    """Load the previous public table so later metric runs can extend it."""
    path = output_dir / "results" / "results.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Invalid previous results table: {path}")
    expected_ids = {sample.sample_id for sample in samples}
    previous_ids = {
        row.get("sample_id")
        for row in rows
        if isinstance(row, dict) and row.get("sample_id") != "AVERAGE"
    }
    if previous_ids != expected_ids:
        raise ValueError(
            "Cannot merge metrics: sample IDs in the previous results table do not "
            "match the current manifest"
        )

    values: dict[str, dict[str, float]] = {}
    for column, metric in _COLUMN_TO_METRIC.items():
        metric_values = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("sample_id") == "AVERAGE":
                continue
            if column in row:
                metric_values[row["sample_id"]] = float(row[column])
        if metric_values:
            values[metric] = metric_values
    return values


def aggregate(
    manifest: str | Path,
    output_dir: str | Path,
    metrics: Iterable[str],
) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    declared_metrics, samples = load_manifest(manifest)
    metric_list = normalize_metrics(metrics)
    values = _load_previous_results(output_root, samples)
    metric_files = {
        metric
        for metric in SUPPORTED_METRICS
        if metric != "jepa_similarity"
        and (output_root / "results" / "metrics" / f"{metric}.json").is_file()
    }
    jepa_file = output_root / "results" / "jepa" / "results.json"
    available_previous = set(values) | metric_files
    if jepa_file.is_file():
        available_previous.add("jepa_similarity")
    undeclared = sorted(
        set(metric_list) - set(declared_metrics) - available_previous
    )
    if undeclared:
        raise ValueError(f"Metrics were not prepared: {', '.join(undeclared)}")

    # Independent metric files are durable across separate evaluation stages.
    # Keep any old metric that was not represented in results.json yet.
    for metric in metric_files:
        metric_path = output_root / "results" / "metrics" / f"{metric}.json"
        if metric not in values:
            values[metric] = _load_metric(output_root, metric)

    for metric in metric_list:
        if metric == "jepa_similarity":
            jepa_path = output_root / "results" / "jepa" / "results.json"
            jepa_score = float(
                json.loads(jepa_path.read_text(encoding="utf-8"))["score"]
            )
            values[metric] = {
                sample.sample_id: jepa_score for sample in samples
            }
        else:
            values[metric] = _load_metric(output_root, metric)

    merged_metrics = [metric for metric in SUPPORTED_METRICS if metric in values]

    rows = []
    for sample in samples:
        row: dict[str, object] = {"sample_id": sample.sample_id}
        for metric in merged_metrics:
            if sample.sample_id not in values[metric]:
                raise ValueError(f"Missing {metric} result for {sample.sample_id}")
            row[METRIC_COLUMNS[metric]] = values[metric][sample.sample_id]
        rows.append(row)

    average: dict[str, object] = {"sample_id": "AVERAGE"}
    for metric in merged_metrics:
        column = METRIC_COLUMNS[metric]
        finite_values = [row[column] for row in rows if np.isfinite(row[column])]
        average[column] = float(np.mean(finite_values)) if finite_values else "-"

    result_dir = output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_dir / "results.json", {"rows": rows, "average": average})

    csv_path = result_dir / "results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    fieldnames = ["sample_id", *[METRIC_COLUMNS[metric] for metric in merged_metrics]]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in [*rows, average]:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    temporary.replace(csv_path)
    return csv_path
