"""Aggregate independent metric result files into the public CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import METRIC_COLUMNS, normalize_metrics
from .manifest import atomic_write_json, load_manifest


def _load_metric(output_dir: Path, metric: str) -> dict[str, float]:
    path = output_dir / "results" / "metrics" / f"{metric}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("metric") != metric or not isinstance(payload.get("values"), dict):
        raise ValueError(f"Invalid metric result: {path}")
    return {sample_id: float(value) for sample_id, value in payload["values"].items()}


def aggregate(
    manifest: str | Path,
    output_dir: str | Path,
    metrics: Iterable[str],
) -> Path:
    output_root = Path(output_dir).expanduser().resolve()
    declared_metrics, samples = load_manifest(manifest)
    metric_list = normalize_metrics(metrics)
    undeclared = sorted(set(metric_list) - set(declared_metrics))
    if undeclared:
        raise ValueError(f"Metrics were not prepared: {', '.join(undeclared)}")

    values = {
        metric: _load_metric(output_root, metric)
        for metric in metric_list
        if metric != "jepa_similarity"
    }
    if "jepa_similarity" in metric_list:
        jepa_path = output_root / "results" / "jepa" / "results.json"
        jepa_score = float(json.loads(jepa_path.read_text(encoding="utf-8"))["score"])
        values["jepa_similarity"] = {
            sample.sample_id: jepa_score for sample in samples
        }

    rows = []
    for sample in samples:
        row: dict[str, object] = {"sample_id": sample.sample_id}
        for metric in metric_list:
            if sample.sample_id not in values[metric]:
                raise ValueError(f"Missing {metric} result for {sample.sample_id}")
            row[METRIC_COLUMNS[metric]] = values[metric][sample.sample_id]
        rows.append(row)

    average: dict[str, object] = {"sample_id": "AVERAGE"}
    for metric in metric_list:
        column = METRIC_COLUMNS[metric]
        average[column] = float(np.mean([row[column] for row in rows]))

    result_dir = output_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_dir / "results.json", {"rows": rows, "average": average})

    csv_path = result_dir / "results.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    fieldnames = ["sample_id", *[METRIC_COLUMNS[metric] for metric in metric_list]]
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
