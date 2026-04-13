#!/usr/bin/env python3
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


BASE_METRICS: List[str] = [
    "Image Quality",
    "Aesthetic Quality",
    "JEPA Similarity",
    "Dynamic Degree",
    "Flow Score",
    "Motion Smoothness",
    "Subject Consistency",
    "Background Consistency",
    "Photometric Consistency",
    "Interaction Quality",
    "Trajectory Accuracy",
    "Depth Accuracy",
    "Perspectivity",
    "Instruction Following",
    "Semantic Alignment",
    "Action Following",
]

DIMENSION_MAP: Dict[str, List[str]] = {
    "Visual Quality": ["Image Quality", "Aesthetic Quality", "JEPA Similarity"],
    "Motion Quality": ["Dynamic Degree", "Flow Score", "Motion Smoothness"],
    "Content Consistency": ["Subject Consistency", "Background Consistency", "Photometric Consistency"],
    "Physics Adherence": ["Interaction Quality", "Trajectory Accuracy"],
    "3D Accuracy": ["Depth Accuracy", "Perspectivity"],
    "Controllability": ["Instruction Following", "Semantic Alignment", "Action Following"],
}

DIMENSION_COLUMNS: List[str] = list(DIMENSION_MAP.keys())
CORE_SCORE_COLUMNS: List[str] = ["EWMScore"] + DIMENSION_COLUMNS
ALL_SCORE_COLUMNS: List[str] = CORE_SCORE_COLUMNS + BASE_METRICS

LEADERBOARD_COLUMNS: List[str] = [
    "Model",
    "open_source",
    "year",
    "EWMScore",
    "Visual Quality",
    "Motion Quality",
    "Content Consistency",
    "Physics Adherence",
    "3D Accuracy",
    "Controllability",
] + BASE_METRICS

MARKDOWN_COLUMNS: List[str] = [
    "Model",
    "open_source",
    "year",
    "EWMScore",
    "Visual Quality",
    "Motion Quality",
    "Content Consistency",
    "Physics Adherence",
    "3D Accuracy",
    "Controllability",
] + sorted(BASE_METRICS)


@dataclass
class LoadResult:
    metrics: Dict[str, float]
    source: str
    warnings: List[str]


def _to_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _select_average_or_compute(rows: List[Dict[str, str]]) -> Tuple[Dict[str, Optional[float]], str]:
    for row in rows:
        vid = (row.get("Video_ID") or "").strip().lower()
        if vid == "average":
            metric_map = {metric: _to_float(row.get(metric)) for metric in BASE_METRICS}
            return metric_map, "AVERAGE row"

    metric_map: Dict[str, Optional[float]] = {}
    for metric in BASE_METRICS:
        vals: List[float] = []
        for row in rows:
            vid = (row.get("Video_ID") or "").strip().lower()
            if vid == "average":
                continue
            val = _to_float(row.get(metric))
            if val is not None:
                vals.append(val)
        metric_map[metric] = _mean(vals) if vals else None
    return metric_map, "mean over per-video rows"


def _infer_scale_factor(metric_values: Dict[str, Optional[float]], mode: str) -> float:
    if mode == "raw":
        return 1.0
    if mode == "percent":
        return 100.0
    # auto
    vals = [v for v in metric_values.values() if v is not None]
    if not vals:
        return 100.0
    if min(vals) >= 0.0 and max(vals) <= 1.5:
        return 100.0
    return 1.0


def load_model_metrics_from_aggregated_csv(
    aggregated_csv: Path,
    scale_mode: str = "auto",
    action_following_default: float = 0.0,
) -> LoadResult:
    rows = _read_csv_rows(aggregated_csv)
    if not rows:
        raise ValueError(f"Empty aggregated CSV: {aggregated_csv}")

    raw_metrics, source = _select_average_or_compute(rows)
    scale = _infer_scale_factor(raw_metrics, scale_mode)

    warnings: List[str] = []
    final_metrics: Dict[str, float] = {}
    for metric in BASE_METRICS:
        value = raw_metrics.get(metric)
        if value is None:
            if metric == "Action Following":
                value = action_following_default
                warnings.append(f"'{metric}' missing -> use default {action_following_default}")
            else:
                value = 0.0
                warnings.append(f"'{metric}' missing -> use default 0.0")
        final_metrics[metric] = value * scale

    warnings.append(f"score source: {source}")
    warnings.append(f"scale factor: x{scale:g} (mode={scale_mode})")
    return LoadResult(metrics=final_metrics, source=source, warnings=warnings)


def compute_dimension_scores(base_metrics: Dict[str, float]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for dim, metrics in DIMENSION_MAP.items():
        vals = [base_metrics[m] for m in metrics]
        scores[dim] = _mean(vals)
    scores["EWMScore"] = _mean([base_metrics[m] for m in BASE_METRICS])
    return scores


def make_my_model_row(
    model_name: str,
    open_source: str,
    year: int,
    base_metrics: Dict[str, float],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "Model": model_name,
        "open_source": open_source,
        "year": year,
    }
    dim_scores = compute_dimension_scores(base_metrics)
    for k, v in dim_scores.items():
        row[k] = v
    for metric, value in base_metrics.items():
        row[metric] = value
    return row


def _read_leaderboard(path: Path) -> List[Dict[str, object]]:
    rows = _read_csv_rows(path)
    parsed: List[Dict[str, object]] = []
    for r in rows:
        item: Dict[str, object] = {}
        for col in LEADERBOARD_COLUMNS:
            if col not in r:
                item[col] = ""
                continue
            if col in ("Model", "open_source"):
                item[col] = r[col]
            elif col == "year":
                v = _to_float(r[col])
                item[col] = int(v) if v is not None else ""
            else:
                v = _to_float(r[col])
                item[col] = v if v is not None else ""
        parsed.append(item)
    return parsed


def _format_num(v: object, digits: int = 2) -> str:
    if isinstance(v, (float, int)):
        return f"{float(v):.{digits}f}"
    return str(v)


def _safe_float(v: object) -> Optional[float]:
    if isinstance(v, (float, int)):
        return float(v)
    return _to_float(str(v))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            out: Dict[str, object] = {}
            for c in columns:
                v = r.get(c, "")
                if c in ("Model", "open_source", "year"):
                    out[c] = v
                else:
                    fv = _safe_float(v)
                    out[c] = "" if fv is None else f"{fv:.2f}"
            writer.writerow(out)


def write_markdown_with_max_tags(path: Path, rows: List[Dict[str, object]], columns: List[str], mark_columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    max_map: Dict[str, float] = {}
    for col in mark_columns:
        vals = [_safe_float(r.get(col, "")) for r in rows]
        nums = [v for v in vals if v is not None]
        if nums:
            max_map[col] = max(nums)

    lines: List[str] = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")

    for r in rows:
        cells: List[str] = []
        for c in columns:
            v = r.get(c, "")
            if c in ("Model", "open_source", "year"):
                cells.append(str(v))
                continue
            fv = _safe_float(v)
            if fv is None:
                cells.append("")
                continue
            text = _format_num(fv, 2)
            if c in mark_columns and c in max_map and abs(fv - max_map[c]) <= 1e-9:
                text = f"{text} (MAX)"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a new leaderboard table with six dimensions and EWMScore from aggregated CSV."
    )
    parser.add_argument("--aggregated_csv", required=True, help="Path to model aggregated CSV (contains per-video rows or AVERAGE row).")
    parser.add_argument("--leaderboard_csv", default="/data/liuwenhao/WorldArena/worldarena_leaderboard.csv", help="Existing leaderboard CSV path.")
    parser.add_argument("--my_model_name", required=True, help="Model name to display in final leaderboard table.")
    parser.add_argument("--my_open_source", default="Open-source", help="open_source field for your model row.")
    parser.add_argument("--my_year", type=int, default=2026, help="year field for your model row.")
    parser.add_argument(
        "--scale_mode",
        choices=["auto", "percent", "raw"],
        default="auto",
        help="How to scale base metrics read from aggregated CSV. auto: if in [0,1.5], multiply by 100.",
    )
    parser.add_argument("--action_following_default", type=float, default=0.0, help="Fallback for missing 'Action Following'.")
    parser.add_argument("--output_csv", default="/data/liuwenhao/WorldArena/worldarena_leaderboard_with_my_model.csv", help="Merged numeric leaderboard CSV output.")
    parser.add_argument("--output_md", default="/data/liuwenhao/WorldArena/worldarena_leaderboard_with_my_model.md", help="Markdown table with MAX tags.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    aggregated_csv = Path(args.aggregated_csv).resolve()
    leaderboard_csv = Path(args.leaderboard_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_md = Path(args.output_md).resolve()

    model_metrics = load_model_metrics_from_aggregated_csv(
        aggregated_csv=aggregated_csv,
        scale_mode=args.scale_mode,
        action_following_default=args.action_following_default,
    )
    my_row = make_my_model_row(
        model_name=args.my_model_name,
        open_source=args.my_open_source,
        year=args.my_year,
        base_metrics=model_metrics.metrics,
    )

    old_rows = _read_leaderboard(leaderboard_csv)
    old_rows = [r for r in old_rows if str(r.get("Model", "")).strip() != args.my_model_name.strip()]
    merged_rows = old_rows + [my_row]

    merged_rows.sort(key=lambda r: (_safe_float(r.get("EWMScore")) or -1e18), reverse=True)

    write_csv(output_csv, merged_rows, MARKDOWN_COLUMNS)
    write_markdown_with_max_tags(
        output_md,
        merged_rows,
        MARKDOWN_COLUMNS,
        mark_columns=ALL_SCORE_COLUMNS,
    )

    print(f"[OK] output_csv={output_csv}")
    print(f"[OK] output_md={output_md}")
    for w in model_metrics.warnings:
        print(f"[INFO] {w}")


if __name__ == "__main__":
    main()
