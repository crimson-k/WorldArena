#!/usr/bin/env python3
"""Aggregate multi-node/multi-shard evaluation outputs and build leaderboard in one step."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


# ----------------------------
# Aggregation constants
# ----------------------------

COLUMN_ORDER: List[str] = [
    "Model_Name",
    "Video_ID",
    "Action Following",
    "Aesthetic Quality",
    "Background Consistency",
    "Depth Accuracy",
    "Dynamic Degree",
    "Flow Score",
    "Image Quality",
    "Instruction Following",
    "Interaction Quality",
    "JEPA Similarity",
    "Motion Smoothness",
    "Perspectivity",
    "Photometric Consistency",
    "Semantic Alignment",
    "Subject Consistency",
    "Trajectory Accuracy",
    "PSNR",
    "SSIM",
    "MSE",
    "LPIPS",
    "FID",
    "FVD",
]

METRIC_KEY_MAP: Dict[str, str] = {
    "subject_consistency": "Subject Consistency",
    "aesthetic_quality": "Aesthetic Quality",
    "image_quality": "Image Quality",
    "psnr": "PSNR",
    "ssim": "SSIM",
    "mse": "MSE",
    "lpips": "LPIPS",
    "fid": "FID",
    "fvd": "FVD",
    "background_consistency": "Background Consistency",
    "dynamic_degree": "Dynamic Degree",
    "interaction_quality": "Interaction Quality",
    "perspectivity": "Perspectivity",
    "instruction_following": "Instruction Following",
    "semantic_alignment": "Semantic Alignment",
    "action_following": "Action Following",
    "flow_score": "Flow Score",
    "depth_accuracy": "Depth Accuracy",
    "trajectory_accuracy": "Trajectory Accuracy",
    "photometric_smoothness": "Photometric Consistency",
    "photometric_consistency": "Photometric Consistency",
    "motion_smoothness": "Motion Smoothness",
    "jepa_similarity": "JEPA Similarity",
}


# ----------------------------
# Leaderboard constants
# ----------------------------

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

# Explicit order for the 16 fine-grained metrics.
ORDERED16_COLUMNS: List[str] = [
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
] + [
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


@dataclass
class LoadResult:
    metrics: Dict[str, float]
    source: str
    warnings: List[str]


@dataclass
class MergeStats:
    leaf_dirs: int = 0
    leaf_video_rows: int = 0
    merged_videos: int = 0
    duplicate_video_rows: int = 0
    metric_conflicts: int = 0
    global_vlm_backfilled: int = 0
    global_jepa_global_score: Optional[float] = None
    global_jepa_per_video_rows: int = 0


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize_episode_token(token: str) -> str:
    lower = token.lower()
    if lower.startswith("episode"):
        rest = token[len("episode"):].lstrip("._")
        if rest:
            return f"episode{rest}"
        return "episode"
    return token


def _parse_video_id_from_path(path: str) -> str:
    norm = path.replace("\\", "/")
    name_with_ext = os.path.basename(norm)
    name, _ = os.path.splitext(name_with_ext)

    parts = norm.split("/")
    anchors = [
        "generated_dataset",
        "generated_dataset_action_following",
        "gt_dataset",
    ]
    for anchor in anchors:
        if anchor in parts:
            idx = parts.index(anchor)
            if len(parts) >= idx + 3:
                task = parts[idx + 1]
                episode_raw = parts[idx + 2]
                episode = _normalize_episode_token(episode_raw)
                if episode.lower().startswith("episode"):
                    suffix = episode[len("episode"):]
                    return f"{task}_episode{suffix}"
                return f"{task}_{episode}"
    if name and any(ch.isdigit() for ch in name):
        return name
    return name or norm


def _normalize_video_id(video_id: str) -> str:
    if not video_id:
        return video_id
    for synthetic_prefix in ("data_", "worldarena_auto_eval_"):
        if video_id.startswith(f"{synthetic_prefix}episode"):
            return video_id[len(synthetic_prefix):]
    return video_id


def _upsert(result: Dict[str, Dict[str, float]], video_id: str, metric_key: str, value: Optional[float]):
    if value is None:
        return
    column = METRIC_KEY_MAP.get(metric_key, metric_key)
    norm_video_id = _normalize_video_id(video_id)
    result.setdefault(norm_video_id, {})[column] = value


def _extract_entry_score(entry: Dict) -> Optional[float]:
    value = (
        entry.get("video_results_normalized")
        if entry.get("video_results_normalized") is not None
        else entry.get("score_normalized")
    )
    if value is None:
        value = (
            entry.get("video_results")
            if entry.get("video_results") is not None
            else entry.get("score")
        )
    if value is None:
        value = (
            entry.get("jepa_similarity_normalized")
            if entry.get("jepa_similarity_normalized") is not None
            else entry.get("jepa_similarity")
        )
    return value


def _ingest_nested_task_metric(metric_key: str, payload: Dict, result: Dict[str, Dict[str, float]]):
    for task_id, episodes in payload.items():
        if not isinstance(episodes, dict):
            continue
        for episode_id, groups in episodes.items():
            values: List[float] = []
            if isinstance(groups, dict):
                for v in groups.values():
                    fv = _to_float(v)
                    if fv is not None:
                        values.append(fv)
            else:
                fv = _to_float(groups)
                if fv is not None:
                    values.append(fv)
            if not values:
                continue
            episode_norm = _normalize_episode_token(str(episode_id))
            video_id = f"{task_id}_{episode_norm}"
            _upsert(result, video_id, metric_key, sum(values) / len(values))


def _ingest_metric_json(path: str, result: Dict[str, Dict[str, float]]):
    data = _read_json(path)
    for metric_key, payload in data.items():
        if isinstance(payload, list) and len(payload) >= 2:
            details = payload[1]
            if not isinstance(details, list):
                continue
            for entry in details:
                video_path = (
                    entry.get("video_path")
                    or entry.get("video")
                    or entry.get("video_name")
                    or entry.get("name")
                )
                if not video_path:
                    continue
                video_id = _parse_video_id_from_path(video_path)
                value = (
                    entry.get("video_results_normalized")
                    if entry.get("video_results_normalized") is not None
                    else entry.get("score_normalized")
                )
                if value is None:
                    value = entry.get("video_results") or entry.get("score")
                _upsert(result, video_id, metric_key, value)
            continue
        if metric_key in {"psnr", "ssim"} and isinstance(payload, dict):
            _ingest_nested_task_metric(metric_key, payload, result)


def _ingest_vlm_json(path: str, result: Dict[str, Dict[str, float]]):
    data = _read_json(path)
    if not isinstance(data, list):
        return
    for item in data:
        video_name = item.get("video")
        if not video_name:
            continue
        video_id = _parse_video_id_from_path(video_name)
        metrics = item.get("metrics", {})
        for metric_key, metric_val in metrics.items():
            if not isinstance(metric_val, dict):
                continue
            value = metric_val.get("score_normalized") or metric_val.get("score")
            normalized_key = metric_key.lower().replace(" ", "_")
            _upsert(result, video_id, normalized_key, value)


def _ingest_jepa_entries(
    entries: List[Dict],
    result: Dict[str, Dict[str, float]],
    allowed_video_ids: Optional[Set[str]] = None,
) -> int:
    count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        video_path = (
            entry.get("video_path")
            or entry.get("video")
            or entry.get("video_name")
            or entry.get("name")
        )
        if not video_path:
            continue
        value = _extract_entry_score(entry)
        if value is None:
            continue
        video_id = _normalize_video_id(_parse_video_id_from_path(str(video_path)))
        if allowed_video_ids is not None and video_id not in allowed_video_ids:
            continue
        _upsert(result, video_id, "jepa_similarity", value)
        count += 1
    return count


def _ingest_jepa_result(
    path: str,
    result: Dict[str, Dict[str, float]],
    allowed_video_ids: Optional[Set[str]] = None,
) -> Dict[str, Optional[float]]:
    if not os.path.exists(path):
        return {"per_video_count": 0, "global_score": None}

    data = _read_json(path)
    per_video_count = 0
    global_score: Optional[float] = None

    if isinstance(data, dict):
        if data.get("score") is not None:
            global_score = data.get("score")
        for payload in data.values():
            if not isinstance(payload, list) or len(payload) < 2:
                continue
            details = payload[1]
            if not isinstance(details, list):
                continue
            per_video_count += _ingest_jepa_entries(details, result, allowed_video_ids=allowed_video_ids)
        if isinstance(data.get("results"), list):
            per_video_count += _ingest_jepa_entries(
                data["results"], result, allowed_video_ids=allowed_video_ids
            )
    elif isinstance(data, list):
        per_video_count += _ingest_jepa_entries(data, result, allowed_video_ids=allowed_video_ids)

    return {"per_video_count": per_video_count, "global_score": global_score}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _safe_float(v: object) -> Optional[float]:
    if isinstance(v, (float, int)):
        return float(v)
    if v is None:
        return None
    return _to_float(str(v))


def _build_average_row(csv_rows: List[Dict[str, str]], model_name: str, fieldnames: List[str]) -> Dict[str, str]:
    avg_row: Dict[str, str] = {col: "" for col in fieldnames}
    avg_row["Model_Name"] = model_name
    avg_row["Video_ID"] = "AVERAGE"
    for col in fieldnames:
        if col in ("Model_Name", "Video_ID"):
            continue
        values: List[float] = []
        for row in csv_rows:
            raw = row.get(col, "")
            if raw is None or raw == "":
                continue
            fv = _to_float(raw)
            if fv is not None:
                values.append(fv)
        if values:
            avg_row[col] = f"{sum(values) / len(values):.6f}"
    return avg_row


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


def _find_eval_leaf_dirs(data_root: Path, model_name: str) -> List[Path]:
    candidates: List[Path] = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name == model_name or p.name.startswith(f"{model_name}_"):
            candidates.append(p)
    leaves = [p for p in candidates if (p / "output" / "generated_results.json").exists()]
    if not leaves:
        single = data_root / model_name
        if (single / "output" / "generated_results.json").exists():
            leaves = [single]
    return sorted(leaves)


def _collect_vlm_json_paths(eval_dir: Path) -> List[Path]:
    output_vlm = eval_dir / "output_VLM"
    if not output_vlm.exists():
        return []
    preferred = output_vlm / eval_dir.name
    paths: List[Path] = []
    if preferred.exists():
        paths.extend(sorted(preferred.glob("*.json")))
    if not paths:
        paths.extend(sorted(output_vlm.glob("*.json")))
    if not paths:
        paths.extend(sorted(output_vlm.glob("*/*.json")))
    return paths


def _collect_global_jepa_candidates(model_root: Path) -> List[Path]:
    cands: List[Path] = [
        model_root / "output_JEDi" / "generated_results.json",
        model_root / "output_JEDi" / "results.json",
    ]
    cands.extend(sorted((model_root / "output_JEDi").glob("*.json")) if (model_root / "output_JEDi").exists() else [])
    seen: Set[Path] = set()
    uniq: List[Path] = []
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _merge_metric_rows(
    merged: Dict[str, Dict[str, float]],
    incoming: Dict[str, Dict[str, float]],
) -> Tuple[int, int]:
    duplicate_video_rows = 0
    metric_conflicts = 0
    for video_id, metrics in incoming.items():
        exists = video_id in merged
        row = merged.setdefault(video_id, {})
        if exists:
            duplicate_video_rows += 1
        for metric_name, value in metrics.items():
            if metric_name not in row:
                row[metric_name] = value
                continue
            old = _to_float(row.get(metric_name))
            new = _to_float(value)
            if old is None and new is not None:
                row[metric_name] = new
                continue
            if old is not None and new is not None and abs(old - new) > 1e-9:
                metric_conflicts += 1
    return duplicate_video_rows, metric_conflicts


def aggregate_multi_node_results(
    video_quality_root: Path,
    model_name: str,
    aggregated_csv_path: Path,
    global_vlm_model_dir: Optional[str] = None,
    global_jepa_path: Optional[Path] = None,
) -> Tuple[Path, MergeStats]:
    data_root = video_quality_root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"data directory not found: {data_root}")

    model_root = data_root / model_name
    if not model_root.exists():
        raise FileNotFoundError(f"model root not found: {model_root}")

    leaf_dirs = _find_eval_leaf_dirs(data_root=data_root, model_name=model_name)
    if not leaf_dirs:
        raise RuntimeError(
            f"No evaluation leaf dirs found for model_name='{model_name}' under {data_root}. "
            "Expected folders like '<model>_shard_00' or '<model>_0_shard_00'."
        )

    stats = MergeStats(leaf_dirs=len(leaf_dirs))
    merged_result: Dict[str, Dict[str, float]] = {}

    # 1) Per leaf: ingest output/generated_results + optional output_action_following + output_VLM + local JEPA.
    for leaf in leaf_dirs:
        leaf_result: Dict[str, Dict[str, float]] = {}

        core_paths = [
            leaf / "output" / "generated_results.json",
            leaf / "output_action_following" / "generated_results.json",
        ]
        for p in core_paths:
            if p.exists():
                _ingest_metric_json(str(p), leaf_result)

        vlm_json_paths = _collect_vlm_json_paths(leaf)
        for vlm_json in vlm_json_paths:
            _ingest_vlm_json(str(vlm_json), leaf_result)

        local_jepa_candidates = [
            leaf / "output_JEDi" / "generated_results.json",
            leaf / "output_JEDi" / "results.json",
        ]
        allowed = set(leaf_result.keys()) if leaf_result else None
        for jp in local_jepa_candidates:
            if not jp.exists():
                continue
            jepa_stats = _ingest_jepa_result(str(jp), leaf_result, allowed_video_ids=allowed)
            if jepa_stats["per_video_count"]:
                break

        stats.leaf_video_rows += len(leaf_result)
        dup, conflicts = _merge_metric_rows(merged_result, leaf_result)
        stats.duplicate_video_rows += dup
        stats.metric_conflicts += conflicts

    # 2) Global VLM backfill (for workflows where VLM is not inside each shard leaf).
    global_vlm_dir = model_root / "output_VLM" / (global_vlm_model_dir or model_name)
    if global_vlm_dir.exists():
        global_vlm: Dict[str, Dict[str, float]] = {}
        for vp in sorted(global_vlm_dir.glob("*.json")):
            _ingest_vlm_json(str(vp), global_vlm)
        for video_id, metrics in global_vlm.items():
            row = merged_result.setdefault(_normalize_video_id(video_id), {})
            touched = False
            for metric_name, value in metrics.items():
                if metric_name not in row:
                    row[metric_name] = value
                    touched = True
            if touched:
                stats.global_vlm_backfilled += 1

    # 3) Global JEPA: prefer per-video injection; fallback global score.
    jepa_global_score: Optional[float] = None
    global_jepa_candidates: List[Path] = []
    if global_jepa_path is not None:
        global_jepa_candidates.append(global_jepa_path)
    else:
        global_jepa_candidates.extend(_collect_global_jepa_candidates(model_root))

    for jp in global_jepa_candidates:
        if not jp.exists():
            continue
        jepa_stats = _ingest_jepa_result(
            str(jp),
            merged_result,
            allowed_video_ids=set(merged_result.keys()) if merged_result else None,
        )
        stats.global_jepa_per_video_rows += int(jepa_stats["per_video_count"] or 0)
        if jepa_stats["global_score"] is not None and jepa_global_score is None:
            jepa_global_score = _to_float(jepa_stats["global_score"])
        if jepa_stats["per_video_count"]:
            break

    if jepa_global_score is not None:
        for _video_id, row in merged_result.items():
            if "JEPA Similarity" not in row or row["JEPA Similarity"] == "":
                row["JEPA Similarity"] = jepa_global_score
    stats.global_jepa_global_score = jepa_global_score

    stats.merged_videos = len(merged_result)

    csv_rows: List[Dict[str, str]] = []
    for video_id in sorted(merged_result.keys()):
        row: Dict[str, str] = {col: "" for col in COLUMN_ORDER}
        row["Model_Name"] = model_name
        row["Video_ID"] = video_id
        for metric_col, value in merged_result[video_id].items():
            if metric_col in COLUMN_ORDER:
                row[metric_col] = value
        csv_rows.append(row)

    if csv_rows:
        csv_rows.append(_build_average_row(csv_rows=csv_rows, model_name=model_name, fieldnames=COLUMN_ORDER))

    aggregated_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregated_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_ORDER)
        writer.writeheader()
        writer.writerows(csv_rows)

    return aggregated_csv_path, stats


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return slug.strip("_") or "my_model"


def build_leaderboard(
    aggregated_csv: Path,
    leaderboard_csv: Path,
    my_model_name: str,
    my_open_source: str,
    my_year: int,
    scale_mode: str,
    action_following_default: float,
    output_csv: Path,
    output_md: Path,
    output_csv_ordered16: Path,
    output_md_ordered16: Path,
) -> LoadResult:
    model_metrics = load_model_metrics_from_aggregated_csv(
        aggregated_csv=aggregated_csv,
        scale_mode=scale_mode,
        action_following_default=action_following_default,
    )
    my_row = make_my_model_row(
        model_name=my_model_name,
        open_source=my_open_source,
        year=my_year,
        base_metrics=model_metrics.metrics,
    )

    old_rows = _read_leaderboard(leaderboard_csv)
    old_rows = [r for r in old_rows if str(r.get("Model", "")).strip() != my_model_name.strip()]
    merged_rows = old_rows + [my_row]
    merged_rows.sort(key=lambda r: (_safe_float(r.get("EWMScore")) or -1e18), reverse=True)

    write_csv(output_csv, merged_rows, MARKDOWN_COLUMNS)
    write_markdown_with_max_tags(
        output_md,
        merged_rows,
        MARKDOWN_COLUMNS,
        mark_columns=ALL_SCORE_COLUMNS,
    )
    write_csv(output_csv_ordered16, merged_rows, ORDERED16_COLUMNS)
    write_markdown_with_max_tags(
        output_md_ordered16,
        merged_rows,
        ORDERED16_COLUMNS,
        mark_columns=ALL_SCORE_COLUMNS,
    )
    return model_metrics


def parse_args() -> argparse.Namespace:
    default_video_quality_root = Path(__file__).resolve().parent.parent
    default_project_root = default_video_quality_root.parent

    parser = argparse.ArgumentParser(
        description=(
            "One-shot script: aggregate multi-node/multi-shard evaluation outputs and "
            "build EWM leaderboard."
        )
    )
    parser.add_argument("--video_quality_root", default=str(default_video_quality_root), help="Path to video_quality root.")
    parser.add_argument("--model_name", required=True, help="Base model folder name under data/.")

    parser.add_argument(
        "--aggregated_csv",
        default=None,
        help="Output path for aggregated per-video CSV. Default: data/<model>/csv_results/<model>_aggregated_results_multi_node.csv",
    )
    parser.add_argument(
        "--global_vlm_model_dir",
        default=None,
        help="Optional global VLM subfolder name under data/<model>/output_VLM/.",
    )
    parser.add_argument(
        "--global_jepa_path",
        default=None,
        help="Optional explicit path to global JEPA json file.",
    )

    parser.add_argument(
        "--leaderboard_csv",
        default=str(default_project_root / "worldarena_leaderboard.csv"),
        help="Existing leaderboard CSV path.",
    )
    parser.add_argument("--my_model_name", default=None, help="Model name shown in final leaderboard. Default: --model_name")
    parser.add_argument("--my_open_source", default="Open-source", help="open_source column value.")
    parser.add_argument("--my_year", type=int, default=2026, help="year column value.")
    parser.add_argument(
        "--scale_mode",
        choices=["auto", "percent", "raw"],
        default="auto",
        help="Score scale mode for aggregated csv when building leaderboard.",
    )
    parser.add_argument("--action_following_default", type=float, default=0.0, help="Fallback for missing Action Following.")

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Output leaderboard csv. Default: <project_root>/worldarena_leaderboard_with_<my_model_name>.csv",
    )
    parser.add_argument(
        "--output_md",
        default=None,
        help="Output leaderboard markdown. Default: <project_root>/worldarena_leaderboard_with_<my_model_name>.md",
    )
    parser.add_argument(
        "--output_csv_ordered16",
        default=None,
        help="Second leaderboard csv with fixed ordered 16 fine-grained metrics. Default: <project_root>/worldarena_leaderboard_with_<my_model_name>_ordered16.csv",
    )
    parser.add_argument(
        "--output_md_ordered16",
        default=None,
        help="Second leaderboard markdown with fixed ordered 16 fine-grained metrics. Default: <project_root>/worldarena_leaderboard_with_<my_model_name>_ordered16.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    video_quality_root = Path(args.video_quality_root).resolve()
    model_name = args.model_name.strip()
    if not model_name:
        raise ValueError("--model_name must be non-empty")

    data_root = video_quality_root / "data"
    model_root = data_root / model_name
    my_model_name = (args.my_model_name or model_name).strip()
    project_root = video_quality_root.parent

    aggregated_csv = (
        Path(args.aggregated_csv).resolve()
        if args.aggregated_csv
        else (model_root / "csv_results" / f"{model_name}_aggregated_results_multi_node.csv").resolve()
    )

    default_slug = _slugify(my_model_name)
    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv
        else (project_root / f"worldarena_leaderboard_with_{default_slug}.csv").resolve()
    )
    output_md = (
        Path(args.output_md).resolve()
        if args.output_md
        else (project_root / f"worldarena_leaderboard_with_{default_slug}.md").resolve()
    )
    output_csv_ordered16 = (
        Path(args.output_csv_ordered16).resolve()
        if args.output_csv_ordered16
        else (project_root / f"worldarena_leaderboard_with_{default_slug}_ordered16.csv").resolve()
    )
    output_md_ordered16 = (
        Path(args.output_md_ordered16).resolve()
        if args.output_md_ordered16
        else (project_root / f"worldarena_leaderboard_with_{default_slug}_ordered16.md").resolve()
    )

    leaderboard_csv = Path(args.leaderboard_csv).resolve()
    global_jepa_path = Path(args.global_jepa_path).resolve() if args.global_jepa_path else None

    aggregated_csv, merge_stats = aggregate_multi_node_results(
        video_quality_root=video_quality_root,
        model_name=model_name,
        aggregated_csv_path=aggregated_csv,
        global_vlm_model_dir=args.global_vlm_model_dir,
        global_jepa_path=global_jepa_path,
    )

    model_metrics = build_leaderboard(
        aggregated_csv=aggregated_csv,
        leaderboard_csv=leaderboard_csv,
        my_model_name=my_model_name,
        my_open_source=args.my_open_source,
        my_year=args.my_year,
        scale_mode=args.scale_mode,
        action_following_default=args.action_following_default,
        output_csv=output_csv,
        output_md=output_md,
        output_csv_ordered16=output_csv_ordered16,
        output_md_ordered16=output_md_ordered16,
    )

    src_videos = model_root / "videos"
    src_video_count = len(list(src_videos.glob("*.mp4"))) if src_videos.exists() else None

    print(f"[DISCOVER] leaf_dirs={merge_stats.leaf_dirs}")
    print(f"[AGG] leaf_video_rows={merge_stats.leaf_video_rows}")
    print(f"[AGG] merged_videos={merge_stats.merged_videos}")
    print(f"[AGG] duplicate_video_rows={merge_stats.duplicate_video_rows}")
    print(f"[AGG] metric_conflicts={merge_stats.metric_conflicts}")
    print(f"[AGG] global_vlm_backfilled_rows={merge_stats.global_vlm_backfilled}")
    print(f"[AGG] global_jepa_per_video_rows={merge_stats.global_jepa_per_video_rows}")
    print(f"[AGG] global_jepa_score={merge_stats.global_jepa_global_score}")
    if src_video_count is not None:
        print(f"[CHECK] source_videos={src_video_count}")
        print(f"[CHECK] coverage={merge_stats.merged_videos}/{src_video_count}")

    print(f"[OK] aggregated_csv={aggregated_csv}")
    print(f"[OK] output_csv={output_csv}")
    print(f"[OK] output_md={output_md}")
    print(f"[OK] output_csv_ordered16={output_csv_ordered16}")
    print(f"[OK] output_md_ordered16={output_md_ordered16}")
    for w in model_metrics.warnings:
        print(f"[INFO] {w}")


if __name__ == "__main__":
    main()
