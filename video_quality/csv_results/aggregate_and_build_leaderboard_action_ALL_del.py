#!/usr/bin/env python3
"""Aggregate multi-node/multi-shard evaluation outputs and build leaderboard in one step."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from xml.sax.saxutils import escape as xml_escape


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

EPISODE_INDEX_PATTERN = re.compile(r"episode[_-]?0*(\d+)", re.IGNORECASE)

TASK_JSONL_CANDIDATES: List[Path] = [
    Path("/ssdfs/datahome/usersht/dev/lwh/episodes_task_label.jsonl"),
    Path("/ssdfs/datahome/usersht/dev/lwh/episodes_val_cam_high_480p.jsonl"),
]


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

EWM_METRIC_COLUMNS: List[str] = COLUMN_ORDER[2:18]
TASK_SCORE_COLUMNS: List[str] = ["Single_Task_EWMScore", "EWMScore"]
TASK_LENGTH_COLUMN = "length"

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


@dataclass
class EpisodeLabel:
    task: str
    length: int


@dataclass
class TaskLabelStats:
    jsonl_path: Path
    input_rows: int
    merged_episode_rows: int
    skipped_non_episode_rows: int
    episode_min: int
    episode_max: int
    task_score_csv: Optional[Path] = None
    task_score_xlsx: Optional[Path] = None
    task_score_rows: int = 0


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


def _extract_episode_index(video_id: str) -> Optional[int]:
    match = EPISODE_INDEX_PATTERN.search(str(video_id))
    if match is None:
        return None
    return int(match.group(1))


def _load_episode_tasks(jsonl_path: Path) -> Dict[int, EpisodeLabel]:
    tasks: Dict[int, EpisodeLabel] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "episode_index" not in item:
                raise KeyError(f"{jsonl_path}:{line_no} is missing 'episode_index'")
            if "task" not in item:
                raise KeyError(f"{jsonl_path}:{line_no} is missing 'task'")
            if "length" not in item:
                raise KeyError(f"{jsonl_path}:{line_no} is missing 'length'")

            episode_index = int(item["episode_index"])
            if episode_index in tasks:
                raise ValueError(f"Duplicate episode_index in {jsonl_path}: {episode_index}")
            tasks[episode_index] = EpisodeLabel(
                task=str(item["task"]),
                length=int(item["length"]),
            )
    return tasks


def _choose_task_jsonl(
    episode_indices: Set[int],
    explicit_jsonl: Optional[Path],
) -> Optional[Tuple[Path, Dict[int, EpisodeLabel]]]:
    """
    Choose a task-label JSONL for possibly incomplete / sparse episode sets.

    Old behavior:
      require set(tasks) == episode_indices

    New behavior:
      require episode_indices ⊆ set(tasks)

    This supports:
      - only episode20..episode39
      - missing middle samples, e.g. episode1..5 and episode7..8
      - arbitrary non-contiguous subsets
      - not starting from 0 or 1
    """
    candidates = [explicit_jsonl] if explicit_jsonl is not None else TASK_JSONL_CANDIDATES
    existing_candidates: List[Path] = []

    best_partial: Optional[Tuple[Path, Dict[int, EpisodeLabel], int]] = None

    for candidate in candidates:
        if candidate is None:
            continue

        jsonl_path = candidate.expanduser().resolve()
        if not jsonl_path.exists():
            if explicit_jsonl is not None:
                raise FileNotFoundError(f"Task JSONL not found: {jsonl_path}")
            continue

        existing_candidates.append(jsonl_path)
        tasks = _load_episode_tasks(jsonl_path)
        task_indices = set(tasks)

        # 核心修改：只要求聚合出来的 episode 都能在 task jsonl 中找到。
        # JSONL 允许包含更多 episode。
        if episode_indices.issubset(task_indices):
            return jsonl_path, tasks

        overlap_count = len(episode_indices & task_indices)
        if best_partial is None or overlap_count > best_partial[2]:
            best_partial = (jsonl_path, tasks, overlap_count)

        if explicit_jsonl is not None:
            missing_labels = sorted(episode_indices - task_indices)
            raise ValueError(
                "Task JSONL does not cover all aggregated CSV episodes. "
                f"missing task labels count={len(missing_labels)}, "
                f"examples={missing_labels[:20]}"
            )

    if not existing_candidates:
        return None

    expected_min = min(episode_indices) if episode_indices else None
    expected_max = max(episode_indices) if episode_indices else None

    if best_partial is not None:
        best_path, _best_tasks, overlap_count = best_partial
        missing_count = len(episode_indices) - overlap_count
        raise ValueError(
            "No task JSONL fully covers aggregated CSV episode_index set. "
            f"episode_range={expected_min}..{expected_max}, "
            f"csv_episode_count={len(episode_indices)}, "
            f"best_jsonl={best_path}, "
            f"matched={overlap_count}, missing={missing_count}. "
            "If this run uses a new task-label file, pass it with --task_jsonl, "
            "or use --skip_task_labels."
        )

    raise ValueError(
        "No task JSONL matches aggregated CSV episode_index set "
        f"({expected_min}..{expected_max}, count={len(episode_indices)}). "
        f"Checked: {', '.join(str(p) for p in existing_candidates)}"
    )


def _choose_video_id(rows: List[Dict[str, str]], episode_index: int) -> str:
    canonical = f"episode{episode_index}"
    for row in rows:
        if row.get("Video_ID") == canonical:
            return canonical
    return rows[0].get("Video_ID", canonical)


def _merge_episode_rows(
    rows: List[Dict[str, str]],
    fieldnames: List[str],
    episode_index: int,
) -> Dict[str, str]:
    merged = {fieldname: "" for fieldname in fieldnames}
    conflicts: List[str] = []

    for row in rows:
        for fieldname in fieldnames:
            value = row.get(fieldname, "")
            if value == "":
                continue
            if merged[fieldname] in ("", value):
                merged[fieldname] = value
            elif fieldname not in ("Video_ID", "task"):
                conflicts.append(fieldname)

    if conflicts:
        conflict_text = ", ".join(sorted(set(conflicts)))
        raise ValueError(
            f"Conflicting non-empty values for episode {episode_index}: {conflict_text}"
        )

    merged["Video_ID"] = _choose_video_id(rows, episode_index)
    return merged


def _score_value(row: Dict[str, str], metric: str) -> float:
    value = _to_float(row.get(metric, ""))
    return value if value is not None else 0.0


def _format_score(value: float) -> str:
    return f"{value:.6f}"


def _mean_score(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _default_task_score_csv_path(aggregated_csv: Path) -> Path:
    return aggregated_csv.with_name(f"{aggregated_csv.stem}_task_ewmscore{aggregated_csv.suffix}")


def _xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_is_number_text(value: str) -> bool:
    return re.fullmatch(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", value) is not None


def _xlsx_cell_xml(ref: str, value: object, style_id: int) -> str:
    style_attr = f' s="{style_id}"' if style_id else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style_attr}/>'

    text = str(value)
    if _xlsx_is_number_text(text):
        return f'<c r="{ref}"{style_attr}><v>{text}</v></c>'

    escaped = xml_escape(text)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return f'<c r="{ref}" t="inlineStr"{style_attr}><is><t{preserve}>{escaped}</t></is></c>'


def _write_colored_task_xlsx(
    output_xlsx: Path,
    rows: List[Dict[str, str]],
    fieldnames: List[str],
) -> Path:
    dark_fill = "08084D"
    max_row = len(rows) + 1
    max_col = len(fieldnames)
    last_cell = f"{_xlsx_col_name(max_col)}{max_row}"

    task_order: List[str] = []
    task_to_style_index: Dict[str, int] = {}
    for row in rows:
        task = row.get("task", "")
        if task not in task_to_style_index:
            task_to_style_index[task] = len(task_order)
            task_order.append(task)

    def row_style(row: Dict[str, str]) -> int:
        task_index = task_to_style_index.get(row.get("task", ""), 0)
        is_colored_task = task_index % 2 == 0
        is_summary = row.get("Video_ID") == "TASK_AVERAGE"
        if is_colored_task and is_summary:
            return 4
        if is_colored_task:
            return 2
        if is_summary:
            return 3
        return 0

    header_cells = [
        _xlsx_cell_xml(f"{_xlsx_col_name(col_idx)}1", fieldname, 1)
        for col_idx, fieldname in enumerate(fieldnames, start=1)
    ]
    row_xml: List[str] = [f'<row r="1">{"".join(header_cells)}</row>']

    for row_idx, row in enumerate(rows, start=2):
        style_id = row_style(row)
        cells = [
            _xlsx_cell_xml(f"{_xlsx_col_name(col_idx)}{row_idx}", row.get(fieldname, ""), style_id)
            for col_idx, fieldname in enumerate(fieldnames, start=1)
        ]
        row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    col_xml = []
    for col_idx, fieldname in enumerate(fieldnames, start=1):
        values = [str(row.get(fieldname, "")) for row in rows[:200]]
        width = min(max(len(fieldname), *(len(value) for value in values), 8) + 2, 38)
        col_xml.append(f'<col min="{col_idx}" max="{col_idx}" width="{width}" customWidth="1"/>')

    worksheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <cols>{"".join(col_xml)}</cols>
  <sheetData>{"".join(row_xml)}</sheetData>
  <autoFilter ref="A1:{last_cell}"/>
</worksheet>'''

    styles_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="4">
    <font><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color theme="1"/><name val="Calibri"/><family val="2"/></font>
    <font><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
    <font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/><family val="2"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF{dark_fill}"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    <xf numFmtId="0" fontId="2" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
    <xf numFmtId="0" fontId="3" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    workbook_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Task EWMScore" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    workbook_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    root_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    content_types_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''

    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)

    return output_xlsx


def write_task_ewmscore_csv(
    output_csv: Path,
    sample_rows: List[Dict[str, str]],
    fieldnames: List[str],
) -> Tuple[int, Path]:
    inserted_columns = TASK_SCORE_COLUMNS + [TASK_LENGTH_COLUMN]
    output_fieldnames = [fieldname for fieldname in fieldnames if fieldname not in inserted_columns]
    task_pos = output_fieldnames.index("task")
    for offset, output_col in enumerate(inserted_columns, start=1):
        output_fieldnames.insert(task_pos + offset, output_col)

    metric_columns = [metric for metric in EWM_METRIC_COLUMNS if metric in fieldnames]
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in sample_rows:
        grouped.setdefault(row.get("task", ""), []).append(row)

    output_rows: List[Dict[str, str]] = []
    for task in sorted(grouped, key=lambda value: value.casefold()):
        task_rows = grouped[task]
        metric_averages = {
            metric: _mean_score([_score_value(row, metric) for row in task_rows])
            for metric in metric_columns
        }
        task_score = _mean_score([metric_averages[metric] for metric in metric_columns])
        task_length = _mean_score([_score_value(row, TASK_LENGTH_COLUMN) for row in task_rows])

        for row in task_rows:
            sample_row = dict(row)
            sample_score = _mean_score([_score_value(sample_row, metric) for metric in metric_columns])
            sample_row["Single_Task_EWMScore"] = _format_score(task_score)
            sample_row["EWMScore"] = _format_score(sample_score)
            output_rows.append(sample_row)

        avg_row = {fieldname: "" for fieldname in output_fieldnames}
        avg_row["Model_Name"] = task_rows[0].get("Model_Name", "")
        avg_row["Video_ID"] = "TASK_AVERAGE"
        avg_row["task"] = task
        avg_row["Single_Task_EWMScore"] = _format_score(task_score)
        avg_row["EWMScore"] = _format_score(task_score)
        avg_row[TASK_LENGTH_COLUMN] = _format_score(task_length)
        for metric in metric_columns:
            avg_row[metric] = _format_score(metric_averages[metric])
        output_rows.append(avg_row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    output_xlsx = _write_colored_task_xlsx(
        output_xlsx=output_csv.with_suffix(".xlsx"),
        rows=output_rows,
        fieldnames=output_fieldnames,
    )

    return len(output_rows), output_xlsx



def add_task_labels_to_aggregated_csv(
    aggregated_csv: Path,
    explicit_jsonl: Optional[Path] = None,
    task_score_csv: Optional[Path] = None,
) -> Optional[TaskLabelStats]:
    with aggregated_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {aggregated_csv}")
        if "Video_ID" not in reader.fieldnames:
            raise KeyError(f"CSV file is missing required column 'Video_ID': {aggregated_csv}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    grouped_rows: Dict[int, List[Dict[str, str]]] = {}
    skipped_non_episode_rows = 0

    for row in rows:
        episode_index = _extract_episode_index(row.get("Video_ID", ""))
        if episode_index is None:
            skipped_non_episode_rows += 1
            continue
        grouped_rows.setdefault(episode_index, []).append(row)

    if not grouped_rows:
        return None

    episode_indices = set(grouped_rows)
    selected = _choose_task_jsonl(episode_indices, explicit_jsonl)
    if selected is None:
        return None

    jsonl_path, labels = selected

    output_fieldnames = [
        fieldname for fieldname in fieldnames if fieldname not in ("task", TASK_LENGTH_COLUMN)
    ]

    video_id_pos = output_fieldnames.index("Video_ID")
    output_fieldnames.insert(video_id_pos + 1, "task")

    merged_rows: List[Tuple[str, int, Dict[str, str], int]] = []
    missing_label_indices: List[int] = []

    for episode_index in sorted(grouped_rows):
        label = labels.get(episode_index)
        if label is None:
            # 理论上 _choose_task_jsonl 已经保证 subset；
            # 这里保留兜底，避免残缺样本直接崩。
            missing_label_indices.append(episode_index)
            label = EpisodeLabel(task="UNKNOWN_TASK", length=0)

        merged = _merge_episode_rows(grouped_rows[episode_index], fieldnames, episode_index)
        merged["task"] = label.task
        merged_rows.append((label.task, episode_index, merged, label.length))

    if missing_label_indices:
        print(
            "[TASK][WARN] missing task labels for "
            f"{len(missing_label_indices)} episodes, examples={missing_label_indices[:20]}. "
            "Use task='UNKNOWN_TASK', length=0."
        )

    merged_rows.sort(key=lambda item: (item[0].casefold(), item[1]))

    sorted_rows = [row for _, _, row, _ in merged_rows]

    task_score_sample_rows = []
    for _, _, row, length in merged_rows:
        task_score_row = dict(row)
        task_score_row[TASK_LENGTH_COLUMN] = str(length)
        task_score_sample_rows.append(task_score_row)

    output_rows = list(sorted_rows)

    if sorted_rows:
        model_name = sorted_rows[0].get("Model_Name", "")
        output_rows.append(
            _build_average_row(
                csv_rows=sorted_rows,
                model_name=model_name,
                fieldnames=output_fieldnames,
            )
        )

    with aggregated_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    task_score_csv = task_score_csv or _default_task_score_csv_path(aggregated_csv)

    task_score_rows, task_score_xlsx = write_task_ewmscore_csv(
        output_csv=task_score_csv,
        sample_rows=task_score_sample_rows,
        fieldnames=output_fieldnames,
    )

    sorted_indices = sorted(episode_indices)

    return TaskLabelStats(
        jsonl_path=jsonl_path,
        input_rows=len(rows),
        merged_episode_rows=len(sorted_rows),
        skipped_non_episode_rows=skipped_non_episode_rows,
        episode_min=sorted_indices[0],
        episode_max=sorted_indices[-1],
        task_score_csv=task_score_csv,
        task_score_xlsx=task_score_xlsx,
        task_score_rows=task_score_rows,
    )


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
        "--task_jsonl",
        default=None,
        help=(
            "Optional explicit JSONL path containing episode_index and task. "
            "If omitted, the script auto-selects the 1000-sample or VAL500 JSONL path."
        ),
    )
    parser.add_argument(
        "--skip_task_labels",
        action="store_true",
        help="Skip merging split per-episode rows and adding the task column to aggregated_csv.",
    )
    parser.add_argument(
        "--task_score_csv",
        default=None,
        help=(
            "Optional output path for the per-task EWMScore CSV. "
            "Default: <aggregated_csv_stem>_task_ewmscore.csv in the same directory."
        ),
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
    task_jsonl_path = Path(args.task_jsonl).resolve() if args.task_jsonl else None
    task_score_csv = Path(args.task_score_csv).resolve() if args.task_score_csv else None

    aggregated_csv, merge_stats = aggregate_multi_node_results(
        video_quality_root=video_quality_root,
        model_name=model_name,
        aggregated_csv_path=aggregated_csv,
        global_vlm_model_dir=args.global_vlm_model_dir,
        global_jepa_path=global_jepa_path,
    )

    task_label_stats: Optional[TaskLabelStats] = None
    if not args.skip_task_labels:
        task_label_stats = add_task_labels_to_aggregated_csv(
            aggregated_csv=aggregated_csv,
            explicit_jsonl=task_jsonl_path,
            task_score_csv=task_score_csv,
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
    if args.skip_task_labels:
        print("[TASK] skipped by --skip_task_labels")
    elif task_label_stats is None:
        print(
            "[TASK] skipped: no matching task JSONL found. "
            f"Auto candidates: {', '.join(str(p) for p in TASK_JSONL_CANDIDATES)}"
        )
    else:
        print(f"[TASK] jsonl={task_label_stats.jsonl_path}")
        print(f"[TASK] input_rows={task_label_stats.input_rows}")
        print(f"[TASK] merged_episode_rows={task_label_stats.merged_episode_rows}")
        print(
            "[TASK] episode_index_range="
            f"{task_label_stats.episode_min}..{task_label_stats.episode_max}"
        )
        print(f"[TASK] skipped_non_episode_rows={task_label_stats.skipped_non_episode_rows}")
        print("[TASK] average_row=appended")
        print(f"[TASK] task_score_csv={task_label_stats.task_score_csv}")
        print(f"[TASK] task_score_xlsx={task_label_stats.task_score_xlsx}")
        print(f"[TASK] task_score_rows={task_label_stats.task_score_rows}")
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
