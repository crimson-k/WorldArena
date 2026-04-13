import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

# Column order for the final CSV
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

# Map metric keys from JSON to CSV column names
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
    # Keep both names for compatibility:
    # - current evaluation output key: photometric_smoothness
    # - legacy/alternate key: photometric_consistency
    "photometric_smoothness": "Photometric Consistency",
    "photometric_consistency": "Photometric Consistency",
    "motion_smoothness": "Motion Smoothness",
    "jepa_similarity": "JEPA Similarity",
}


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_video_id_from_path(path: str) -> str:
    # Normalize separators and strip extension when present
    norm = path.replace("\\", "/")
    name_with_ext = os.path.basename(norm)
    name, _ext = os.path.splitext(name_with_ext)

    def _normalize_episode_token(token: str) -> str:
        lower = token.lower()
        if lower.startswith("episode"):
            rest = token[len("episode"):].lstrip("._")
            if rest:
                return f"episode{rest}"
            return "episode"
        return token

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
                # Avoid duplicating "episode" prefix
                if episode.lower().startswith("episode"):
                    suffix = episode[len("episode"):]
                    video_id = f"{task}_episode{suffix}"
                else:
                    video_id = f"{task}_{episode}"
                return video_id
    # If path already encodes id (e.g., adjust_bottle_episode40.mp4)
    if name and any(char.isdigit() for char in name):
        return name
    # Fallback to base name
    return name or norm


def _normalize_video_id(video_id: str) -> str:
    """Normalize id variants so rows from different evaluators can merge.

    Example:
      data_episode0_s000000 -> episode0_s000000
      worldarena_auto_eval_episode0_s000000 -> episode0_s000000
    """
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


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_episode_token(token: str) -> str:
    lower = token.lower()
    if lower.startswith("episode"):
        rest = token[len("episode") :].lstrip("._")
        if rest:
            return f"episode{rest}"
        return "episode"
    return token


def _ingest_nested_task_metric(metric_key: str, payload: Dict, result: Dict[str, Dict[str, float]]):
    """Ingest nested metric dicts like:
    {task: {episode: {gid: score}}} or {task: {episode: score}}
    """
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
    # Expected format: {metric_name: [overall, [{"video_path":..., "video_results_normalized":...}, ...]], ...}
    for metric_key, payload in data.items():
        if isinstance(payload, list) and len(payload) >= 2:
            details = payload[1]
            if not isinstance(details, list):
                continue
            for entry in details:
                video_path = entry.get("video_path") or entry.get("video") or entry.get("video_name") or entry.get("name")
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

        # psnr/ssim are often stored as nested dict in generated_results.json
        if metric_key in {"psnr", "ssim"} and isinstance(payload, dict):
            _ingest_nested_task_metric(metric_key, payload, result)


def _ingest_vlm_json(path: str, result: Dict[str, Dict[str, float]]):
    # Expected format: list of {"video": "name.mp4", "metrics": {<Metric>: {"score_normalized": ...}}}
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
    """Load JEPA result file.

    Returns:
        {
          "per_video_count": int,  # number of per-video JEPA rows ingested
          "global_score": Optional[float],  # global score when available
        }
    """
    if not os.path.exists(path):
        return {"per_video_count": 0, "global_score": None}

    data = _read_json(path)
    per_video_count = 0
    global_score: Optional[float] = None

    if isinstance(data, dict):
        if data.get("score") is not None:
            global_score = data.get("score")

        # Common metric-json style: {"jepa_similarity": [overall, [{...}, ...]]}
        for payload in data.values():
            if not isinstance(payload, list) or len(payload) < 2:
                continue
            details = payload[1]
            if not isinstance(details, list):
                continue
            per_video_count += _ingest_jepa_entries(details, result, allowed_video_ids=allowed_video_ids)

        # Alternate style: {"results": [{...}, ...]}
        if isinstance(data.get("results"), list):
            per_video_count += _ingest_jepa_entries(
                data["results"], result, allowed_video_ids=allowed_video_ids
            )

    elif isinstance(data, list):
        per_video_count += _ingest_jepa_entries(data, result, allowed_video_ids=allowed_video_ids)

    return {"per_video_count": per_video_count, "global_score": global_score}


def aggregate_results(
    base_dir: str,
    model_name: str,
    csv_name: str = "aggregated_results.csv",
    vlm_model_dir: Optional[str] = None,
    jepa_result_path: Optional[str] = None,
    include_average_row: bool = True,
) -> str:
    """
    Aggregate metric outputs into a single CSV.

    Args:
        base_dir: Path to video_quality directory.
        model_name: Model name to populate the CSV Model_Name column.
        csv_name: Output CSV file name.
        vlm_model_dir: Optional folder name under output_VLM (defaults to model_name).
        jepa_result_path: Optional path to JEPA result JSON.
            Supports per-video `output_JEDi/generated_results.json` and global `output_JEDi/results.json`.

    Returns:
        Path to the written CSV file.
    """

    result: Dict[str, Dict[str, float]] = {}

    # Core metric JSONs
    core_metric_files = [
        os.path.join(base_dir, "output", "generated_results.json"),
        os.path.join(base_dir, "output_action_following", "generated_results.json"),
    ]
    for path in core_metric_files:
        if os.path.exists(path):
            _ingest_metric_json(path, result)

    # VLM metrics
    vlm_dir = os.path.join(base_dir, "output_VLM", vlm_model_dir or model_name)
    if os.path.isdir(vlm_dir):
        for json_path in glob.glob(os.path.join(vlm_dir, "*.json")):
            _ingest_vlm_json(json_path, result)

    # JEPA metrics: prefer per-video records, fallback to global score.
    jepa_score = None
    allowed_jepa_video_ids = set(result.keys()) if result else None

    if jepa_result_path:
        ingest_stats = _ingest_jepa_result(
            jepa_result_path,
            result,
            allowed_video_ids=allowed_jepa_video_ids,
        )
        if ingest_stats["global_score"] is not None:
            jepa_score = ingest_stats["global_score"]
    else:
        candidates = [
            os.path.join(base_dir, "output_JEDi", "generated_results.json"),
            os.path.join(base_dir, "output_JEDi", "results.json"),
        ] + glob.glob(os.path.join(base_dir, "output_JEDi", "*.json"))
        seen_candidates = set()
        for cand in candidates:
            if cand in seen_candidates:
                continue
            seen_candidates.add(cand)
            ingest_stats = _ingest_jepa_result(
                cand,
                result,
                allowed_video_ids=allowed_jepa_video_ids,
            )
            if ingest_stats["global_score"] is not None and jepa_score is None:
                jepa_score = ingest_stats["global_score"]
            if ingest_stats["per_video_count"] > 0:
                break

    # Prepare rows
    csv_rows: List[Dict[str, str]] = []
    for video_id in sorted(result.keys()):
        row: Dict[str, str] = {col: "" for col in COLUMN_ORDER}
        row["Model_Name"] = model_name
        row["Video_ID"] = video_id
        for metric_col, value in result[video_id].items():
            if metric_col in COLUMN_ORDER:
                row[metric_col] = value
        if jepa_score is not None and row["JEPA Similarity"] == "":
            row["JEPA Similarity"] = jepa_score
        csv_rows.append(row)

    if include_average_row and csv_rows:
        csv_rows.append(_build_average_row(csv_rows=csv_rows, model_name=model_name, fieldnames=COLUMN_ORDER))

    # Ensure output directory exists
    csv_dir = os.path.join(base_dir, "csv_results")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, csv_name)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMN_ORDER)
        writer.writeheader()
        writer.writerows(csv_rows)

    return csv_path


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
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if values:
            avg_row[col] = f"{sum(values) / len(values):.6f}"
    return avg_row


def _default_width(num_shards: int) -> int:
    if num_shards <= 1:
        return 2
    return max(2, len(str(num_shards - 1)))


def _parse_shard_ids(num_shards: Optional[int], shard_ids: Optional[str]) -> List[str]:
    if shard_ids:
        ids = [s.strip() for s in shard_ids.split(",") if s.strip()]
        if not ids:
            raise ValueError("--shard_ids is set but empty after parsing.")
        return ids

    if num_shards is None:
        raise ValueError("Sharded mode requires --num_shards or --shard_ids.")
    if num_shards < 1:
        raise ValueError("--num_shards must be >= 1.")
    width = _default_width(num_shards)
    return [f"{idx:0{width}d}" for idx in range(num_shards)]


def aggregate_sharded_results(
    video_quality_root: str,
    model_name: str,
    csv_name: Optional[str] = None,
    num_shards: Optional[int] = None,
    shard_ids: Optional[str] = None,
    global_vlm_model_dir: Optional[str] = None,
    jepa_result_path: Optional[str] = None,
) -> str:
    root = Path(video_quality_root).resolve()
    data_root = root / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"data directory not found: {data_root}")

    suffixes = _parse_shard_ids(num_shards=num_shards, shard_ids=shard_ids)

    default_jepa_candidates = [
        data_root / model_name / "output_JEDi" / "generated_results.json",
        data_root / model_name / "output_JEDi" / "results.json",
    ]
    if jepa_result_path:
        merged_jepa_path = jepa_result_path
    else:
        merged_jepa_path = None
        for p in default_jepa_candidates:
            if p.exists():
                merged_jepa_path = str(p)
                break

    shard_csv_paths: List[Path] = []
    for suffix in suffixes:
        shard_model_name = f"{model_name}_shard_{suffix}"
        shard_base_dir = data_root / shard_model_name
        if not shard_base_dir.exists():
            raise FileNotFoundError(f"Shard base_dir not found: {shard_base_dir}")

        shard_csv_name = f"{shard_model_name}_aggregated_results.csv"
        shard_csv_path = Path(
            aggregate_results(
                base_dir=str(shard_base_dir),
                model_name=shard_model_name,
                csv_name=shard_csv_name,
                vlm_model_dir=shard_model_name,
                jepa_result_path=merged_jepa_path,
                include_average_row=False,
            )
        )
        shard_csv_paths.append(shard_csv_path)
        print(f"[per-shard] {shard_model_name} -> {shard_csv_path}")

    if not shard_csv_paths:
        raise RuntimeError("No shard CSV files generated.")

    merged_rows: List[Dict[str, str]] = []
    seen_video_ids = set()
    duplicate_count = 0
    header: Optional[List[str]] = None

    for shard_csv in shard_csv_paths:
        with shard_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if header is None:
                header = reader.fieldnames
            if not header:
                raise RuntimeError(f"Empty CSV header in {shard_csv}")

            for row in reader:
                video_id = (row.get("Video_ID") or "").strip()
                if not video_id:
                    continue
                if video_id in seen_video_ids:
                    duplicate_count += 1
                    continue
                seen_video_ids.add(video_id)
                row["Model_Name"] = model_name
                merged_rows.append(row)

    if not header:
        raise RuntimeError("Failed to determine CSV header while merging shard CSVs.")

    # Inject global VLM scores when shard folders do not contain output_VLM.
    global_vlm_dir = data_root / model_name / "output_VLM" / (global_vlm_model_dir or model_name)
    vlm_jsons = sorted(global_vlm_dir.glob("*.json")) if global_vlm_dir.exists() else []
    if vlm_jsons:
        vlm_result: Dict[str, Dict[str, float]] = {}
        for vlm_json in vlm_jsons:
            _ingest_vlm_json(str(vlm_json), vlm_result)
        row_map = {(_normalize_video_id((r.get("Video_ID") or "").strip())): r for r in merged_rows}
        updated = 0
        for video_id, metrics in vlm_result.items():
            row = row_map.get(_normalize_video_id(video_id))
            if not row:
                continue
            touched = False
            for metric_col, value in metrics.items():
                if metric_col in header:
                    row[metric_col] = value
                    touched = True
            if touched:
                updated += 1
        print(f"[merged] injected global VLM rows={updated} json_files={len(vlm_jsons)} from {global_vlm_dir}")
    else:
        print(f"[WARN] No global VLM json found under: {global_vlm_dir}")

    merged_rows.sort(key=lambda r: (r.get("Video_ID") or ""))
    if merged_rows:
        merged_rows.append(_build_average_row(csv_rows=merged_rows, model_name=model_name, fieldnames=header))
    out_csv_name = csv_name or f"{model_name}_aggregated_results_all_shards.csv"
    out_csv = data_root / model_name / "csv_results" / out_csv_name
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(
        f"[merged] output={out_csv} shards={len(shard_csv_paths)} rows={len(merged_rows)} "
        f"duplicates_skipped={duplicate_count}"
    )
    return str(out_csv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate video_quality evaluation outputs into a CSV.")
    parser.add_argument("--base_dir", default=os.path.dirname(__file__), help="Single-model base dir. Used when not in sharded mode.")
    parser.add_argument("--model_name", required=True, help="Model name for CSV rows and VLM folder")
    parser.add_argument("--csv_name", default="aggregated_results.csv", help="Output CSV file name")
    parser.add_argument("--vlm_model_dir", default=None, help="Override VLM output folder name if different from model name")
    parser.add_argument(
        "--jepa_result_path",
        default=None,
        help=(
            "Optional JEPA JSON path. Supports per-video output_JEDi/generated_results.json "
            "or global output_JEDi/results.json."
        ),
    )
    parser.add_argument("--video_quality_root", default=str(Path(__file__).resolve().parent.parent), help="Path to video_quality root. Used in sharded mode.")
    parser.add_argument("--num_shards", type=int, default=None, help="Enable sharded mode with N shards.")
    parser.add_argument("--shard_ids", default=None, help="Comma-separated shard suffixes, e.g. 00,01,02. Enables sharded mode.")
    parser.add_argument("--global_vlm_model_dir", default=None, help="In sharded mode, use data/<model>/output_VLM/<global_vlm_model_dir> for VLM injection.")
    args = parser.parse_args()

    sharded_mode = (args.num_shards is not None) or (args.shard_ids is not None)
    if sharded_mode:
        csv_path = aggregate_sharded_results(
            video_quality_root=args.video_quality_root,
            model_name=args.model_name,
            csv_name=args.csv_name,
            num_shards=args.num_shards,
            shard_ids=args.shard_ids,
            global_vlm_model_dir=args.global_vlm_model_dir,
            jepa_result_path=args.jepa_result_path,
        )
    else:
        csv_path = aggregate_results(
            base_dir=args.base_dir,
            model_name=args.model_name,
            csv_name=args.csv_name,
            vlm_model_dir=args.vlm_model_dir,
            jepa_result_path=args.jepa_result_path,
            include_average_row=True,
        )
    print(f"CSV written to: {csv_path}")
