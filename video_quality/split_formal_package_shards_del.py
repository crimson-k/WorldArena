#!/usr/bin/env python3
"""Split a formal WorldArena eval package in two levels:
1) split into server folders
2) split each server folder into shards

Usage:
  python video_quality/split_formal_package_shards.py <package_dir>

Only one CLI argument is required. Split sizes are controlled by env vars:
  - WORLD_ARENA_NUM_SERVERS (default: 5)
  - WORLD_ARENA_SHARDS_PER_SERVER (default: 8)
  - WORLD_ARENA_NUM_SHARDS (legacy alias for SHARDS_PER_SERVER)

Output layout:
  <package_parent>/<package_name>_0/
    summary.json
    <model>_test*/
    gt_first_frames/          optional
    shards/
      shard_00/
        summary.json
        <model>_test*/
        gt_first_frames/      optional
      ...
  <package_parent>/<package_name>_1/
  ...

Important:
  - This version does NOT require source samples to be continuous.
  - It does NOT require episode ids to start from 0 or 1.
  - It matches each summary item by id2 = stem(gt_path).
  - If a required video is missing, that summary item is skipped instead of aborting.
  - Skipped items are saved to <package_parent>/<package_name>_servers_skipped_records.json.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_NUM_SERVERS = 5
DEFAULT_SHARDS_PER_SERVER = 8

TEST_SUFFIXES = ("_test", "_test_1", "_test_2", "_test_vlm")


def die(msg: str) -> None:
    print(f"[split-formal-shards][ERROR] {msg}", file=sys.stderr)
    raise SystemExit(1)


def warn(msg: str) -> None:
    print(f"[split-formal-shards][WARN] {msg}")


def info(msg: str) -> None:
    print(f"[split-formal-shards] {msg}")


def parse_positive_int(env_key: str, default: int) -> int:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        return default

    try:
        value = int(raw)
    except ValueError:
        die(f"{env_key} is not an integer: {raw}")

    if value <= 0:
        die(f"{env_key} must be > 0, got: {value}")

    return value


def parse_split_config() -> Tuple[int, int]:
    num_servers = parse_positive_int("WORLD_ARENA_NUM_SERVERS", DEFAULT_NUM_SERVERS)

    raw_shards = os.environ.get("WORLD_ARENA_SHARDS_PER_SERVER", "").strip()
    if raw_shards:
        try:
            shards_per_server = int(raw_shards)
        except ValueError:
            die(f"WORLD_ARENA_SHARDS_PER_SERVER is not an integer: {raw_shards}")

        if shards_per_server <= 0:
            die(f"WORLD_ARENA_SHARDS_PER_SERVER must be > 0, got: {shards_per_server}")
    else:
        shards_per_server = parse_positive_int(
            "WORLD_ARENA_NUM_SHARDS",
            DEFAULT_SHARDS_PER_SERVER,
        )

    return num_servers, shards_per_server


def indexed_name(prefix: str, index: int, total: int) -> str:
    width = max(2, len(str(max(0, total - 1))))
    return f"{prefix}_{index:0{width}d}"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_summary(summary_path: Path) -> List[dict]:
    if not summary_path.exists():
        die(f"summary.json not found: {summary_path}")

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        die(f"Invalid JSON in {summary_path}: {exc}")

    if not isinstance(data, list):
        die(f"summary.json must be a JSON list: {summary_path}")

    return data


def extract_ids(item: dict, idx: int) -> Dict[str, str]:
    gt_path = item.get("gt_path")

    if not isinstance(gt_path, str) or not gt_path.strip():
        die(f"summary item #{idx} missing valid gt_path")

    parts = Path(gt_path).parts
    id2 = Path(gt_path).stem.strip()

    if not id2:
        die(f"summary item #{idx} invalid gt_path stem: {gt_path}")

    # 兼容旧逻辑：WorldArena 原始 preprocess 一般依赖 parts[-5]。
    # 如果 gt_path 少于 5 段，这里不直接报错，而是让后续文件匹配逻辑尽力处理。
    id1 = parts[-5] if len(parts) >= 5 else ""

    return {"id1": id1, "id2": id2}


def detect_data_layout(package_dir: Path) -> Tuple[Dict[str, Path], Dict[str, str]]:
    package_name = package_dir.name

    entries = [
        p
        for p in sorted(package_dir.iterdir())
        if p.is_dir() and p.name != "shards"
    ]

    by_suffix: Dict[str, List[Path]] = {s: [] for s in TEST_SUFFIXES}
    gt_first_frames: Optional[Path] = None

    for p in entries:
        if p.name == "gt_first_frames":
            gt_first_frames = p
            continue

        for suffix in TEST_SUFFIXES:
            if p.name.endswith(suffix):
                by_suffix[suffix].append(p)
                break

    canonical_dirs: Dict[str, Path] = {}
    alias_map: Dict[str, str] = {}

    for suffix in TEST_SUFFIXES:
        candidates = by_suffix[suffix]
        if not candidates:
            continue

        expected_name = f"{package_name}{suffix}"
        expected = package_dir / expected_name

        if expected in candidates:
            chosen = expected
        elif len(candidates) == 1:
            chosen = candidates[0]
        else:
            names = ", ".join(p.name for p in candidates)
            die(f"Ambiguous folders for suffix {suffix}: {names}")

        canonical_dirs[chosen.name] = chosen

        if expected_name != chosen.name:
            alias_map[expected_name] = chosen.name

    if gt_first_frames is not None:
        canonical_dirs["gt_first_frames"] = gt_first_frames

    if not canonical_dirs:
        die(f"No supported data folders found under: {package_dir}")

    if not any(name.endswith("_test") for name in canonical_dirs):
        die("Need at least one *_test folder.")

    if not any(name.endswith("_test_vlm") for name in canonical_dirs):
        die("Need at least one *_test_vlm folder.")

    return canonical_dirs, alias_map


def expected_file_name(dir_name: str, id1: str, id2: str) -> str:
    if dir_name == "gt_first_frames":
        return f"{id2}.png"

    if dir_name.endswith("_test_vlm"):
        return f"{id2}.mp4"

    if dir_name.endswith("_test") or dir_name.endswith("_test_1") or dir_name.endswith("_test_2"):
        if id1:
            return f"{id1}_{id2}.mp4"
        return f"{id2}.mp4"

    die(f"Unsupported dir type: {dir_name}")
    return ""


def resolve_source_file(
    dir_name: str,
    src_dir: Path,
    item: dict,
    idx: int,
    id1: str,
    id2: str,
) -> Tuple[Optional[Path], Optional[str], bool, str]:
    """Resolve source file for one sample in one data directory.

    Returns:
      (src_file, dst_file_name, required, reason)

    required means:
      - True: if not found, this whole sample should be skipped.
      - False: if not found, just skip copying this optional file.
    """

    # Optional image folder.
    if dir_name == "gt_first_frames":
        dst_name = f"{id2}.png"
        expected = src_dir / dst_name

        if expected.exists():
            return expected, dst_name, False, ""

        image_value = item.get("image")
        if isinstance(image_value, str) and image_value.strip():
            fallback_img = Path(image_value)
            if fallback_img.exists():
                return fallback_img, dst_name, False, ""

        return None, dst_name, False, f"missing optional image for id2={id2}"

    # VLM folder: expected <id2>.mp4.
    if dir_name.endswith("_test_vlm"):
        dst_name = f"{id2}.mp4"
        expected = src_dir / dst_name

        if expected.exists():
            return expected, dst_name, True, ""

        return None, dst_name, True, f"missing required vlm video: {expected}"

    # Main/action folders:
    # Preferred target name follows summary id1/id2 if id1 is available.
    # If source file was produced using another prefix, we still copy it to dst_name.
    if dir_name.endswith("_test") or dir_name.endswith("_test_1") or dir_name.endswith("_test_2"):
        dst_name = f"{id1}_{id2}.mp4" if id1 else f"{id2}.mp4"

        candidates: List[Path] = []

        if id1:
            candidates.append(src_dir / f"{id1}_{id2}.mp4")

        candidates.append(src_dir / f"{id2}.mp4")

        # Fallback for cases like:
        #   fixed_scene_task_episode7.mp4
        #   usersht_episode7.mp4
        # when id1 in summary does not equal the prefix used by prepare script.
        candidates.extend(sorted(src_dir.glob(f"*_{id2}.mp4")))

        seen = set()
        unique_candidates: List[Path] = []
        for c in candidates:
            key = str(c)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(c)

        existing = [c for c in unique_candidates if c.exists()]

        if not existing:
            return None, dst_name, True, (
                f"missing required eval video for id2={id2}, "
                f"tried={[str(c) for c in unique_candidates[:10]]}"
            )

        if len(existing) > 1:
            exact = src_dir / dst_name
            if exact.exists():
                return exact, dst_name, True, ""

            names = [str(x.name) for x in existing[:10]]
            return None, dst_name, True, (
                f"ambiguous eval video for id2={id2} under {src_dir}, candidates={names}"
            )

        return existing[0], dst_name, True, ""

    return None, None, True, f"unsupported dir type: {dir_name}"


def prepare_aliases(
    base_dir: Path,
    alias_map: Dict[str, str],
    expected_prefix: str,
    data_dir_names: List[str],
) -> None:
    merged_alias_map = dict(alias_map)

    for suffix in TEST_SUFFIXES:
        matches = [name for name in data_dir_names if name.endswith(suffix)]
        if not matches:
            continue

        if len(matches) > 1:
            die(f"Ambiguous target dirs for suffix {suffix}: {matches}")

        target_name = matches[0]
        dynamic_alias = f"{expected_prefix}{suffix}"

        if dynamic_alias != target_name:
            merged_alias_map[dynamic_alias] = target_name

    for alias_name, target_name in merged_alias_map.items():
        alias_path = base_dir / alias_name

        if alias_path.exists() or alias_path.is_symlink():
            if alias_path.is_dir() and not alias_path.is_symlink():
                shutil.rmtree(alias_path)
            else:
                alias_path.unlink()

        os.symlink(target_name, alias_path)


def create_server_layout(
    server_dirs: List[Path],
    num_servers: int,
    shards_per_server: int,
    data_dirs: Dict[str, Path],
    alias_map: Dict[str, str],
) -> Tuple[List[Path], List[List[Path]]]:
    if len(server_dirs) != num_servers:
        die(
            f"create_server_layout got server_dirs={len(server_dirs)}, "
            f"num_servers={num_servers}"
        )

    concrete_server_dirs: List[Path] = []
    server_shard_dirs: List[List[Path]] = []

    for server_dir in server_dirs:
        server_dir.mkdir(parents=True, exist_ok=True)

        for dname in data_dirs:
            (server_dir / dname).mkdir(parents=True, exist_ok=True)

        prepare_aliases(
            base_dir=server_dir,
            alias_map=alias_map,
            expected_prefix=server_dir.name,
            data_dir_names=list(data_dirs.keys()),
        )

        shard_root = server_dir / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)

        shards_for_server: List[Path] = []

        for shard_idx in range(shards_per_server):
            shard_dir = shard_root / indexed_name("shard", shard_idx, shards_per_server)
            shard_dir.mkdir(parents=True, exist_ok=True)

            for dname in data_dirs:
                (shard_dir / dname).mkdir(parents=True, exist_ok=True)

            prepare_aliases(
                base_dir=shard_dir,
                alias_map=alias_map,
                expected_prefix=server_dir.name,
                data_dir_names=list(data_dirs.keys()),
            )

            shards_for_server.append(shard_dir)

        concrete_server_dirs.append(server_dir)
        server_shard_dirs.append(shards_for_server)

    return concrete_server_dirs, server_shard_dirs


def build_valid_entries(
    summary: List[dict],
    data_dirs: Dict[str, Path],
) -> Tuple[List[dict], List[dict]]:
    valid_entries: List[dict] = []
    skipped_records: List[dict] = []

    for idx, item in enumerate(summary):
        ids = extract_ids(item, idx)
        id1, id2 = ids["id1"], ids["id2"]

        file_plan = []
        missing_required = []
        optional_warnings = []

        for dname, src_dir in data_dirs.items():
            src_file, dst_name, required, reason = resolve_source_file(
                dir_name=dname,
                src_dir=src_dir,
                item=item,
                idx=idx,
                id1=id1,
                id2=id2,
            )

            if src_file is None:
                if required:
                    missing_required.append(
                        {
                            "dir": dname,
                            "reason": reason,
                        }
                    )
                else:
                    optional_warnings.append(
                        {
                            "dir": dname,
                            "reason": reason,
                        }
                    )
                continue

            if dst_name is None:
                missing_required.append(
                    {
                        "dir": dname,
                        "reason": "resolved dst_name is None",
                    }
                )
                continue

            file_plan.append(
                {
                    "dir": dname,
                    "src": src_file,
                    "dst_name": dst_name,
                }
            )

        if missing_required:
            skipped_records.append(
                {
                    "summary_index": idx,
                    "id1": id1,
                    "id2": id2,
                    "gt_path": item.get("gt_path"),
                    "missing_required": missing_required,
                    "optional_warnings": optional_warnings,
                }
            )
            continue

        if optional_warnings:
            warn(
                f"sample#{idx} id2={id2} has optional missing files: "
                f"{optional_warnings}"
            )

        valid_entries.append(
            {
                "summary_index": idx,
                "item": item,
                "id1": id1,
                "id2": id2,
                "file_plan": file_plan,
                "optional_warnings": optional_warnings,
            }
        )

    return valid_entries, skipped_records


def main() -> None:
    if len(sys.argv) != 2:
        die("Usage: python video_quality/split_formal_package_shards.py <package_dir>")

    package_dir = Path(sys.argv[1]).expanduser().resolve()
    if not package_dir.is_dir():
        die(f"package_dir is not a directory: {package_dir}")

    num_servers, shards_per_server = parse_split_config()

    summary = read_summary(package_dir / "summary.json")
    if not summary:
        die(f"summary.json is empty: {package_dir / 'summary.json'}")

    data_dirs, alias_map = detect_data_layout(package_dir)

    valid_entries, skipped_records = build_valid_entries(
        summary=summary,
        data_dirs=data_dirs,
    )

    if not valid_entries:
        skipped_preview = json.dumps(skipped_records[:5], ensure_ascii=False, indent=2)
        die(
            "No valid samples left after checking package files. "
            f"Skipped preview:\n{skipped_preview}"
        )

    if num_servers > len(valid_entries):
        die(
            f"num_servers ({num_servers}) > valid samples ({len(valid_entries)}). "
            f"original samples={len(summary)}, skipped={len(skipped_records)}"
        )

    min_samples_per_server = len(valid_entries) // num_servers
    if min_samples_per_server < shards_per_server:
        die(
            "Not enough valid samples per server for requested shard count. "
            f"original_samples={len(summary)}, valid_samples={len(valid_entries)}, "
            f"skipped={len(skipped_records)}, num_servers={num_servers}, "
            f"min_samples_per_server={min_samples_per_server}, "
            f"shards_per_server={shards_per_server}"
        )

    package_name = package_dir.name
    package_parent = package_dir.parent

    server_dirs = [package_parent / f"{package_name}_{i}" for i in range(num_servers)]

    for server_dir in server_dirs:
        if server_dir.exists():
            shutil.rmtree(server_dir)

    server_dirs, server_shard_dirs = create_server_layout(
        server_dirs=server_dirs,
        num_servers=num_servers,
        shards_per_server=shards_per_server,
        data_dirs=data_dirs,
        alias_map=alias_map,
    )

    server_summaries: List[List[dict]] = [[] for _ in range(num_servers)]
    shard_summaries: List[List[List[dict]]] = [
        [[] for _ in range(shards_per_server)] for _ in range(num_servers)
    ]

    copied_files = 0

    for valid_idx, entry in enumerate(valid_entries):
        item = entry["item"]
        file_plan = entry["file_plan"]

        server_idx = valid_idx % num_servers
        local_index = len(server_summaries[server_idx])
        shard_idx = local_index % shards_per_server

        server_summaries[server_idx].append(item)
        shard_summaries[server_idx][shard_idx].append(item)

        for plan in file_plan:
            dname = plan["dir"]
            src_file = plan["src"]
            dst_name = plan["dst_name"]

            server_dst = server_dirs[server_idx] / dname / dst_name
            shard_dst = server_shard_dirs[server_idx][shard_idx] / dname / dst_name

            link_or_copy(src_file, server_dst)
            link_or_copy(src_file, shard_dst)
            copied_files += 2

    for server_idx in range(num_servers):
        server_dir = server_dirs[server_idx]

        (server_dir / "summary.json").write_text(
            json.dumps(server_summaries[server_idx], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for shard_idx in range(shards_per_server):
            shard_dir = server_shard_dirs[server_idx][shard_idx]
            (shard_dir / "summary.json").write_text(
                json.dumps(shard_summaries[server_idx][shard_idx], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    split_meta_path = package_parent / f"{package_name}_servers_split_meta.json"
    skipped_path = package_parent / f"{package_name}_servers_skipped_records.json"

    meta = {
        "package_dir": str(package_dir),
        "original_num_samples": len(summary),
        "valid_num_samples": len(valid_entries),
        "skipped_num_samples": len(skipped_records),
        "num_servers": num_servers,
        "shards_per_server": shards_per_server,
        "copied_files": copied_files,
        "data_dirs": sorted(data_dirs.keys()),
        "alias_map": alias_map,
        "server_split_rule": "valid_summary_round_robin",
        "shard_split_rule": "server_local_round_robin",
        "missing_policy": "skip_samples_with_missing_required_videos",
        "optional_file_policy": "skip_missing_optional_gt_first_frames",
        "skipped_records_file": str(skipped_path),
    }

    split_meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    skipped_path.write_text(
        json.dumps(skipped_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    info(f"done: {package_dir}")
    info(
        f"original_samples={len(summary)}, valid_samples={len(valid_entries)}, "
        f"skipped={len(skipped_records)}, num_servers={num_servers}, "
        f"shards_per_server={shards_per_server}"
    )

    if skipped_records:
        warn(f"skipped records saved to: {skipped_path}")
        warn(f"first skipped record: {json.dumps(skipped_records[0], ensure_ascii=False)}")

    for server_idx in range(num_servers):
        server_name = indexed_name("server", server_idx, num_servers)
        server_total = len(server_summaries[server_idx])
        shard_counts = [
            len(shard_summaries[server_idx][i])
            for i in range(shards_per_server)
        ]
        info(f"{server_name}: {server_total} samples, shard_counts={shard_counts}")

    info(f"output servers under: {package_parent}")
    info(f"meta: {split_meta_path}")
    info(f"skipped: {skipped_path}")


if __name__ == "__main__":
    main()