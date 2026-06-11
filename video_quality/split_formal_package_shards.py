#!/usr/bin/env python3
"""Split a formal WorldArena eval package in two levels:
1) split into server folders
2) split each server folder into shards

Usage:
  python video_quality/split_formal_package_shards.py <package_dir>

Only one CLI argument is required. Split sizes are controlled by env vars:
  - WORLD_ARENA_NUM_SERVERS (default: 4)
  - WORLD_ARENA_SHARDS_PER_SERVER (default: 8)
  - WORLD_ARENA_NUM_SHARDS (legacy alias for SHARDS_PER_SERVER)

Output layout (same level as <package_dir>):
  <package_parent>/<package_name>_0/
    summary.json
    <model>_test*/
    gt_first_frames/
    shards/
      shard_00/
        summary.json
        <model>_test*/
        gt_first_frames/
      ...
  <package_parent>/<package_name>_1/
  ...
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_NUM_SERVERS = 5
DEFAULT_SHARDS_PER_SERVER = 8
TEST_SUFFIXES = ("_test", "_test_1", "_test_2", "_test_vlm")


def die(msg: str) -> None:
    print(f"[split-formal-shards][ERROR] {msg}", file=sys.stderr)
    raise SystemExit(1)


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
        # Backward compatibility with old env var.
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
    if len(parts) < 5:
        die(f"summary item #{idx} gt_path has <5 parts: {gt_path}")
    id1 = parts[-5]
    id2 = Path(parts[-1]).stem
    if not id2:
        die(f"summary item #{idx} invalid gt_path stem: {gt_path}")
    return {"id1": id1, "id2": id2}


def detect_data_layout(package_dir: Path) -> Tuple[Dict[str, Path], Dict[str, str]]:
    package_name = package_dir.name
    entries = [
        p
        for p in sorted(package_dir.iterdir())
        if p.is_dir() and p.name != "shards"
    ]

    by_suffix: Dict[str, List[Path]] = {s: [] for s in TEST_SUFFIXES}
    gt_first_frames: Path | None = None
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
        return f"{id1}_{id2}.mp4"
    die(f"Unsupported dir type: {dir_name}")
    return ""


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
    if num_servers > len(summary):
        die(f"num_servers ({num_servers}) > samples ({len(summary)})")

    min_samples_per_server = len(summary) // num_servers
    if min_samples_per_server < shards_per_server:
        die(
            "Not enough samples per server for requested shard count. "
            f"samples={len(summary)}, num_servers={num_servers}, "
            f"min_samples_per_server={min_samples_per_server}, "
            f"shards_per_server={shards_per_server}"
        )

    data_dirs, alias_map = detect_data_layout(package_dir)

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

    missing: List[str] = []
    for idx, item in enumerate(summary):
        ids = extract_ids(item, idx)
        id1, id2 = ids["id1"], ids["id2"]

        server_idx = idx % num_servers
        local_index = len(server_summaries[server_idx])
        shard_idx = local_index % shards_per_server

        server_summaries[server_idx].append(item)
        shard_summaries[server_idx][shard_idx].append(item)

        for dname, src_dir in data_dirs.items():
            fname = expected_file_name(dname, id1, id2)
            src_file = src_dir / fname

            # 增加对 gt_first_frames 的回退机制：如果找不到，尝试使用 summary.json 中的 image 字段
            if not src_file.exists() and dname == "gt_first_frames" and "image" in item:
                fallback_img = Path(item["image"])
                if fallback_img.exists():
                    src_file = fallback_img
                    
            if not src_file.exists():
                missing.append(f"sample#{idx} {dname}/{fname}")
                continue

            server_dst = server_dirs[server_idx] / dname / fname
            shard_dst = server_shard_dirs[server_idx][shard_idx] / dname / fname
            link_or_copy(src_file, server_dst)
            link_or_copy(src_file, shard_dst)

    if missing:
        preview = "\n".join(f"  - {x}" for x in missing[:30])
        die(
            "Missing files while splitting:\n"
            f"{preview}\n"
            f"(total missing: {len(missing)})"
        )

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

    meta = {
        "package_dir": str(package_dir),
        "num_samples": len(summary),
        "num_servers": num_servers,
        "shards_per_server": shards_per_server,
        "data_dirs": sorted(data_dirs.keys()),
        "alias_map": alias_map,
        "server_split_rule": "global_summary_round_robin",
        "shard_split_rule": "server_local_round_robin",
    }
    split_meta_path = package_parent / f"{package_name}_servers_split_meta.json"
    split_meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    info(f"done: {package_dir}")
    info(
        f"num_samples={len(summary)}, num_servers={num_servers}, "
        f"shards_per_server={shards_per_server}"
    )
    for server_idx in range(num_servers):
        server_name = indexed_name("server", server_idx, num_servers)
        server_total = len(server_summaries[server_idx])
        shard_counts = [len(shard_summaries[server_idx][i]) for i in range(shards_per_server)]
        info(f"{server_name}: {server_total} samples, shard_counts={shard_counts}")
    info(f"output servers under: {package_parent}")
    info(f"meta: {split_meta_path}")


if __name__ == "__main__":
    main()
