#!/usr/bin/env python3
"""Prepare a WorldArena local-eval package from raw generated videos + summary.

This script converts a raw folder (for example: videos/videos_1/videos_2 + summary.json)
into a formal structure that works with:
  - run_evaluation.sh (8 non-GT metrics etc., via preprocess scripts)
  - run_action_following.sh
  - run_VLM_judge.sh

Output structure:
  <output_root>/
    <model_name>_test/
    <model_name>_test_1/
    <model_name>_test_2/
    <model_name>_test_vlm/
    summary.json
    report.json

Naming rules in output:
  - *_test / *_test_1 / *_test_2 use: <id1>_<id2>.mp4
  - *_test_vlm uses: <id2>.mp4
Where:
  - id2 = stem(gt_path) from input summary
  - id1 = configurable fixed token (default: fixed_scene_task)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_GT_TEMPLATE = "{id1}/placeholder/placeholder/placeholder/{id2}.mp4"


@dataclass
class SourceSet:
    main: Path
    alt1: Path
    alt2: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build formal WorldArena eval package with standardized names and summary."
    )
    parser.add_argument("--input_root", type=Path, required=True, help="Raw input root directory.")
    parser.add_argument("--output_root", type=Path, required=True, help="Output root directory.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name prefix for output folders.")
    parser.add_argument("--summary_name", type=str, default="summary.json", help="Input summary file name.")

    parser.add_argument("--src_main", type=str, default="videos", help="Main raw video folder under input_root.")
    parser.add_argument("--src_alt1", type=str, default="videos_1", help="Alt-1 raw video folder under input_root.")
    parser.add_argument("--src_alt2", type=str, default="videos_2", help="Alt-2 raw video folder under input_root.")

    parser.add_argument(
        "--id1",
        type=str,
        default="fixed_scene_task",
        help="id1 token used for <id1>_<id2>.mp4 and gt_path template.",
    )
    parser.add_argument(
        "--gt_path_template",
        type=str,
        default=DEFAULT_GT_TEMPLATE,
        help=(
            "Template for rewritten summary gt_path. Supports {id1}, {id2}, {index}. "
            "Default guarantees parts[-5] == id1."
        ),
    )

    parser.add_argument(
        "--image_mode",
        choices=("keep", "extract_if_missing", "extract_all"),
        default="extract_if_missing",
        help=(
            "How to set summary image path. "
            "'keep': keep original image path as-is; "
            "'extract_if_missing': keep existing image if file exists, else extract first frame; "
            "'extract_all': always extract first frame to output_root/gt_first_frames."
        ),
    )

    parser.add_argument(
        "--transfer",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="How to place videos in output folders.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_root if it exists.")
    parser.add_argument("--dry_run", action="store_true", help="Validate and print plan without writing files.")
    return parser.parse_args()


def ensure_clean_output(output_root: Path, overwrite: bool, dry_run: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"Output root exists: {output_root}. Use --overwrite to replace.")
        if not dry_run:
            shutil.rmtree(output_root)
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)


def load_summary(summary_path: Path) -> List[dict]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Summary must be a JSON list.")
    return data


def validate_summary_records(records: List[dict]) -> None:
    required = ("gt_path", "image", "prompt")
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Record #{idx} is not an object.")
        for key in required:
            if key not in item:
                raise ValueError(f"Record #{idx} missing key: {key}")
        if not isinstance(item["gt_path"], str):
            raise ValueError(f"Record #{idx} gt_path must be string.")
        if not isinstance(item["image"], str):
            raise ValueError(f"Record #{idx} image must be string.")
        prompt = item["prompt"]
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"Record #{idx} prompt must be string or list.")


def build_stem_index(folder: Path) -> Dict[str, Path]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Video folder not found: {folder}")
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() != ".mp4":
            continue
        stem = f.stem
        if stem in index:
            duplicates.setdefault(stem, [index[stem]])
            duplicates[stem].append(f)
        else:
            index[stem] = f
    if duplicates:
        sample = next(iter(duplicates.items()))
        raise ValueError(
            f"Duplicate mp4 stem in {folder}: {sample[0]} -> {[str(x.name) for x in sample[1][:5]]}"
        )
    return index


def has_valid_video_folder(folder: Path) -> bool:
    return folder.exists() and folder.is_dir() and any(folder.glob("*.mp4"))


def safe_prompt(prompt):
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        if not prompt:
            return ""
        first = prompt[0]
        return first if isinstance(first, str) else str(first)
    return str(prompt)


def transfer_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unsupported transfer mode: {mode}")


def extract_first_frame(video_path: Path, png_path: Path) -> bool:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV (cv2) is required for image extraction.") from exc

    png_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    return bool(cv2.imwrite(str(png_path), frame))


def compute_id2(gt_path: str) -> str:
    stem = Path(gt_path).stem.strip()
    if not stem:
        raise ValueError(f"Invalid gt_path stem: {gt_path}")
    return stem


def build_new_gt_path(template: str, id1: str, id2: str, index: int) -> str:
    value = template.format(id1=id1, id2=id2, index=index)
    parts = Path(value).parts
    if len(parts) < 5:
        raise ValueError(
            f"gt_path_template produced path with <5 segments: {value}. "
            "Need >=5 so preprocess scripts can read parts[-5]."
        )
    if parts[-5] != id1:
        raise ValueError(
            f"gt_path_template produced parts[-5]={parts[-5]!r}, expected {id1!r}. "
            f"Path: {value}"
        )
    return value


def main() -> None:
    args = parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    summary_path = input_root / args.summary_name

    src = SourceSet(
        main=(input_root / args.src_main),
        alt1=(input_root / args.src_alt1),
        alt2=(input_root / args.src_alt2),
    )

    records = load_summary(summary_path)
    validate_summary_records(records)

    main_idx = build_stem_index(src.main)

    action_following_enabled = has_valid_video_folder(src.alt1) and has_valid_video_folder(src.alt2)

    if action_following_enabled:
        print(f"[INFO] action_following enabled: found {src.alt1} and {src.alt2}")
        alt1_idx = build_stem_index(src.alt1)
        alt2_idx = build_stem_index(src.alt2)
    else:
        print(f"[INFO] action_following disabled: missing {src.alt1} or {src.alt2}")
        alt1_idx = {}
        alt2_idx = {}

    dst_test = output_root / f"{args.model_name}_test"
    dst_test_1 = output_root / f"{args.model_name}_test_1"
    dst_test_2 = output_root / f"{args.model_name}_test_2"
    dst_vlm = output_root / f"{args.model_name}_test_vlm"
    dst_first_frames = output_root / "gt_first_frames"

    planned = []
    new_summary = []
    missing_main = 0
    missing_alt1 = 0
    missing_alt2 = 0
    extracted_images = 0
    kept_images = 0

    for i, item in enumerate(records):
        idx = i + 1
        old_gt = item["gt_path"]
        id2 = compute_id2(old_gt)
        id1 = args.id1

        src_main = main_idx.get(id2)
        src_alt1 = alt1_idx.get(id2) if action_following_enabled else None
        src_alt2 = alt2_idx.get(id2) if action_following_enabled else None

        if src_main is None:
            missing_main += 1
            continue

        if action_following_enabled:
            if src_alt1 is None:
                missing_alt1 += 1
                continue
            if src_alt2 is None:
                missing_alt2 += 1
                continue

        name_eval = f"{id1}_{id2}.mp4"
        name_vlm = f"{id2}.mp4"

        planned.append((src_main, dst_test / name_eval))
        planned.append((src_main, dst_vlm / name_vlm))

        if action_following_enabled:
            planned.append((src_alt1, dst_test_1 / name_eval))
            planned.append((src_alt2, dst_test_2 / name_eval))

        new_gt_path = build_new_gt_path(args.gt_path_template, id1=id1, id2=id2, index=idx)

        old_image = str(item["image"])
        image_path_out = old_image

        if args.image_mode == "keep":
            kept_images += 1
        else:
            old_exists = Path(old_image).exists() if old_image else False
            need_extract = args.image_mode == "extract_all" or (args.image_mode == "extract_if_missing" and not old_exists)
            if need_extract:
                png_path = dst_first_frames / f"{id2}.png"
                image_path_out = str(png_path.resolve())
                extracted_images += 1
            else:
                kept_images += 1

        prompt_out = safe_prompt(item["prompt"])
        new_summary.append(
            {
                "gt_path": new_gt_path,
                "image": image_path_out,
                "prompt": [prompt_out],
            }
        )

    if missing_main or (action_following_enabled and (missing_alt1 or missing_alt2)):
        raise RuntimeError(
            "Source video matching failed. "
            f"missing_main={missing_main}, missing_alt1={missing_alt1}, missing_alt2={missing_alt2}, "
            f"action_following_enabled={action_following_enabled}. "
            "Expected each enabled source folder to contain <id2>.mp4 for every summary item."
        )

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"input_root={input_root}")
        print(f"output_root={output_root}")
        print(f"records_in={len(records)}")
        print(f"records_out={len(new_summary)}")
        print(f"planned_transfers={len(planned)}")
        print(f"image_mode={args.image_mode}, extracted_images={extracted_images}, kept_images={kept_images}")
        print(f"dst_test={dst_test}")
        if action_following_enabled:
            print(f"dst_test_1={dst_test_1}")
            print(f"dst_test_2={dst_test_2}")
        else:
            print("dst_test_1=(disabled)")
            print("dst_test_2=(disabled)")
        print(f"dst_vlm={dst_vlm}")
        if new_summary:
            print("summary_sample=", new_summary[0])
        return

    ensure_clean_output(output_root, overwrite=args.overwrite, dry_run=False)
    output_dirs = [dst_test, dst_vlm, dst_first_frames]
    if action_following_enabled:
        output_dirs.extend([dst_test_1, dst_test_2])

    for p in output_dirs:
        p.mkdir(parents=True, exist_ok=True)

    for src_path, dst_path in planned:
        transfer_file(src_path, dst_path, mode=args.transfer)

    if args.image_mode in ("extract_if_missing", "extract_all"):
        by_id2 = {Path(row["gt_path"]).stem: row for row in new_summary}
        failed_extract = 0
        for id2, row in by_id2.items():
            out_img = Path(row["image"])
            if out_img.exists():
                continue
            src_video = main_idx[id2]
            ok = extract_first_frame(src_video, out_img)
            if not ok:
                failed_extract += 1
        if failed_extract:
            print(f"[WARN] failed to extract first frame for {failed_extract} videos.")

    out_summary = output_root / "summary.json"
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(new_summary, f, ensure_ascii=False, indent=2)

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "records_in": len(records),
        "records_out": len(new_summary),
        "transfer_mode": args.transfer,
        "image_mode": args.image_mode,
        "id1": args.id1,
        "gt_path_template": args.gt_path_template,
        "folders": {
            "test": str(dst_test),
            "test_1": str(dst_test_1) if action_following_enabled else None,
            "test_2": str(dst_test_2) if action_following_enabled else None,
            "test_vlm": str(dst_vlm),
        },
        "action_following_enabled": action_following_enabled,
        "summary": str(out_summary),
    }
    with (output_root / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
