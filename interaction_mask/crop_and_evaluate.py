"""Crop matched RoboTwin GT/generated videos around GT gripper points and evaluate."""

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from interaction_mask.crop_gripper import crop_video
from interaction_mask.project_gripper import decode_rgb, point_xyz_from_endpose, project_points

DEFAULT_GT_ROOT = Path(
    "/data1/fangxuebin/boundless-world-model/converted_dataset_bwm_6tasks_256x192"
)
DEFAULT_ROBOTWIN_ROOT = Path("/data1/common_data/RoboTwin2.0/dataset")
EPISODE_PATTERN = re.compile(r"(?:.+__)?episode[_-]?(\d+)$")


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


@dataclass
class Pair:
    task: str
    episode: int
    generated: Path
    gt: Path
    hdf5: Path
    gt_info: VideoInfo
    generated_info: VideoInfo
    active_sides: tuple
    activity: dict
    coordinates: dict


def video_info(path):
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        metadata = reader.get_meta_data()
        frames = iter(reader)
        first = next(frames, None)
        if first is None:
            raise ValueError(f"Video has no decodable frames: {path}")
        height, width = first.shape[:2]
        count = 1 + sum(1 for _ in frames)
        fps = float(metadata.get("fps", 0))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video metadata for {path}: {metadata}")
        return VideoInfo(width, height, fps, count)
    finally:
        reader.close()


def load_gt_manifest(gt_root):
    manifests = [gt_root / "metadata_test.jsonl", gt_root / "metadata_train.jsonl"]
    manifests = [path for path in manifests if path.is_file()]
    if not manifests:
        raise FileNotFoundError(f"No metadata_test/train.jsonl in {gt_root}")
    rows = {}
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task = row["task"]
                episode = int(row["source_episode_index"])
                relative = Path(row["video"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe GT video path at {manifest}:{line_no}: {relative}")
                key = (task, episode)
                if key in rows and rows[key] != relative:
                    raise ValueError(f"Conflicting GT manifest entries for {key}")
                rows[key] = relative
    return rows


def index_generated_videos(root):
    if not root.is_dir():
        raise NotADirectoryError(root)
    indexed = {}
    for path in sorted(root.rglob("*.mp4")):
        relative = path.relative_to(root)
        if len(relative.parts) < 2:
            raise ValueError(f"Generated video must be under a task directory: {path}")
        match = EPISODE_PATTERN.fullmatch(path.stem)
        if not match:
            raise ValueError(f"Unrecognized generated video filename: {path.name}")
        key = (relative.parts[0], int(match.group(1)))
        if key in indexed:
            raise ValueError(f"Multiple generated videos for {key}: {indexed[key]} and {path}")
        indexed[key] = path
    if not indexed:
        raise ValueError(f"No generated MP4 videos found in {root}")
    return indexed


def source_for_pair(task, episode, relative_gt_video, gt_root, robotwin_root):
    gt_video = gt_root / relative_gt_video
    if len(relative_gt_video.parts) < 4 or relative_gt_video.parts[0] != "videos":
        raise ValueError(f"Unexpected GT manifest video path: {relative_gt_video}")
    subset = relative_gt_video.parts[2]
    hdf5 = robotwin_root / task / subset / "data" / f"episode{episode}.hdf5"
    if not gt_video.is_file():
        raise FileNotFoundError(f"Missing converted GT video: {gt_video}")
    if not hdf5.is_file():
        raise FileNotFoundError(f"Missing RoboTwin HDF5: {hdf5}")
    return gt_video, hdf5


def detect_active_grippers(hdf5_path, task):
    """Detect the moving arm from endpose, gripper and joint-action signals."""
    signals = {}
    with h5py.File(hdf5_path, "r") as file:
        for side in ("left", "right"):
            xyz = file[f"endpose/{side}_endpose"][:, :3]
            gripper = file[f"endpose/{side}_gripper"][:]
            joint_action = file[f"joint_action/{side}_arm"][:]
            path_length = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
            xyz_range = float(np.linalg.norm(np.ptp(xyz, axis=0)))
            gripper_range = float(np.ptp(gripper))
            joint_range = float(np.linalg.norm(np.ptp(joint_action, axis=0)))
            score = (
                min(path_length / 0.02, 20.0)
                + min(xyz_range / 0.01, 20.0)
                + min(gripper_range / 0.05, 20.0)
                + min(joint_range / 0.1, 20.0)
            )
            signals[side] = {
                "score": score,
                "xyz_path_length": path_length,
                "xyz_range": xyz_range,
                "gripper_range": gripper_range,
                "joint_action_range": joint_range,
            }

    if task == "place_burger_fries":
        return ("left", "right"), {"method": "known_dual_arm_task", "signals": signals}
    ranked = sorted(signals, key=lambda arm: signals[arm]["score"], reverse=True)
    first, second = ranked
    first_score, second_score = signals[first]["score"], signals[second]["score"]
    if first_score < 1.0 or second_score >= first_score * 0.5:
        raise ValueError(
            f"Cannot uniquely identify active gripper in {hdf5_path}: "
            f"left={signals['left']['score']:.3f}, right={signals['right']['score']:.3f}"
        )
    return (first,), {"method": "trajectory_activity", "signals": signals}


def project_episode(hdf5_path, camera, side, image_size,
                    point_type="tcp", tcp_offset=0.12):
    """Make crop_gripper-compatible pixel coordinates in the GT video space."""
    with h5py.File(hdf5_path, "r") as file:
        observation = file[f"observation/{camera}"]
        rgb = observation["rgb"]
        pose = file[f"endpose/{side}_endpose"][:]
        if len(rgb) != len(pose):
            raise ValueError(f"RGB and {side} endpose frame counts differ in {hdf5_path}")
        source_height, source_width = decode_rgb(rgb, 0).shape[:2]
        point_xyz = point_xyz_from_endpose(pose, point_type, tcp_offset)
        intrinsic = observation["intrinsic_cv"][:]
        extrinsic = observation["extrinsic_cv"][:]
        uv, camera_xyz, in_front = project_points(
            point_xyz, intrinsic, extrinsic
        )
    target_width, target_height = image_size
    uv *= np.array([target_width / source_width, target_height / source_height])
    in_frame = (
        in_front & (uv[:, 0] >= 0) & (uv[:, 0] < target_width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < target_height)
    )
    frames = []
    for index in range(len(pose)):
        if not np.isfinite(uv[index]).all():
            raise ValueError(f"Projected {side} coordinate is invalid at frame {index}: {hdf5_path}")
        point = [float(uv[index, 0]), float(uv[index, 1])]
        frames.append({
            "frame_index": index,
            "source_frame_index": index,
            side: {
                "uv": point,
                "xyz": [float(x) for x in point_xyz[index]],
                "endpose_xyz": [float(x) for x in pose[index, :3]],
                "in_front": bool(in_front[index]),
                "in_frame": bool(in_frame[index]),
                "camera_xyz": [float(x) for x in camera_xyz[index]],
            },
        })
    return {
        "hdf5": str(hdf5_path.resolve()),
        "camera": camera,
        "side": side,
        "point_type": point_type,
        "tcp_offset_local_xyz": [float(tcp_offset), 0.0, 0.0] if point_type == "tcp" else [0.0, 0.0, 0.0],
        "image_size": [target_width, target_height],
        "frame_count": len(pose),
        "frames": frames,
    }


def coordinates_for_generated(gt_coordinates, frame_count):
    """Interpolate crop centers over the full GT trajectory without resampling video frames."""
    if frame_count == gt_coordinates["frame_count"]:
        return gt_coordinates
    side = gt_coordinates["side"]
    gt_frames = gt_coordinates["frames"]
    gt_uv = np.asarray([row[side]["uv"] for row in gt_frames])
    gt_positions = np.arange(len(gt_frames))
    positions = np.linspace(0, len(gt_frames) - 1, frame_count)
    uv = np.column_stack([
        np.interp(positions, gt_positions, gt_uv[:, axis]) for axis in range(2)
    ])
    width, height = gt_coordinates["image_size"]
    frames = []
    for index, (position, point) in enumerate(zip(positions, uv)):
        lower = int(np.floor(position))
        upper = min(lower + 1, len(gt_frames) - 1)
        in_front = gt_frames[lower][side]["in_front"] and gt_frames[upper][side]["in_front"]
        frames.append({
            "frame_index": index,
            "gt_frame_position": float(position),
            side: {
                "uv": [float(point[0]), float(point[1])],
                "in_front": bool(in_front),
                "in_frame": bool(in_front and 0 <= point[0] < width and 0 <= point[1] < height),
            },
        })
    return {
        **gt_coordinates,
        "frame_count": frame_count,
        "coordinate_sampling": "linear_interpolation_over_gt_episode",
        "frames": frames,
    }


def prepare_pair(pair, output_dir, gt_assets_root, camera, crop_width, crop_height):
    task_dir = Path(pair.task)
    stem = f"episode{pair.episode}.mp4"
    results = []
    for side in pair.active_sides:
        coords = pair.coordinates[side]
        coordinate_path = gt_assets_root / "coordinates" / side / task_dir / f"episode{pair.episode}.json"
        coordinate_path.parent.mkdir(parents=True, exist_ok=True)
        if coordinate_path.exists():
            if json.loads(coordinate_path.read_text(encoding="utf-8")) != coords:
                raise ValueError(f"Shared GT coordinates differ: {coordinate_path}")
        else:
            coordinate_path.write_text(
                json.dumps(coords, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
        generated_coords = coordinates_for_generated(coords, pair.generated_info.frames)
        generated_coordinate_path = coordinate_path
        if generated_coords is not coords:
            generated_coordinate_path = (
                output_dir / "generated_coordinates" / side / task_dir / f"episode{pair.episode}.json"
            )
            generated_coordinate_path.parent.mkdir(parents=True, exist_ok=True)
            generated_coordinate_path.write_text(
                json.dumps(generated_coords, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )

        gt_output = gt_assets_root / f"gt_crops_{side}" / task_dir / stem
        generated_output = output_dir / f"generated_crops_{side}" / task_dir / stem
        gt_metadata = gt_output.with_suffix(".crop.json")
        if gt_output.exists() and gt_metadata.exists():
            saved = json.loads(gt_metadata.read_text(encoding="utf-8"))
            if (saved["video_source"] != str(pair.gt.resolve())
                    or saved["crop_size"] != [crop_width, crop_height]
                    or saved["frame_count"] != pair.gt_info.frames
                    or not math.isclose(saved["fps"], pair.gt_info.fps, rel_tol=1e-4)):
                raise ValueError(f"Shared GT crop does not match this run: {gt_output}")
        else:
            crop_video(
                pair.gt, coordinate_path, gt_output, gt_metadata, side=side,
                width=crop_width, height=crop_height, draw_point=False,
            )
        crop_video(
            pair.generated, generated_coordinate_path, generated_output,
            generated_output.with_suffix(".crop.json"), side=side,
            width=crop_width, height=crop_height, draw_point=False,
        )
        results.append({
            "task": pair.task, "episode_index": pair.episode,
            "active_sides": list(pair.active_sides), "active_arm_detection": pair.activity,
            "source_hdf5": str(pair.hdf5), "gt_video": str(pair.gt),
            "generated_video": str(pair.generated),
            "projected_coordinates": str(coordinate_path),
            "generated_coordinates": str(generated_coordinate_path),
            "gt_crop": str(gt_output), "generated_crop": str(generated_output),
            "gt_frame_count": pair.gt_info.frames,
            "generated_frame_count": pair.generated_info.frames,
            "gt_fps": pair.gt_info.fps,
            "generated_fps": pair.generated_info.fps,
            "crop_size": [crop_width, crop_height],
            "side": side, "camera": camera,
            "point_type": pair.coordinates[side]["point_type"],
            "tcp_offset_local_xyz": pair.coordinates[side]["tcp_offset_local_xyz"],
        })
    return results


def write_average_results(output_dir, side_csvs):
    """Average metric rows across active arms, including both burger/fries arms."""
    side_rows = []
    metric_names = []
    for side, path in side_csvs.items():
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            for column in reader.fieldnames or []:
                if column != "sample_id" and column not in metric_names:
                    metric_names.append(column)
            for row in reader:
                if row.get("sample_id") == "AVERAGE":
                    continue
                values = {}
                for name in metric_names:
                    raw = row.get(name)
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if not math.isnan(value):
                        values[name] = value
                side_rows.append({"sample_id": row["sample_id"], "side": side, "metrics": values})

    by_sample = {}
    for row in side_rows:
        by_sample.setdefault(row["sample_id"], []).append(row)
    averaged = []
    for sample_id, rows in sorted(by_sample.items()):
        result = {"sample_id": sample_id, "arms": ",".join(sorted(row["side"] for row in rows))}
        for metric in metric_names:
            values = [row["metrics"][metric] for row in rows
                      if metric in row["metrics"] and math.isfinite(row["metrics"][metric])]
            result[metric] = sum(values) / len(values) if values else "-"
        averaged.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    arm_csv = output_dir / "results_by_arm.csv"
    with arm_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "side", *metric_names])
        writer.writeheader()
        for row in sorted(side_rows, key=lambda x: (x["sample_id"], x["side"])):
            writer.writerow({
                "sample_id": row["sample_id"], "side": row["side"],
                **{name: row["metrics"].get(name, "-") for name in metric_names},
            })

    average_csv = output_dir / "results.csv"
    fields = ["sample_id", "arms", *metric_names]
    with average_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in averaged:
            writer.writerow(row)
        mean_row = {"sample_id": "AVERAGE", "arms": "per-sample arm mean"}
        for metric in metric_names:
            values = [row[metric] for row in averaged
                      if row[metric] != "-" and math.isfinite(row[metric])]
            mean_row[metric] = sum(values) / len(values) if values else "-"
        writer.writerow(mean_row)
    return average_csv, arm_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", required=True, type=Path)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--robotwin-dataset", type=Path, default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shared-gt-root", type=Path,
                        help="Reuse GT coordinates and crops across generated roots")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--point-type", choices=("tcp", "endpose"), default="tcp",
                        help="Project the RoboTwin TCP center (default) or saved endpose origin")
    parser.add_argument("--tcp-offset", type=float, default=0.12,
                        help="TCP offset in meters along local +X (default: RoboTwin 0.12 m)")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--task", help="Optional task filter")
    parser.add_argument("--episode-index", type=int, help="Optional source/generated episode filter")
    parser.add_argument("--metrics", help="Metric list forwarded to stack_and_evaluate")
    parser.add_argument("--trim-gt-to-generated", action="store_true",
                        help="Trim extra GT tail frames when evaluating the crops")
    parser.add_argument("--gpus", help="GPU IDs forwarded to the evaluator, e.g. 0,1")
    parser.add_argument("--processes-per-gpu", type=int, default=1)
    parser.add_argument("--config", type=Path, help="Optional video_quality config")
    parser.add_argument("--eval-python", help="Python executable with video_quality dependencies")
    parser.add_argument("--jepa-python", help="Python executable with videojedi dependencies")
    parser.add_argument("--eval-env", default="WorldArena",
                        help="Conda environment used for evaluation when --eval-python is omitted")
    parser.add_argument("--summary-only", action="store_true",
                        help="Prepare crops and stack_and_evaluate input summaries only")
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    generated_root = args.generated_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    robotwin_root = args.robotwin_dataset.expanduser().resolve()
    gt_assets_root = args.shared_gt_root.expanduser().resolve() if args.shared_gt_root else output_dir
    if output_dir.exists():
        parser.error(f"--output-dir must be a new directory: {output_dir}")
    for input_root in (generated_root, gt_root, robotwin_root):
        if output_dir.is_relative_to(input_root) or input_root.is_relative_to(output_dir):
            parser.error("--output-dir must be separate from input roots")
        if gt_assets_root.is_relative_to(input_root) or input_root.is_relative_to(gt_assets_root):
            parser.error("--shared-gt-root must be separate from input roots")

    gt_manifest = load_gt_manifest(gt_root)
    generated = index_generated_videos(generated_root)
    keys = sorted(generated)
    if args.task is not None:
        keys = [key for key in keys if key[0] == args.task]
    if args.episode_index is not None:
        keys = [key for key in keys if key[1] == args.episode_index]
    if not keys:
        parser.error("No generated videos match the selected task/episode filters")

    # Validate every pair before creating output files.
    pairs = []
    for index, (task, episode) in enumerate(keys, 1):
        key = (task, episode)
        if key not in gt_manifest:
            raise ValueError(f"No GT manifest match for task/episode {key}")
        gt_path, hdf5_path = source_for_pair(
            task, episode, gt_manifest[key], gt_root, robotwin_root
        )
        gt_info = video_info(gt_path)
        generated_info = video_info(generated[key])
        if (gt_info.width, gt_info.height) != (generated_info.width, generated_info.height):
            raise ValueError(f"Video dimensions differ for {key}: GT={gt_info}, generated={generated_info}")
        with h5py.File(hdf5_path, "r") as file:
            source_frames = len(file[f"observation/{args.camera}/rgb"])
        if gt_info.frames != source_frames:
            raise ValueError(f"GT/HDF5 frame count differs for {key}: {gt_info.frames}!={source_frames}")
        active_sides, activity = detect_active_grippers(hdf5_path, task)
        coordinates = {
            side: project_episode(
                hdf5_path, args.camera, side,
                (gt_info.width, gt_info.height),
                point_type=args.point_type, tcp_offset=args.tcp_offset,
            )
            for side in active_sides
        }
        pairs.append(Pair(task, episode, generated[key], gt_path, hdf5_path,
                          gt_info, generated_info,
                          active_sides, activity, coordinates))
        print(f"[match] {index}/{len(keys)} {task} episode{episode}: "
              f"active={','.join(active_sides)} point={args.point_type} "
              f"GT={gt_info.frames}@{gt_info.fps:g}fps "
              f"generated={generated_info.frames}@{generated_info.fps:g}fps", flush=True)

    output_dir.mkdir(parents=True)
    results = []
    for index, pair in enumerate(pairs, 1):
        print(f"[crop] {index}/{len(pairs)} {pair.task} episode{pair.episode}", flush=True)
        results.extend(prepare_pair(
            pair, output_dir, gt_assets_root, args.camera, args.width, args.height
        ))
    (output_dir / "crop_pairs.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    evaluation_sides = sorted({row["side"] for row in results})
    side_csvs = {}
    for side in evaluation_sides:
        command = []
        if args.eval_python:
            command = [args.eval_python]
        else:
            if not shutil.which("conda"):
                parser.error("Conda is required unless --eval-python is supplied")
            command = ["conda", "run", "--no-capture-output", "-n", args.eval_env, "python"]
        command += ["-m", "video_quality.stack_and_evaluate",
                    "--generated-root", str(output_dir / f"generated_crops_{side}"),
                    "--gt-root", str(gt_assets_root / f"gt_crops_{side}"),
                    "--output-dir", str(output_dir / f"evaluation_{side}"),
                    "--processes-per-gpu", str(args.processes_per_gpu)]
        if args.metrics:
            command += ["--metrics", args.metrics]
        if args.trim_gt_to_generated:
            command.append("--trim-gt-to-generated")
        if args.gpus:
            command += ["--gpus", args.gpus]
        if args.config:
            command += ["--config", str(args.config.expanduser().resolve())]
        if args.jepa_python:
            command += ["--jepa-python", args.jepa_python]
        if args.summary_only:
            command.append("--summary-only")
        print(f"[evaluate:{side}]", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        result_csv = output_dir / f"evaluation_{side}" / "results" / "results.csv"
        if result_csv.is_file():
            side_csvs[side] = result_csv
    if not args.summary_only:
        average_csv, arm_csv = write_average_results(output_dir, side_csvs)
        print(f"Per-arm metrics: {arm_csv}")
        print(f"Averaged metrics: {average_csv}")
    if args.summary_only:
        print(f"Prepared {len(results)} active-arm crop pairs and evaluation summaries in {output_dir}")
    else:
        print(f"Prepared and evaluated {len(pairs)} episodes across {len(evaluation_sides)} active-arm groups in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
