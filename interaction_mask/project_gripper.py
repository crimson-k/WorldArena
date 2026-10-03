"""Project RoboTwin end-effector positions onto a camera's original RGB frames."""

import argparse
import io
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def project_points(xyz, intrinsic, extrinsic):
    """Return (pixels, camera_xyz, in_front) using CV world-to-camera matrices.

    xyz: (T, 3); intrinsic: (3, 3) or (T, 3, 3);
    extrinsic: (3, 4) or (T, 3, 4). Invalid pixels are NaN.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (T, 3)")
    count = len(xyz)
    matrices = []
    for value, shape in ((intrinsic, (3, 3)), (extrinsic, (3, 4))):
        value = np.asarray(value, dtype=np.float64)
        if value.shape == shape:
            value = np.broadcast_to(value, (count, *shape))
        if value.shape != (count, *shape):
            raise ValueError(f"Expected {shape} or {(count, *shape)}, got {value.shape}")
        matrices.append(value)
    intrinsic, extrinsic = matrices
    homogeneous = np.concatenate([xyz, np.ones((count, 1))], axis=1)
    camera_xyz = np.einsum("tij,tj->ti", extrinsic, homogeneous)
    projected = np.einsum("tij,tj->ti", intrinsic, camera_xyz)
    valid = (
        np.isfinite(camera_xyz).all(axis=1)
        & np.isfinite(projected).all(axis=1)
        & (camera_xyz[:, 2] > 1e-8)
        & (np.abs(projected[:, 2]) > 1e-8)
    )
    pixels = np.full((count, 2), np.nan)
    pixels[valid] = projected[valid, :2] / projected[valid, 2:3]
    return pixels, camera_xyz, valid


def point_xyz_from_endpose(endpose, point_type="tcp", tcp_offset=0.12):
    """Return world XYZ for the saved EE origin or RoboTwin's TCP center.

    RoboTwin stores get_*_ee_pose() in endpose. Its get_*_tcp_pose() applies
    a fixed offset along the pose's local +X axis, so the quaternion rotates
    that offset into world coordinates for every frame.
    """
    endpose = np.asarray(endpose, dtype=np.float64)
    if endpose.ndim != 2 or endpose.shape[1] != 7:
        raise ValueError("endpose must have shape (T, 7): xyz + quaternion [w, x, y, z]")
    if point_type == "endpose":
        return endpose[:, :3].copy()
    if point_type != "tcp":
        raise ValueError("point_type must be 'endpose' or 'tcp'")
    if not np.isfinite(tcp_offset):
        raise ValueError("tcp_offset must be finite")

    quaternion = endpose[:, 3:7]
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(norm <= 1e-12) or not np.isfinite(norm).all():
        raise ValueError("Encountered an invalid endpose quaternion")
    w, x, y, z = (quaternion / norm[:, None]).T
    rotation = np.empty((len(endpose), 3, 3), dtype=np.float64)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    local_offset = np.array([tcp_offset, 0.0, 0.0])
    world_offset = np.einsum("tij,j->ti", rotation, local_offset)
    return endpose[:, :3] + world_offset


def decode_rgb(dataset, index):
    """Decode RoboTwin JPEG bytes or an RGB array into an RGB array."""
    value = dataset[index]
    if isinstance(value, np.ndarray) and value.ndim == 3:
        if value.shape[2] != 3:
            raise ValueError(f"Expected an RGB image at frame {index}, got {value.shape}")
        return value
    encoded = value.tobytes() if hasattr(value, "tobytes") else bytes(value)
    with Image.open(io.BytesIO(encoded)) as image:
        return np.asarray(image.convert("RGB"))


def finite_list(array):
    return [float(x) if np.isfinite(x) else None for x in array]


def render_episode(hdf5_path, video_path, json_path, camera="head_camera", video=None, fps=None,
                   expected_size=None, point_type="tcp", tcp_offset=0.12):
    """Render one episode to explicit paths and return them."""
    hdf5_path, video_path, json_path = Path(hdf5_path), Path(video_path), Path(json_path)
    if json_path.exists() or video_path.exists():
        raise FileExistsError(f"Output already exists: {video_path} or {json_path}")
    reader = writer = None
    video_frames = None
    try:
        with h5py.File(hdf5_path, "r") as file:
            observation = file[f"observation/{camera}"]
            rgb = observation["rgb"]
            count = len(rgb)
            if not count:
                raise ValueError("RGB stream is empty")
            source_h, source_w = decode_rgb(rgb, 0).shape[:2]
            input_size = (source_w, source_h)
            if video is not None:
                reader = imageio.get_reader(str(video), format="ffmpeg")
                metadata = reader.get_meta_data()
                video_frames = iter(reader)
                try:
                    first = next(video_frames)
                except StopIteration as exc:
                    raise ValueError(f"Video has no readable frames: {video}") from exc
                height, width = first.shape[:2]
                if first.ndim != 3 or first.shape[2] < 3:
                    raise ValueError(f"Expected RGB video frames, got {first.shape}")
                first = first[:, :, :3]
                input_size = (width, height)
                output_fps = fps if fps is not None else metadata.get("fps", 30.0)
            else:
                first = decode_rgb(rgb, 0)
                height, width = first.shape[:2]
                output_fps = fps if fps is not None else 30.0
            if expected_size is not None and input_size != tuple(expected_size):
                raise ValueError(f"Expected video size {tuple(expected_size)}, got {input_size}: {video}")
            if not np.isfinite(output_fps) or output_fps <= 0:
                raise ValueError("FPS must be positive; specify --fps")

            annotations = {}
            for side in ("left", "right"):
                pose = file[f"endpose/{side}_endpose"][:]
                if pose.shape != (count, 7):
                    raise ValueError(f"{side} pose shape {pose.shape} does not match RGB length {count}")
                point_xyz = point_xyz_from_endpose(pose, point_type, tcp_offset)
                uv, camera_xyz, in_front = project_points(
                    point_xyz, observation["intrinsic_cv"][:], observation["extrinsic_cv"][:]
                )
                uv *= np.array([width / source_w, height / source_h])
                in_frame = (in_front & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                            & (uv[:, 1] >= 0) & (uv[:, 1] < height))
                annotations[side] = (point_xyz, pose[:, :3], camera_xyz, uv, in_front, in_frame)

            video_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(
                str(video_path), fps=float(output_fps), codec="libx264",
                macro_block_size=1, ffmpeg_log_level="error",
            )
            records = []
            for index in range(count):
                if reader is not None:
                    if index == 0:
                        frame = first.copy()
                    else:
                        try:
                            frame = next(video_frames)
                        except StopIteration as exc:
                            raise ValueError(
                                f"Video ends at frame {index}; HDF5 contains {count} frames"
                            ) from exc
                        frame = frame[:, :, :3]
                elif index == 0:
                    frame = first.copy()
                else:
                    frame = decode_rgb(rgb, index)
                if frame.shape[:2] != (height, width):
                    raise ValueError(f"Image dimensions changed at frame {index}")
                if frame.shape[2] != 3:
                    raise ValueError(f"Expected RGB frame at index {index}, got {frame.shape}")
                canvas = Image.fromarray(frame)
                draw = ImageDraw.Draw(canvas)
                font = ImageFont.load_default()
                record = {"frame_index": index}
                for side, color in (("left", (0, 220, 0)), ("right", (255, 140, 0))):
                    xyz, endpose_xyz, camera_xyz, uv, in_front, in_frame = annotations[side]
                    record[side] = {
                        "xyz": finite_list(xyz[index]),
                        "endpose_xyz": finite_list(endpose_xyz[index]),
                        "camera_xyz": finite_list(camera_xyz[index]),
                        "uv": finite_list(uv[index]),
                        "in_front": bool(in_front[index]),
                        "in_frame": bool(in_frame[index]),
                    }
                    if in_frame[index]:
                        x, y = np.rint(uv[index]).astype(int)
                        radius = max(2, round(min(width, height) / 48))
                        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                                     outline=color, width=max(1, radius // 2))
                        draw.text((x + radius + 2, max(0, y - radius - 10)), side[0].upper(),
                                  fill=color, font=font)
                writer.append_data(np.asarray(canvas))
                records.append(record)
            if video_frames is not None:
                try:
                    next(video_frames)
                except StopIteration:
                    pass
                else:
                    raise ValueError("Video has more frames than HDF5; temporal alignment is required")
            writer.close()
            writer = None
            result = {
                "hdf5": str(hdf5_path.resolve()), "camera": camera,
                "video_source": str(Path(video).resolve()) if video else f"observation/{camera}/rgb",
                "source_image_size": [source_w, source_h], "image_size": [width, height],
                "fps": output_fps, "frame_count": count,
                "point_type": point_type,
                "tcp_offset_local_xyz": [float(tcp_offset), 0.0, 0.0] if point_type == "tcp" else [0.0, 0.0, 0.0],
                "point_definition": (
                    f"RoboTwin TCP center: stored endpose origin plus {tcp_offset:g} m along local +X"
                    if point_type == "tcp" else "stored endpose origin"
                ),
                "projection": "camera_xyz = extrinsic_cv @ [xyz, 1]; uv = K @ camera_xyz, divided by depth",
                "pixel_convention": "origin top-left; u right; v down; scaled to output image",
                "visibility_note": "in_frame checks depth and bounds only, not occlusion",
                "frames": records,
            }
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        video_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        raise
    finally:
        if reader is not None:
            reader.close()
        if writer is not None:
            writer.close()
    return video_path, json_path


def annotate_episode(hdf5_path, output_dir, camera="head_camera", video=None, fps=None,
                     point_type="tcp", tcp_offset=0.12):
    """Write one episode's annotated MP4 and JSON coordinate sidecar."""
    hdf5_path, output_dir = Path(hdf5_path), Path(output_dir)
    stem = f"{hdf5_path.stem}_{camera}_gripper"
    return render_episode(
        hdf5_path, output_dir / f"{stem}.mp4", output_dir / f"{stem}.json",
        camera=camera, video=video, fps=fps, point_type=point_type, tcp_offset=tcp_offset,
    )


def annotate_converted_dataset(converted_dir, robotwin_dir, output_dir,
                               camera="head_camera", expected_size=(256, 192),
                               task_filter=None, episode_filter=None,
                               point_type="tcp", tcp_offset=0.12):
    """Annotate each unique resized video referenced by converted dataset manifests."""
    converted_dir, robotwin_dir, output_dir = map(Path, (converted_dir, robotwin_dir, output_dir))
    manifests = [converted_dir / "metadata_train.jsonl", converted_dir / "metadata_test.jsonl"]
    manifests = [path for path in manifests if path.is_file()]
    if not manifests:
        manifest = converted_dir / "metadata.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"No metadata manifest found in {converted_dir}")
        manifests = [manifest]

    episodes = {}
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as lines:
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                relative = Path(item["video"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe video path in {manifest}:{line_number}: {relative}")
                if len(relative.parts) < 4 or relative.parts[0] != "videos":
                    raise ValueError(f"Unexpected manifest video path: {relative}")
                key = relative.as_posix()
                identity = (item["task"], int(item["source_episode_index"]))
                if key in episodes and episodes[key][0] != identity:
                    raise ValueError(f"Conflicting episode references for {key}")
                episodes[key] = (identity, relative)

    if not episodes:
        raise ValueError("Dataset manifests contain no videos")
    if task_filter is not None:
        episodes = {
            key: value for key, value in episodes.items()
            if value[0][0] == task_filter
        }
    if episode_filter is not None:
        episodes = {
            key: value for key, value in episodes.items()
            if value[0][1] == episode_filter
        }
    if not episodes:
        raise ValueError("No videos match the requested task/episode filter")
    results = []
    for index, (identity, relative) in enumerate(sorted(episodes.values(), key=lambda x: x[1].as_posix()), 1):
        task, source_index = identity
        subset = relative.parts[2]
        source_hdf5 = robotwin_dir / task / subset / "data" / f"episode{source_index}.hdf5"
        source_video = converted_dir / relative
        destination_video = output_dir / relative
        destination_json = output_dir / "coordinates" / relative.with_suffix(".json").relative_to("videos")
        if not source_hdf5.is_file():
            raise FileNotFoundError(f"Missing RoboTwin episode: {source_hdf5}")
        if not source_video.is_file():
            raise FileNotFoundError(f"Missing converted video: {source_video}")
        if destination_video.exists() and destination_json.exists():
            print(f"[{index}/{len(episodes)}] skip existing {task} episode {source_index}", flush=True)
            continue
        if destination_video.exists() or destination_json.exists():
            raise FileExistsError(
                f"Only one output exists for {task} episode {source_index}; "
                f"remove or complete {destination_video} and {destination_json}"
            )
        print(f"[{index}/{len(episodes)}] {task} episode {source_index}", flush=True)
        results.append(render_episode(
            source_hdf5, destination_video, destination_json,
            camera=camera, video=source_video, expected_size=expected_size,
            point_type=point_type, tcp_offset=tcp_offset,
        ))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hdf5", type=Path, help="Annotate a single RoboTwin HDF5 episode")
    source.add_argument("--converted-dataset", type=Path,
                        help="Annotate all videos referenced by metadata_train/test.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--point-type", choices=("tcp", "endpose"), default="tcp",
                        help="Point to project; TCP is the RoboTwin gripper center")
    parser.add_argument("--tcp-offset", type=float, default=0.12,
                        help="TCP offset in meters along local +X (default: RoboTwin 0.12 m)")
    parser.add_argument("--video", type=Path, help="Optional matching camera video; defaults to HDF5 RGB")
    parser.add_argument("--fps", type=float, help="Output FPS; default: input video FPS, or 30 for HDF5")
    parser.add_argument("--robotwin-dataset", type=Path,
                        default=Path("/data1/common_data/RoboTwin2.0/dataset"),
                        help="Source RoboTwin dataset root for --converted-dataset")
    parser.add_argument("--width", type=int, default=256, help="Expected batch video width")
    parser.add_argument("--height", type=int, default=192, help="Expected batch video height")
    parser.add_argument("--task", help="Limit converted-dataset mode to one task")
    parser.add_argument("--episode-index", type=int,
                        help="Limit converted-dataset mode to one source episode index")
    args = parser.parse_args()
    if args.converted_dataset is not None:
        if args.video is not None or args.fps is not None:
            parser.error("--video and --fps apply only with --hdf5")
        paths = annotate_converted_dataset(
            args.converted_dataset, args.robotwin_dataset, args.output_dir,
            camera=args.camera, expected_size=(args.width, args.height),
            task_filter=args.task, episode_filter=args.episode_index,
            point_type=args.point_type, tcp_offset=args.tcp_offset,
        )
        print(f"Annotated {len(paths)} episode videos.")
        print(f"Outputs: {args.output_dir / 'videos'} and {args.output_dir / 'coordinates'}")
    else:
        paths = annotate_episode(
            args.hdf5, args.output_dir, args.camera, args.video, args.fps,
            point_type=args.point_type, tcp_offset=args.tcp_offset,
        )
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
