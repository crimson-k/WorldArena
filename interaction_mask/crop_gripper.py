"""Crop a video around a per-frame gripper point and pad out-of-frame regions."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def crop_video(video_path, coordinates_path, output_video, output_coordinates,
               side="left", width=160, height=120, draw_point=True,
               frame_limit=None, source_frame_indices=None, output_fps=None):
    video_path = Path(video_path)
    coordinates_path = Path(coordinates_path)
    output_video = Path(output_video)
    output_coordinates = Path(output_coordinates)
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if width <= 0 or height <= 0:
        raise ValueError("crop width and height must be positive")
    if output_video.exists() or output_coordinates.exists():
        raise FileExistsError(f"Output already exists: {output_video} or {output_coordinates}")

    annotations = json.loads(coordinates_path.read_text(encoding="utf-8"))
    records = annotations.get("frames")
    if not isinstance(records, list) or not records:
        raise ValueError(f"No frame coordinates in {coordinates_path}")
    if frame_limit is not None:
        if frame_limit <= 0 or frame_limit > len(records):
            raise ValueError(f"frame_limit must be between 1 and {len(records)}")
        records = records[:frame_limit]
    if source_frame_indices is None:
        source_frame_indices = list(range(len(records)))
    else:
        source_frame_indices = [int(index) for index in source_frame_indices]
        if len(source_frame_indices) != len(records):
            raise ValueError("source_frame_indices must match the coordinate frame count")
        if any(index < 0 for index in source_frame_indices):
            raise ValueError("source_frame_indices must be non-negative")
        if any(a > b for a, b in zip(source_frame_indices, source_frame_indices[1:])):
            raise ValueError("source_frame_indices must be non-decreasing")

    reader = imageio.get_reader(str(video_path), format="ffmpeg")
    writer = None
    try:
        metadata = reader.get_meta_data()
        source_fps = float(metadata.get("fps", 30.0))
        fps = float(output_fps) if output_fps is not None else source_fps
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid output fps: {fps}")
        frames = iter(reader)
        first = next(frames, None)
        if first is None:
            raise ValueError(f"Video has no frames: {video_path}")
        source_frame = first
        current_source_index = 0
        source_height, source_width = first.shape[:2]
        declared_size = annotations.get("image_size")
        if declared_size and list(declared_size) != [source_width, source_height]:
            raise ValueError(
                f"Video size {(source_width, source_height)} does not match coordinate image_size "
                f"{tuple(declared_size)}"
            )

        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_coordinates.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(output_video), fps=float(fps), codec="libx264rgb", pixelformat="rgb24",
            ffmpeg_params=["-crf", "0", "-preset", "fast"],
            macro_block_size=1, ffmpeg_log_level="error",
        )
        output_records = []
        for index, (record, source_frame_index) in enumerate(zip(records, source_frame_indices)):
            while current_source_index < source_frame_index:
                source_frame = next(frames, None)
                current_source_index += 1
                if source_frame is None:
                    raise ValueError(
                        f"Video ends before source frame {source_frame_index}; "
                        f"coordinates contain {len(records)} frames"
                    )
            frame = source_frame
            if frame.shape[:2] != (source_height, source_width):
                raise ValueError(f"Video dimensions changed at frame {index}")
            if record.get("frame_index", index) != index:
                raise ValueError(f"Nonsequential frame_index at coordinate row {index}")
            point = record.get(side, {}).get("uv")
            if (not isinstance(point, list) or len(point) != 2
                    or point[0] is None or point[1] is None
                    or not np.isfinite(point).all()):
                raise ValueError(f"Missing or invalid {side} point at frame {index}")
            u, v = map(float, point)

            # Pixel centers span [0, size - 1]. Choose the nearest integer
            # window origin so the point stays within half a pixel of center.
            left = int(np.rint(u - (width - 1) / 2))
            top = int(np.rint(v - (height - 1) / 2))
            right, bottom = left + width, top + height
            crop = np.zeros((height, width, 3), dtype=np.uint8)
            x0, y0 = max(0, left), max(0, top)
            x1, y1 = min(source_width, right), min(source_height, bottom)
            if x1 > x0 and y1 > y0:
                dx, dy = x0 - left, y0 - top
                crop[dy:dy + (y1 - y0), dx:dx + (x1 - x0)] = frame[y0:y1, x0:x1, :3]

            point_in_crop = [u - left, v - top]
            if draw_point:
                canvas = Image.fromarray(crop)
                draw = ImageDraw.Draw(canvas)
                cx, cy = point_in_crop
                radius = 3
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                             outline=(255, 255, 255), width=1)
                draw.ellipse((cx - 1, cy - 1, cx + 1, cy + 1), fill=(0, 255, 0))
                crop = np.asarray(canvas)
            writer.append_data(crop)
            output_records.append({
                "frame_index": index,
                "source_frame_index": source_frame_index,
                "side": side,
                "source_uv": [u, v],
                "crop_origin_xy": [left, top],
                "crop_uv": point_in_crop,
                "source_in_frame": bool(record[side].get("in_frame", False)),
                "source_image_size": [source_width, source_height],
                "crop_size": [width, height],
                "source_crop_bounds_xyxy": [x0, y0, x1, y1],
            })

        if frame_limit is None and source_frame_indices == list(range(len(records))) \
                and next(frames, None) is not None:
            raise ValueError("Video has more frames than coordinate JSON")
        writer.close()
        writer = None
        output = {
            "video_source": str(video_path.resolve()),
            "coordinates_source": str(coordinates_path.resolve()),
            "side": side,
            "crop_size": [width, height],
            "point_centered": True,
            "out_of_frame_fill": "black",
            "point_drawn_at_crop_center": draw_point,
            "fps": fps,
            "source_fps": source_fps,
            "frame_count": len(output_records),
            "frames": output_records,
        }
        output_coordinates.write_text(
            json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        if writer is not None:
            writer.close()
        output_video.unlink(missing_ok=True)
        output_coordinates.unlink(missing_ok=True)
        raise
    finally:
        reader.close()
    return output_video, output_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--coordinates", required=True, type=Path)
    parser.add_argument("--output-video", required=True, type=Path)
    parser.add_argument("--output-coordinates", type=Path,
                        help="Defaults to output-video with a .json suffix")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--no-draw-point", action="store_true",
                        help="Do not draw a marker at the crop center")
    args = parser.parse_args()
    output_coordinates = args.output_coordinates or args.output_video.with_suffix(".json")
    video, coordinates = crop_video(
        args.video, args.coordinates, args.output_video, output_coordinates,
        side=args.side, width=args.width, height=args.height,
        draw_point=not args.no_draw_point,
    )
    print(video)
    print(coordinates)


if __name__ == "__main__":
    main()
