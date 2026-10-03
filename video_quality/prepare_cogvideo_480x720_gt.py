"""Create spatially matched GT videos for the CogVideo 480x720 evaluation."""

from __future__ import annotations

from pathlib import Path
import math
import shutil
import subprocess

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_ROOT = PROJECT_ROOT / "inferred_results/CogVideo_480x720"
DATASET_ROOT = Path("/data1/fangxuebin/boundless-world-model/converted_dataset_bwm_6tasks/videos")
OUTPUT_ROOT = PROJECT_ROOT / "evaluation_runs/inferred_all_metrics/CogVideo_480x720/gt_videos"
WIDTH, HEIGHT = 720, 480


def video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required; run this script in the WorldArena environment")

    generated_videos = sorted(GENERATED_ROOT.rglob("*.mp4"))
    if not generated_videos:
        raise FileNotFoundError(f"No generated MP4 videos found under {GENERATED_ROOT}")

    for index, generated in enumerate(generated_videos, 1):
        relative = generated.relative_to(GENERATED_ROOT)
        task = relative.parts[-2]
        source = DATASET_ROOT / task / "aloha-agilex_clean_50" / generated.name
        if not source.is_file():
            raise FileNotFoundError(f"No matching GT for {generated}: {source}")

        source_info = video_info(source)
        if source_info["width"] <= 0 or source_info["height"] <= 0:
            raise ValueError(f"Invalid source size for {source}: {source_info}")
        scaled_height = round(source_info["height"] * WIDTH / source_info["width"])
        if scaled_height < HEIGHT:
            raise ValueError(f"Width resize cannot fill a {HEIGHT}-pixel crop for {source}: {source_info}")

        target = OUTPUT_ROOT / task / generated.name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-vf", f"scale={WIDTH}:-2:flags=lanczos,crop={WIDTH}:{HEIGHT}:(iw-ow)/2:(ih-oh)/2",
            "-an", "-c:v", "libx264", "-crf", "0", "-preset", "fast",
            "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", str(temporary),
        ]
        subprocess.run(command, check=True)
        target_info = video_info(temporary)
        if (target_info["width"], target_info["height"]) != (WIDTH, HEIGHT):
            raise ValueError(f"Unexpected processed size for {temporary}: {target_info}")
        if target_info["frames"] != source_info["frames"]:
            raise ValueError(f"Frame count changed for {source}: {source_info} -> {target_info}")
        if not math.isclose(target_info["fps"], source_info["fps"], rel_tol=1e-4):
            raise ValueError(f"FPS changed for {source}: {source_info} -> {target_info}")
        temporary.replace(target)
        print(f"[{index}/{len(generated_videos)}] {task}/{generated.name} {target_info}", flush=True)

    print(f"Prepared {len(generated_videos)} GT videos in {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
