import csv
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import sys

import cv2
import numpy as np

from video_quality.aggregate import aggregate
from video_quality.constants import normalize_worldarena
from video_quality.manifest import Sample, atomic_write_json, load_manifest, prepare
from video_quality.runner import (
    _checkpoint,
    _prepare_missing_trajectories,
    run_distributed_evaluate,
    run_evaluate,
    run_jepa,
)
from video_quality.timing import MetricTimingLogger, format_duration


def write_video(path: Path, value: int, frames: int = 3):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 24)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV test video writer is unavailable")
    for _ in range(frames):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()


def write_summary(root: Path, gt_value: int = 20, generated_value: int = 30) -> Path:
    gt = root / "episode0.mp4"
    generated = root / "generated.mp4"
    write_video(gt, gt_value)
    write_video(generated, generated_value)
    summary = root / "summary.json"
    summary.write_text(
        json.dumps([{"gt_path": str(gt), "generated_video": str(generated)}]),
        encoding="utf-8",
    )
    return summary


def write_stacked_video(path: Path, top_value: int, bottom_value: int, frames: int = 3):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (32, 48)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV test video writer is unavailable")
    for _ in range(frames):
        top = np.full((24, 32, 3), top_value, dtype=np.uint8)
        bottom = np.full((24, 32, 3), bottom_value, dtype=np.uint8)
        writer.write(np.concatenate([top, bottom], axis=0))
    writer.release()


class ManifestTests(unittest.TestCase):
    def test_prepare_uses_separate_png_and_jpeg_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = prepare(
                write_summary(root), root / "out", ["psnr", "image_quality"]
            )
            metrics, samples = load_manifest(manifest)
            self.assertEqual(metrics, ["psnr", "image_quality"])
            sample = samples[0]
            self.assertEqual(len(list(Path(sample.gt_png_frames).glob("*.png"))), 3)
            self.assertEqual(
                len(list(Path(sample.generated_png_frames).glob("*.png"))), 3
            )
            self.assertEqual(len(list(Path(sample.gt_frames).glob("*.jpg"))), 3)
            self.assertEqual(len(list(Path(sample.generated_frames).glob("*.jpg"))), 3)
            payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
            self.assertNotIn("gt_sha256", payload["samples"][0])
            self.assertFalse(any((root / "out").rglob("*.metadata.json")))

    def test_prepare_crops_vertical_stacked_video_without_reencoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "stacked.mp4"
            write_stacked_video(video, 20, 180)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "task__episode_000040",
                            "stacked_video": str(video),
                        }
                    ]
                ),
                encoding="utf-8",
            )

            manifest = prepare(
                summary, root / "out", ["psnr", "image_quality"]
            )
            metrics, samples = load_manifest(manifest)
            self.assertEqual(metrics, ["psnr", "image_quality"])
            sample = samples[0]
            self.assertEqual(sample.sample_id, "task__episode_000040")
            self.assertEqual(sample.layout, "vertical_gt_top")
            gt = cv2.imread(str(Path(sample.gt_png_frames) / "frame_00000.png"))
            generated = cv2.imread(
                str(Path(sample.generated_png_frames) / "frame_00000.png")
            )
            self.assertEqual(gt.shape, (24, 32, 3))
            self.assertEqual(generated.shape, (24, 32, 3))
            self.assertLess(float(gt.mean()), float(generated.mean()))
            self.assertEqual(len(list(Path(sample.gt_frames).glob("*.jpg"))), 3)
            self.assertEqual(
                len(list(Path(sample.generated_frames).glob("*.jpg"))), 3
            )

    def test_duplicate_gt_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a" / "same.mp4"
            second = root / "b" / "same.mp4"
            first.parent.mkdir()
            second.parent.mkdir()
            generated_a = root / "ga.mp4"
            generated_b = root / "gb.mp4"
            for path in (first, second, generated_a, generated_b):
                write_video(path, 0)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    [
                        {"gt_path": str(first), "generated_video": str(generated_a)},
                        {"gt_path": str(second), "generated_video": str(generated_b)},
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
                prepare(summary, root / "out", ["psnr"])


class TimingLogTests(unittest.TestCase):
    def test_metric_timing_writes_text_and_structured_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logger = MetricTimingLogger(root, root / "manifest.json", ["psnr"], 4)
            logger.start_stage(["psnr", "ssim"])
            logger.finish_stage()
            logger.finish_run()

            payload = json.loads(
                (root / "logs" / "metric_timings.json").read_text(encoding="utf-8")
            )
            text = (root / "logs" / "metric_timings.log").read_text(
                encoding="utf-8"
            )
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["world_size"], 4)
            self.assertEqual(payload["stages"][0]["metrics"], ["psnr", "ssim"])
            self.assertIn("START psnr+ssim", text)
            self.assertIn("RUN END", text)
            self.assertEqual(format_duration(3661.2), "01:01:01")


class AggregateTests(unittest.TestCase):
    def test_independent_metric_files_and_average(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "run_manifest.json"
            atomic_write_json(
                manifest,
                {
                    "version": 2,
                    "summary_json": "/unused",
                    "metrics": ["psnr", "ssim"],
                    "samples": [
                        {
                            "sample_id": sample_id,
                            "gt_path": "/unused",
                            "generated_video": "/unused",
                            "gt_frames": None,
                            "generated_frames": None,
                            "gt_png_frames": None,
                            "generated_png_frames": None,
                        }
                        for sample_id in ("a", "b")
                    ],
                },
            )
            atomic_write_json(
                root / "results" / "metrics" / "psnr.json",
                {"metric": "psnr", "values": {"a": 10, "b": 30}},
            )
            atomic_write_json(
                root / "results" / "metrics" / "ssim.json",
                {"metric": "ssim", "values": {"a": 0.5, "b": 0.9}},
            )
            csv_path = aggregate(manifest, root, ["psnr", "ssim"])
            with csv_path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[-1]["sample_id"], "AVERAGE")
            self.assertEqual(rows[-1]["PSNR"], "20.000000")
            self.assertEqual(rows[-1]["SSIM"], "0.700000")

    def test_aggregate_merges_metrics_from_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "run_manifest.json"
            atomic_write_json(
                manifest,
                {
                    "version": 2,
                    "summary_json": "/unused",
                    "metrics": ["image_quality"],
                    "samples": [
                        {
                            "sample_id": sample_id,
                            "gt_path": "/unused",
                            "generated_video": "/unused",
                            "gt_frames": None,
                            "generated_frames": None,
                            "gt_png_frames": None,
                            "generated_png_frames": None,
                        }
                        for sample_id in ("a", "b")
                    ],
                },
            )
            atomic_write_json(
                root / "results" / "results.json",
                {
                    "rows": [
                        {"sample_id": "a", "PSNR": 10.0},
                        {"sample_id": "b", "PSNR": 20.0},
                    ],
                    "average": {"sample_id": "AVERAGE", "PSNR": 15.0},
                },
            )
            atomic_write_json(
                root / "results" / "metrics" / "image_quality.json",
                {"metric": "image_quality", "values": {"a": 0.4, "b": 0.6}},
            )

            csv_path = aggregate(manifest, root, ["image_quality"])
            payload = json.loads((root / "results" / "results.json").read_text())
            self.assertEqual(set(payload["rows"][0]), {"sample_id", "PSNR", "Image Quality"})
            self.assertEqual(payload["average"]["PSNR"], 15.0)
            self.assertEqual(payload["average"]["Image Quality"], 0.5)
            with csv_path.open(encoding="utf-8", newline="") as stream:
                header = next(csv.reader(stream))
            self.assertEqual(header, ["sample_id", "PSNR", "Image Quality"])

    def test_worldarena_normalization(self):
        self.assertEqual(normalize_worldarena("trajectory_accuracy", 40.8540), 1.0)
        self.assertEqual(normalize_worldarena("trajectory_accuracy", -1), 0.0)
        self.assertEqual(normalize_worldarena("depth_accuracy", 0.2228), 1.0)
        self.assertEqual(normalize_worldarena("depth_accuracy", 4.3711), 0.0)


class ReferencePipelineTests(unittest.TestCase):
    def test_psnr_ssim_call_original_worldarena_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            manifest = prepare(
                write_summary(root, gt_value=40, generated_value=40),
                output,
                ["psnr", "ssim"],
            )
            config = root / "config.yaml"
            config.write_text("ckpt: {}\n", encoding="utf-8")
            run_evaluate(manifest, output, ["psnr", "ssim"], config)
            payload = json.loads(
                (output / "results" / "metrics" / "psnr.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(math.isinf(payload["values"]["episode0"]))


class DistributedEvaluationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("WORLD_ARENA_RUN_DISTRIBUTED_TESTS") == "1",
        "set WORLD_ARENA_RUN_DISTRIBUTED_TESTS=1 to open a local process-group port",
    )
    def test_two_workers_shard_and_gather_psnr_ssim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_rows = []
            for index, value in enumerate((30, 60)):
                gt = root / f"episode{index}.mp4"
                generated = root / f"generated{index}.mp4"
                write_video(gt, value)
                write_video(generated, value)
                summary_rows.append(
                    {"gt_path": str(gt), "generated_video": str(generated)}
                )
            summary = root / "summary.json"
            summary.write_text(json.dumps(summary_rows), encoding="utf-8")
            output = root / "output"
            manifest = prepare(summary, output, ["psnr", "ssim"])
            config = root / "config.yaml"
            config.write_text("ckpt: {}\n", encoding="utf-8")

            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    "2",
                    "-m",
                    "video_quality.cli",
                    "evaluate",
                    "--manifest",
                    str(manifest),
                    "--output-dir",
                    str(output),
                    "--config",
                    str(config),
                    "--metrics",
                    "psnr,ssim",
                ],
                check=True,
                env=environment,
                cwd=str(Path(__file__).resolve().parents[2]),
            )
            for metric in ("psnr", "ssim"):
                payload = json.loads(
                    (output / "results" / "metrics" / f"{metric}.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(set(payload["values"]), {"episode0", "episode1"})

    @patch("video_quality.runner.subprocess.run")
    def test_gpu_launcher_builds_one_worker_per_selected_gpu(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "run_manifest.json"
            atomic_write_json(
                manifest,
                {
                    "version": 2,
                    "summary_json": "/unused",
                    "metrics": ["psnr"],
                    "samples": [
                        {
                            "sample_id": f"sample{index}",
                            "gt_path": "/unused",
                            "generated_video": "/unused",
                            "gt_frames": None,
                            "generated_frames": None,
                            "gt_png_frames": "/unused",
                            "generated_png_frames": "/unused",
                        }
                        for index in range(3)
                    ],
                },
            )
            config = root / "config.yaml"
            config.write_text("ckpt: {}\n", encoding="utf-8")
            run_distributed_evaluate(
                manifest, root / "output", ["psnr"], config, [2, 4, 7]
            )
            command = run.call_args.args[0]
            self.assertIn("torch.distributed.run", command)
            self.assertEqual(command[command.index("--nproc_per_node") + 1], "3")
            self.assertEqual(
                run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,4,7"
            )


    @patch("video_quality.runner.subprocess.run")
    def test_gpu_launcher_supports_multiple_workers_per_gpu(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "run_manifest.json"
            atomic_write_json(
                manifest,
                {
                    "version": 2,
                    "summary_json": "/unused",
                    "metrics": ["psnr"],
                    "samples": [
                        {
                            "sample_id": f"sample{index}",
                            "gt_path": "/unused",
                            "generated_video": "/unused",
                            "gt_frames": None,
                            "generated_frames": None,
                            "gt_png_frames": "/unused",
                            "generated_png_frames": "/unused",
                        }
                        for index in range(4)
                    ],
                },
            )
            config = root / "config.yaml"
            config.write_text("ckpt: {}\n", encoding="utf-8")
            run_distributed_evaluate(
                manifest,
                root / "output",
                ["psnr"],
                config,
                [2, 4],
                processes_per_gpu=2,
            )
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--nproc_per_node") + 1], "4")
            self.assertEqual(
                run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,4"
            )


class RelativeConfigPathTests(unittest.TestCase):
    def test_checkpoint_paths_are_relative_to_project_root(self):
        resolved = _checkpoint({"ckpt": {"readme": "README.md"}}, "ckpt", "readme")
        self.assertEqual(
            Path(resolved), Path(__file__).resolve().parents[2] / "README.md"
        )


class TrajectoryReuseTests(unittest.TestCase):
    def test_existing_both_sides_do_not_import_or_run_sam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_frames = root / "gt" / "episode0" / "video"
            generated_frames = root / "generated" / "episode0" / "1" / "video"
            gt_frames.mkdir(parents=True)
            generated_frames.mkdir(parents=True)
            (gt_frames.parent / "traj").mkdir()
            (generated_frames.parent / "traj").mkdir()
            np.save(gt_frames.parent / "traj" / "traj.npy", np.zeros((2, 2, 2)))
            np.save(
                generated_frames.parent / "traj" / "traj.npy", np.zeros((2, 2, 2))
            )
            sample = Sample(
                "episode0", "/unused", "/unused", str(gt_frames), str(generated_frames)
            )
            _prepare_missing_trajectories([sample], {"ckpt": {}})

    def test_only_missing_side_runs_sam(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt_frames = root / "gt" / "episode0" / "video"
            generated_frames = root / "generated" / "episode0" / "1" / "video"
            gt_frames.mkdir(parents=True)
            generated_frames.mkdir(parents=True)
            (gt_frames.parent / "traj").mkdir()
            np.save(gt_frames.parent / "traj" / "traj.npy", np.zeros((2, 2, 2)))
            model = root / "sam"
            model.mkdir()
            calls = []

            class Detector:
                def __init__(self, model_path):
                    self.model_path = model_path

            def process(**kwargs):
                calls.append(kwargs)
                return True

            fake_module = types.SimpleNamespace(
                GripperDetector=Detector, process_video_with_tracking=process
            )
            sample = Sample(
                "episode0", "/unused", "/unused", str(gt_frames), str(generated_frames)
            )
            with patch.dict(
                sys.modules,
                {"video_quality.processing.detection_tracking": fake_module},
            ):
                _prepare_missing_trajectories(
                    [sample], {"ckpt": {"sam3_model_ckpt": str(model)}}
                )
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["data_type"], "val")


class JepaWrapperTests(unittest.TestCase):
    @patch("video_quality.runner.subprocess.run")
    def test_jepa_uses_manual_directories_without_manifest_staging(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            generated = root / "generated"
            model = root / "model"
            real.mkdir()
            generated.mkdir()
            model.mkdir()
            jepa_config = root / "jepa.yaml"
            jepa_config.write_text("pretrain: {}\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(
                "ckpt:\n  jepa_similarity:\n"
                f"    model_dir: {model}\n    config: {jepa_config}\n",
                encoding="utf-8",
            )
            run_jepa(real, generated, root / "out", config, "jepa-python")
            command = run.call_args.args[0]
            self.assertIn(str(real.resolve()), command)
            self.assertIn(str(generated.resolve()), command)
            self.assertFalse((root / "out" / "cache" / "jepa_pairs").exists())

    @patch("video_quality.runner.subprocess.run")
    def test_jepa_accepts_stacked_summary(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            summary = root / "summary.json"
            summary.write_text("[]\n", encoding="utf-8")
            jepa_config = root / "jepa.yaml"
            jepa_config.write_text("pretrain: {}\n", encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(
                "ckpt:\n  jepa_similarity:\n"
                f"    model_dir: {model}\n    config: {jepa_config}\n",
                encoding="utf-8",
            )
            run_jepa(
                None,
                None,
                root / "out",
                config,
                "jepa-python",
                stacked_summary=summary,
            )
            command = run.call_args.args[0]
            self.assertIn("--stacked_summary", command)
            self.assertIn(str(summary.resolve()), command)


if __name__ == "__main__":
    unittest.main()
