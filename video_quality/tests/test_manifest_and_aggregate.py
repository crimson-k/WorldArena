import csv
import json
import math
from pathlib import Path
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
from video_quality.runner import _prepare_missing_trajectories, run_evaluate, run_jepa


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


if __name__ == "__main__":
    unittest.main()
