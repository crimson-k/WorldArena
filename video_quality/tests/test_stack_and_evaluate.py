import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from video_quality.stack_and_evaluate import evaluate, index_videos, main, merge_results, plan_pairs


def write_video(path, values, fps=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("Test encoder unavailable")
    for value in values:
        frame = np.full((24, 32, 3), value, np.uint8)
        frame[5:15, 7:19] = (10, 70, 150)
        writer.write(frame)
    writer.release()


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


class VideoEvaluationTests(unittest.TestCase):
    def test_generated_only_writes_summary_without_video_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated" / "task"
            generated.mkdir(parents=True)
            (generated / "episode_000016.mp4").touch()
            output = root / "out"
            main([
                "--generated-root", str(root / "generated"),
                "--output-dir", str(output),
                "--metrics", "psnr,image_quality,depth_accuracy",
                "--skip-gt-metrics", "--summary-only",
            ])
            summary = json.loads((output / "generated_summary.json").read_text())
            self.assertEqual(summary, [{
                "sample_id": "task__episode16",
                "generated_video": str(generated / "episode_000016.mp4"),
            }])
            self.assertFalse((output / "stacked_videos").exists())

    def test_mismatched_metadata_is_marked_without_failing(self):
        generated, gt = {("task", 0): Path("gen")}, {("task", 0): Path("gt")}
        info = dict(width=32, height=24, frames=3, fps=8.0)
        for changed in ({"width": 64}, {"frames": 2}, {"fps": 16.0}):
            with patch("video_quality.stack_and_evaluate.video_info", side_effect=[info, {**info, **changed}]):
                self.assertFalse(plan_pairs(generated, gt, False)[0]["aligned"])

    def test_matching_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_video(root / "task_a" / "camera" / "episode_000002.mp4", [30])
            write_video(root / "task_b" / "episode2.mp4", [40])
            self.assertEqual(set(index_videos(root)), {("task_a", 2), ("task_b", 2)})
            write_video(root / "task_a" / "episode2.mp4", [50])
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                index_videos(root)

    def test_missing_gt_still_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing GT"):
            plan_pairs({("task", 0): Path("gen")}, {}, False)

    def test_matching_pair_evaluates_separate_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt" / "task" / "episode0.mp4"
            generated = root / "generated" / "task" / "episode0.mp4"
            write_video(gt, [40, 80, 120])
            write_video(generated, [50, 90, 130])
            output = root / "out"
            main(["--gt-root", str(root / "gt"), "--generated-root", str(root / "generated"),
                  "--output-dir", str(output), "--metrics", "psnr,ssim"])
            self.assertFalse((output / "stacked_videos").exists())
            summary = json.loads((output / "video_summary.json").read_text())
            self.assertEqual(summary[0]["gt_path"], str(gt))
            self.assertEqual(summary[0]["generated_video"], str(generated))
            a, b = read_frames(gt), read_frames(generated)
            expected = {
                "psnr": np.mean([peak_signal_noise_ratio(x, y) for x, y in zip(a, b)]),
                "ssim": np.mean([structural_similarity(x, y, channel_axis=-1) for x, y in zip(a, b)]),
            }
            for metric, score in expected.items():
                result = json.loads((output / f"results/metrics/{metric}.json").read_text())
                self.assertAlmostEqual(result["values"]["task__episode0"], score, places=10)

    def test_mismatched_pair_gets_dash_and_is_not_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for episode, gt_values, generated_values in [
                (0, [40, 80, 120], [50, 90, 130]),
                (1, [20, 40, 60], [25, 45]),
                (2, [30, 60], [35, 65]),
            ]:
                gt_fps = 16 if episode == 2 else 8
                write_video(root / "gt" / "task" / f"episode{episode}.mp4", gt_values, gt_fps)
                write_video(root / "generated" / "task" / f"episode{episode}.mp4", generated_values)
            output = root / "out"
            main(["--gt-root", str(root / "gt"), "--generated-root", str(root / "generated"),
                  "--output-dir", str(output), "--metrics", "psnr,ssim"])
            with (output / "results/results.csv").open(newline="") as stream:
                rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
            self.assertNotEqual(rows["task__episode0"]["PSNR"], "-")
            for episode in (1, 2):
                self.assertEqual(rows[f"task__episode{episode}"]["PSNR"], "-")
                self.assertEqual(rows[f"task__episode{episode}"]["SSIM"], "-")
            self.assertEqual(rows["AVERAGE"]["PSNR"], rows["task__episode0"]["PSNR"])
            self.assertFalse((output / "stacked_videos").exists())
            gt_results = json.loads((output / "results/metrics/psnr.json").read_text())
            self.assertEqual(set(gt_results["values"]), {"task__episode0"})
            self.assertEqual({path.name for path in output.iterdir()},
                             {"video_summary.json", "video_pairs.json", "run_manifest.json",
                              "cache", "logs", "results"})
            self.assertFalse(any("stack" in path.name.lower() for path in output.rglob("*")))

    @unittest.skipUnless(Path(sys.executable).with_name("ffmpeg").is_file(), "FFmpeg unavailable")
    def test_trim_only_extra_gt_tail_and_keep_other_mismatches_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                (0, [40, 80, 120, 160], [50, 90, 130], 8),
                (1, [20, 40, 60], [25, 45], 16),
                (2, [30, 60], [35, 65, 95], 8),
            ]
            for episode, gt_values, generated_values, gt_fps in cases:
                write_video(root / "gt/task" / f"episode{episode}.mp4", gt_values, gt_fps)
                write_video(root / "generated/task" / f"episode{episode}.mp4", generated_values)
            output = root / "out"
            main(["--gt-root", str(root / "gt"), "--generated-root", str(root / "generated"),
                  "--output-dir", str(output), "--metrics", "psnr,ssim",
                  "--trim-gt-to-generated", "--ffmpeg",
                  str(Path(sys.executable).with_name("ffmpeg"))])
            original = read_frames(root / "gt/task/episode0.mp4")
            trimmed_path = output / "cache/trimmed_gt_videos/task__episode0.mp4"
            trimmed = read_frames(trimmed_path)
            self.assertEqual(len(trimmed), 3)
            for before, after in zip(original[:3], trimmed):
                np.testing.assert_array_equal(after, before)
            self.assertFalse((output / "cache/trimmed_gt_videos/task__episode1.mp4").exists())
            self.assertFalse((output / "cache/trimmed_gt_videos/task__episode2.mp4").exists())
            self.assertFalse((output / "stacked_videos").exists())
            with (output / "results/results.csv").open(newline="") as stream:
                rows = {row["sample_id"]: row for row in csv.DictReader(stream)}
            self.assertNotEqual(rows["task__episode0"]["PSNR"], "-")
            for episode in (1, 2):
                self.assertEqual(rows[f"task__episode{episode}"]["PSNR"], "-")
                self.assertEqual(rows[f"task__episode{episode}"]["SSIM"], "-")
            summary = json.loads((output / "cache/aligned_summary.json").read_text())
            self.assertEqual(summary[0]["gt_path"], str(trimmed_path))
            self.assertTrue((output / "video_summary.json").is_file())

    def test_all_mismatched_pairs_have_dash_without_running_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_video(root / "gt/task/episode0.mp4", [20, 40, 60])
            write_video(root / "generated/task/episode0.mp4", [25, 45])
            output = root / "out"
            with patch("video_quality.stack_and_evaluate.subprocess.run") as run:
                main(["--gt-root", str(root / "gt"), "--generated-root", str(root / "generated"),
                      "--output-dir", str(output), "--metrics", "psnr"])
            run.assert_not_called()
            result = json.loads((output / "results/results.json").read_text())
            self.assertEqual(result["rows"][0]["PSNR"], "-")
            self.assertEqual(result["average"]["PSNR"], "-")

    def test_merge_keeps_non_basic_metrics_for_mismatched_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated_dir = root / "generated_metrics/results"
            gt_dir = root / "gt_metrics/results"
            generated_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            (generated_dir / "results.json").write_text(json.dumps({"rows": [
                {"sample_id": "a", "Image Quality": 1.0, "JEPA Similarity": 0.7,
                 "Trajectory Accuracy": 0.8, "Depth Accuracy": 0.9},
                {"sample_id": "b", "Image Quality": 2.0, "JEPA Similarity": 0.6,
                 "Trajectory Accuracy": 0.5, "Depth Accuracy": 0.4},
            ]}))
            (gt_dir / "results.json").write_text(json.dumps({"rows": [
                {"sample_id": "a", "PSNR": 30.0},
            ]}))
            (gt_dir / "metrics").mkdir()
            (gt_dir / "metrics/psnr.json").write_text(json.dumps({
                "metric": "psnr", "values": {"a": 30.0},
            }))
            merge_results([{"sample_id": "a", "aligned": True},
                           {"sample_id": "b", "aligned": False}],
                          ["psnr", "image_quality", "jepa_similarity",
                           "trajectory_accuracy", "depth_accuracy"], root,
                          root / "generated_metrics", root / "gt_metrics")
            result = json.loads((root / "results/results.json").read_text())
            self.assertEqual(result["rows"][1],
                             {"sample_id": "b", "PSNR": "-", "Image Quality": 2.0,
                              "JEPA Similarity": 0.6, "Trajectory Accuracy": 0.5,
                              "Depth Accuracy": 0.4})
            self.assertEqual(result["average"]["PSNR"], 30.0)
            self.assertTrue((root / "results/metrics/psnr.json").is_file())

    def test_mismatched_pairs_still_run_jepa_trajectory_and_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for episode, count in ((0, 3), (1, 2)):
                write_video(root / "gt/task" / f"episode{episode}.mp4", [20, 40, 60])
                write_video(root / "generated/task" / f"episode{episode}.mp4", [25, 45, 65][:count])
            with patch("video_quality.stack_and_evaluate.evaluate") as run, \
                 patch("video_quality.stack_and_evaluate.merge_results"):
                main(["--gt-root", str(root / "gt"),
                      "--generated-root", str(root / "generated"),
                      "--output-dir", str(root / "out"),
                      "--metrics", "psnr,ssim,jepa_similarity,trajectory_accuracy,depth_accuracy"])
            self.assertEqual(run.call_count, 2)
            self.assertEqual({row["sample_id"] for row in run.call_args_list[0].args[0]},
                             {"task__episode0", "task__episode1"})
            self.assertEqual(run.call_args_list[0].args[1],
                             ["jepa_similarity", "trajectory_accuracy", "depth_accuracy"])
            self.assertEqual({row["sample_id"] for row in run.call_args_list[1].args[0]},
                             {"task__episode0"})
            self.assertEqual(run.call_args_list[1].args[1], ["psnr", "ssim"])

    def test_jepa_uses_matching_links_to_original_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt, generated = root / "original_gt.mp4", root / "original_generated.mp4"
            gt.touch()
            generated.touch()
            output = root / "out"
            args = SimpleNamespace(config=root / "config.yaml", gpus=None,
                                   processes_per_gpu=1, jepa_python="python")
            with patch("video_quality.stack_and_evaluate.subprocess.run") as run:
                evaluate([{"sample_id": "task__episode0", "gt_path": str(gt),
                           "generated_video": str(generated)}],
                         ["jepa_similarity"], output, args)
            command = run.call_args.args[0]
            self.assertIn("--jepa-real-dir", command)
            self.assertIn("--jepa-gen-dir", command)
            self.assertNotIn("--jepa-stacked-summary", command)
            self.assertEqual((output / "cache/jepa_pairs/gt/task__episode0.mp4").resolve(), gt)
            self.assertEqual((output / "cache/jepa_pairs/generated/task__episode0.mp4").resolve(), generated)


if __name__ == "__main__":
    unittest.main()
