import math
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from video_quality.WorldArena.basic_metrics import compute_basic_metrics


class WorldArenaBasicMetricTests(unittest.TestCase):
    def test_original_basic_metrics_on_png_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "gt" / "default" / "episode0" / "video"
            generated = root / "generated" / "default" / "episode0" / "1" / "video"
            gt.mkdir(parents=True)
            generated.mkdir(parents=True)
            frame = np.full((16, 16, 3), 80, dtype=np.uint8)
            cv2.imwrite(str(gt / "frame_00000.png"), frame)
            cv2.imwrite(str(generated / "frame_00000.png"), frame)

            result = compute_basic_metrics(str(root / "gt"), str(root / "generated"))
            self.assertTrue(math.isinf(result["psnr"]["default"]["episode0"]["1"]))
            self.assertEqual(result["ssim"]["default"]["episode0"]["1"], 1.0)


if __name__ == "__main__":
    unittest.main()
