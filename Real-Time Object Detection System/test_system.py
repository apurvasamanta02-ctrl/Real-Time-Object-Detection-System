# =============================================================================
# test_system.py — Automated Unit & Integration Tests for System Modules
# =============================================================================

import unittest
import numpy as np
import config
from tracker import CentroidTracker
from detector import YOLODetector, Detection
from utils import FPSCounter, class_color, find_track_id


class TestConfig(unittest.TestCase):
    """Test configuration defaults and parameters."""

    def test_config_parameters(self):
        self.assertIsInstance(config.CONFIDENCE_THRESHOLD, float)
        self.assertIsInstance(config.IOU_THRESHOLD, float)
        self.assertIsInstance(config.CAMERA_INDEX, int)
        self.assertIsInstance(config.CLASS_FILTER, list)
        self.assertIsInstance(config.CLASS_EXCLUDE, list)
        self.assertGreater(config.BBOX_SMOOTH_ALPHA, 0.0)
        self.assertLessEqual(config.BBOX_SMOOTH_ALPHA, 1.0)


class TestTracker(unittest.TestCase):
    """Test CentroidTracker logic and EMA smoothing."""

    def setUp(self):
        self.tracker = CentroidTracker(max_disappeared=5, max_distance=50, smooth_alpha=0.5)

    def test_registration_and_tracking(self):
        # Frame 1 detection
        det1 = [Detection(bbox=(100, 100, 200, 200), class_id=0, class_name="person", confidence=0.9)]
        res1 = self.tracker.update(det1)
        self.assertEqual(len(res1), 1)
        self.assertIn(0, res1)

        # Frame 2 detection slightly moved
        det2 = [Detection(bbox=(102, 102, 202, 202), class_id=0, class_name="person", confidence=0.9)]
        res2 = self.tracker.update(det2)
        self.assertEqual(len(res2), 1)
        self.assertIn(0, res2)

        # Check EMA smoothing (blend between (100,100,200,200) and (102,102,202,202))
        smoothed_bbox = res2[0]
        self.assertEqual(smoothed_bbox, (101, 101, 201, 201))

    def test_deregistration(self):
        det = [Detection(bbox=(50, 50, 80, 80), class_id=0, class_name="cup", confidence=0.8)]
        self.tracker.update(det)
        self.assertEqual(self.tracker.count, 1)

        # Update 6 frames with no detection -> should deregister (max_disappeared=5)
        for _ in range(6):
            self.tracker.update([])
        self.assertEqual(self.tracker.count, 0)


class TestUtils(unittest.TestCase):
    """Test utility functions and FPS counter."""

    def test_fps_counter(self):
        fps_ctr = FPSCounter(window=10)
        fps_ctr.tick()
        self.assertGreaterEqual(fps_ctr.fps, 0.0)

    def test_class_color(self):
        color1 = class_color(0)
        color2 = class_color(1)
        self.assertEqual(len(color1), 3)
        self.assertEqual(len(color2), 3)

    def test_find_track_id(self):
        tracked = {0: (100, 100, 200, 200), 1: (300, 300, 400, 400)}
        matched = find_track_id((102, 101, 199, 201), tracked, threshold=10)
        self.assertEqual(matched, 0)

        unmatched = find_track_id((500, 500, 600, 600), tracked, threshold=10)
        self.assertIsNone(unmatched)


if __name__ == "__main__":
    unittest.main()
