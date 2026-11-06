"""Behavior checks for camera zoom, segmentation noise, and scene resets."""

import unittest

import cv2
import numpy as np

from track_soccer import FieldTracker, field_region


def textured_field():
    rng = np.random.default_rng(7)
    frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    frame[60:330, 40:600] = (45, 110, 70)
    for x, y in rng.integers((60, 85), (580, 310), size=(300, 2)):
        shade = int(rng.integers(70, 220))
        cv2.circle(frame, (int(x), int(y)), 2, (shade, shade, shade), -1)
    return frame


def iou(first, second):
    return np.logical_and(first > 0, second > 0).sum() / np.logical_or(first > 0, second > 0).sum()


class FieldTrackerTests(unittest.TestCase):
    def test_separate_landscaping_does_not_expand_pitch(self):
        frame = textured_field()
        frame[20:45, 300:620] = (45, 110, 70)
        _, polygon = field_region(frame)
        self.assertIsNotNone(polygon)
        self.assertGreaterEqual(int(polygon[:, :, 1].min()), 55)

    def test_zoom_and_pan_align_history(self):
        frame = textured_field()
        tracker = FieldTracker()
        tracker.update(frame)
        expected = cv2.getRotationMatrix2D((320, 180), 0.5, 1.06)
        expected[:, 2] += (8, -3)
        zoomed = cv2.warpAffine(frame, expected, (640, 360))
        mask, polygon = tracker.update(zoomed)
        self.assertIsNotNone(polygon)
        self.assertIsNotNone(tracker.last_transform)
        points = np.array([[[100., 100.], [540., 280.], [320., 180.]]], dtype=np.float32)
        error = np.linalg.norm(cv2.transform(points, expected) - cv2.transform(points, tracker.last_transform), axis=2)
        self.assertLess(float(error.max()), 1.5)
        fresh, _ = field_region(zoomed)
        self.assertGreater(iou(mask, fresh), 0.97)

    def test_color_noise_does_not_immediately_move_boundary(self):
        frame = textured_field()
        tracker = FieldTracker()
        original, _ = tracker.update(frame)
        noisy = frame.copy()
        noisy[60:68, 40:600] = (40, 40, 40)
        raw, _ = field_region(noisy)
        smoothed, _ = tracker.update(noisy)
        self.assertIsNotNone(tracker.last_transform)
        self.assertGreater(iou(original, smoothed), iou(original, raw))

    def test_missing_field_clears_history_and_reacquires(self):
        tracker = FieldTracker()
        tracker.update(textured_field())
        mask, polygon = tracker.update(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertIsNone(polygon)
        self.assertFalse(mask.any())
        self.assertIsNone(tracker.probability)
        mask, polygon = tracker.update(textured_field())
        self.assertIsNotNone(polygon)
        self.assertTrue(mask.any())
        self.assertIsNone(tracker.last_transform)


if __name__ == "__main__":
    unittest.main()
