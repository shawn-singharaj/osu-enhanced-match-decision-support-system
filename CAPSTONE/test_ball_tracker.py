import unittest

import numpy as np

from ball_tracker import BallSelector, BallTracker


def ball(x, confidence=0.06):
    return [[x, 100, x + 10, 110, confidence, 32]]


class BallTrackerTests(unittest.TestCase):
    def test_low_confidence_fast_ball_keeps_id_without_overlap(self):
        tracker = BallTracker()
        first = tracker.update(ball(100), 0)
        following = tracker.update(ball(125), 1)
        self.assertEqual(len(following), 1)
        self.assertEqual(first[0, 4], following[0, 4])
        self.assertAlmostEqual(float(following[0, 5]), 0.06)

    def test_missing_frames_do_not_produce_measured_positions(self):
        tracker = BallTracker()
        first = tracker.update(ball(100), 0)
        tracker.update(ball(110), 1)
        self.assertEqual(tracker.update([], 2).shape, (0, 7))
        following = tracker.update(ball(130), 3)
        self.assertEqual(first[0, 4], following[0, 4])

    def test_old_track_expires(self):
        tracker = BallTracker(max_missed=2)
        first = tracker.update(ball(100), 0)
        following = tracker.update(ball(100), 4)
        self.assertNotEqual(first[0, 4], following[0, 4])

    def test_duplicate_candidates_are_suppressed(self):
        tracker = BallTracker()
        detections = np.array(ball(100) + ball(101, 0.2))
        output = tracker.update(detections, 0)
        self.assertEqual(len(output), 1)
        self.assertAlmostEqual(float(output[0, 5]), 0.2)


class BallSelectorTests(unittest.TestCase):
    def test_stationary_penalty_spot_is_not_selected_over_moving_ball(self):
        selector = BallSelector()
        selector.update([[100,100,110,110,1,0.3,32], [300,100,310,110,2,0.2,32]])
        selected = selector.update([[120,100,130,110,1,0.1,32], [300,100,310,110,2,0.4,32]])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0, 4], 1)

    def test_foot_candidate_is_penalized_without_rejecting_nearby_ball(self):
        selector = BallSelector()
        selector.update([[100,100,110,110,1,0.3,32]])
        selected = selector.update(
            [[120,100,130,110,1,0.12,32], [140,100,150,110,2,0.25,32]],
            [[138,60,155,110]],
        )
        self.assertEqual(selected[0, 4], 1)

    def test_selected_stopped_ball_remains_visible(self):
        selector = BallSelector(start=(105,105))
        for _ in range(30):
            selected = selector.update([[100,100,110,110,1,0.2,32]])
            self.assertEqual(len(selected), 1)

    def test_start_position_overrides_distant_higher_confidence_spot(self):
        selector = BallSelector(start=(105,105))
        selected = selector.update([[100,100,110,110,1,0.08,32], [300,100,310,110,2,0.9,32]])
        self.assertEqual(selected[0, 4], 1)

    def test_camera_pan_updates_search_position(self):
        selector = BallSelector(start=(105,105))
        selector.update([[100,100,110,110,1,0.2,32]])
        camera = np.array([[1.,0.,100.], [0.,1.,0.]])
        selected = selector.update([[200,100,210,110,1,0.2,32]], camera_transform=camera)
        self.assertEqual(len(selected), 1)

    def test_missing_detection_does_not_display_prediction(self):
        selector = BallSelector(start=(105,105))
        selector.update([[100,100,110,110,1,0.2,32]])
        self.assertEqual(selector.update([]).shape, (0,7))


if __name__ == '__main__':
    unittest.main()
