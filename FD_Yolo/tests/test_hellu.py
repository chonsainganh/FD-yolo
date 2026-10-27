import unittest

import numpy as np

from Hellu import _fit_frame, frames_to_skip, mark_fallen_boxes, seconds_until_frame


class MarkFallenBoxesTest(unittest.TestCase):
    def test_paints_a_red_outline_only_around_the_person_at_or_above_the_threshold(self):
        image = np.zeros((100, 400, 3), dtype=np.uint8)
        boxes = {
            1: np.array([10, 40, 80, 90], dtype=np.float32),
            2: np.array([250, 40, 320, 90], dtype=np.float32),
        }

        marked = mark_fallen_boxes(image.copy(), boxes, {1: 0.91, 2: 0.2}, threshold=0.5)

        self.assertEqual(tuple(int(v) for v in marked[40, 10]), (0, 0, 255))
        self.assertEqual(tuple(int(v) for v in marked[60, 45]), (0, 0, 0))
        self.assertTrue(np.all(marked[:, 200:] == 0))

    def test_leaves_the_frame_alone_when_nobody_reaches_the_threshold(self):
        image = np.zeros((100, 400, 3), dtype=np.uint8)
        boxes = {1: np.array([10, 40, 80, 90], dtype=np.float32)}

        marked = mark_fallen_boxes(image.copy(), boxes, {1: None}, threshold=0.5)

        self.assertTrue(np.all(marked == 0))


class FitFrameTest(unittest.TestCase):
    def test_enlarges_a_webcam_frame_until_one_edge_meets_the_screen(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        fitted = _fit_frame(image, max_width=1880, max_height=960)

        self.assertEqual(fitted.shape[1], 1280)
        self.assertEqual(fitted.shape[0], 960)

    def test_shrinks_a_frame_larger_than_the_screen(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        fitted = _fit_frame(image, max_width=1280, max_height=720)

        self.assertEqual(fitted.shape[1], 1280)
        self.assertEqual(fitted.shape[0], 720)


class FrameSkipTest(unittest.TestCase):
    def test_skips_frames_that_fell_behind_the_wall_clock(self):
        # 79ms into a 30 FPS file, the next unread frame is 1. Frame 2 is the one due.
        self.assertEqual(frames_to_skip(0.079, fps=30, next_index=1), 1)

    def test_does_not_skip_when_the_next_frame_is_still_early(self):
        self.assertEqual(frames_to_skip(0.02, fps=30, next_index=1), 0)
        self.assertAlmostEqual(seconds_until_frame(0.02, fps=30, next_index=1), 1 / 30 - 0.02)

    def test_reads_the_first_frame_immediately(self):
        self.assertEqual(frames_to_skip(0.0, fps=30, next_index=0), 0)
        self.assertEqual(seconds_until_frame(0.0, fps=30, next_index=0), 0.0)
