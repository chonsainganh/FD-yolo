import unittest
from pathlib import Path

import numpy as np

from src.utils.keypoints import (
    extract_keypoints_from_video,
    frame_count_at_fps,
    person_keypoints,
    resample_sequence,
    resample_to_fps,
    tracked_people,
)


class _Tensor:
    """Đủ các phép mà person_keypoints gọi trên tensor YOLO."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def __len__(self):
        return len(self.value)

    def argmax(self):
        index = int(np.argmax(self.value))

        class _Index:
            def item(self):
                return index

        return _Index()

    def __getitem__(self, item):
        return _Tensor(self.value[item])


class _Result:
    def __init__(self, data, xyn, conf):
        self.keypoints = type("Keypoints", (), {"data": _Tensor(data), "xyn": _Tensor(xyn)})()
        self.boxes = type("Boxes", (), {"conf": _Tensor(conf)})()


class ResampleSequenceTest(unittest.TestCase):
    def test_keeps_frames_when_length_already_matches(self):
        sequence = np.arange(12, dtype=np.float32).reshape(4, 1, 3)

        resampled = resample_sequence(sequence, 4)

        np.testing.assert_array_equal(resampled, sequence)

    def test_picks_evenly_spaced_frames(self):
        sequence = np.arange(5, dtype=np.float32).reshape(5, 1, 1)

        resampled = resample_sequence(sequence, 3)

        np.testing.assert_array_equal(resampled[:, 0, 0], np.array([0, 2, 4], dtype=np.float32))

    def test_rounds_halfway_indices_to_even(self):
        sequence = np.arange(3, dtype=np.float32).reshape(3, 1, 1)

        resampled = resample_sequence(sequence, 5)

        np.testing.assert_array_equal(resampled[:, 0, 0], np.array([0, 0, 1, 2, 2], dtype=np.float32))


class ResampleToFpsTest(unittest.TestCase):
    def test_counts_one_second_of_video_as_ten_frames(self):
        self.assertEqual(frame_count_at_fps(30, source_fps=30, target_fps=10), 10)

    def test_keeps_every_frame_when_the_source_fps_is_unknown(self):
        self.assertEqual(frame_count_at_fps(7, source_fps=0, target_fps=10), 7)

    def test_resample_keeps_the_first_and_last_frame(self):
        sequence = np.arange(30, dtype=np.float32).reshape(30, 1, 1)

        resampled = resample_to_fps(sequence, source_fps=30, target_fps=10)

        self.assertEqual(len(resampled), 10)
        self.assertEqual(float(resampled[0, 0, 0]), 0.0)
        self.assertEqual(float(resampled[-1, 0, 0]), 29.0)
        self.assertTrue(np.all(np.diff(resampled[:, 0, 0]) >= 0))


class PersonKeypointsTest(unittest.TestCase):
    def test_returns_zeros_when_nobody_is_detected(self):
        result = type("Result", (), {"keypoints": None})()

        keypoints = person_keypoints(result, 17)

        self.assertEqual(keypoints.shape, (17, 3))
        self.assertTrue(np.all(keypoints == 0))

    def test_keeps_the_person_with_the_highest_box_confidence(self):
        data = np.zeros((2, 17, 3), dtype=np.float32)
        xyn = np.zeros((2, 17, 2), dtype=np.float32)
        xyn[1, :, 0] = 0.25
        xyn[1, :, 1] = 0.75
        data[1, :, 2] = 0.8
        result = _Result(data, xyn, np.array([0.2, 0.9], dtype=np.float32))

        keypoints = person_keypoints(result, 17)

        np.testing.assert_allclose(keypoints[:, 0], 0.25)
        np.testing.assert_allclose(keypoints[:, 1], 0.75)
        np.testing.assert_allclose(keypoints[:, 2], 0.8)


class TrackedPeopleTest(unittest.TestCase):
    def test_returns_normalized_keypoints_with_the_track_id(self):
        data = np.zeros((1, 2, 3), dtype=np.float32)
        data[0, :, 2] = 0.6
        xyn = np.zeros((1, 2, 2), dtype=np.float32)
        xyn[0, :, 0] = 0.1
        xyn[0, :, 1] = 0.2
        result = _Result(data, xyn, np.array([0.9], dtype=np.float32))
        result.boxes.id = _Tensor(np.array([7.0]))

        people = tracked_people(result)

        self.assertEqual(len(people), 1)
        track_id, keypoints = people[0]
        self.assertEqual(track_id, 7)
        np.testing.assert_allclose(keypoints[:, 0], 0.1)
        np.testing.assert_allclose(keypoints[:, 1], 0.2)
        np.testing.assert_allclose(keypoints[:, 2], 0.6)

    def test_returns_nobody_when_tracking_ids_are_missing(self):
        data = np.zeros((1, 2, 3), dtype=np.float32)
        xyn = np.zeros((1, 2, 2), dtype=np.float32)
        result = _Result(data, xyn, np.array([0.9], dtype=np.float32))

        self.assertEqual(tracked_people(result), [])


class ExtractKeypointsValidationTest(unittest.TestCase):
    def test_rejects_a_non_integer_sequence_length(self):
        with self.assertRaises(TypeError):
            extract_keypoints_from_video(Path("missing.mp4"), sequence_length=True)

    def test_rejects_a_missing_video_before_loading_a_model(self):
        with self.assertRaises(FileNotFoundError):
            extract_keypoints_from_video(Path("missing-video.mp4"), sequence_length=8)


if __name__ == "__main__":
    unittest.main()
