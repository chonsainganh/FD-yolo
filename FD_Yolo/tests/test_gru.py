import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.detectors.gru import (
    FallGRU,
    SequenceFallDetector,
    collect_ready_windows,
    drop_stale_tracks,
    fall_decision,
    load_checkpoint,
    publish_score,
    save_checkpoint,
    window_probability,
)
from src.train import fit, split_indices


def _toy_sequences():
    sequences = []
    labels = []
    for _ in range(4):
        sequences.append(np.ones((6, 17, 3), dtype=np.float32))
        labels.append(1)
        sequences.append(np.zeros((5, 17, 3), dtype=np.float32))
        labels.append(0)
    return sequences, labels


class SplitIndicesTest(unittest.TestCase):
    def test_holds_out_the_tail_of_each_label_and_keeps_one_training_example(self):
        labels = [1, 1, 1, 1, 1, 0]

        train, val = split_indices(labels, val_fraction=0.2)

        self.assertEqual(val, [4])
        self.assertEqual(train, [0, 1, 2, 3, 5])

    def test_keeps_a_singleton_label_in_training(self):
        train, val = split_indices([1, 0, 0], val_fraction=0.2)

        self.assertIn(0, train)
        self.assertNotIn(0, val)


class FitTest(unittest.TestCase):
    def test_learns_to_separate_fall_and_normal_sequences(self):
        sequences, labels = _toy_sequences()

        model, history = fit(
            sequences,
            labels,
            epochs=40,
            hidden_size=16,
            lr=0.05,
            seed=0,
            device="cpu",
        )

        self.assertEqual(history["val_accuracy"], 1.0)
        self.assertLess(history["train_loss"], 0.2)
        scores = _probabilities(model, sequences)
        self.assertGreater(min(scores[0::2]), 0.8)
        self.assertLess(max(scores[1::2]), 0.2)


class CheckpointTest(unittest.TestCase):
    def test_roundtrip_keeps_config_and_logits(self):
        model = FallGRU(num_keypoints=17, hidden_size=8)
        sequence = np.random.default_rng(0).random((4, 17, 3), dtype=np.float32)
        before = _logit(model, sequence)
        config = {
            "target_fps": 10,
            "num_keypoints": 17,
            "hidden_size": 8,
            "num_layers": 1,
            "min_frames": 10,
            "decision_threshold": 0.5,
            "pose_model": "yolo26n-pose.pt",
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            save_checkpoint(path, model, config)
            loaded, loaded_config = load_checkpoint(path)
            sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))

        self.assertEqual(loaded_config["target_fps"], 10)
        self.assertEqual(sidecar["min_frames"], 10)
        torch.testing.assert_close(before, _logit(loaded, sequence))


class StepMatchesSequenceTest(unittest.TestCase):
    def test_frame_by_frame_hidden_state_matches_the_full_sequence(self):
        torch.manual_seed(0)
        model = FallGRU(num_keypoints=17, hidden_size=8)
        sequence = torch.randn(5, 17, 3)
        full = _logit(model, sequence.numpy())

        hidden = None
        for frame in sequence:
            logit, hidden = model.step(frame.numpy(), hidden)

        torch.testing.assert_close(logit, full, atol=1e-4, rtol=1e-4)


class SequenceFallDetectorTest(unittest.TestCase):
    def test_keeps_one_pose_per_sample_tick_and_still_remembers_the_person(self):
        detector = SequenceFallDetector(_small_model(), min_frames=10, target_fps=10)
        frame = np.zeros((2, 3), dtype=np.float32)

        self.assertTrue(detector.update(frame, 0.0))
        self.assertFalse(detector.update(frame, 0.05))

        self.assertEqual(detector.accepted_frames, 1)
        self.assertEqual(detector.last_seen, 0.05)
        self.assertIsNone(detector.snapshot())

        self.assertTrue(detector.update(frame, 0.1))
        self.assertEqual(detector.accepted_frames, 2)

    def test_a_late_timestamp_enters_once(self):
        detector = SequenceFallDetector(_small_model(), min_frames=10, target_fps=10)
        frame = np.zeros((2, 3), dtype=np.float32)

        self.assertTrue(detector.update(frame, 0.0))
        self.assertTrue(detector.update(frame, 0.35))
        self.assertFalse(detector.update(frame, 0.39))

        self.assertEqual(detector.accepted_frames, 2)

    def test_enqueue_does_not_run_the_network(self):
        model = _small_model()
        calls = []
        forward = model.forward

        def counting(sequences, lengths):
            calls.append(1)
            return forward(sequences, lengths)

        model.forward = counting
        detector = SequenceFallDetector(model, min_frames=2, target_fps=10)
        frame = np.zeros((2, 3), dtype=np.float32)
        detector.update(frame, 0.0)
        detector.update(frame, 0.1)

        self.assertEqual(calls, [])
        self.assertIsNotNone(detector.snapshot())
        window_probability(model, detector.snapshot())
        self.assertEqual(calls, [1])

    def test_snapshot_keeps_the_old_window_after_the_queue_slides(self):
        detector = SequenceFallDetector(_small_model(), min_frames=2, target_fps=10)
        first = np.ones((2, 3), dtype=np.float32)
        second = np.zeros((2, 3), dtype=np.float32)
        third = np.full((2, 3), 0.25, dtype=np.float32)
        detector.update(first, 0.0)
        detector.update(second, 0.1)
        held = detector.snapshot()
        saved = held.copy()

        detector.update(third, 0.2)

        np.testing.assert_array_equal(held, saved)
        latest = detector.snapshot()
        np.testing.assert_array_equal(latest[0], second)
        np.testing.assert_array_equal(latest[1], third)

    def test_score_uses_the_ten_pose_copy(self):
        model = _small_model()
        detector = SequenceFallDetector(model, min_frames=2, target_fps=10)
        first = np.ones((2, 3), dtype=np.float32)
        second = np.zeros((2, 3), dtype=np.float32)
        detector.update(first, 0.0)

        self.assertIsNone(detector.score())

        detector.update(second, 0.1)
        expected = window_probability(model, np.stack([first, second]))

        self.assertAlmostEqual(detector.score(), expected)
        self.assertEqual(collect_ready_windows({1: detector}), [])


class ScorePublishTest(unittest.TestCase):
    def test_next_score_uses_the_latest_window_and_skips_the_ones_in_between(self):
        detector = SequenceFallDetector(_small_model(), min_frames=2, target_fps=10)
        states = {3: detector}
        first = np.ones((2, 3), dtype=np.float32)
        second = np.zeros((2, 3), dtype=np.float32)
        third = np.full((2, 3), 0.25, dtype=np.float32)
        fourth = np.full((2, 3), 0.5, dtype=np.float32)
        detector.update(first, 0.0)

        self.assertEqual(collect_ready_windows(states), [])
        self.assertEqual(detector.scored_revision, detector.revision)

        detector.update(second, 0.1)
        jobs = collect_ready_windows(states)
        track_id, queued, revision, snap = jobs[0]
        self.assertIs(queued, detector)
        detector.update(third, 0.2)
        detector.update(fourth, 0.3)

        self.assertTrue(publish_score(states, track_id, detector, revision, 0.2))
        self.assertEqual(detector.probability, 0.2)
        self.assertNotEqual(detector.scored_revision, detector.revision)

        _, _, next_revision, latest = collect_ready_windows(states)[0]

        self.assertNotEqual(next_revision, revision)
        np.testing.assert_array_equal(snap[0], first)
        np.testing.assert_array_equal(snap[1], second)
        np.testing.assert_array_equal(latest[0], third)
        np.testing.assert_array_equal(latest[1], fourth)

    def test_publish_ignores_a_track_that_was_removed(self):
        detector = SequenceFallDetector(_small_model(), min_frames=2, target_fps=10)
        detector.update(np.zeros((2, 3), dtype=np.float32), 0.0)

        self.assertFalse(publish_score({}, 3, detector, detector.revision, 0.4))
        self.assertIsNone(detector.probability)


class FallDecisionTest(unittest.TestCase):
    def test_waits_until_any_track_has_a_probability(self):
        fallen, text = fall_decision([None, None], threshold=0.5)

        self.assertFalse(fallen)
        self.assertEqual(text, "Cho du lieu")

    def test_marks_a_fall_from_the_highest_track_probability(self):
        fallen, text = fall_decision([None, 0.2, 0.81], threshold=0.5)

        self.assertTrue(fallen)
        self.assertEqual(text, "NGA 0.81")


class DropStaleTracksTest(unittest.TestCase):
    def test_drops_tracks_missing_longer_than_the_grace_period(self):
        states = {1: _Seen(0.0), 2: _Seen(9.5), 3: _Seen(9.2)}

        drop_stale_tracks(states, seen_ids={3}, now=10.0, grace=1.0)

        self.assertEqual(set(states), {2, 3})


class _Seen:
    def __init__(self, last_seen):
        self.last_seen = last_seen


def _small_model():
    return FallGRU(num_keypoints=2, hidden_size=4)


def _logit(model, sequence):
    tensor = torch.from_numpy(np.asarray(sequence, dtype=np.float32)).unsqueeze(0)
    lengths = torch.tensor([tensor.shape[1]])
    model.eval()
    with torch.no_grad():
        return model(tensor, lengths).squeeze(0)


def _window_probability(model, frames):
    return float(torch.sigmoid(_logit(model, np.stack(frames))).item())


def _probabilities(model, sequences):
    model.eval()
    scores = []
    with torch.no_grad():
        for sequence in sequences:
            scores.append(float(torch.sigmoid(_logit(model, sequence))))
    return scores


if __name__ == "__main__":
    unittest.main()
