import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.detectors.gru import (
    FallGRU,
    SequenceFallDetector,
    drop_stale_tracks,
    fall_decision,
    load_checkpoint,
    save_checkpoint,
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
            "pose_model": "yolo11x-pose.pt",
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
    def test_samples_at_target_fps_and_waits_for_min_frames(self):
        model = FallGRU(num_keypoints=2, hidden_size=4)
        detector = SequenceFallDetector(model, target_fps=10, min_frames=2)
        frame = np.zeros((2, 3), dtype=np.float32)

        self.assertIsNone(detector.update(frame, 0.0))
        self.assertIsNone(detector.update(frame, 0.05))
        self.assertEqual(detector.accepted_frames, 1)

        probability = detector.update(frame, 0.10)

        self.assertEqual(detector.accepted_frames, 2)
        self.assertIsNotNone(probability)
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
        self.assertEqual(detector.update(frame, 0.15), probability)


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


def _logit(model, sequence):
    tensor = torch.from_numpy(np.asarray(sequence, dtype=np.float32)).unsqueeze(0)
    lengths = torch.tensor([tensor.shape[1]])
    model.eval()
    with torch.no_grad():
        return model(tensor, lengths).squeeze(0)


def _probabilities(model, sequences):
    model.eval()
    scores = []
    with torch.no_grad():
        for sequence in sequences:
            scores.append(float(torch.sigmoid(_logit(model, sequence))))
    return scores


if __name__ == "__main__":
    unittest.main()
