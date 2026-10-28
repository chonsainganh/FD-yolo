import tempfile
import unittest
from pathlib import Path

from src.data.labels import labeled_videos


class LabeledVideosTest(unittest.TestCase):
    def test_reads_fall_labels_from_folder_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "falls").mkdir()
            (root / "normal").mkdir()
            (root / "no_fall_static" / "nested").mkdir(parents=True)
            (root / "other").mkdir()
            (root / "falls" / "b.mp4").write_bytes(b"b")
            (root / "falls" / "a.MP4").write_bytes(b"a")
            (root / "falls" / "note.txt").write_text("khong phai video", encoding="utf-8")
            (root / "normal" / "stand.avi").write_bytes(b"n")
            (root / "no_fall_static" / "nested" / "lie.mov").write_bytes(b"s")
            (root / "other" / "ignore.mp4").write_bytes(b"x")

            found = labeled_videos(root)

        self.assertEqual(
            [(path.name, label) for path, label in found],
            [("a.MP4", 1), ("b.mp4", 1), ("lie.mov", 0), ("stand.avi", 0)],
        )

    def test_reads_fall_and_adl_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Fall").mkdir()
            (root / "ADL").mkdir()
            (root / "Fall" / "subject1-01.mp4").write_bytes(b"f")
            (root / "ADL" / "subject1-01.mp4").write_bytes(b"a")

            found = labeled_videos(root)

        self.assertEqual(
            [(path.parent.name, path.name, label) for path, label in found],
            [("ADL", "subject1-01.mp4", 0), ("Fall", "subject1-01.mp4", 1)],
        )

    def test_rejects_a_missing_dataset_root(self):
        with self.assertRaises(FileNotFoundError):
            labeled_videos(Path("missing-dataset-root"))

    def test_rejects_a_root_with_no_labeled_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                labeled_videos(Path(tmp))
