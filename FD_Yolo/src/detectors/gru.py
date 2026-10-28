"""GRU phân loại té ngã trên chuỗi keypoints."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class FallGRU(nn.Module):
    """Một GRU đọc chuỗi (khung, khớp, xy + độ tin cậy) và trả logit té ngã."""

    def __init__(self, num_keypoints: int = 17, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        if num_keypoints < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("num_keypoints, hidden_size và num_layers phải lớn hơn hoặc bằng 1.")
        self.num_keypoints = int(num_keypoints)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.gru = nn.GRU(
            input_size=self.num_keypoints * 3,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(self.hidden_size, 1)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """sequences là (batch, khung, khớp, 3). Trả logit kích thước (batch,)."""
        batch, timesteps, keypoints, channels = sequences.shape
        flat = sequences.reshape(batch, timesteps, keypoints * channels)
        packed = nn.utils.rnn.pack_padded_sequence(
            flat,
            lengths.detach().cpu().long(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(-1)

    def step(self, keypoints: np.ndarray, hidden: torch.Tensor | None):
        """Đẩy một khung và trả (logit, hidden mới)."""
        self.eval()
        device = next(self.parameters()).device
        frame = torch.as_tensor(np.asarray(keypoints), dtype=torch.float32, device=device)
        frame = frame.reshape(1, 1, -1)
        with torch.no_grad():
            output, hidden = self.gru(frame, hidden)
            logit = self.head(output[:, -1, :]).squeeze(-1).squeeze(0)
        return logit, hidden


class SampleClock:
    """Nhịp lấy mẫu. Mỗi khoảng 1/target_fps của timestamp thì nhận một khung."""

    def __init__(self, target_fps: float):
        if target_fps <= 0:
            raise ValueError("target_fps phải lớn hơn 0.")
        self.interval = 1.0 / float(target_fps)
        self._next = None

    def reset(self) -> None:
        self._next = None

    def accept(self, timestamp: float) -> bool:
        timestamp = float(timestamp)
        if self._next is None:
            self._next = timestamp
        if timestamp + 1e-6 < self._next:
            return False
        overshoot = timestamp - self._next
        if overshoot < 0:
            overshoot = 0.0
        steps = int(overshoot / self.interval) + 1
        self._next += steps * self.interval
        return True


class SequenceFallDetector:
    """Một người, một hàng đợi tư thế. update chỉ nạp theo nhịp. GRU chấm trên bản sao."""

    def __init__(self, model: FallGRU, min_frames: int = 10, target_fps: float = 10.0):
        if min_frames < 1:
            raise ValueError("min_frames phải lớn hơn hoặc bằng 1.")
        self.model = model
        self.min_frames = int(min_frames)
        self.clock = SampleClock(target_fps)
        self.reset()

    def reset(self) -> None:
        self._frames: list[np.ndarray] = []
        self.probability = None
        self.accepted_frames = 0
        self.last_seen = None
        self.revision = 0
        self.scored_revision = 0
        self.clock.reset()

    def update(self, keypoints: np.ndarray, timestamp: float) -> bool:
        """Ghi nhận người này. Đúng nhịp thì nạp tư thế và trượt cửa sổ. Không gọi GRU."""
        self.last_seen = float(timestamp)
        if not self.clock.accept(timestamp):
            return False
        self.accepted_frames += 1
        self.revision += 1
        self._frames.append(np.asarray(keypoints, dtype=np.float32).copy())
        if len(self._frames) > self.min_frames:
            del self._frames[0]
        return True

    def has_window(self) -> bool:
        return len(self._frames) >= self.min_frames

    def snapshot(self) -> np.ndarray | None:
        """Bản sao (khung, khớp, 3) khi đủ min_frames. None khi hàng đợi còn thiếu."""
        if not self.has_window():
            return None
        return np.stack(self._frames)

    def score(self) -> float | None:
        """Chấm bản sao hiện tại trên luồng này. Luồng realtime chấm ngoài lock."""
        sequence = self.snapshot()
        if sequence is None:
            return None
        self.probability = window_probability(self.model, sequence)
        self.scored_revision = self.revision
        return self.probability


def window_probability(model: FallGRU, frames) -> float:
    """Chấm một cửa sổ từ trạng thái 0. Bộ nhớ GRU chỉ tóm tắt các khung trong bản sao này."""
    sequence = np.asarray(frames, dtype=np.float32)
    if sequence.ndim != 3:
        sequence = np.stack(frames).astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(sequence)).unsqueeze(0)
    lengths = torch.tensor([sequence.shape[0]])
    model.eval()
    with torch.no_grad():
        logit = model(tensor, lengths).squeeze(0)
    return float(torch.sigmoid(logit).item())


def collect_ready_windows(states: dict) -> list[tuple[int, SequenceFallDetector, int, np.ndarray]]:
    """Mỗi track đủ khung và chưa chấm revision này thì trả bản sao.

    Queue chưa đủ khung được đánh dấu đã xem để luồng GRU không quay vòng.
    """
    jobs = []
    for track_id, detector in list(states.items()):
        if detector.revision == detector.scored_revision:
            continue
        snapshot = detector.snapshot()
        if snapshot is None:
            detector.scored_revision = detector.revision
            continue
        jobs.append((int(track_id), detector, int(detector.revision), snapshot))
    return jobs


def publish_score(states: dict, track_id: int, detector: SequenceFallDetector, revision: int, probability: float) -> bool:
    """Ghi xác suất của đúng bản sao đã chấm. Track đã bị xóa hoặc đã reset thì bỏ qua."""
    if states.get(track_id) is not detector or detector.revision < revision:
        return False
    detector.probability = float(probability)
    detector.scored_revision = int(revision)
    return True


def fall_decision(probabilities: list[float | None], threshold: float) -> tuple[bool, str]:
    """Chọn xác suất cao nhất trong các track đã đủ dữ liệu."""
    scores = [float(score) for score in probabilities if score is not None]
    if not scores:
        return False, "Cho du lieu"
    score = max(scores)
    if score >= threshold:
        return True, f"NGA {score:.2f}"
    return False, f"Binh thuong {score:.2f}"


def drop_stale_tracks(states: dict, seen_ids: set, now: float, grace: float) -> None:
    """Xóa track biến mất lâu hơn grace giây. Track vừa nhìn thấy được giữ."""
    stale = [
        track_id
        for track_id, detector in states.items()
        if track_id not in seen_ids and now - detector.last_seen > grace
    ]
    for track_id in stale:
        del states[track_id]


def save_checkpoint(path: str | Path, model: FallGRU, config: dict) -> None:
    """Lưu trọng số vào file .pt và config đọc được ở file .json cùng tên."""
    path = Path(path)
    if path.suffix.lower() != ".pt":
        path = path.with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
    path.with_suffix(".json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_checkpoint(path: str | Path) -> tuple[FallGRU, dict]:
    """Nạp FallGRU và config từ checkpoint.pt."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "Không thấy checkpoint: "
            f"{path}. Train bằng lệnh python -m src.train --output {path}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    config = payload["config"]
    model = FallGRU(
        num_keypoints=int(config["num_keypoints"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config.get("num_layers", 1)),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config
