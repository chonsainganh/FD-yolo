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


class SequenceFallDetector:
    """Bộ nhớ một người: lấy mẫu theo target_fps rồi cập nhật GRU."""

    def __init__(self, model: FallGRU, target_fps: float = 10.0, min_frames: int = 10):
        if target_fps <= 0:
            raise ValueError("target_fps phải lớn hơn 0.")
        if min_frames < 1:
            raise ValueError("min_frames phải lớn hơn hoặc bằng 1.")
        self.model = model
        self.interval = 1.0 / float(target_fps)
        self.min_frames = int(min_frames)
        self.reset()

    def reset(self) -> None:
        self._hidden = None
        self._last_accepted = None
        self._probability = None
        self.accepted_frames = 0
        self.last_seen = None

    def update(self, keypoints: np.ndarray, timestamp: float) -> float | None:
        """Trả xác suất té khi đã đủ khung. Khung đến sớm hơn nhịp FPS thì giữ kết quả cũ."""
        timestamp = float(timestamp)
        self.last_seen = timestamp
        due = self._last_accepted is None or timestamp - self._last_accepted >= self.interval
        if not due:
            if self.accepted_frames < self.min_frames:
                return None
            return self._probability

        self._last_accepted = timestamp
        self.accepted_frames += 1
        logit, self._hidden = self.model.step(keypoints, self._hidden)
        self._probability = float(torch.sigmoid(logit).item())
        if self.accepted_frames < self.min_frames:
            return None
        return self._probability


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
            f"{path}. Train bằng lệnh python -m src.train --data data/videos --output {path}"
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
