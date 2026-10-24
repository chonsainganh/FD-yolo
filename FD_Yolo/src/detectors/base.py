"""Hợp đồng dùng chung cho detector theo thời gian."""

from typing import Protocol

import numpy as np


class FallDetector(Protocol):
    """Cập nhật một người và trả xác suất ngã.

    None nghĩa là chưa đủ khung ở nhịp target_fps để quyết định.
    """

    def update(self, keypoints: np.ndarray, timestamp: float) -> float | None:
        """keypoints là (num_keypoints, 3) đã chuẩn hóa cùng cách với lúc train."""
