"""Hợp đồng dùng chung cho detector theo thời gian."""

from typing import Protocol

import numpy as np


class FallDetector(Protocol):
    """Một người theo thời gian.

    update nạp tư thế theo nhịp và trả True khi khung đó vào queue.
    Xác suất nằm ở probability sau khi GRU chấm bản sao của queue.
    """

    def update(self, keypoints: np.ndarray, timestamp: float) -> bool:
        """keypoints là (num_keypoints, 3) đã chuẩn hóa cùng cách với lúc train."""
