"""Trích keypoints YOLO Pose từ video và resample về độ dài cố định hoặc về FPS đích."""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def extract_keypoints_from_video(
    video_path: str | Path,
    sequence_length: int,
    model: YOLO | str | Path = "yolo26n-pose.pt",
    conf: float = 0.25,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Trích keypoints tư thế người từ video và chuẩn hóa về đúng sequence_length khung.

    Mỗi khung lấy người có độ tin cậy bbox cao nhất. Tọa độ x, y được chuẩn hóa
    theo kích thước khung hình (0–1); kênh thứ ba là độ tin cậy của khớp.

    Returns
    -------
    numpy.ndarray
        Mảng float32 kích thước (sequence_length, num_keypoints, 3).
    """
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, (int, np.integer)):
        raise TypeError("sequence_length phải là số nguyên dương.")
    if sequence_length < 1:
        raise ValueError("sequence_length phải lớn hơn hoặc bằng 1.")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    sequence = resample_sequence(_top_person_frames(video_path, model, conf), int(sequence_length))
    _save_sequence(sequence, save_path)
    return sequence


def extract_pose_sequence(
    video_path: str | Path,
    target_fps: float = 10.0,
    model: YOLO | str | Path = "yolo26n-pose.pt",
    conf: float = 0.25,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Trích một người mỗi khung rồi lấy mẫu lại về target_fps.

    FPS nguồn không đọc được thì giữ mọi khung. x, y nằm trong khoảng 0–1.
    """
    if target_fps <= 0:
        raise ValueError("target_fps phải lớn hơn 0.")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    sequence = resample_to_fps(_top_person_frames(video_path, model, conf), video_fps(video_path), target_fps)
    _save_sequence(sequence, save_path)
    return sequence


def _top_person_frames(video_path: Path, model, conf: float) -> np.ndarray:
    pose_model = model if isinstance(model, YOLO) else YOLO(str(model))
    kpt_shape = getattr(pose_model.model, "kpt_shape", (17, 3))
    num_keypoints = int(kpt_shape[0])
    frames: list[np.ndarray] = []
    for result in pose_model.predict(
        source=str(video_path),
        stream=True,
        verbose=False,
        conf=conf,
    ):
        frames.append(person_keypoints(result, num_keypoints))
    if not frames:
        raise ValueError(f"Video không có khung hình: {video_path}")
    return np.stack(frames)


def _save_sequence(sequence: np.ndarray, save_path: str | Path | None) -> None:
    if save_path is None:
        return
    save_path = Path(save_path)
    if save_path.suffix.lower() != ".npy":
        save_path = save_path.with_suffix(".npy")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, sequence)


def video_fps(video_path: str | Path) -> float:
    """FPS ghi trong metadata. Trả về 0 khi file không khai báo FPS hợp lệ."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0:
        return 0.0
    return float(fps)


def person_keypoints(result, num_keypoints: int) -> np.ndarray:
    """Lấy keypoints của người tin cậy nhất trong một khung. Thiếu người thì trả về 0."""
    keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)
    detected = result.keypoints
    if detected is None or detected.data is None or len(detected.data) == 0:
        return keypoints

    data = detected.data.detach().cpu().numpy()
    if result.boxes is not None and result.boxes.conf is not None and len(result.boxes.conf) > 0:
        index = int(result.boxes.conf.argmax().item())
    else:
        index = int(data[:, :, 2].mean(axis=1).argmax())
    index = min(index, data.shape[0] - 1)

    xy = detected.xyn[index].detach().cpu().numpy()
    count = min(num_keypoints, xy.shape[0], data.shape[1])
    keypoints[:count, :2] = xy[:count]
    keypoints[:count, 2] = data[index, :count, 2]
    return keypoints


def track_boxes(result) -> dict[int, np.ndarray]:
    """Hộp xyxy theo track_id, đơn vị pixel của khung vừa đọc. Không có id thì dict rỗng."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return {}
    track_ids = getattr(boxes, "id", None)
    xyxy = getattr(boxes, "xyxy", None)
    if track_ids is None or xyxy is None:
        return {}
    ids = track_ids.detach().cpu().numpy().astype(int)
    coords = xyxy.detach().cpu().numpy()
    count = min(len(ids), len(coords))
    return {int(ids[index]): coords[index] for index in range(count)}


def tracked_people(result) -> list[tuple[int, np.ndarray]]:
    """Trả về (track_id, keypoints chuẩn hóa 0–1). Không có id theo dõi thì danh sách rỗng."""
    detected = getattr(result, "keypoints", None)
    if detected is None or getattr(detected, "data", None) is None or len(detected.data) == 0:
        return []
    boxes = getattr(result, "boxes", None)
    track_ids = getattr(boxes, "id", None) if boxes is not None else None
    if track_ids is None:
        return []

    ids = track_ids.detach().cpu().numpy().astype(int)
    data = detected.data.detach().cpu().numpy()
    xy = detected.xyn.detach().cpu().numpy()
    count = min(len(ids), data.shape[0], xy.shape[0])
    people = []
    for index in range(count):
        keypoints = np.zeros((xy.shape[1], 3), dtype=np.float32)
        keypoint_count = min(xy.shape[1], data.shape[1])
        keypoints[:keypoint_count, :2] = xy[index, :keypoint_count]
        keypoints[:keypoint_count, 2] = data[index, :keypoint_count, 2]
        people.append((int(ids[index]), keypoints))
    return people


def resample_sequence(sequence: np.ndarray, sequence_length: int) -> np.ndarray:
    """Lấy đều các khung để chuỗi có đúng sequence_length bước thời gian."""
    frame_count = sequence.shape[0]
    if frame_count == sequence_length:
        return sequence

    indices = np.linspace(0, frame_count - 1, sequence_length)
    indices = np.clip(np.rint(indices).astype(int), 0, frame_count - 1)
    return sequence[indices]


def frame_count_at_fps(frame_count: int, source_fps: float, target_fps: float) -> int:
    """Số khung sau khi đổi một đoạn video từ source_fps sang target_fps."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, (int, np.integer)):
        raise TypeError("frame_count phải là số nguyên dương.")
    if frame_count < 1:
        raise ValueError("frame_count phải lớn hơn hoặc bằng 1.")
    if target_fps <= 0:
        raise ValueError("target_fps phải lớn hơn 0.")
    if source_fps <= 0:
        return int(frame_count)
    duration = float(frame_count) / float(source_fps)
    return max(1, int(round(duration * float(target_fps))))


def resample_to_fps(sequence: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Lấy mẫu lại chuỗi keypoints về target_fps. FPS nguồn không hợp lệ thì giữ nguyên."""
    count = frame_count_at_fps(int(sequence.shape[0]), source_fps, target_fps)
    return resample_sequence(sequence, count)
