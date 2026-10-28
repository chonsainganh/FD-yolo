import argparse
import sys
import threading
import time

import cv2
from ultralytics import YOLO

from src.detectors.gru import (
    SequenceFallDetector,
    collect_ready_windows,
    drop_stale_tracks,
    fall_decision,
    load_checkpoint,
    publish_score,
    window_probability,
)
from src.utils.keypoints import track_boxes, tracked_people


def frames_to_skip(elapsed: float, fps: float, next_index: int) -> int:
    """Số khung cần bỏ để khung sắp đọc trùng đồng hồ thật. 0 khi khung tới vẫn chưa tới giờ."""
    if fps <= 1.0 or next_index < 0:
        return 0
    target = int(elapsed * fps)
    if target <= next_index:
        return 0
    return target - next_index


def seconds_until_frame(elapsed: float, fps: float, next_index: int) -> float:
    """Số giây phải chờ để không phát nhanh hơn file. 0 khi khung này đã tới giờ."""
    if fps <= 1.0:
        return 0.0
    return max(0.0, (next_index / fps) - elapsed)


def _screen_size():

    try:
        import ctypes

        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
        if width > 0 and height > 0:
            return max(1, width - 40), max(1, height - 120)
    except (AttributeError, OSError, ValueError):
        pass
    return 1280, 720


def _fit_frame(image, max_width, max_height):
    """Đưa khung vừa vùng hiển thị và giữ tỉ lệ. Ảnh nhỏ được phóng lên tới mép màn hình."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if abs(scale - 1.0) < 0.001:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, new_size, interpolation=interpolation)


def _annotate(canvas, text, fallen, skipped_now):
    """Vẽ trạng thái ngã và số khung vừa bị bỏ vì YOLO chưa theo kịp đồng hồ."""
    color = (0, 0, 255) if fallen else (0, 180, 0)
    detail = f"{text} | skip {skipped_now}"
    cv2.rectangle(canvas, (8, 8), (8 + 18 * len(detail), 48), (0, 0, 0), -1)
    cv2.putText(canvas, detail, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return canvas


def mark_fallen_boxes(image, boxes, probabilities, threshold):
    """Tô viền đỏ quanh người có xác suất đạt ngưỡng."""
    height, width = image.shape[:2]
    for track_id, xyxy in boxes.items():
        score = probabilities.get(track_id)
        if score is None or float(score) < threshold:
            continue
        x1 = min(width - 1, max(0, int(round(float(xyxy[0])))))
        y1 = min(height - 1, max(0, int(round(float(xyxy[1])))))
        x2 = min(width - 1, max(0, int(round(float(xyxy[2])))))
        y2 = min(height - 1, max(0, int(round(float(xyxy[3])))))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return image


def _run_ingest(source, stop_event, pose_model, conf, fall_model, fall_config, detectors, lock, score_ready):
    """Đọc khung, YOLO lấy tư thế, nạp queue. Không chờ GRU."""
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Không mở được nguồn hình: {source}")
        stop_event.set()
        score_ready.set()
        return

    is_file = not isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    paced = is_file and fps > 1.0
    clock_start = time.perf_counter()
    next_index = 0
    window = "Fall detection"
    max_width, max_height = _screen_size()
    threshold = float(fall_config["decision_threshold"])
    min_frames = int(fall_config["min_frames"])
    target_fps = float(fall_config.get("target_fps", 10.0))

    try:
        while not stop_event.is_set():
            skipped_now = 0
            if paced:
                elapsed = time.perf_counter() - clock_start
                delay = seconds_until_frame(elapsed, fps, next_index)
                if delay > 0.0:
                    if stop_event.wait(delay):
                        break
                    continue
                elapsed = time.perf_counter() - clock_start
                skipped_now = frames_to_skip(elapsed, fps, next_index)
                missed_end = False
                for _ in range(skipped_now):
                    if not cap.grab():
                        missed_end = True
                        break
                    next_index += 1
                if missed_end:
                    break
            ok, frame = cap.read()
            if not ok:
                break
            frame = frame.copy()
            if paced:
                timestamp = next_index / fps
            else:
                timestamp = time.perf_counter() - clock_start
            frame_id = next_index
            next_index += 1
            result = pose_model.track(source=frame, persist=True, verbose=False, conf=conf)[0]
            boxes = track_boxes(result)
            with lock:
                seen = set()
                for track_id, keypoints in tracked_people(result):
                    seen.add(track_id)
                    detector = detectors.get(track_id)
                    if detector is None:
                        detector = SequenceFallDetector(
                            fall_model,
                            min_frames=min_frames,
                            target_fps=target_fps,
                        )
                        detectors[track_id] = detector
                    if detector.update(keypoints, timestamp) and detector.has_window():
                        score_ready.set()
                drop_stale_tracks(detectors, seen, timestamp, grace=1.0)
                probabilities = {
                    track_id: detectors[track_id].probability
                    for track_id in seen
                    if track_id in detectors
                }
            fallen, text = fall_decision(list(probabilities.values()), threshold)
            display = _fit_frame(
                _annotate(
                    mark_fallen_boxes(result.plot(), boxes, probabilities, threshold),
                    text,
                    fallen,
                    skipped_now,
                ),
                max_width,
                max_height,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                stop_event.set()
                score_ready.set()
                break
    finally:
        cap.release()
        stop_event.set()
        score_ready.set()
        cv2.destroyAllWindows()


def _score_windows(model, detectors, lock, score_ready, stop_event):
    """GRU chấm bản sao 10 tư thế mới nhất. Queue vẫn được nạp trong lúc chấm."""
    while not stop_event.is_set():
        with lock:
            jobs = collect_ready_windows(detectors)
            if not jobs:
                score_ready.clear()
        if not jobs:
            if stop_event.is_set():
                break
            score_ready.wait(timeout=0.2)
            continue
        for track_id, detector, revision, snapshot in jobs:
            if stop_event.is_set():
                break
            probability = window_probability(model, snapshot)
            with lock:
                publish_score(detectors, track_id, detector, revision, probability)


def run_realtime(source=0, checkpoint="runs/fall-gru/checkpoint.pt", pose=None, conf=0.1):
    """Chạy luồng đọc hình cùng YOLO, và luồng GRU.

    Parameters
    ----------
    source : int | str
        Chỉ số webcam (0) hoặc đường dẫn video.
    checkpoint : str | Path
        Trọng số GRU và config đã lưu lúc train.
    pose : str | Path | None
        YOLO Pose. None thì lấy pose_model trong checkpoint. Lệnh Hellu.py mặc định truyền yolo26n-pose.pt.
    conf : float
        Ngưỡng điểm của YOLO. Hộp dưới ngưỡng này bị bỏ.
    """
    fall_model, fall_config = load_checkpoint(checkpoint)
    pose_name = fall_config["pose_model"] if pose is None else pose
    stop_event = threading.Event()
    detectors = {}
    lock = threading.Lock()
    score_ready = threading.Event()
    pose_model = pose_name if isinstance(pose_name, YOLO) else YOLO(str(pose_name))
    ingest = threading.Thread(
        target=_run_ingest,
        name="pose-ingest",
        args=(source, stop_event, pose_model, conf, fall_model, fall_config, detectors, lock, score_ready),
        daemon=False,
    )
    scorer = threading.Thread(
        target=_score_windows,
        name="gru-score",
        args=(fall_model, detectors, lock, score_ready, stop_event),
        daemon=False,
    )
    ingest.start()
    scorer.start()
    try:
        while ingest.is_alive() and scorer.is_alive():
            ingest.join(timeout=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        score_ready.set()
        ingest.join()
        scorer.join()


def _parse_source(text):
    """'0' thành webcam 0; chuỗi khác giữ nguyên làm đường dẫn."""
    return int(text) if text.isdigit() else text


def _use_utf8_stdio():
    """Windows console mặc định không in được tiếng Việt trong --help và thông báo lỗi."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv=None):
    """python Hellu.py [--checkpoint runs/fall-gru/checkpoint.pt] [webcam hoặc video]."""
    _use_utf8_stdio()
    parser = argparse.ArgumentParser(description="Phát hiện ngã realtime bằng GRU trên tư thế.")
    parser.add_argument("source", nargs="?", default="0", help="Chỉ số webcam hoặc đường dẫn video.")
    parser.add_argument("--checkpoint", default="runs/fall-gru/checkpoint.pt")
    parser.add_argument("--pose", default="yolo26n-pose.pt", help="YOLO Pose dùng để lấy khớp.")
    parser.add_argument("--conf", type=float, default=0.1, help="Ngưỡng điểm YOLO. Hộp dưới ngưỡng bị bỏ.")
    args = parser.parse_args(argv)
    try:
        run_realtime(
            source=_parse_source(args.source),
            checkpoint=args.checkpoint,
            pose=args.pose,
            conf=args.conf,
        )
    except FileNotFoundError as exc:
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main(sys.argv[1:])
