import argparse
import sys
import threading
import time

import cv2
from ultralytics import YOLO

from src.detectors.gru import (
    SequenceFallDetector,
    drop_stale_tracks,
    fall_decision,
    load_checkpoint,
)
from src.utils.keypoints import extract_keypoints_from_video, tracked_people

__all__ = ["extract_keypoints_from_video", "run_realtime"]

# YOLO xong thì lấy đúng bản đang nằm trong biến.
latest_frame = None
latest_frame_id = 0
frame_condition = threading.Condition()


def _read_frames(source, stop_event):
    """Luồng 1: đọc liên tục và ghi đè latest_frame.
    """
    global latest_frame, latest_frame_id

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Không mở được nguồn hình: {source}")
        stop_event.set()
        with frame_condition:
            frame_condition.notify_all()
        return

    is_file = not isinstance(source, int)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    interval = (1.0 / fps) if is_file and fps > 1.0 else 0.0
    next_tick = time.perf_counter()

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            snapshot = frame.copy()
            with frame_condition:
                latest_frame = snapshot
                latest_frame_id += 1
                frame_condition.notify()

            if interval <= 0.0:
                continue
            next_tick += interval
            delay = next_tick - time.perf_counter()
            if delay > 0.0:
                if stop_event.wait(delay):
                    break
            else:
                next_tick = time.perf_counter()
    finally:
        cap.release()
        stop_event.set()
        with frame_condition:
            frame_condition.notify_all()


def _screen_size():
    """Vùng hiển thị dùng được của màn hình chính, chừa thanh tiêu đề và taskbar."""
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
    """Thu nhỏ khung cho vừa cửa sổ hiển thị và giữ tỉ lệ. Không phóng to ảnh nhỏ hơn."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 0.999:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _annotate(canvas, text, fallen, frame_id, skipped):
    """Vẽ trạng thái ngã và số frame bị bỏ qua lên ảnh BGR."""
    color = (0, 0, 255) if fallen else (0, 180, 0)
    detail = f"{text} | frame {frame_id} | bo qua {skipped}"
    cv2.rectangle(canvas, (8, 8), (8 + 18 * len(detail), 48), (0, 0, 0), -1)
    cv2.putText(canvas, detail, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return canvas


def _run_ai(model, stop_event, conf, fall_model, fall_config):
    """Luồng 2: lấy khung đang nằm trong biến, yolo tìm người, GRU tính xác suất.
    """
    global latest_frame_id

    window = "Fall detection"
    max_width, max_height = _screen_size()
    detectors = {}
    threshold = float(fall_config["decision_threshold"])
    target_fps = float(fall_config["target_fps"])
    min_frames = int(fall_config["min_frames"])

    while True:
        with frame_condition:
            while latest_frame_id == 0 and not stop_event.is_set():
                frame_condition.wait(timeout=0.2)
            if latest_frame is None or (stop_event.is_set() and latest_frame_id == 0):
                break
            frame = latest_frame
            frame_id = latest_frame_id
            skipped = frame_id - 1
            latest_frame_id = 0

        result = model.track(source=frame, persist=True, verbose=False, conf=conf)[0]
        now = time.perf_counter()
        seen = set()
        probabilities = []
        for track_id, keypoints in tracked_people(result):
            seen.add(track_id)
            detector = detectors.get(track_id)
            if detector is None:
                detector = SequenceFallDetector(
                    fall_model,
                    target_fps=target_fps,
                    min_frames=min_frames,
                )
                detectors[track_id] = detector
            probabilities.append(detector.update(keypoints, now))
        drop_stale_tracks(detectors, seen, now, grace=1.0)
        fallen, text = fall_decision(probabilities, threshold)

        display = _fit_frame(
            _annotate(result.plot(), text, fallen, frame_id, skipped),
            max_width,
            max_height,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            stop_event.set()
            with frame_condition:
                frame_condition.notify_all()
            break

    cv2.destroyAllWindows()


def run_realtime(source=0, checkpoint="runs/fall-gru/checkpoint.pt", pose=None, conf=0.25):
    """Chạy luồng đọc hình và luồng AI.

    Parameters
    ----------
    source : int | str
        Chỉ số webcam (0) hoặc đường dẫn video.
    checkpoint : str | Path
        Trọng số GRU và config đã lưu lúc train.
    pose : str | Path | None
        YOLO Pose. None thì lấy pose_model trong checkpoint. Lệnh Hellu.py mặc định truyền yolo11x-pose.pt.
    conf : float
        Ngưỡng điểm của YOLO. Hộp dưới ngưỡng này bị bỏ.
    """
    global latest_frame, latest_frame_id

    fall_model, fall_config = load_checkpoint(checkpoint)
    pose_name = fall_config["pose_model"] if pose is None else pose

    with frame_condition:
        latest_frame = None
        latest_frame_id = 0

    stop_event = threading.Event()
    reader = threading.Thread(
        target=_read_frames,
        name="camera-reader",
        args=(source, stop_event),
        daemon=False,
    )
    reader.start()
    worker = None
    try:
        pose_model = pose_name if isinstance(pose_name, YOLO) else YOLO(str(pose_name))
        worker = threading.Thread(
            target=_run_ai,
            name="ai-worker",
            args=(pose_model, stop_event, conf, fall_model, fall_config),
            daemon=False,
        )
        worker.start()
        while reader.is_alive() and worker.is_alive():
            reader.join(timeout=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        with frame_condition:
            frame_condition.notify_all()
        reader.join()
        if worker is not None:
            worker.join()


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
    parser.add_argument("--pose", default="yolo11x-pose.pt", help="YOLO Pose dùng để lấy khớp. Mặc định là model của project mẫu.")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng điểm YOLO. Hộp dưới ngưỡng bị bỏ.")
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
