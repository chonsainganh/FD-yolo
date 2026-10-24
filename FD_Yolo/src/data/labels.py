"""Gán nhãn video theo tên thư mục."""

from pathlib import Path

LABEL_BY_FOLDER = {
    "falls": 1,
    "normal": 0,
    "no_fall_static": 0,
}

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def labeled_videos(root: str | Path) -> list[tuple[Path, int]]:
    """Trả về (đường dẫn, nhãn) cho video trong falls, normal và no_fall_static.

    Nhãn 1 là té ngã. Nhãn 0 là hoạt động thường hoặc tư thế nằm yên.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dữ liệu: {root}")

    found: list[tuple[Path, int]] = []
    for name, label in LABEL_BY_FOLDER.items():
        folder = root / name
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                found.append((path, label))

    if not found:
        raise ValueError(
            f"Không có video trong {root}. Cần thư mục falls, normal hoặc no_fall_static."
        )
    found.sort(key=lambda item: str(item[0]).lower())
    return found
