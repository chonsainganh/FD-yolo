"""Train GRU phát hiện ngã từ video đã gán nhãn theo thư mục."""

import argparse
import random
import sys

import numpy as np
import torch
from torch import nn

from src.data.labels import labeled_videos
from src.detectors.gru import FallGRU, save_checkpoint


def split_indices(labels: list[int], val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    """Giữ phần đuôi của từng nhãn làm validation. Nhãn chỉ có một mẫu thì để lại train."""
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction phải nằm trong [0, 1).")

    groups = {0: [], 1: []}
    for index, label in enumerate(labels):
        label = int(label)
        if label not in groups:
            raise ValueError("Nhãn chỉ được là 0 hoặc 1.")
        groups[label].append(index)

    train: list[int] = []
    val: list[int] = []
    for label in (0, 1):
        indices = groups[label]
        if val_fraction <= 0 or len(indices) <= 1:
            train.extend(indices)
            continue
        val_count = int(round(len(indices) * val_fraction))
        val_count = min(max(val_count, 1), len(indices) - 1)
        val.extend(indices[-val_count:])
        train.extend(indices[:-val_count])
    return sorted(train), sorted(val)


def fit(
    sequences: list[np.ndarray],
    labels: list[int],
    *,
    epochs: int = 20,
    hidden_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    val_fraction: float = 0.2,
    batch_size: int = 8,
    device: str | None = None,
) -> tuple[FallGRU, dict]:
    """Học logit té ngã. Mỗi phần tử sequences có dạng (khung, khớp, 3)."""
    if epochs < 1:
        raise ValueError("epochs phải lớn hơn hoặc bằng 1.")
    if len(sequences) == 0 or len(sequences) != len(labels):
        raise ValueError("Mỗi chuỗi keypoints cần đúng một nhãn.")

    labels = [int(label) for label in labels]
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Nhãn chỉ được là 0 hoặc 1.")
    if sum(labels) in (0, len(labels)):
        raise ValueError("Cần cả mẫu té (1) và mẫu không té (0).")
    if any(seq.ndim != 3 or seq.shape[0] < 1 or seq.shape[2] != 3 for seq in sequences):
        raise ValueError("Mỗi chuỗi phải có dạng (số khung, số khớp, 3).")
    keypoint_counts = {int(seq.shape[1]) for seq in sequences}
    if len(keypoint_counts) != 1:
        raise ValueError("Mọi chuỗi phải có cùng số khớp.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_index, val_index = split_indices(labels, val_fraction)
    train_labels = [labels[index] for index in train_index]
    if sum(train_labels) in (0, len(train_labels)):
        raise ValueError("Tập train cần cả hai nhãn. Thêm video hoặc giảm phần validation.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    model = FallGRU(num_keypoints=keypoint_counts.pop(), hidden_size=hidden_size).to(torch_device)
    positive = sum(train_labels)
    negative = len(train_labels) - positive
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative / positive], dtype=torch.float32, device=torch_device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_train = None
    best_val = None
    best_accuracy = None
    rng = random.Random(seed)
    for _ in range(epochs):
        model.train()
        order = list(train_index)
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch_ids = order[start : start + batch_size]
            features, lengths = _collate([sequences[index] for index in batch_ids])
            target = torch.tensor(
                [labels[index] for index in batch_ids],
                dtype=torch.float32,
                device=torch_device,
            )
            optimizer.zero_grad()
            logits = model(features.to(torch_device), lengths)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch_ids)

        train_loss = total / len(train_index)
        val_loss, val_accuracy = _evaluate(
            model, sequences, labels, val_index, criterion, torch_device
        )
        if val_loss is None:
            improved = best_state is None or train_loss < best_train
        else:
            improved = best_state is None or val_loss < best_val
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_train = train_loss
            best_val = val_loss
            best_accuracy = val_accuracy

    model.load_state_dict(best_state)
    model.to(torch_device)
    model.eval()
    return model, {
        "train_loss": best_train,
        "val_loss": best_val,
        "val_accuracy": best_accuracy,
        "epochs": epochs,
    }


def train(
    data_dir,
    output,
    epochs: int = 20,
    target_fps: float = 10.0,
    hidden_size: int = 64,
    lr: float = 1e-3,
    pose_model: str = "yolo26n-pose.pt",
    seed: int = 0,
    conf: float = 0.25,
    min_frames: int | None = None,
) -> dict:
    """Trích chuỗi keypoints từ video rồi train và lưu checkpoint kèm config."""
    from ultralytics import YOLO

    from src.utils.keypoints import extract_pose_sequence

    samples = labeled_videos(data_dir)
    pose = pose_model if isinstance(pose_model, YOLO) else YOLO(str(pose_model))
    sequences = []
    labels = []
    for index, (path, label) in enumerate(samples, start=1):
        print(f"Trích keypoints {index}/{len(samples)}: {path}")
        sequences.append(
            extract_pose_sequence(path, target_fps=target_fps, model=pose, conf=conf)
        )
        labels.append(label)

    model, history = fit(
        sequences,
        labels,
        epochs=epochs,
        hidden_size=hidden_size,
        lr=lr,
        seed=seed,
    )
    if min_frames is None:
        min_frames = max(1, int(round(target_fps)))
    config = {
        "target_fps": float(target_fps),
        "num_keypoints": model.num_keypoints,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "min_frames": int(min_frames),
        "decision_threshold": 0.5,
        "pose_model": str(pose_model),
        "feature_order": "xy_conf",
    }
    save_checkpoint(output, model, config)
    val_loss = history["val_loss"]
    val_accuracy = history["val_accuracy"]
    val_loss_text = "không có" if val_loss is None else f"{val_loss:.4f}"
    val_accuracy_text = "không có" if val_accuracy is None else f"{val_accuracy:.3f}"
    print(
        f"Đã lưu {output}. Train loss {history['train_loss']:.4f}. "
        f"Val loss {val_loss_text}. Val accuracy {val_accuracy_text}."
    )
    return history


def _collate(sequences: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [int(sequence.shape[0]) for sequence in sequences]
    keypoints = int(sequences[0].shape[1])
    batch = np.zeros((len(sequences), max(lengths), keypoints, 3), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        batch[index, : sequence.shape[0]] = sequence
    return torch.from_numpy(batch), torch.tensor(lengths, dtype=torch.long)


def _evaluate(model, sequences, labels, indices, criterion, device):
    if not indices:
        return None, None
    model.eval()
    features, lengths = _collate([sequences[index] for index in indices])
    target = torch.tensor([labels[index] for index in indices], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(features.to(device), lengths)
        loss = float(criterion(logits, target).item())
        correct = (torch.sigmoid(logits) >= 0.5) == target.bool()
        accuracy = float(correct.float().mean().item())
    return loss, accuracy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GRU phát hiện ngã trên chuỗi keypoints.")
    parser.add_argument("--data", default="data/videos", help="Thư mục chứa falls, normal, no_fall_static.")
    parser.add_argument("--output", default="runs/fall-gru/checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pose", default="yolo26n-pose.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser


def _use_utf8_stdio() -> None:
    """Windows console mặc định không in được tiếng Việt trong --help và log train."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass


def main(argv: list[str] | None = None) -> None:
    _use_utf8_stdio()
    args = build_parser().parse_args(argv)
    train(
        data_dir=args.data,
        output=args.output,
        epochs=args.epochs,
        target_fps=args.target_fps,
        hidden_size=args.hidden_size,
        lr=args.lr,
        pose_model=args.pose,
        seed=args.seed,
        conf=args.conf,
    )


if __name__ == "__main__":
    main()
