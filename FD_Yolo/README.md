# Phát hiện ngã theo keypoints

`yolo26n-pose.pt` trích 17 khớp mỗi người. GRU học phân biệt té và không té từ nhãn thư mục, trên chuỗi tư thế lấy mẫu 10 FPS.

## Dữ liệu

Video đặt theo thư mục, nhãn lấy từ tên thư mục. Train mặc định đọc bộ GMDCSA24:

| Thư mục | Nhãn |
|---|---|
| `GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master/Fall` | 1 |
| `GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos-master/ADL` | 0 |

`data/videos` vẫn nằm trong project (`falls` = 1, `normal` và `no_fall_static` = 0) nhưng không phải dữ liệu của lệnh train mặc định.

Mỗi video thành một chuỗi keypoints của người có bbox tin cậy nhất. Chuỗi được resample về **10 FPS**. Giá trị đó nằm trong `checkpoint.json` cùng trọng số.

## Train

```text
py -m src.train --epochs 20 --output runs/fall-gru/checkpoint.pt
```

`runs/fall-gru/checkpoint.pt` (trọng số)
`runs/fall-gru/checkpoint.json` (config: `target_fps`, `min_frames`, `pose_model`, ngưỡng quyết định).

## Realtime

```text
py Hellu.py --checkpoint runs/fall-gru/checkpoint.pt
py Hellu.py falldemo.mp4 --checkpoint runs/fall-gru/checkpoint.pt
```

