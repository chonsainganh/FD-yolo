# Phát hiện ngã theo keypoints

YOLO Pose trích 17 khớp mỗi người. GRU học phân biệt té và không té từ nhãn thư mục, trên chuỗi tư thế lấy mẫu 10 FPS.

## Dữ liệu

Video đặt theo thư mục, nhãn lấy từ tên thư mục:

| Thư mục | Nhãn |
|---|---|
| `data/videos/falls` | 1 |
| `data/videos/normal` | 0 |
| `data/videos/no_fall_static` | 0 |

Nguồn: https://drive.google.com/drive/folders/1A83twlTOPE_-QZElOKOnG8TzfI6Do3TY

Mỗi video thành một chuỗi keypoints của người có bbox tin cậy nhất. Chuỗi được resample về **10 FPS**. Giá trị đó nằm trong `checkpoint.json` cùng trọng số.

## Train

```text
py -m src.train --data data/videos --epochs 20 --output runs/fall-gru/checkpoint.pt
```

`runs/fall-gru/checkpoint.pt` (trọng số)
`runs/fall-gru/checkpoint.json` (config: `target_fps`, `min_frames`, `pose_model`, ngưỡng quyết định).

## Realtime

```text
py Hellu.py --checkpoint runs/fall-gru/checkpoint.pt
py Hellu.py falldemo.mp4 --checkpoint runs/fall-gru/checkpoint.pt
```

Mỗi `track_id` có một bộ nhớ GRU riêng, cập nhật ở 10 FPS. Xác suất chỉ hiện sau `min_frames` khung (mặc định một giây). Bộ dữ liệu trên chưa được kiểm chứng cho nhiều người trong cùng một khung.
