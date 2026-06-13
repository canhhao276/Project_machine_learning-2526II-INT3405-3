# 🚦 Hệ Thống Phát Hiện Vi Phạm & Điều Khiển Đèn Giao Thông Thích Ứng

Dự án thị giác máy tính xây dựng bằng Python. Hệ thống sử dụng **YOLOv8** để nhận diện phương tiện và đèn tín hiệu, **ByteTrack** để bám vết đa đối tượng, **SVM** để phân loại trạng thái đèn và mật độ giao thông, **OpenCV** để xử lý video. Hỗ trợ ba chế độ hoạt động độc lập: phát hiện vượt đèn đỏ (`violation`), điều khiển đèn thích ứng (`adaptive`), và so sánh song song (`compare`).

---

## 📂 Cấu trúc thư mục dự án

```text
ML_project/
├── main.py                                  # File chạy chính, hỗ trợ 3 chế độ hoạt động
├── config.yaml                              # Cấu hình toàn bộ hệ thống
├── requirements.txt                         # Danh sách thư viện phụ thuộc
├── data/
│   ├── input/                               # Đặt video đầu vào tại đây (.mp4)
│   ├── output/                              # Video kết quả, violations.json, violations.csv
│   └── traffic_light/                       # Dữ liệu huấn luyện SVM
│       ├── dataset.yaml                     # Cấu hình fine-tune YOLOv8 đèn giao thông
│       ├── train_features.csv               # Đặc trưng màu đèn (tập huấn luyện)
│       ├── val_features.csv                 # Đặc trưng màu đèn (tập kiểm thử)
│       └── density_features.csv             # Đặc trưng mật độ xe
├── Luutru_Vipham/                           # Thư mục lưu trữ vi phạm (tự tạo khi chạy)
│   └── YYYY-MM-DD/
│       └── Vuot_Den_Do/
│           └── VehicleID_42_car_F00123_.../
│               └── VehicleID_42_car_F00123_....mp4
└── src/
    ├── config.py                            # Đọc và kiểm tra cấu hình config.yaml
    ├── core/
    │   ├── tracker.py                       # VehicleTracker: YOLOv8 + ByteTrack, quản lý track ID
    │   ├── geometry.py                      # Cross product kiểm tra giao cắt vạch dừng
    │   ├── zones.py                         # Định nghĩa StopLine, RightTurnZone
    │   ├── violation_detector.py            # Điều phối logic phát hiện vi phạm
    │   ├── violation_logic.py               # Quản lý trạng thái từng xe, cooldown per ID
    │   ├── adaptive_light_controller.py     # State machine điều khiển đèn thích ứng
    │   └── bytetrack_traffic.yaml           # Cấu hình ByteTrack
    ├── models/
    │   ├── vehicle_detector.py              # YOLOv8 nhận diện phương tiện (COCO ID 2,3,5,7)
    │   ├── traffic_light_detector.py        # Dual-mode: YOLOv8 fine-tuned + HSV fallback
    │   ├── train_svm.py                     # Huấn luyện SVM phân loại màu đèn
    │   ├── train_density_svm.py             # Huấn luyện SVM phân loại mật độ xe
    │   ├── extract_features.py              # Trích xuất đặc trưng màu đèn từ video
    │   └── collect_density_features.py      # Thu thập đặc trưng mật độ xe từ video
    ├── storage/
    │   └── video_clip_extractor.py          # Ring buffer + tự động cắt clip vi phạm
    ├── utils/
    │   ├── config_loader.py                 # Nạp cấu hình từ config.yaml
    │   ├── video_reader.py                  # Đọc, ghi và quản lý luồng video
    │   ├── visualization.py                 # Vẽ overlay toàn cảnh lên frame
    │   └── overlay.py                       # HUD đèn, bounding box, cảnh báo vi phạm
    └── api/
        └── exporter.py                      # Xuất log JSON, CSV và POST Webhook
```

---

## 🛠️ Giải thích chi tiết chức năng từng file

| Đường dẫn | Vai trò / Chức năng |
| :--- | :--- |
| **`main.py`** | Điểm khởi chạy chính. Điều phối vòng lặp frame, phân nhánh 3 chế độ, ghép frame 1920×540 trong chế độ `compare`. |
| **`config.yaml`** | Cấu hình tập trung: tọa độ vạch dừng, vùng rẽ phải, đường dẫn model, ngưỡng confidence, tham số lưu clip. |
| **`src/config.py`** | Nạp và kiểm tra kiểu dữ liệu `config.yaml`, tự động tạo thư mục `data/input`, `data/output`. |
| **`src/core/tracker.py`** | Bọc YOLOv8 + ByteTrack, ánh xạ raw ID → compact ID, lưu `track_history` tối đa 50 điểm để tính tốc độ. |
| **`src/core/geometry.py`** | Hàm `has_crossed_line()` dùng cross product kiểm tra xe có cắt qua vạch dừng không. |
| **`src/core/zones.py`** | Định nghĩa `StopLine` và `RightTurnZone` từ tọa độ trong config. |
| **`src/core/violation_detector.py`** | Điều phối phát hiện vi phạm: kiểm tra đèn đỏ → giao cắt vạch dừng → hủy nếu rẽ phải. |
| **`src/core/violation_logic.py`** | Quản lý trạng thái từng xe (track_id), kiểm soát cooldown để tránh ghi trùng vi phạm. |
| **`src/core/adaptive_light_controller.py`** | State machine 3 trạng thái (GREEN→YELLOW→RED), điều chỉnh thời gian pha đèn theo mật độ SVM (Green Lock, rút ngắn pha). |
| **`src/models/vehicle_detector.py`** | YOLOv8 phát hiện 4 lớp xe COCO (car, motorcycle, bus, truck). Mở rộng bounding box xe máy lên 60% để bao gồm người lái. |
| **`src/models/traffic_light_detector.py`** | **Dual-mode**: ưu tiên YOLOv8 fine-tuned 4 lớp (green/yellow/red/off) + SVM 27 đặc trưng; tự động fallback sang phân tích HSV nếu lỗi. Có temporal smoothing và hybrid correction cho đèn vàng. |
| **`src/models/train_svm.py`** | Huấn luyện SVM (kernel RBF) phân loại màu đèn từ `train_features.csv`, lưu `svm_traffic_light.pkl`. |
| **`src/models/train_density_svm.py`** | Huấn luyện SVM phân loại mật độ 3 mức (Low/Medium/High) từ `density_features.csv`, lưu `svm_traffic_density.pkl`. |
| **`src/models/extract_features.py`** | Trích xuất vector đặc trưng 27 chiều (màu sắc toàn cục + theo vùng) từ ảnh crop đèn để huấn luyện SVM. |
| **`src/models/collect_density_features.py`** | Chạy qua video thực, thu thập 5 đặc trưng mật độ (motorcycle_count, car_count, stopped_vehicles, pcu_load, average_speed) và gán nhãn bán tự động theo ngưỡng PCU. |
| **`src/storage/video_clip_extractor.py`** | Ring buffer giữ N frame gần nhất. Khi có vi phạm → tự động ghép clip (1s trước + 2s sau) lưu theo cấu trúc `Luutru_Vipham/YYYY-MM-DD/...`. Hủy clip nếu xe rẽ phải. |
| **`src/utils/video_reader.py`** | Generator đọc frame hiệu suất cao, chuẩn hóa về `target_resolution`, khởi tạo `VideoWriter` đầu ra. |
| **`src/utils/visualization.py`** | Vẽ toàn bộ overlay: bounding box, vạch dừng xanh, vùng rẽ phải, HUD đèn, khung đỏ nhấp nháy, PCU load. |
| **`src/utils/overlay.py`** | Các hàm vẽ chi tiết: HUD trạng thái đèn, thông báo vi phạm, đếm ngược thời gian pha. |
| **`src/api/exporter.py`** | Ghi log vi phạm đồng thời vào `.json` và `.csv`; POST REST Webhook tự động; hủy bản ghi khi xe rẽ phải. |

---

## 🚀 Hướng dẫn cài đặt và thiết lập môi trường

### 1. Tạo và kích hoạt môi trường ảo

```powershell
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows PowerShell)
.\venv\Scripts\Activate
```

> *Nếu PowerShell báo lỗi quyền thực thi, chạy lệnh sau một lần:*
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

### 2. Cài đặt thư viện

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## 🏃 Hướng dẫn khởi chạy hệ thống

### Bước 1: Chuẩn bị video đầu vào

Đặt file video vào thư mục `data/input/`. Độ phân giải khuyến nghị: `1280×720`.

### Bước 2: Khởi chạy pipeline

```powershell
# Chế độ phát hiện vượt đèn đỏ (mặc định)
python main.py --mode violation --video data/input/traffic.mp4

# Chế độ điều khiển đèn thích ứng theo mật độ xe
python main.py --mode adaptive --video data/input/traffic.mp4

# Chế độ so sánh song song cả 2 chức năng (frame 1920×540)
python main.py --mode compare --video data/input/traffic.mp4

# Bật xem trước thời gian thực (nhấn Q để dừng)
python main.py --mode violation --video data/input/traffic.mp4 --preview
```

### Bước 3: Xem kết quả đầu ra

Sau khi xử lý xong, kết quả được lưu tại:

| File | Mô tả |
| :--- | :--- |
| `data/output/result_video.mp4` | Video gốc với overlay trực quan |
| `data/output/violations.json` | Log vi phạm định dạng JSON |
| `data/output/violations.csv` | Log vi phạm định dạng CSV |
| `Luutru_Vipham/YYYY-MM-DD/Vuot_Den_Do/.../` | Clip video từng vi phạm (1s trước + 2s sau) |
