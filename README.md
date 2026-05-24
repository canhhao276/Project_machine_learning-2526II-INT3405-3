# Hệ Thống Phát Hiện Vượt Đèn Đỏ (Traffic Red Light Violation Detection System)

Dự án thị giác máy tính chuyên nghiệp được xây dựng bằng Python, tối ưu cho môi trường **AI CORE ENGINEER**. Hệ thống sử dụng **YOLOv8** để nhận diện phương tiện và đèn tín hiệu, thuật toán bám vết **ByteTrack** để quản lý ID xe, **OpenCV** để xử lý hình ảnh và cấu trúc logic xác định vi phạm luật giao thông khi đèn đỏ.

---

## 📂 Cấu trúc thư mục dự án (Project Structure)

```text
ML_project/
├── requirements.txt                 # Định nghĩa các thư viện phụ thuộc của dự án
├── main.py                         # File chạy chính (Application Entry Point)
├── config.yaml                     # Cấu hình hệ thống (Stop line, model paths, thresholds)
├── README.md                       # Tài liệu hướng dẫn sử dụng chi tiết
├── data/                           # Thư mục lưu trữ dữ liệu
│   ├── input/                      # Nơi đặt video đầu vào để phân tích (.mp4)
│   └── output/                     # Kết quả phân tích (video kết quả, logs vi phạm)
└── src/                            # Thư mục mã nguồn chính (Source Code)
    ├── __init__.py
    ├── config.py                   # Đọc và nạp cấu hình hệ thống từ config.yaml
    ├── core/                       # Logic xử lý nghiệp vụ chính
    │   ├── __init__.py
    │   ├── tracker.py              # Xử lý bám vết vật thể (Wrapper ByteTrack của YOLOv8)
    │   └── violation_detector.py   # Logic phát hiện vượt đèn đỏ tại vạch dừng (Stop Line)
    ├── models/                     # Nhận dạng đối tượng bằng Deep Learning
    │   ├── __init__.py
    │   ├── vehicle_detector.py     # Nhận dạng xe cộ (ô tô, xe máy, xe buýt, xe tải)
    │   └── traffic_light_detector.py # Nhận dạng hộp đèn & phân tích màu HSV (Xanh/Vàng/Đỏ)
    ├── utils/                      # Tiện ích bổ trợ xử lý hình ảnh & luồng
    │   ├── __init__.py
    │   ├── visualization.py        # Vẽ overlays (bounding box, HUD, cảnh báo vi phạm)
    │   └── video_reader.py         # Đọc, ghi và quản lý luồng video bằng OpenCV
    └── api/                        # Cung cấp đầu ra tích hợp cho module khác
        ├── __init__.py
        └── exporter.py             # Xuất log vi phạm định dạng JSON, CSV và Webhook API
```

---

## 🛠️ Giải thích chi tiết chức năng từng File

| Đường dẫn file | Vai trò / Chức năng chi tiết |
| :--- | :--- |
| **`requirements.txt`** | Chứa danh sách các gói thư viện cần thiết như `ultralytics` (YOLOv8), `opencv-python`, `pyyaml` phục vụ quá trình huấn luyện và suy luận. |
| **`config.yaml`** | File cấu hình tập trung chứa tọa độ vạch dừng chân (Stop line), ngưỡng nhận diện `conf`, cổng xuất dữ liệu Webhook, và đường dẫn mô hình. |
| **`main.py`** | Điểm khởi chạy chính. Điều phối vòng lặp đọc video, truyền khung hình qua mô hình phát hiện, bám vết, chạy bộ kiểm tra vi phạm, vẽ hình trực quan và lưu kết quả. |
| **`src/config.py`** | Tự động hóa quá trình nạp và định kiểu dữ liệu cho cấu hình `config.yaml`, tạo sẵn các thư mục cần thiết (`data/input`, `data/output`) để tránh lỗi. |
| **`src/models/vehicle_detector.py`** | Sử dụng YOLOv8 để nhận diện nhanh các loại xe phổ biến (Class IDs 2, 3, 5, 7 trong bộ COCO). |
| **`src/models/traffic_light_detector.py`** | **Dual-mode Traffic Light Classifier**: Tự động nhận diện và sử dụng mô hình Deep Learning fine-tune (YOLOv8 4-class) để phân loại trực tiếp màu đèn (`RED`, `YELLOW`, `GREEN`) bằng AI học sâu học đầu-cuối. Nếu dùng mô hình chung, hệ thống tự động fallback sang thuật toán phân tích không gian màu HSV cải tiến kết hợp định vị đèn dọc/ngang để đảm bảo độ chính xác tuyệt đối. |
| **`src/core/tracker.py`** | Khởi tạo ByteTrack tích hợp sâu từ Ultralytics giúp bám vết ID xe qua từng khung hình cực kỳ mượt mà, loại bỏ việc biên dịch thư viện C++ phức tạp trên Windows. |
| **`src/core/violation_detector.py`** | Logic phát hiện vi phạm cốt lõi. Áp dụng thuật toán giao cắt giữa vector di chuyển của xe (tính bằng điểm đáy bánh xe qua 2 frame liên tiếp) với đoạn thẳng vạch dừng khi đèn tín hiệu ở trạng thái **RED**. |
| **`src/utils/visualization.py`** | Thiết kế giao diện overlays cao cấp (HUD điều khiển hiển thị trạng thái đèn hiện tại, vẽ khung vi phạm màu đỏ nhấp nháy, vạch dừng xanh bắt mắt và thông điệp cảnh báo lớn phía dưới). |
| **`src/utils/video_reader.py`** | Trình đọc/ghi video hiệu suất cao bằng OpenCV kế thừa cơ chế generator trong Python giúp tối ưu bộ nhớ RAM khi chạy video dài hoặc phân giải 4K. |
| **`src/api/exporter.py`** | Module tích hợp API ngoài. Hỗ trợ ghi log vi phạm đồng thời vào file `.json` và file cơ sở dữ liệu bảng `.csv` kèm cơ chế POST REST API Webhook tự động đến server quản trị tập trung. |

---

## 🚀 Hướng dẫn cài đặt và thiết lập môi trường

Do bạn sử dụng **Windows** và trình soạn thảo **VS Code**, hãy chạy các câu lệnh Powershell dưới đây tại thư mục gốc của dự án:

### 1. Tạo và kích hoạt Môi trường ảo (Python Virtual Environment)
Tạo môi trường ảo giúp dự án hoạt động cô lập, tránh xung đột thư viện toàn cục:
```powershell
# Tạo môi trường ảo với tên venv
python -m venv venv

# Kích hoạt môi trường ảo trên Windows (Powershell)
.\venv\Scripts\Activate
```
> *Lưu ý:* Nếu bạn gặp lỗi bảo mật quyền chạy script của Powershell, hãy chạy lệnh sau một lần: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`.

### 2. Cài đặt các thư viện phụ thuộc
Nâng cấp trình quản lý gói `pip` và tải toàn bộ các thư viện được định nghĩa sẵn:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🏃 Hướng dẫn khởi chạy hệ thống

### Bước 1: Chuẩn bị dữ liệu đầu vào
- Tạo hoặc sử dụng thư mục đã tự động tạo `data/input/`.
- Copy video giao thông của bạn vào thư mục này và đặt tên là `traffic.mp4` (hoặc tên bất kỳ).

### Bước 2: Khởi chạy Pipeline phân tích
Chạy lệnh cơ bản (chế độ headless - lưu video kết quả và logs):
```powershell
python main.py
```

Chạy kèm tham số tùy chỉnh video đầu vào và hiển thị khung hình xem trước thời gian thực (Live Preview):
```powershell
python main.py --video data/input/your_traffic_video.mp4 --preview
```
*(Trong quá trình xem trước, bạn có thể nhấn phím `q` để dừng chương trình bất cứ lúc nào).*

### Bước 3: Xem kết quả đầu ra
Sau khi xử lý xong, bạn sẽ nhận được:
- Video đã vẽ overlays trực quan tại: `data/output/result_video.mp4`
- Nhật ký dữ liệu vi phạm định dạng JSON tại: `data/output/violations.json`
- Bảng thống kê dữ liệu vi phạm định dạng CSV tại: `data/output/violations.csv`
