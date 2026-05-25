"""
========================================================================
[NHIỆM VỤ KỸ THUẬT CỦA THÀNH VIÊN 3: STORAGE & TESTING ENGINEER]
========================================================================
Module này thực hiện:
  1. Thiết kế cấu trúc thư mục phân loại ảnh vi phạm tự động:
         Luutru_Vipham/
         └── YYYY-MM-DD/
             └── Vuot_Den_Do/
                 ├── VehicleID_42_car_F0123_20250525_143201.jpg
                 └── ...
  2. Crop ảnh xe vi phạm bằng cv2.imwrite, đặt tên file chuẩn và lưu đúng thư mục.
  3. Ghi log JSON + CSV, đo FPS trung bình, thống kê Precision/Recall cuối session.
========================================================================
"""

import os
import csv
import json
import time
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
#  CẤU HÌNH MẶC ĐỊNH
# ─────────────────────────────────────────────────────────────
DEFAULT_ROOT_DIR      = "Luutru_Vipham"   # Thư mục gốc lưu ảnh vi phạm
DEFAULT_JSON_PATH     = "data/output/violations.json"
DEFAULT_CSV_PATH      = "data/output/violations.csv"
CROP_PADDING          = 20                # Pixel padding thêm xung quanh bbox khi crop
THUMBNAIL_MAX_SIZE    = (320, 240)        # Kích thước thumbnail tối đa (tùy chọn)


# ─────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────

def _safe_crop(frame: np.ndarray, bbox: List[int], padding: int = CROP_PADDING) -> Optional[np.ndarray]:
    """
    Cắt vùng bbox từ frame, thêm padding an toàn (không vượt biên).
    Trả về None nếu bbox không hợp lệ.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Thêm padding và kẹp trong biên ảnh
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()


def _build_save_path(root_dir: str, vehicle_id: int, vehicle_type: str,
                     frame_idx: int, timestamp: float) -> str:
    """
    Xây dựng đường dẫn lưu file ảnh vi phạm theo cấu trúc:
        <root_dir>/<YYYY-MM-DD>/Vuot_Den_Do/<filename>.jpg

    Tên file: VehicleID_{id}_{type}_F{frame:05d}_{datetime}.jpg
    Ví dụ:    VehicleID_42_car_F00123_20250525_143201.jpg
    """
    date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
    time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")

    save_dir = os.path.join(root_dir, date_str, "Vuot_Den_Do")
    os.makedirs(save_dir, exist_ok=True)

    filename = f"VehicleID_{vehicle_id}_{vehicle_type}_F{frame_idx:05d}_{time_str}.jpg"
    return os.path.join(save_dir, filename)


# ─────────────────────────────────────────────────────────────
#  FPS TRACKER
# ─────────────────────────────────────────────────────────────

class FPSTracker:
    """
    Đo tốc độ xử lý FPS theo thời gian thực (rolling window) và tổng thể.

    Cách dùng:
        fps_tracker = FPSTracker()
        fps_tracker.start()
        for frame in ...:
            fps_tracker.tick()
            current_fps = fps_tracker.get_fps()
        avg_fps = fps_tracker.get_average_fps()
    """

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._timestamps: List[float] = []
        self._start_time: Optional[float] = None
        self._total_ticks: int = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Bắt đầu đo (gọi trước vòng lặp chính)."""
        self._start_time = time.perf_counter()
        self._total_ticks = 0
        self._timestamps.clear()

    def tick(self) -> None:
        """Ghi nhận một frame đã xử lý xong."""
        now = time.perf_counter()
        with self._lock:
            self._timestamps.append(now)
            self._total_ticks += 1
            # Giữ rolling window
            if len(self._timestamps) > self._window_size:
                self._timestamps.pop(0)

    def get_fps(self) -> float:
        """FPS hiện tại (rolling window)."""
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / elapsed

    def get_average_fps(self) -> float:
        """FPS trung bình toàn bộ session."""
        if self._start_time is None or self._total_ticks == 0:
            return 0.0
        elapsed = time.perf_counter() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._total_ticks / elapsed

    def get_total_frames(self) -> int:
        return self._total_ticks


# ─────────────────────────────────────────────────────────────
#  PRECISION / RECALL CALCULATOR
# ─────────────────────────────────────────────────────────────

class MetricsCalculator:
    """
    Tính Precision và Recall sau khi đối chiếu thủ công:

        Precision = TP / (TP + FP)
        Recall    = TP / (TP + FN)
        F1        = 2 * P * R / (P + R)

    Cách dùng:
        calc = MetricsCalculator()
        calc.set_ground_truth(total_actual_violations=12)
        calc.set_detections(true_positives=10, false_positives=2)
        report = calc.get_report()
    """

    def __init__(self):
        self._tp: int = 0   # Đúng: hệ thống bắt đúng xe vi phạm thật
        self._fp: int = 0   # Sai dương: hệ thống báo vi phạm nhưng thực ra không vi phạm
        self._fn: int = 0   # Sai âm: xe vi phạm thật nhưng hệ thống bỏ sót

    def set_ground_truth(self, total_actual_violations: int) -> None:
        """Nhập tổng số vi phạm thực tế (đếm thủ công khi xem video)."""
        self._fn = max(0, total_actual_violations - self._tp)

    def set_detections(self, true_positives: int, false_positives: int) -> None:
        """Nhập kết quả đối chiếu thủ công: TP và FP."""
        self._tp = true_positives
        self._fp = false_positives

    def get_precision(self) -> float:
        denom = self._tp + self._fp
        return self._tp / denom if denom > 0 else 0.0

    def get_recall(self) -> float:
        denom = self._tp + self._fn
        return self._tp / denom if denom > 0 else 0.0

    def get_f1(self) -> float:
        p, r = self.get_precision(), self.get_recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def get_report(self) -> Dict[str, Any]:
        return {
            "true_positives":  self._tp,
            "false_positives": self._fp,
            "false_negatives": self._fn,
            "precision":       round(self.get_precision(), 4),
            "recall":          round(self.get_recall(), 4),
            "f1_score":        round(self.get_f1(), 4),
        }

    def print_report(self, fps_tracker: Optional["FPSTracker"] = None) -> None:
        """In bảng thống kê ra console."""
        sep = "=" * 55
        print(f"\n{sep}")
        print("  BẢNG THỐNG KÊ ĐÁNH GIÁ HỆ THỐNG ")
        print(sep)
        r = self.get_report()
        print(f"  True  Positives  (TP): {r['true_positives']:>6}")
        print(f"  False Positives  (FP): {r['false_positives']:>6}  ← Báo nhầm")
        print(f"  False Negatives  (FN): {r['false_negatives']:>6}  ← Bỏ sót")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Precision            : {r['precision']:>8.2%}")
        print(f"  Recall               : {r['recall']:>8.2%}")
        print(f"  F1-Score             : {r['f1_score']:>8.4f}")
        if fps_tracker:
            print(f"  ─────────────────────────────────────────────")
            print(f"  Tổng frames xử lý   : {fps_tracker.get_total_frames():>6}")
            print(f"  FPS trung bình      : {fps_tracker.get_average_fps():>8.2f}")
        print(sep + "\n")


# ─────────────────────────────────────────────────────────────
#  VIOLATION EXPORTER  (lớp chính – Thành viên 2 gọi vào đây)
# ─────────────────────────────────────────────────────────────

class ViolationExporter:
    """
    Module lưu trữ và thống kê vi phạm vượt đèn đỏ.

    Pipeline mỗi khi phát hiện vi phạm:
      1. Crop ảnh xe từ frame gốc  →  lưu vào Luutru_Vipham/<date>/Vuot_Den_Do/
      2. Ghi bản ghi vào violations.json  (append)
      3. Ghi bản ghi vào violations.csv   (append)
      4. Gọi Webhook nếu có cấu hình      (non-blocking thread)

    Tích hợp với main.py:
      - Khởi tạo 1 lần:  exporter = ViolationExporter(...)
      - Mỗi frame:       exporter.fps_tracker.tick()
      - Mỗi vi phạm:     exporter.export_event(violation, frame)
      - Cuối session:    exporter.print_summary()
    """

    def __init__(
        self,
        json_path:   str = DEFAULT_JSON_PATH,
        webhook_url: str = "",
        root_dir:    str = DEFAULT_ROOT_DIR,
        save_crops:  bool = True,
    ):
        self.json_path   = json_path
        self.csv_path    = os.path.splitext(json_path)[0] + ".csv"
        self.webhook_url = webhook_url
        self.root_dir    = root_dir
        self.save_crops  = save_crops

        # Đảm bảo thư mục output tồn tại
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)

        # Bộ nhớ nội tại
        self._violations: List[Dict[str, Any]] = []
        self._exported_ids: set = set()

        # Khởi tạo file CSV và JSON (đảm bảo file luôn tồn tại từ đầu)
        self._init_csv()
        self._init_json()

        # FPS tracker (main.py gọi fps_tracker.tick() mỗi frame)
        self.fps_tracker = FPSTracker(window_size=30)
        self.fps_tracker.start()

        # Metrics (điền sau khi đối chiếu thủ công)
        self.metrics = MetricsCalculator()

        print(f"[Exporter] Khởi động. Ảnh vi phạm sẽ lưu tại: {os.path.abspath(root_dir)}/")
        print(f"[Exporter] JSON log : {os.path.abspath(json_path)}")
        print(f"[Exporter] CSV  log : {os.path.abspath(self.csv_path)}")

    # ── KHỞI TẠO FILE CSV ────────────────────────────────────

    def _init_csv(self) -> None:
        # Luôn ghi đè (overwrite) tệp CSV cũ để bắt đầu phiên mới sạch sẽ
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writeheader()

    @staticmethod
    def _csv_fields() -> List[str]:
        return [
            "vehicle_id", "vehicle_type", "frame", "timestamp",
            "datetime", "confidence", "bbox", "crop_path"
        ]

    # ── KHỞI TẠO FILE JSON ───────────────────────────────────

    def _init_json(self) -> None:
        # Luôn ghi đè (overwrite) tệp JSON cũ thành mảng rỗng [] mới hoàn toàn cho phiên mới
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    # ── HÀM CHÍNH: TIẾP NHẬN VI PHẠM TỪ THÀNH VIÊN 2 ───────

    def export_event(
        self,
        violation_event: Dict[str, Any],
        frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Tiếp nhận dữ liệu vi phạm từ ViolationDetector và:
          - Crop + lưu ảnh xe vi phạm (nếu frame được truyền vào)
          - Append JSON log
          - Append CSV log
          - Gọi Webhook (non-blocking)

        Tham số:
            violation_event: dict từ ViolationDetector.process_frame()
                {vehicle_id, vehicle_type, bbox, timestamp, frame, confidence}
            frame: numpy array của frame gốc (để crop ảnh).
                   Nếu None thì chỉ ghi log, không lưu ảnh.

        Trả về True nếu thành công.
        """
        try:
            v_id       = violation_event["vehicle_id"]
            v_type     = violation_event.get("vehicle_type", "unknown")
            bbox       = violation_event.get("bbox", [0, 0, 0, 0])
            timestamp  = violation_event.get("timestamp", time.time())
            frame_idx  = violation_event.get("frame", 0)
            confidence = violation_event.get("confidence", 0.0)

            # ── 1. CROP VÀ LƯU ẢNH ───────────────────────────
            crop_path = ""
            if self.save_crops and frame is not None:
                crop_img = _safe_crop(frame, bbox, padding=CROP_PADDING)
                if crop_img is not None:
                    crop_path = _build_save_path(
                        self.root_dir, v_id, v_type, frame_idx, timestamp
                    )
                    # Vẽ thông tin lên ảnh crop trước khi lưu
                    _annotate_crop(crop_img, v_id, v_type, frame_idx, confidence)
                    cv2.imwrite(crop_path, crop_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    print(f"[Exporter] ✓ Đã lưu ảnh: {crop_path}")

            # ── 2. XÂY DỰNG BẢN GHI ─────────────────────────
            record = {
                "vehicle_id":   v_id,
                "vehicle_type": v_type,
                "frame":        frame_idx,
                "timestamp":    round(timestamp, 3),
                "datetime":     datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "confidence":   round(confidence, 4),
                "bbox":         bbox,
                "crop_path":    crop_path,
            }
            self._violations.append(record)

            # ── 3. GHI JSON (toàn bộ list, overwrite) ────────
            self._write_json()

            # ── 4. APPEND CSV ─────────────────────────────────
            self._append_csv(record)

            # ── 5. WEBHOOK (non-blocking) ─────────────────────
            if self.webhook_url:
                threading.Thread(
                    target=self._post_webhook,
                    args=(record,),
                    daemon=True
                ).start()

            return True

        except Exception as e:
            print(f"[Exporter] ✗ Lỗi khi export: {e}")
            return False

    # ── GHI FILE ─────────────────────────────────────────────

    def _write_json(self) -> None:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._violations, f, ensure_ascii=False, indent=2)

    def _append_csv(self, record: Dict[str, Any]) -> None:
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fields())
            writer.writerow({
                "vehicle_id":   record["vehicle_id"],
                "vehicle_type": record["vehicle_type"],
                "frame":        record["frame"],
                "timestamp":    record["timestamp"],
                "datetime":     record["datetime"],
                "confidence":   record["confidence"],
                "bbox":         str(record["bbox"]),
                "crop_path":    record["crop_path"],
            })

    def _post_webhook(self, record: Dict[str, Any]) -> None:
        try:
            import requests
            payload = {k: v for k, v in record.items() if k != "crop_path"}
            resp = requests.post(self.webhook_url, json=payload, timeout=5)
            if resp.status_code == 200:
                print(f"[Webhook] ✓ Đã gửi vi phạm ID={record['vehicle_id']}")
            else:
                print(f"[Webhook] ✗ HTTP {resp.status_code}")
        except Exception as e:
            print(f"[Webhook] ✗ Lỗi: {e}")

    # ── TRUY VẤN ─────────────────────────────────────────────

    def get_latest_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Trả về `limit` bản ghi vi phạm gần nhất."""
        return self._violations[-limit:]

    def get_total_violations(self) -> int:
        return len(self._violations)

    # ── THỐNG KÊ & IN KẾT QUẢ ────────────────────────────────

    def print_summary(
        self,
        total_actual: Optional[int] = None,
        true_positives: Optional[int] = None,
        false_positives: Optional[int] = None,
    ) -> None:
        """
        In báo cáo tổng kết cuối session.

        Nếu truyền vào các thông số đối chiếu thủ công thì in cả Precision/Recall.

        Ví dụ (sau khi xem video và đếm tay):
            exporter.print_summary(
                total_actual    = 12,   # Thực tế có 12 xe vượt đèn đỏ
                true_positives  = 10,   # Hệ thống bắt đúng 10
                false_positives = 2,    # Hệ thống báo nhầm 2
            )
        """
        sep = "=" * 55
        print(f"\n{sep}")
        print("  TÓM TẮT SESSION : STORAGE & TESTING")
        print(sep)
        print(f"  Tổng vi phạm hệ thống ghi nhận : {self.get_total_violations()}")
        print(f"  Thư mục ảnh vi phạm            : {os.path.abspath(self.root_dir)}/")
        print(f"  File JSON log                  : {os.path.abspath(self.json_path)}")
        print(f"  File CSV  log                  : {os.path.abspath(self.csv_path)}")

        if true_positives is not None and false_positives is not None:
            self.metrics.set_detections(true_positives, false_positives)
        if total_actual is not None:
            self.metrics.set_ground_truth(total_actual)

        if true_positives is not None:
            self.metrics.print_report(self.fps_tracker)
        else:
            avg = self.fps_tracker.get_average_fps()
            total = self.fps_tracker.get_total_frames()
            print(f"\n  Tổng frames xử lý : {total}")
            print(f"  FPS trung bình     : {avg:.2f}")
            print(f"\n  (Để tính Precision/Recall: đếm thủ công trên video,")
            print(f"   rồi gọi exporter.print_summary(total_actual=...,")
            print(f"   true_positives=..., false_positives=...))")
            print(sep)

    def reset(self) -> None:
        """Xóa toàn bộ bộ nhớ trong session (không xóa file đã ghi)."""
        self._violations.clear()
        self._exported_ids.clear()
        self.fps_tracker.start()
        print("[Exporter] Đã reset trạng thái.")


# ─────────────────────────────────────────────────────────────
#  HELPER: VẼ WATERMARK LÊN ẢNH CROP
# ─────────────────────────────────────────────────────────────

def _annotate_crop(
    img: np.ndarray,
    vehicle_id: int,
    vehicle_type: str,
    frame_idx: int,
    confidence: float,
) -> None:
    """
    Vẽ thông tin lên góc trái trên ảnh crop để nhận dạng nhanh khi duyệt thư mục.
    """
    label = f"ID:{vehicle_id} {vehicle_type.upper()} F:{frame_idx} {confidence:.2f}"
    h, w = img.shape[:2]

    # Nền mờ
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    cv2.putText(
        img, label,
        (4, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 0, 255), 1, cv2.LINE_AA
    )

    # Viền đỏ cảnh báo vi phạm
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 220), 2)
