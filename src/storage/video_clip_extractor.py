"""
========================================================================
[THÀNH VIÊN 3: STORAGE & TESTING ENGINEER — VIDEO CLIP EXTRACTOR]
========================================================================
Module cắt đoạn video ngắn (1s trước + 2s sau) tại thời điểm xe vi phạm.

Cơ chế hoạt động:
  1. Ring buffer giữ N frames gần nhất (N = FPS × clip_before_sec).
  2. Khi nhận sự kiện vi phạm → sao chép buffer hiện tại + đánh dấu
     cần thu thêm FPS × clip_after_sec frames nữa.
  3. Khi đủ frames → ghép thành video .mp4 bằng cv2.VideoWriter.

Cấu trúc lưu trữ:
    Luutru_Vipham/
    └── YYYY-MM-DD/
        └── Vuot_Den_Do/
            └── clips/
                └── VehicleID_42_car_F00123_20250607_143201.mp4
========================================================================
"""

import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


class ViolationClipExtractor:
    """
    Bộ cắt đoạn video vi phạm tự động.

    Sử dụng ring buffer (deque) để luôn giữ sẵn N frames trước đó.
    Khi phát hiện vi phạm, bắt đầu thu thêm M frames tiếp theo rồi
    ghép toàn bộ thành 1 file video clip ngắn.

    Tham số:
        fps (float): FPS gốc của video đầu vào
        clip_before_sec (float): Số giây lưu trước thời điểm vi phạm (mặc định 1s)
        clip_after_sec (float): Số giây lưu sau thời điểm vi phạm (mặc định 2s)
        root_dir (str): Thư mục gốc lưu trữ vi phạm
        frame_size (Tuple[int,int]): Kích thước frame (width, height)
    """

    def __init__(
        self,
        fps: float = 30.0,
        clip_before_sec: float = 1.0,
        clip_after_sec: float = 2.0,
        root_dir: str = "Luutru_Vipham",
        frame_size: Tuple[int, int] = (1280, 720),
    ):
        self.fps = max(fps, 1.0)
        self.clip_before_sec = clip_before_sec
        self.clip_after_sec = clip_after_sec
        self.root_dir = root_dir
        self.frame_size = frame_size

        # Số frames cần giữ trước vi phạm
        self._buffer_size = int(self.fps * self.clip_before_sec)
        # Số frames cần thu sau vi phạm
        self._after_frames = int(self.fps * self.clip_after_sec)

        # Ring buffer luôn giữ N frames gần nhất
        self._ring_buffer: deque = deque(maxlen=max(self._buffer_size, 1))

        # Danh sách các clip đang thu (chưa hoàn thành)
        # Mỗi phần tử: {
        #   "violation_info": dict,
        #   "before_frames": list[np.ndarray],
        #   "after_frames":  list[np.ndarray],
        #   "remaining":     int,   ← số frames còn phải thu
        # }
        self._pending_clips: List[Dict[str, Any]] = []

        # Clip đã hoàn thành (path lưu)
        self._completed_clips: List[str] = []

        print(f"[ClipExtractor] Ring buffer size: {self._buffer_size} frames "
              f"({clip_before_sec}s × {self.fps:.0f} FPS)")
        print(f"[ClipExtractor] After-violation capture: {self._after_frames} frames "
              f"({clip_after_sec}s × {self.fps:.0f} FPS)")

    # ── GỌI MỖI FRAME ───────────────────────────────────────

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Đẩy frame mới vào ring buffer và cập nhật các clip đang pending.
        GỌI HÀM NÀY MỖI FRAME trong vòng lặp chính của main.py.
        """
        # 1. Thêm vào ring buffer
        self._ring_buffer.append(frame.copy())

        # 2. Cập nhật các clip đang thu after-frames
        finished_indices = []
        for i, clip_data in enumerate(self._pending_clips):
            if clip_data["remaining"] > 0:
                clip_data["after_frames"].append(frame.copy())
                clip_data["remaining"] -= 1

            # Kiểm tra hoàn thành
            if clip_data["remaining"] <= 0:
                finished_indices.append(i)

        # 3. Ghi các clip đã hoàn thành
        for i in reversed(finished_indices):
            clip_data = self._pending_clips.pop(i)
            self._write_clip(clip_data)

    # ── KHI PHÁT HIỆN VI PHẠM ───────────────────────────────

    def trigger_violation(self, violation_event: Dict[str, Any]) -> None:
        """
        Bắt đầu thu clip cho một vi phạm mới.
        Gọi hàm này ngay khi ViolationDetector phát hiện vi phạm.

        violation_event: dict chứa vehicle_id, vehicle_type, frame, timestamp, bbox, confidence
        """
        # Sao chép toàn bộ buffer hiện tại làm phần "trước vi phạm"
        before_frames = list(self._ring_buffer)

        clip_data = {
            "violation_info": violation_event.copy(),
            "before_frames": before_frames,
            "after_frames": [],
            "remaining": self._after_frames,
        }
        self._pending_clips.append(clip_data)

        v_id = violation_event.get("vehicle_id", "?")
        print(f"[ClipExtractor] ▶ Bắt đầu thu clip cho xe #{v_id} "
              f"(đã có {len(before_frames)} frames trước, cần thêm {self._after_frames} frames)")

    # ── HỦY CLIP KHI VI PHẠM BỊ CANCEL ──────────────────────

    def cancel_clip(self, vehicle_id: int) -> None:
        """Hủy clip đang pending cho xe bị hủy vi phạm (rẽ phải)."""
        before = len(self._pending_clips)
        self._pending_clips = [
            c for c in self._pending_clips
            if c["violation_info"].get("vehicle_id") != vehicle_id
        ]
        if len(self._pending_clips) < before:
            print(f"[ClipExtractor] ✗ Đã hủy clip đang thu cho xe #{vehicle_id}")

    # ── GHI FILE VIDEO ───────────────────────────────────────

    def _write_clip(self, clip_data: Dict[str, Any]) -> None:
        """Ghép before_frames + after_frames → file video .mp4"""
        info = clip_data["violation_info"]
        all_frames = clip_data["before_frames"] + clip_data["after_frames"]

        if len(all_frames) == 0:
            return

        # Xây dựng đường dẫn lưu
        clip_path = self._build_clip_path(info)

        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = all_frames[0].shape[:2]
            writer = cv2.VideoWriter(clip_path, fourcc, self.fps, (w, h))

            for frm in all_frames:
                writer.write(frm)
            writer.release()

            self._completed_clips.append(clip_path)
            v_id = info.get("vehicle_id", "?")
            print(f"[ClipExtractor] ✓ Đã lưu clip: {clip_path} "
                  f"({len(all_frames)} frames ≈ {len(all_frames)/self.fps:.1f}s)")

        except Exception as e:
            print(f"[ClipExtractor] ✗ Lỗi ghi clip: {e}")

    def _build_clip_path(self, violation_info: Dict[str, Any]) -> str:
        """Xây dựng đường dẫn clip theo cấu trúc thư mục chuẩn."""
        timestamp = violation_info.get("timestamp", 0)
        v_id = violation_info.get("vehicle_id", 0)
        v_type = violation_info.get("vehicle_type", "unknown")
        frame_idx = violation_info.get("frame", 0)

        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
        time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        folder_name = f"VehicleID_{v_id}_{v_type}_F{frame_idx:05d}_{time_str}"

        clip_dir = os.path.join(self.root_dir, date_str, "Vuot_Den_Do", folder_name)
        os.makedirs(clip_dir, exist_ok=True)

        filename = f"{folder_name}.mp4"
        return os.path.join(clip_dir, filename)

    # ── KẾT THÚC SESSION ────────────────────────────────────

    def finalize(self) -> List[str]:
        """
        Ghi tất cả các clip đang pending (dù chưa đủ after-frames).
        Gọi hàm này khi video kết thúc hoặc pipeline dừng.
        Trả về danh sách đường dẫn tất cả clip đã lưu.
        """
        for clip_data in self._pending_clips:
            self._write_clip(clip_data)
        self._pending_clips.clear()

        total = len(self._completed_clips)
        print(f"[ClipExtractor] Finalized. Tổng cộng {total} clip(s) đã lưu.")
        return self._completed_clips.copy()

    def get_completed_clips(self) -> List[str]:
        """Trả về danh sách tất cả clip đã ghi xong."""
        return self._completed_clips.copy()

    def get_pending_count(self) -> int:
        """Số clip đang chờ thu thêm frames."""
        return len(self._pending_clips)
