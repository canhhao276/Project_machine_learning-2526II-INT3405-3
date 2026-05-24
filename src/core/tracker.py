"""Bám vết phương tiện giao thông bằng YOLOv8 + ByteTrack."""

import argparse
import os
import sys
import time
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Tuple

VEHICLE_CLASS_NAMES: Dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

TRACK_PALETTE: List[Tuple[int, int, int]] = [
    (230, 100, 50),   (0, 200, 100),   (255, 180, 0),   (180, 0, 255),
    (0, 255, 255),    (255, 0, 150),   (100, 255, 50),  (50, 150, 255),
    (200, 200, 0),    (0, 100, 255),   (255, 50, 50),   (100, 0, 200),
    (0, 220, 180),    (220, 120, 255), (128, 255, 0),   (255, 200, 150),
    (80, 80, 255),    (255, 128, 0),   (0, 180, 100),   (200, 50, 200),
]

def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Trả về màu BGR duy nhất cho mỗi track ID."""
    return TRACK_PALETTE[track_id % len(TRACK_PALETTE)]

class VehicleTracker:
    """Bộ theo vết phương tiện giao thông sử dụng ByteTrack."""

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        conf_threshold: float = 0.15,
        iou_threshold: float = 0.5,
        track_buffer: int = 60,
    ):
        """Khởi tạo bộ theo vết."""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.track_buffer = track_buffer

        self.model = YOLO(model_path)
        self.vehicle_classes = list(VEHICLE_CLASS_NAMES.keys())
        
        self.tracker_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "bytetrack_traffic.yaml",
        )

        self._raw_to_compact: Dict[int, int] = {}
        self._next_compact_id: int = 1
        self._lost_frames: Dict[int, int] = defaultdict(int)
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.max_trail_length = 50

    def track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Bám vết phương tiện trong frame hiện tại."""
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.vehicle_classes,
            tracker=self.tracker_config,
            verbose=False,
        )

        tracked_objects = []
        current_raw_ids: set = set()

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            if boxes.id is not None:
                raw_ids = boxes.id.cpu().numpy().astype(int).tolist()
                xyxy    = boxes.xyxy.cpu().numpy().tolist()
                confs   = boxes.conf.cpu().numpy().tolist()
                clss    = boxes.cls.cpu().numpy().astype(int).tolist()

                for idx, raw_id in enumerate(raw_ids):
                    current_raw_ids.add(raw_id)
                    x1, y1, x2, y2 = xyxy[idx]
                    cls_id = clss[idx]

                    if raw_id not in self._raw_to_compact:
                        self._raw_to_compact[raw_id] = self._next_compact_id
                        self._next_compact_id += 1
                    compact_id = self._raw_to_compact[raw_id]

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    self.track_history[compact_id].append((cx, cy))
                    if len(self.track_history[compact_id]) > self.max_trail_length:
                        self.track_history[compact_id].pop(0)

                    tracked_objects.append({
                        "id": compact_id,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "center": (cx, cy),
                        "conf": float(confs[idx]),
                        "class_id": cls_id,
                        "class_name": VEHICLE_CLASS_NAMES.get(cls_id, "unknown"),
                    })

        for rid in list(self._raw_to_compact.keys()):
            if rid not in current_raw_ids:
                self._lost_frames[rid] += 1
                if self._lost_frames[rid] > self.track_buffer:
                    cid = self._raw_to_compact.pop(rid)
                    self._lost_frames.pop(rid, None)
                    self.track_history.pop(cid, None)
            else:
                self._lost_frames[rid] = 0

        return tracked_objects

    def draw_tracks(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> np.ndarray:
        """Vẽ thông tin bám vết lên khung hình."""
        for obj in tracked_objects:
            track_id   = obj["id"]
            x1, y1, x2, y2 = obj["box"]
            cls_name   = obj["class_name"]
            conf       = obj["conf"]
            color      = get_track_color(track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id} {cls_name} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2,
            )
            cv2.rectangle(
                frame,
                (x1, y1 - th - baseline - 6),
                (x1 + tw + 4, y1),
                color, cv2.FILLED,
            )
            cv2.putText(
                frame, label,
                (x1 + 2, y1 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2,
            )

            trail = self.track_history.get(track_id, [])
            if len(trail) >= 2:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
                cv2.circle(frame, trail[-1], 4, color, cv2.FILLED)

        return frame

    @staticmethod
    def draw_tracking_hud(
        frame: np.ndarray,
        fps: float,
        tracked_objects: List[Dict[str, Any]],
        total_unique_ids: int,
    ) -> np.ndarray:
        """Vẽ bảng thống kê bám vết HUD."""
        counts: Dict[str, int] = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        for obj in tracked_objects:
            counts[obj["class_name"]] = counts.get(obj["class_name"], 0) + 1

        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (320, 165), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(
            frame, "BYTETRACK VEHICLE TRACKER",
            (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 2,
        )

        cv2.putText(
            frame, f"FPS: {fps:.1f}",
            (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2,
        )

        cv2.putText(
            frame, f"Active: {len(tracked_objects)}  |  Total IDs: {total_unique_ids}",
            (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        y = 108
        for cls_name, count in counts.items():
            cv2.putText(
                frame, f"  {cls_name}: {count}",
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1,
            )
            y += 20

        return frame

    def reset(self) -> None:
        """Reset trạng thái bám vết."""
        self.model = YOLO(self.model_path)
        self.track_history.clear()
        self._raw_to_compact.clear()
        self._next_compact_id = 1
        self._lost_frames.clear()

    def run_on_video(self, video_path: str) -> None:
        """Chạy demo bám vết trực tiếp trên video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            sys.exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Video] {width}x{height} | FPS: {video_fps:.1f} | Frames: {total_frames}")

        cv2.namedWindow("ByteTrack Vehicle Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ByteTrack Vehicle Tracking", 960, 540)

        frame_idx = 0
        fps = 0.0
        all_seen_ids: set = set()

        try:
            while cap.isOpened():
                t_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                tracked_objects = self.track(frame)

                for obj in tracked_objects:
                    all_seen_ids.add(obj["id"])

                frame = self.draw_tracks(frame, tracked_objects)
                frame = self.draw_tracking_hud(frame, fps, tracked_objects, len(all_seen_ids))

                cv2.imshow("ByteTrack Vehicle Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                elapsed = time.perf_counter() - t_start
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    print(f"Frame {frame_idx}/{total_frames} ({progress:.1f}%) | FPS: {fps:.1f} | Active: {len(tracked_objects)}")

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bám vết phương tiện giao thông bằng YOLOv8 + ByteTrack")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    tracker = VehicleTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    tracker.run_on_video(args.video)
