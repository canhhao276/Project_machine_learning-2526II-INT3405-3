"""Phát hiện phương tiện giao thông bằng YOLOv8."""

import argparse
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

VEHICLE_COLORS: Dict[str, tuple] = {
    "car":        (0, 200, 0),
    "motorcycle": (0, 180, 255),
    "bus":        (255, 120, 0),
    "truck":      (180, 0, 255),
}

DEFAULT_COLOR = (200, 200, 200)

class VehicleDetector:
    """Phát hiện phương tiện giao thông sử dụng YOLOv8."""

    # COCO Class IDs: 2: car, 3: motorcycle, 5: bus, 7: truck
    VEHICLE_CLASSES: Dict[int, str] = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        """Khởi tạo bộ phát hiện phương tiện."""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Phát hiện phương tiện trong khung hình."""
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=list(self.VEHICLE_CLASSES.keys()),
            verbose=False,
        )

        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                detections.append({
                    "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": self.VEHICLE_CLASSES.get(cls_id, "unknown"),
                })

        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Vẽ bounding box và nhãn lên khung hình."""
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["class_name"]
            conf  = det["conf"]
            color = VEHICLE_COLORS.get(label, DEFAULT_COLOR)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
            )

            cv2.rectangle(
                frame,
                (x1, y1 - th - baseline - 4),
                (x1 + tw, y1),
                color, cv2.FILLED,
            )

            cv2.putText(
                frame, text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2,
            )

        return frame

    @staticmethod
    def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
        """Vẽ chỉ số FPS lên khung hình."""
        fps_text = f"FPS: {fps:.1f}"
        (tw, th), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2,
        )

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (20 + tw, 20 + th + 6), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame, fps_text,
            (15, 15 + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 100), 2,
        )
        return frame

    @staticmethod
    def draw_summary(frame: np.ndarray, counts: Dict[str, int]) -> np.ndarray:
        """Hiển thị bảng đếm số lượng xe."""
        h, w = frame.shape[:2]
        y_offset = 20

        for cls_name, count in counts.items():
            color = VEHICLE_COLORS.get(cls_name, DEFAULT_COLOR)
            text = f"{cls_name}: {count}"

            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
            )
            x_pos = w - tw - 20

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (x_pos - 5, y_offset - 2),
                (w - 10, y_offset + th + 6),
                (30, 30, 30), cv2.FILLED,
            )
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            cv2.putText(
                frame, text,
                (x_pos, y_offset + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2,
            )
            y_offset += th + 14

        return frame

    def run_on_video(self, video_path: str) -> None:
        """Chạy bộ phát hiện trực tiếp trên video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] Cannot open video: {video_path}")
            sys.exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps    = cap.get(cv2.CAP_PROP_FPS)
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"[Video] Resolution: {width}x{height} | FPS: {video_fps:.1f} | Total frames: {total_frames}")

        frame_idx = 0
        fps = 0.0

        try:
            while cap.isOpened():
                t_start = time.perf_counter()

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                detections = self.detect(frame)

                counts: Dict[str, int] = {name: 0 for name in self.VEHICLE_CLASSES.values()}
                for det in detections:
                    counts[det["class_name"]] += 1

                frame = self.draw_detections(frame, detections)
                frame = self.draw_fps(frame, fps)
                frame = self.draw_summary(frame, counts)

                cv2.imshow("YOLOv8 Vehicle Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                elapsed = time.perf_counter() - t_start
                fps = 1.0 / elapsed if elapsed > 0 else 0.0

                if frame_idx % 100 == 0:
                    progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    print(f"Frame {frame_idx}/{total_frames} ({progress:.1f}%) | FPS: {fps:.1f} | Dets: {len(detections)}")

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phát hiện phương tiện giao thông bằng YOLOv8")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    detector = VehicleDetector(model_path=args.model, conf_threshold=args.conf)
    detector.run_on_video(args.video)

