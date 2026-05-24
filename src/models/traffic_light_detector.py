import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class TrafficLightDetector:
    """Phát hiện và phân tích trạng thái đèn giao thông."""
    TRAFFIC_LIGHT_CLASS = 9

    def __init__(self, model_path: str = "yolov8s.pt", conf_threshold: float = 0.25, roi: List[int] = None, static_lights: List[List[int]] = None):
        """Khởi tạo bộ phát hiện đèn giao thông."""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        self.roi = roi
        self.static_lights = static_lights
        
        # Tự động nhận diện loại mô hình
        class_names = list(self.model.names.values())
        has_color_classes = any(
            any(c in name.lower() for c in ['red', 'green', 'yellow', 'stop', 'go', 'warning'])
            for name in class_names
        )
        
        if len(class_names) < 15 and has_color_classes:
            self.is_custom_model = True
            print(f"[TrafficLightDetector] Custom model detected: {self.model.names}")
        else:
            self.is_custom_model = False
            self.traffic_light_class_id = self.TRAFFIC_LIGHT_CLASS
            for cid, cname in self.model.names.items():
                if 'traffic' in cname.lower() or 'light' in cname.lower():
                    self.traffic_light_class_id = cid
                    break
            print(f"[TrafficLightDetector] COCO model detected (Class ID: {self.traffic_light_class_id})")

        self.last_lights = []
        self.lost_frames = 0
        self.max_lost_frames = 60  # Giữ trạng thái cũ trong ~2 giây khi mất dấu
        self.state_history = []    # Lịch sử trạng thái để làm mịn

    def detect_and_classify(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Phát hiện hộp đèn và phân tích màu sắc."""
        lights = []
        h_f, w_f = frame.shape[:2]
        
        # Chế độ hộp đèn cố định
        if self.static_lights and len(self.static_lights) > 0:
            for box in self.static_lights:
                x1, y1, x2, y2 = [int(v) for v in box]
                
                # Cắt crop an toàn
                crop_x1, crop_y1 = max(0, x1), max(0, y1)
                crop_x2, crop_y2 = min(w_f, x2), min(h_f, y2)
                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Phân loại màu sắc bằng HSV & mật độ sáng
                state = self._classify_state(crop)
                
                lights.append({
                    "box": [x1, y1, x2, y2],
                    "conf": 1.0,
                    "state": state
                })
        else:
            # Chế độ dò tìm bằng YOLO
            if self.is_custom_model:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    verbose=False
                )
            else:
                results = self.model.predict(
                    source=frame,
                    conf=self.conf_threshold,
                    classes=[self.traffic_light_class_id],
                    verbose=False
                )
            
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id].lower()
                    
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    # Lọc theo ROI
                    if self.roi and len(self.roi) == 4:
                        rx1, ry1, rx2, ry2 = self.roi
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                            continue
                            
                      # Lọc nhiễu hình học
                    w_box = x2 - x1
                    h_box = y2 - y1
                    if w_box < 5 or h_box < 5: 
                        continue
                        
                    aspect_ratio = max(w_box, h_box) / min(w_box, h_box)
                    if aspect_ratio > 4.5: 
                        continue
                        
                    if y1 > h_f * 0.85: 
                        continue
                    
                    # Thêm padding để lấy trọn viền phát quang
                    pad = 3
                    crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
                    crop_x2, crop_y2 = min(w_f, x2 + pad), min(h_f, y2 + pad)
                    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    # Nhận diện màu bằng HSV + Mật độ sáng
                    hsv_state = self._classify_state(crop)
                    
                    # Xác định màu sắc dự đoán từ nhãn YOLO nếu là custom
                    yolo_state = "UNKNOWN"
                    if self.is_custom_model:
                        if 'red' in class_name or 'stop' in class_name:
                            yolo_state = "RED"
                        elif 'green' in class_name or 'go' in class_name:
                            yolo_state = "GREEN"
                        elif 'yellow' in class_name or 'warning' in class_name:
                            yolo_state = "YELLOW"
                    
                    # Quyết định lai (Hybrid Decision)
                    if self.is_custom_model and conf >= 0.45 and yolo_state != "UNKNOWN":
                        state = yolo_state
                    elif hsv_state != "UNKNOWN":
                        state = hsv_state
                    else:
                        state = yolo_state if yolo_state != "UNKNOWN" else "UNKNOWN"
                    
                    lights.append({
                        "box": [x1, y1, x2, y2],
                        "conf": conf,
                        "state": state
                    })

        # Bộ đệm làm mịn trạng thái (Temporal Smoothing)
        if len(lights) > 0:
            self.last_lights = lights
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            if self.lost_frames <= self.max_lost_frames and len(self.last_lights) > 0:
                lights = self.last_lights
                
        # Thực hiện làm mịn bằng Median Filter (cỡ 3 frames)
        if len(lights) > 0:
            main_light = lights[0]
            current_state = main_light["state"]
            
            # Chỉ làm mịn đối với các trạng thái màu hợp lệ
            if current_state != "UNKNOWN":
                self.state_history.append(current_state)
                if len(self.state_history) > 3:
                    self.state_history.pop(0)
                
                if len(self.state_history) >= 2:
                    smoothed_state = max(set(self.state_history), key=self.state_history.count)
                    main_light["state"] = smoothed_state
                    
        # Vẽ preview lên khung hình
        for l in lights:
            x1, y1, x2, y2 = l["box"]
            state = l["state"]
            color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0), "UNKNOWN": (255, 255, 255)}
            box_color = color_map.get(state, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"TL_{state}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)
                        
        return lights

    def _classify_state(self, crop: np.ndarray) -> str:
        """Nhận diện màu sắc kết hợp HSV và mật độ sáng."""
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return "UNKNOWN"
            
        h, w = crop.shape[:2]
        
        # Tiền xử lý Gaussian Blur nhẹ loại bỏ nhiễu
        if h >= 6 and w >= 6:
            processed = cv2.GaussianBlur(crop, (3, 3), 0)
        else:
            processed = crop.copy()
            
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        # Ngưỡng HSV cho Đỏ, Vàng, Xanh lá
        lower_red1 = np.array([0, 12, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([165, 12, 60])
        upper_red2 = np.array([180, 255, 255])
        
        lower_yellow = np.array([11, 35, 60])
        upper_yellow = np.array([33, 255, 255])
        
        lower_green = np.array([35, 35, 60])
        upper_green = np.array([90, 255, 255])
        
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)
        
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        min_pixels_threshold = int(h * w * 0.015)
        
        if max_pixels >= min_pixels_threshold and max_pixels > 0:
            if max_pixels == red_pixels: return "RED"
            if max_pixels == yellow_pixels: return "YELLOW"
            if max_pixels == green_pixels: return "GREEN"
            
        # Dự phòng 1: Lấy mẫu màu vi mô quanh điểm sáng nhất
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        _, max_val, _, max_loc = cv2.minMaxLoc(gray)
        
        if max_val >= 180:
            mx, my = max_loc
            x_start = max(0, mx - 1)
            x_end = min(w, mx + 2)
            y_start = max(0, my - 1)
            y_end = min(h, my + 2)
            neighborhood_hsv = hsv[y_start:y_end, x_start:x_end]
            
            n_red = cv2.countNonZero(cv2.bitwise_or(
                cv2.inRange(neighborhood_hsv, lower_red1, upper_red1),
                cv2.inRange(neighborhood_hsv, lower_red2, upper_red2)
            ))
            n_yellow = cv2.countNonZero(cv2.inRange(neighborhood_hsv, lower_yellow, upper_yellow))
            n_green = cv2.countNonZero(cv2.inRange(neighborhood_hsv, lower_green, upper_green))
            
            if max(n_red, n_yellow, n_green) > 0:
                if n_red >= n_yellow and n_red >= n_green: return "RED"
                if n_yellow >= n_red and n_yellow >= n_green: return "YELLOW"
                return "GREEN"
                
        # Dự phòng 2: Phân tích mật độ sáng 3 vùng
        if w > h:
            w3 = w // 3
            mean1 = np.mean(gray[:, :w3]) if w3 > 0 else 0
            mean2 = np.mean(gray[:, w3:2*w3]) if w3 > 0 else 0
            mean3 = np.mean(gray[:, 2*w3:]) if w3 > 0 else 0
        else:
            h3 = h // 3
            mean1 = np.mean(gray[:h3, :]) if h3 > 0 else 0
            mean2 = np.mean(gray[h3:2*h3, :]) if h3 > 0 else 0
            mean3 = np.mean(gray[2*h3:, :]) if h3 > 0 else 0
            
        if max(mean1, mean2, mean3) < 45: 
            return "UNKNOWN"
            
        if mean1 == max(mean1, mean2, mean3): return "RED"
        if mean2 == max(mean1, mean2, mean3): return "YELLOW"
        return "GREEN"

    def get_global_traffic_light_state(self, lights: List[Dict[str, Any]]) -> str:
        """Trả về trạng thái đèn giao thông chung (Ưu tiên: ĐỎ > VÀNG > XANH)."""
        if not lights: 
            return "UNKNOWN"
            
        states = [l["state"] for l in lights if l["state"] != "UNKNOWN"]
        if not states: 
            return "UNKNOWN"
            
        if "RED" in states: return "RED"
        elif "YELLOW" in states: return "YELLOW"
        elif "GREEN" in states: return "GREEN"
        return "UNKNOWN"