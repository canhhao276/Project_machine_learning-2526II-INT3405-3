import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class TrafficLightDetector:
    TRAFFIC_LIGHT_CLASS = 9

    def __init__(self, model_path: str = "yolov8s.pt", conf_threshold: float = 0.25, roi: List[int] = None, static_lights: List[List[int]] = None):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        self.roi = roi
        self.static_lights = static_lights
        
        class_names = list(self.model.names.values())
        has_color_classes = any(
            any(c in name.lower() for c in ['red', 'green', 'yellow', 'stop', 'go', 'warning'])
            for name in class_names
        )
        
        if len(class_names) < 15 and has_color_classes:
            self.is_custom_model = True
        else:
            self.is_custom_model = False
            self.traffic_light_class_id = self.TRAFFIC_LIGHT_CLASS
            for cid, cname in self.model.names.items():
                if 'traffic' in cname.lower() or 'light' in cname.lower():
                    self.traffic_light_class_id = cid
                    break

        self.last_lights = []
        self.lost_frames = 0
        self.max_lost_frames = 60  
        self.state_history = []    

    def detect_and_classify(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        lights = []
        h_f, w_f = frame.shape[:2]
        
        if self.static_lights and len(self.static_lights) > 0:
            yolo_lights = []
            if self.is_custom_model:
                results = self.model.predict(source=frame, conf=0.01, verbose=False)
                if len(results) > 0 and results[0].boxes is not None:
                    for b in results[0].boxes:
                        bx = b.xyxy[0].cpu().numpy()
                        bconf = float(b.conf[0].cpu().numpy())
                        bcid = int(b.cls[0].cpu().numpy())
                        bcname = self.model.names[bcid].lower()
                        
                        bstate = "UNKNOWN"
                        if 'red' in bcname or 'stop' in bcname: bstate = "RED"
                        elif 'green' in bcname or 'go' in bcname: bstate = "GREEN"
                        elif 'yellow' in bcname or 'warning' in bcname: bstate = "YELLOW"
                            
                        yolo_lights.append({
                            "box": [int(bx[0]), int(bx[1]), int(bx[2]), int(bx[3])],
                            "state": bstate,
                            "conf": bconf
                        })

            for box in self.static_lights:
                x1, y1, x2, y2 = [int(v) for v in box]
                crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                
                if self.is_custom_model:
                    state = "UNKNOWN"
                    if len(yolo_lights) > 0:
                        best_match = None
                        best_dist = 9999
                        cx_static, cy_static = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        for yl in yolo_lights:
                            cx_yolo = (yl["box"][0] + yl["box"][2]) / 2
                            cy_yolo = (yl["box"][1] + yl["box"][3]) / 2
                            dist = np.sqrt((cx_static - cx_yolo)**2 + (cy_static - cy_yolo)**2)
                            
                            if dist < 60 and yl["state"] != "UNKNOWN" and dist < best_dist:
                                best_dist = dist
                                best_match = yl
                                    
                        if best_match is not None:
                            state = best_match["state"]
                    
                    if state == "UNKNOWN":
                        state = self._classify_state(crop)
                else:
                    state = self._classify_state(crop)
                
                lights.append({"box": [x1, y1, x2, y2], "conf": 1.0, "state": state})
        else:
            results = self.model.predict(
                source=frame, 
                conf=self.conf_threshold, 
                classes=None if self.is_custom_model else [self.traffic_light_class_id], 
                verbose=False
            )
            
            if len(results) > 0:
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id].lower()
                    
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    
                    if self.roi and len(self.roi) == 4:
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        if not (self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3]):
                            continue
                            
                    w_box, h_box = x2 - x1, y2 - y1
                    if w_box < 5 or h_box < 5: continue
                    if max(w_box, h_box) / min(w_box, h_box) > 4.5: continue
                    if y1 > h_f * 0.85: continue
                    
                    crop = frame[max(0, y1-3):min(h_f, y2+3), max(0, x1-3):min(w_f, x2+3)]
                    
                    if self.is_custom_model:
                        yolo_state = "UNKNOWN"
                        if 'red' in class_name or 'stop' in class_name: yolo_state = "RED"
                        elif 'green' in class_name or 'go' in class_name: yolo_state = "GREEN"
                        elif 'yellow' in class_name or 'warning' in class_name: yolo_state = "YELLOW"
                        state = yolo_state
                        
                        if state == "UNKNOWN":
                            state = self._classify_state(crop)
                    else:
                        state = self._classify_state(crop)
                    
                    lights.append({"box": [x1, y1, x2, y2], "conf": conf, "state": state})

        if len(lights) > 0:
            self.last_lights = lights
            self.lost_frames = 0
        else:
            self.lost_frames += 1
            if self.lost_frames <= self.max_lost_frames and len(self.last_lights) > 0:
                lights = self.last_lights
                
        if len(lights) > 0:
            main_light = lights[0]
            current_state = main_light["state"]
            
            if current_state != "UNKNOWN":
                self.state_history.append(current_state)
                if len(self.state_history) > 3: self.state_history.pop(0)
                if len(self.state_history) >= 2:
                    main_light["state"] = max(set(self.state_history), key=self.state_history.count)
                    
        for l in lights:
            x1, y1, x2, y2 = l["box"]
            state = l["state"]
            color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0), "UNKNOWN": (255, 255, 255)}
            box_color = color_map.get(state, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"TL_{state}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)
                        
        return lights

    def _classify_state(self, crop: np.ndarray) -> str:
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4: return "UNKNOWN"
        h, w = crop.shape[:2]
        processed = cv2.GaussianBlur(crop, (3, 3), 0) if h >= 6 and w >= 6 else crop.copy()
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 12, 60]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([165, 12, 60]), np.array([180, 255, 255]))
        )
        mask_yellow = cv2.inRange(hsv, np.array([11, 35, 60]), np.array([33, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([35, 35, 60]), np.array([90, 255, 255]))
        
        r_p, y_p, g_p = cv2.countNonZero(mask_red), cv2.countNonZero(mask_yellow), cv2.countNonZero(mask_green)
        max_pixels = max(r_p, y_p, g_p)
        
        if max_pixels >= int(h * w * 0.015) and max_pixels > 0:
            if max_pixels == r_p: return "RED"
            if max_pixels == y_p: return "YELLOW"
            if max_pixels == g_p: return "GREEN"
            
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        _, max_val, _, max_loc = cv2.minMaxLoc(gray)
        
        if max_val >= 180:
            mx, my = max_loc
            neighborhood_hsv = hsv[max(0, my-1):min(h, my+2), max(0, mx-1):min(w, mx+2)]
            n_red = cv2.countNonZero(cv2.bitwise_or(
                cv2.inRange(neighborhood_hsv, np.array([0, 12, 60]), np.array([10, 255, 255])),
                cv2.inRange(neighborhood_hsv, np.array([165, 12, 60]), np.array([180, 255, 255]))
            ))
            n_yellow = cv2.countNonZero(cv2.inRange(neighborhood_hsv, np.array([11, 35, 60]), np.array([33, 255, 255])))
            n_green = cv2.countNonZero(cv2.inRange(neighborhood_hsv, np.array([35, 35, 60]), np.array([90, 255, 255])))
            
            if max(n_red, n_yellow, n_green) > 0:
                if n_red >= n_yellow and n_red >= n_green: return "RED"
                if n_yellow >= n_red and n_yellow >= n_green: return "YELLOW"
                return "GREEN"
                
        if w > h:
            w3 = w // 3
            mean1, mean2, mean3 = np.mean(gray[:, :w3]), np.mean(gray[:, w3:2*w3]), np.mean(gray[:, 2*w3:])
        else:
            h3 = h // 3
            mean1, mean2, mean3 = np.mean(gray[:h3, :]), np.mean(gray[h3:2*h3, :]), np.mean(gray[2*h3:, :])
            
        if max(mean1, mean2, mean3) < 45: return "UNKNOWN"
        if mean1 == max(mean1, mean2, mean3): return "RED"
        if mean2 == max(mean1, mean2, mean3): return "YELLOW"
        return "GREEN"

    def get_global_traffic_light_state(self, lights: List[Dict[str, Any]]) -> str:
        if not lights: return "UNKNOWN"
        states = [l["state"] for l in lights if l["state"] != "UNKNOWN"]
        if not states: return "UNKNOWN"
        if "RED" in states: return "RED"
        if "YELLOW" in states: return "YELLOW"
        if "GREEN" in states: return "GREEN"
        return "UNKNOWN"