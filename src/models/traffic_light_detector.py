import cv2
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import pickle

class TrafficLightDetector:
    def __init__(self, model_path: str = "yolov8s.pt", conf_threshold: float = 0.25, roi: List[int] = None, static_lights: List[List[int]] = None):
        self.static_lights = static_lights
        
        # Load SVM model and scaler directly
        svm_path = Path(__file__).parent / "svm_traffic_light.pkl"
        with open(svm_path, 'rb') as f:
            self.svm_data = pickle.load(f)
        print(f"[TrafficLightDetector] Loaded SVM classifier from: {svm_path}", flush=True)
        
        self.last_lights = []
        self.lost_frames = 0
        self.max_lost_frames = 60  
        self.state_history = []    

    def detect_and_classify(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        lights = []
        h_f, w_f = frame.shape[:2]
        
        if self.static_lights:
            for box in self.static_lights:
                x1, y1, x2, y2 = [int(v) for v in box]
                crop = frame[max(0, y1):min(h_f, y2), max(0, x1):min(w_f, x2)]
                state = self._classify_state(crop)
                lights.append({"box": [x1, y1, x2, y2], "conf": 1.0, "state": state})

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
                if len(self.state_history) > 3: 
                    self.state_history.pop(0)
                if len(self.state_history) >= 2:
                    main_light["state"] = max(set(self.state_history), key=self.state_history.count)
                    
        for l in lights:
            x1, y1, x2, y2 = l["box"]
            state = l["state"]
            color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0), "UNKNOWN": (255, 255, 255)}
            box_color = color_map.get(state, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"TL_{state} (SVM)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)
                        
        return lights

    def _classify_state(self, crop: np.ndarray) -> str:
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4: 
            return "UNKNOWN"
        
        h, w = crop.shape[:2]
        
        # 1. Global features
        mean_b, mean_g, mean_r = cv2.mean(crop)[:3]
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_hsv, std_hsv = cv2.meanStdDev(hsv_crop)
        mean_h, mean_s, mean_v = mean_hsv[0][0], mean_hsv[1][0], mean_hsv[2][0]
        std_h, std_s, std_v = std_hsv[0][0], std_hsv[1][0], std_hsv[2][0]
        
        global_feats = [mean_r, mean_g, mean_b, mean_h, mean_s, mean_v, std_h, std_s, std_v]
        
        # 2. Spatial features
        if h >= w:
            h3 = h // 3
            part1 = crop[:h3, :]
            part2 = crop[h3:2*h3, :]
            part3 = crop[2*h3:, :]
        else:
            w3 = w // 3
            part1 = crop[:, :w3]
            part2 = crop[:, w3:2*w3]
            part3 = crop[:, 2*w3:]
            
        def get_part_means(part):
            if part.size == 0:
                return [0.0] * 6
            p_mean_b, p_mean_g, p_mean_r = cv2.mean(part)[:3]
            p_hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
            p_mean_h, p_mean_s, p_mean_v = cv2.mean(p_hsv)[:3]
            return [p_mean_r, p_mean_g, p_mean_b, p_mean_h, p_mean_s, p_mean_v]
            
        part1_feats = get_part_means(part1)
        part2_feats = get_part_means(part2)
        part3_feats = get_part_means(part3)
        
        features_list = global_feats + part1_feats + part2_feats + part3_feats
        features = np.array([features_list])
        
        try:
            features_scaled = self.svm_data['scaler'].transform(features)
            pred_idx = self.svm_data['svm'].predict(features_scaled)[0]
            
            # ['green', 'off', 'red', 'yellow']
            state_map = {0: "GREEN", 1: "UNKNOWN", 2: "RED", 3: "YELLOW"}
            state = state_map.get(pred_idx, "UNKNOWN")
            
            # Hybrid correction for Yellow: Check yellow pixels in the middle bulb section (Part 2)
            if h >= w:
                h3 = h // 3
                part2 = crop[h3:2*h3, :]
                part2_hsv = cv2.cvtColor(part2, cv2.COLOR_BGR2HSV)
                mask_yellow_p2 = cv2.inRange(part2_hsv, np.array([10, 30, 60]), np.array([25, 255, 255]))
                y_p2 = cv2.countNonZero(mask_yellow_p2)
                if y_p2 > 100:
                    state = "YELLOW"
            
            if state != "UNKNOWN":
                return state
        except Exception as e:
            pass
            
        return self._classify_state_hsv(crop)

    def _classify_state_hsv(self, crop: np.ndarray) -> str:
        h, w = crop.shape[:2]
        processed = cv2.GaussianBlur(crop, (3, 3), 0) if h >= 6 and w >= 6 else crop.copy()
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 30, 60]), np.array([9, 255, 255])),
            cv2.inRange(hsv, np.array([160, 30, 60]), np.array([180, 255, 255]))
        )
        mask_yellow = cv2.inRange(hsv, np.array([10, 30, 60]), np.array([25, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([35, 30, 60]), np.array([90, 255, 255]))
        
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
                cv2.inRange(neighborhood_hsv, np.array([0, 30, 60]), np.array([9, 255, 255])),
                cv2.inRange(neighborhood_hsv, np.array([160, 30, 60]), np.array([180, 255, 255]))
            ))
            n_yellow = cv2.countNonZero(cv2.inRange(neighborhood_hsv, np.array([10, 30, 60]), np.array([25, 255, 255])))
            n_green = cv2.countNonZero(cv2.inRange(neighborhood_hsv, np.array([35, 30, 60]), np.array([90, 255, 255])))
            
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