import cv2
import numpy as np
from typing import List, Dict, Any

class Visualizer:
    """
    Visualizer phục vụ Thành viên 1 hiển thị kết quả AI (Hộp đèn cố định và tracking xe).
    ========================================================================
    [PHẦN VẼ VẠCH DỪNG ẢO VÀ ĐA GIÁC RẼ PHẢI LÀ NHIỆM VỤ CỦA THÀNH VIÊN 2]
    ========================================================================
    """
    def __init__(self, stop_line: List[int]):
        self.stop_line = stop_line
        self.COLORS = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 255, 0),
            "UNKNOWN": (128, 128, 128),
            "VEHICLE": (200, 200, 0)
        }

    def draw_scene(
        self, 
        frame: np.ndarray, 
        tracked_vehicles: List[Dict[str, Any]], 
        traffic_lights: List[Dict[str, Any]],
        global_light_state: str,
        violated_ids: List[int],
        active_violations: List[Dict[str, Any]]
    ) -> np.ndarray:
        canvas = frame.copy()
        
        # 1. Vẽ Traffic Lights (AI của Thành viên 1)
        for light in traffic_lights:
            box = light["box"]
            state = light["state"]
            color = self.COLORS.get(state, self.COLORS["UNKNOWN"])
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(
                canvas, f"TL: {state}", (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
        # 2. Vẽ Tracked Vehicles (AI của Thành viên 1)
        for vehicle in tracked_vehicles:
            v_id = vehicle["id"]
            box = vehicle["box"]
            cls_name = vehicle["class_name"]
            
            color = self.COLORS["VEHICLE"]
            cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            label = f"ID: {v_id} {cls_name.upper()}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (box[0], box[1] - h - 10), (box[0] + w, box[1]), color, -1)
            cv2.putText(
                canvas, label, (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
        # 3. HUD hiển thị trạng thái đèn
        hud_bg = canvas.copy()
        cv2.rectangle(hud_bg, (10, 10), (450, 110), (30, 30, 30), -1)
        canvas = cv2.addWeighted(hud_bg, 0.4, canvas, 0.6, 0)
        
        cv2.putText(
            canvas, "TRAFFIC SIGNAL CONTROLLER", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
        )
        
        light_color = self.COLORS.get(global_light_state, self.COLORS["UNKNOWN"])
        cv2.putText(
            canvas, f"STATUS: {global_light_state}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, light_color, 2
        )
        cv2.circle(canvas, (320, 62), 15, light_color, -1)
        
        # [PHẦN VẼ VẠCH DỪNG ẢO, ĐA GIÁC RẼ PHẢI VÀ WARNING BANNERS LÀ NHIỆM VỤ CỦA THÀNH VIÊN 2]
        
        return canvas
