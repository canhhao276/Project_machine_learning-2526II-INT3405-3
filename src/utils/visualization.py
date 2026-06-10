import cv2
import numpy as np
from typing import List, Dict, Any
from src.utils.overlay import OverlayManager
from src.core.zones import RightTurnZone

class Visualizer:
    """
    Visualizer phục vụ hiển thị kết quả AI ở hai chế độ riêng biệt:
    1. 'violation': Chỉ hiển thị hộp đèn vật lý và bắt lỗi vượt đèn đỏ thực tế.
    2. 'adaptive': Chỉ hiển thị giả lập đèn ảo thông minh thích ứng theo lưu lượng xe và Queue ROI.
    """
    def __init__(self, stop_line: List[int], right_turn_zone: List[List[int]] = None):
        self.stop_line = stop_line
        
        # Xác định đa giác vùng rẽ phải
        rt_zone_pts = right_turn_zone if right_turn_zone else [[1000, 600], [1200, 600], [1200, 720], [1000, 720]]
        self.right_turn_zone_poly = RightTurnZone(rt_zone_pts).get_polygon()
        
        # Vùng Queue ROI cho việc đánh giá mật độ xe
        self.roi_polygon = np.array([[530, 486], [1100, 486], [1280, 720], [380, 720]], dtype=np.int32)
        
        self.COLORS = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 255, 0),
            "UNKNOWN": (128, 128, 128),
            "VEHICLE": (200, 200, 0)
        }
        
        # Bảng màu hiển thị mật độ xe trong Queue ROI
        self.DENSITY_COLORS = {
            0: (0, 255, 0),      # Low/Empty - Xanh lá
            1: (0, 165, 255),    # Medium/Normal - Cam
            2: (0, 0, 255)       # High/Congested - Đỏ
        }

    def draw_scene(
        self, 
        frame: np.ndarray, 
        tracked_vehicles: List[Dict[str, Any]], 
        traffic_lights: List[Dict[str, Any]],
        global_light_state: str,
        violated_ids: List[int],
        active_violations: List[Dict[str, Any]],
        mode: str = "violation",  # "violation" hoặc "adaptive"
        density_label: str = "Low/Empty",
        density_class: int = 0,
        countdown: float = 0.0,
        prolong_next_green: bool = False,
        pcu_load: float = 0.0,
        stopped_count: int = 0
    ) -> np.ndarray:
        canvas = frame.copy()
        
        # ==========================================
        # CHẾ ĐỘ 1: BẮT VƯỢT ĐÈN ĐỎ THỰC TẾ (VIOLATION)
        # ==========================================
        if mode == "violation":
            # 1. Vẽ vạch dừng ảo (màu dựa trên đèn gốc)
            pt1 = (self.stop_line[0], self.stop_line[1])
            pt2 = (self.stop_line[2], self.stop_line[3])
            OverlayManager.draw_stop_line(canvas, pt1, pt2, global_light_state)
            
            # 2. Vẽ đa giác rẽ phải
            OverlayManager.draw_right_turn_zone(canvas, self.right_turn_zone_poly)
            
            # 3. Vẽ hộp đèn vật lý do AI nhận dạng từ video
            for light in traffic_lights:
                box = light["box"]
                state = light["state"]
                color = self.COLORS.get(state, self.COLORS["UNKNOWN"])
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(
                    canvas, f"TL: {state}", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
                
            # 4. Vẽ xe bám vết & cảnh báo nếu xe vi phạm vượt đèn đỏ
            for vehicle in tracked_vehicles:
                v_id = vehicle["id"]
                box = vehicle["box"]
                cls_name = vehicle["class_name"]
                
                if v_id in violated_ids:
                    OverlayManager.draw_violation_alert(canvas, box)
                else:
                    color = self.COLORS["VEHICLE"]
                    cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 2)
                    label = f"ID: {v_id} {cls_name.upper()}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(canvas, (box[0], box[1] - h - 10), (box[0] + w, box[1]), color, -1)
                    cv2.putText(
                        canvas, label, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                    )
                    
            # 5. Vẽ HUD thông tin Đèn đỏ thực tế
            hud_bg = canvas.copy()
            cv2.rectangle(hud_bg, (10, 10), (450, 110), (30, 30, 30), -1)
            canvas = cv2.addWeighted(hud_bg, 0.4, canvas, 0.6, 0)
            cv2.rectangle(canvas, (10, 10), (450, 110), (100, 100, 100), 1)
            
            cv2.putText(
                canvas, "RED LIGHT VIOLATION DETECTION", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            light_color = self.COLORS.get(global_light_state, self.COLORS["UNKNOWN"])
            cv2.putText(
                canvas, f"PHYSICAL LIGHT: {global_light_state}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, light_color, 2
            )
            cv2.circle(canvas, (330, 64), 12, light_color, -1)
            
            # Hiển thị số lượng vi phạm hiện tại
            cv2.putText(
                canvas, f"Total Violations: {len(violated_ids)}", (20, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
            
        # ==========================================
        # CHẾ ĐỘ 2: ĐÈN GIAO THÔNG THÍCH ỨNG (ADAPTIVE)
        # ==========================================
        elif mode == "adaptive":
            # 1. Vẽ vùng Queue ROI bán trong suốt dựa trên mức độ ùn tắc
            roi_color = self.DENSITY_COLORS.get(density_class, (0, 255, 0))
            roi_overlay = canvas.copy()
            cv2.fillPoly(roi_overlay, [self.roi_polygon], roi_color)
            
            alpha = 0.18
            cv2.addWeighted(roi_overlay, alpha, canvas, 1 - alpha, 0, canvas)
            cv2.polylines(canvas, [self.roi_polygon], isClosed=True, color=roi_color, thickness=2)
            
            cv2.putText(
                canvas, "QUEUE ROI", (740, 510),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 2
            )
            
            # 2. Vẽ vạch dừng ảo (màu dựa trên đèn ảo giả lập)
            pt1 = (self.stop_line[0], self.stop_line[1])
            pt2 = (self.stop_line[2], self.stop_line[3])
            OverlayManager.draw_stop_line(canvas, pt1, pt2, global_light_state)
            
            # 3. Vẽ đa giác rẽ phải
            OverlayManager.draw_right_turn_zone(canvas, self.right_turn_zone_poly)
            
            # 4. Vẽ Đèn Giao Thông Giả Lập trực tiếp đè lên đèn vật lý
            cv2.rectangle(canvas, (1210, 25), (1265, 185), (20, 20, 20), -1)
            cv2.rectangle(canvas, (1210, 25), (1265, 185), (80, 80, 80), 2)
            
            red_color = (0, 0, 255) if global_light_state == "RED" else (0, 0, 40)
            yellow_color = (0, 255, 255) if global_light_state == "YELLOW" else (0, 40, 40)
            green_color = (0, 255, 0) if global_light_state == "GREEN" else (0, 40, 0)
            
            cv2.circle(canvas, (1237, 55), 14, (60, 60, 60), 1)
            cv2.circle(canvas, (1237, 105), 14, (60, 60, 60), 1)
            cv2.circle(canvas, (1237, 155), 14, (60, 60, 60), 1)
            
            cv2.circle(canvas, (1237, 55), 13, red_color, -1)
            cv2.circle(canvas, (1237, 105), 13, yellow_color, -1)
            cv2.circle(canvas, (1237, 155), 13, green_color, -1)
            
            # Vẽ đồng hồ đếm ngược (Countdown)
            cv2.rectangle(canvas, (1210, 1), (1265, 22), (15, 15, 15), -1)
            cv2.rectangle(canvas, (1210, 1), (1265, 22), (80, 80, 80), 1)
            countdown_sec = int(np.ceil(countdown))
            cv2.putText(
                canvas, f"{countdown_sec:02d}", (1217, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            
            # 5. Vẽ xe bám vết (Không hiện cảnh báo vi phạm để tránh "vi phạm ảo")
            for vehicle in tracked_vehicles:
                v_id = vehicle["id"]
                box = vehicle["box"]
                cls_name = vehicle["class_name"]
                
                cx, cy = vehicle["center"]
                is_queued = cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(cy)), False) >= 0
                
                color = self.COLORS["VEHICLE"]
                cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                status_suffix = " [QUEUE]" if is_queued else ""
                label = f"ID: {v_id} {cls_name.upper()}{status_suffix}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(canvas, (box[0], box[1] - h - 10), (box[0] + w, box[1]), color, -1)
                cv2.putText(
                    canvas, label, (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
                
            # 6. HUD hiển thị trạng thái của Bộ điều khiển thích ứng
            hud_bg = canvas.copy()
            cv2.rectangle(hud_bg, (10, 10), (460, 150), (30, 30, 30), -1)
            canvas = cv2.addWeighted(hud_bg, 0.4, canvas, 0.6, 0)
            cv2.rectangle(canvas, (10, 10), (460, 150), (100, 100, 100), 1)
            
            cv2.putText(
                canvas, "ADAPTIVE TRAFFIC CONTROL SYSTEM", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
            light_color = self.COLORS.get(global_light_state, self.COLORS["UNKNOWN"])
            cv2.putText(
                canvas, f"VIRTUAL LIGHT: {global_light_state} ({countdown_sec}s)", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, light_color, 2
            )
            cv2.circle(canvas, (330, 56), 10, light_color, -1)
            
            cv2.putText(
                canvas, f"TRAFFIC DENSITY: {density_label.upper()}", (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2
            )
            cv2.putText(
                canvas, f"PCU Load: {pcu_load:.1f} | Stopped Vehicles: {stopped_count}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
            )
            constraint_text = "ACTIVE (Min 5.0s per phase)"
            constraint_color = (0, 255, 0)
            cv2.putText(
                canvas, f"Switch Constraint: {constraint_text}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, constraint_color, 1
            )
            
        return canvas
