from src.core.geometry import is_point_in_polygon
from typing import List, Dict, Any
import time

from src.core.zones import StopLine, RightTurnZone
from src.core.violation_logic import ViolationManager

class ViolationDetector:
    """
    ========================================================================
    [NHIỆM VỤ KỸ THUẬT CỦA THÀNH VIÊN 2: LOGIC DEVELOPER (XỬ LÝ HÌNH HỌC)]
    ========================================================================
    Nhiệm vụ:
    1. Xác định tọa độ pixel và vẽ vùng đa giác rẽ phải đè lên video.
    2. Lập trình logic bắt lỗi: Nếu đèn ĐỎ và xe cắt Stop Line và xe KHÔNG rẽ phải -> Vi phạm.
    3. Truyền dữ liệu vi phạm cho module của Thành viên 3.
    """
    
    def __init__(self, stop_line: List[int], right_turn_zone: List[List[int]], movement_direction: str = "down"):
        self.stop_line_zone = StopLine(stop_line)
        # Default polygon if none provided
        rt_zone_pts = right_turn_zone if right_turn_zone else [[1000, 600], [1200, 600], [1200, 720], [1000, 720]]
        self.right_turn_zone = RightTurnZone(rt_zone_pts)
        
        self.manager = ViolationManager(
            stop_line=self.stop_line_zone, 
            right_turn_zone=self.right_turn_zone, 
            movement_direction=movement_direction
        )
        
        self.violations_log = []
        # Property to expose violated vehicle IDs easily to visualizer
        self._violated_vehicles = set()

    @property
    def violated_vehicles(self):
        return self._violated_vehicles

    def process_frame(self, tracked_vehicles: List[Dict[str, Any]], traffic_light_state: str, frame_idx: int) -> List[Dict[str, Any]]:
        new_violations = []
        
        for vehicle in tracked_vehicles:
            is_violation = self.manager.update_and_check(vehicle, traffic_light_state)
            
            if is_violation:
                v_id = vehicle["id"]
                self._violated_vehicles.add(v_id)
                
                violation_record = {
                    "vehicle_id": v_id,
                    "vehicle_type": vehicle.get("class_name", "unknown"),
                    "bbox": vehicle["box"],
                    "timestamp": time.time(),
                    "frame": frame_idx,
                    "confidence": float(vehicle.get("conf", 0.0))
                }
                self.violations_log.append(violation_record)
                new_violations.append(violation_record)
                
        return new_violations

    def get_all_violations(self) -> List[Dict[str, Any]]:
        return self.violations_log
        
    def reset(self) -> None:
        self.violations_log.clear()
        self._violated_vehicles.clear()
        self.manager.reset()


