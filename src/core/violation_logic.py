from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from src.core.geometry import get_bottom_center, is_point_in_polygon, has_crossed_line
from src.core.zones import StopLine, RightTurnZone

class ViolationManager:
    """
    Core logic to determine if a vehicle violated the red light.

    - Dùng vector cross product thay vì so sánh Y đơn giản (hỗ trợ camera nghiêng).
    - Direction filtering: chỉ bắt xe đi đúng hướng (từ trên xuống), bỏ qua xe quay đầu/lùi.
    - Frame-by-frame crossing: kiểm tra từng cặp frame liên tiếp thay vì chỉ so đầu-cuối.
    - Cooldown per ID: mỗi xe chỉ bị báo vi phạm 1 lần duy nhất.
    """
    
    # Ngưỡng di chuyển tối thiểu (pixel) để xác nhận xe thực sự đang di chuyển
    # Tránh bắt nhầm xe đứng yên mà bbox nhảy do tracking lỗi
    MIN_MOVEMENT_THRESHOLD = 5
    
    def __init__(self, stop_line: StopLine, right_turn_zone: RightTurnZone, movement_direction: str = "down"):
        self.stop_line = stop_line
        self.right_turn_zone = right_turn_zone
        self.movement_direction = movement_direction
        
        # Lưu vị trí tâm xe qua các frame, key = vehicle_id
        self.vehicle_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.HISTORY_LENGTH = 15
        
        # Set chứa các ID đã bị báo vi phạm => cooldown vĩnh viễn trong 1 session
        self.reported_ids: Set[int] = set()

    def update_and_check(self, vehicle: Dict[str, Any], light_state: str) -> bool:
        """
        Cập nhật lịch sử xe và kiểm tra vi phạm.
        Returns True nếu phát hiện vi phạm MỚI.
        """
        v_id = vehicle["id"]
        bbox = vehicle["box"]
        bottom_center = get_bottom_center(bbox)
        
        # === BƯỚC 1: Cập nhật lịch sử tọa độ ===
        self.vehicle_history[v_id].append(bottom_center)
        if len(self.vehicle_history[v_id]) > self.HISTORY_LENGTH:
            self.vehicle_history[v_id].pop(0)
            
        # === BƯỚC 2: Cooldown - đã báo rồi thì bỏ qua ===
        if v_id in self.reported_ids:
            return False
            
        # === BƯỚC 3: Chỉ xét khi đèn ĐỎ ===
        if light_state.upper() != "RED":
            return False
            
        # === BƯỚC 4: Cần ít nhất 2 điểm lịch sử ===
        history = self.vehicle_history[v_id]
        if len(history) < 2:
            return False
            
        # === BƯỚC 5: Direction filtering - chỉ bắt xe đi đúng hướng ===
        if not self._is_moving_correct_direction(history):
            return False
            
        # === BƯỚC 6: Kiểm tra cắt vạch bằng vector crossing ===
        if not self._check_vector_crossing(history):
            return False
            
        # === BƯỚC 7: Kiểm tra xe có đang rẽ phải hợp lệ không ===
        if self._is_right_turning(history):
            return False
            
        # === TẤT CẢ ĐIỀU KIỆN THỎA MÃN => VI PHẠM ===
        self.reported_ids.add(v_id)
        return True

    def _is_moving_correct_direction(self, history: List[Tuple[int, int]]) -> bool:
        """
        Kiểm tra xe có đang di chuyển đúng hướng không.
        Tính delta_y giữa vị trí cũ nhất và mới nhất trong lịch sử.
        Chống bắt nhầm xe quay đầu, xe lùi, hoặc tracking lỗi nhảy bbox.
        """
        oldest = history[0]
        newest = history[-1]
        delta_y = newest[1] - oldest[1]
        
        if self.movement_direction == "down":
            # Xe phải đi xuống (delta_y > ngưỡng)
            return delta_y > self.MIN_MOVEMENT_THRESHOLD
        elif self.movement_direction == "up":
            # Xe phải đi lên (delta_y < -ngưỡng)
            return delta_y < -self.MIN_MOVEMENT_THRESHOLD
        return False

    def _check_vector_crossing(self, history: List[Tuple[int, int]]) -> bool:
        """
        Kiểm tra xe đã cắt qua vạch dừng bằng thuật toán vector crossing (cross product).
        Duyệt từng cặp frame liên tiếp (prev, curr) để phát hiện thời điểm chính xác cắt vạch.
        Hoạt động đúng cả khi camera nghiêng (stop line không ngang tuyệt đối).
        """
        line_start = (self.stop_line.x1, self.stop_line.y1)
        line_end = (self.stop_line.x2, self.stop_line.y2)
        
        for i in range(1, len(history)):
            prev_pt = history[i - 1]
            curr_pt = history[i]
            if has_crossed_line(prev_pt, curr_pt, line_start, line_end):
                return True
        return False
        
    def reset(self):
        """Xóa toàn bộ lịch sử và trạng thái vi phạm."""
        self.vehicle_history.clear()
        self.reported_ids.clear()

    def _is_right_turning(self, history):

        recent_points = history[-5:]

        inside_count = 0

        polygon = self.right_turn_zone.get_polygon()

        for pt in recent_points:
            if is_point_in_polygon(pt, polygon):
                inside_count += 1

        if inside_count < 3:
            return False

        dx = recent_points[-1][0] - recent_points[0][0]
        dy = recent_points[-1][1] - recent_points[0][1]

        return dx > 15 and dy > 10
        

