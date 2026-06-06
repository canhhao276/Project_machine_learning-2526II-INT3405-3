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
        
        # Set chứa các ID đã bị báo vi phạm trong session
        self.violated_ids: Set[int] = set()
        self.canceled_ids: Set[int] = set()

    def update_and_check(self, vehicle: Dict[str, Any], light_state: str) -> str:
        """
        Cập nhật lịch sử xe và kiểm tra trạng thái vi phạm.
        Trả về "violation", "cancel" hoặc "none".
        """
        v_id = vehicle["id"]
        bbox = vehicle["box"]
        bottom_center = get_bottom_center(bbox)
        
        # === BƯỚC 1: Cập nhật lịch sử tọa độ ===
        self.vehicle_history[v_id].append(bottom_center)
        if len(self.vehicle_history[v_id]) > self.HISTORY_LENGTH:
            self.vehicle_history[v_id].pop(0)
            
        # Nếu xe đã bị hủy vi phạm trước đó thì bỏ qua luôn
        if v_id in self.canceled_ids:
            return "none"

        history = self.vehicle_history[v_id]
        if len(history) < 2:
            return "none"

        if light_state.upper() != "RED":
            return "none"

        if not self._is_moving_correct_direction(history):
            return "none"

        # Nếu xe đã tham gia vùng rẽ phải, hủy vi phạm cũ nếu có
        if self._has_entered_right_turn_zone(history):
            if v_id in self.violated_ids:
                self.violated_ids.remove(v_id)
                self.canceled_ids.add(v_id)
                return "cancel"
            return "none"

        if v_id in self.violated_ids:
            return "none"

        if not self._check_vector_crossing(history):
            return "none"

        if self._is_right_turning(history):
            return "none"

        self.violated_ids.add(v_id)
        return "violation"

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
        self.violated_ids.clear()
        self.canceled_ids.clear()

    def _has_entered_right_turn_zone(self, history: List[Tuple[int, int]]) -> bool:
        """Kiểm tra nếu xe đã vào vùng rẽ phải trong lịch sử gần nhất."""
        polygon = self.right_turn_zone.get_polygon()
        for pt in history[-self.HISTORY_LENGTH:]:
            if is_point_in_polygon(pt, polygon):
                return True
        return False

    def _is_right_turning(self, history):

        recent_points = history[-5:]

        inside_count = 0

        polygon = self.right_turn_zone.get_polygon()

        for pt in recent_points:
            if is_point_in_polygon(pt, polygon):
                inside_count += 1

        if inside_count < 2:
            return False

        dx = recent_points[-1][0] - recent_points[0][0]
        dy = recent_points[-1][1] - recent_points[0][1]

        return dx > 15 and dy > 10
        

