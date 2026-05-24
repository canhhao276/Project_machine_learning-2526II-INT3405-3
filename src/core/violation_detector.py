from typing import List, Dict, Any

class ViolationDetector:
    """
    ========================================================================
    [NHIỆM VỤ KỸ THUẬT CỦA THÀNH VIÊN 2: LOGIC DEVELOPER (XỬ LÝ HÌNH HỌC)]
    ========================================================================
    Nhiệm vụ:
    1. Xác định tọa độ pixel và vẽ vùng đa giác rẽ phải đè lên video.
    2. Lập trình logic bắt lỗi: Nếu đèn ĐỎ và xe cắt Stop Line và xe KHÔNG rẽ phải -> Vi phạm.
    3. Truyền dữ liệu vi phạm cho module của Thành viên 3.
    
    Hãy viết toàn bộ thuật toán bắt lỗi của bạn tại đây!
    """
    
    def __init__(self, stop_line: List[int], movement_direction: str = "down"):
        self.violations_log = []
        self.violated_vehicles = set()

    def process_frame(self, tracked_vehicles: List[Dict[str, Any]], traffic_light_state: str, frame_idx: int) -> List[Dict[str, Any]]:
        # Để trống hoàn toàn thuật toán bắt lỗi cho Thành viên 2 tự lập trình
        return []

    def get_all_violations(self) -> List[Dict[str, Any]]:
        return self.violations_log
        
    def reset(self) -> None:
        self.violations_log.clear()
        self.violated_vehicles.clear()
