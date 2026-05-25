import numpy as np
from typing import List, Tuple

class StopLine:
    """
    Represents a virtual stop line on the road.
    Lưu trữ 2 đầu mút của đoạn thẳng vạch dừng.
    Logic crossing mới dùng vector cross product trên toàn bộ đoạn thẳng
    thay vì chỉ so sánh y_threshold, nên hoạt động đúng cả khi camera nghiêng.
    """
    def __init__(self, coordinates: List[int]):
        # coordinates: [x1, y1, x2, y2]
        self.x1, self.y1, self.x2, self.y2 = coordinates

    def get_coords(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return ((self.x1, self.y1), (self.x2, self.y2))


class RightTurnZone:
    """Represents a polygon area where vehicles are allowed to turn right on red."""
    def __init__(self, points: List[List[int]]):
        self.polygon = np.array(points, np.int32)
        # Reshape for OpenCV polylines format
        self.polygon = self.polygon.reshape((-1, 1, 2))

    def get_polygon(self) -> np.ndarray:
        return self.polygon
