import cv2
import numpy as np
from typing import Tuple, List

def get_bottom_center(bbox: List[int]) -> Tuple[int, int]:
    """
    Return bottom-center point of bbox.
    Dùng bánh xe chạm mặt đường thay vì centroid.
    """
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int(y2))

def is_point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
    """Check if a point is inside a polygon using OpenCV."""
    result = cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False)
    return result >= 0

def cross_product_sign(point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
    """
    Tính cross product để xác định điểm nằm ở phía nào của đường thẳng.
    Trả về > 0 nếu điểm ở bên trái (phía trên với đường ngang trái->phải),
             < 0 nếu ở bên phải (phía dưới),
             = 0 nếu nằm trên đường thẳng.
    Công thức: (B-A) x (P-A) = (Bx-Ax)*(Py-Ay) - (By-Ay)*(Px-Ax)
    """
    return ((line_end[0] - line_start[0]) * (point[1] - line_start[1]) -
            (line_end[1] - line_start[1]) * (point[0] - line_start[0]))

def has_crossed_line(prev_point: Tuple[int, int], curr_point: Tuple[int, int],
                     line_start: Tuple[int, int], line_end: Tuple[int, int]) -> bool:
    """
    Kiểm tra xem đoạn thẳng (prev_point -> curr_point) có cắt qua 
    đoạn thẳng (line_start -> line_end) hay không, sử dụng cross product.
    Hoạt động chính xác với đường nghiêng (camera perspective).
    """
    d1 = cross_product_sign(prev_point, line_start, line_end)
    d2 = cross_product_sign(curr_point, line_start, line_end)
    # Nếu 2 điểm nằm ở 2 phía khác nhau của đường thẳng => đã cắt qua
    if d1 * d2 > 0:
        return False
    # Kiểm tra thêm: đoạn thẳng di chuyển cũng phải cắt đoạn stop line (không phải đường thẳng vô hạn)
    d3 = cross_product_sign(line_start, prev_point, curr_point)
    d4 = cross_product_sign(line_end, prev_point, curr_point)
    if d3 * d4 > 0:
        return False
    return True

