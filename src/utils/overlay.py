import cv2
import numpy as np
from typing import Tuple, List

class OverlayManager:
    """Handles rendering of geometric zones and alerts."""
    
    @staticmethod
    def draw_stop_line(frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], light_state: str):
        # Color line based on traffic light state
        if light_state == "RED":
            color = (0, 0, 255) # BGR
        elif light_state == "GREEN":
            color = (0, 255, 0)
        elif light_state == "YELLOW":
            color = (0, 255, 255)
        else:
            color = (255, 255, 255)
            
        cv2.line(frame, pt1, pt2, color, thickness=3)
        # Add label
        cv2.putText(frame, "STOP LINE", (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def draw_right_turn_zone(frame: np.ndarray, polygon: np.ndarray):
        # Draw translucent polygon
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (255, 0, 0)) # Blue for right turn zone
        
        # Blend overlay with original frame
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw border
        cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Add text in the middle of polygon
        M = cv2.moments(polygon)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, "RIGHT TURN", (cX - 50, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    @staticmethod
    def draw_violation_alert(frame: np.ndarray, bbox: List[int]):
        x1, y1, x2, y2 = bbox
        # Draw thick red bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        
        # Flashy text
        label = "VIOLATION!"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - h - 15), (x1 + w, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
