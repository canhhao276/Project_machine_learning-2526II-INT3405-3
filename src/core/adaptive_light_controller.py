import os
import pickle
import numpy as np
from pathlib import Path

class AdaptiveLightController:
    def __init__(self, fps: float = 30.0, default_green: float = 15.0, default_yellow: float = 3.0, default_red: float = 15.0):
        self.fps = fps
        self.default_green = default_green
        self.default_yellow = default_yellow
        self.default_red = default_red
        
        # Load SVM Density Classifier
        model_path = Path(__file__).parent.parent / "models" / "svm_traffic_density.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.scaler = data['scaler']
                self.svm = data['svm']
                self.classes = data['classes']
            print(f"[AdaptiveLightController] Loaded density SVM model from {model_path}")
        else:
            self.scaler = None
            self.svm = None
            self.classes = ['Low/Empty', 'Medium/Normal', 'High/Congested']
            print(f"[AdaptiveLightController] Warning: Density SVM model not found at {model_path}. Using fallback rule-based density.")

        # State Machine Initialization
        # Bắt đầu bằng ĐÈN ĐỎ để có thể theo dõi xe dồn ứ trước vạch dừng
        self.current_state = "RED" 
        self.time_remaining = self.default_red
        
        # Biến đếm thời gian thực tế đã trôi qua trong pha hiện tại
        self.time_elapsed = 0.0
        
        # Cờ ghi nhận việc log trạng thái khóa giữ xanh do ùn tắc
        self.green_lock_logged = False
        
        self.last_predicted_density = 0
        self.last_prediction_probs = [0.0, 0.0, 0.0]
        
        self.has_synced_initial_state = False

    def initialize_state(self, physical_state: str):
        if not self.has_synced_initial_state and physical_state in ["RED", "GREEN", "YELLOW"]:
            self.current_state = physical_state
            if physical_state == "RED":
                self.time_remaining = self.default_red
            elif physical_state == "GREEN":
                self.time_remaining = self.default_green
            elif physical_state == "YELLOW":
                self.time_remaining = self.default_yellow
            self.has_synced_initial_state = True
            print(f"[Controller] Đã đồng bộ trạng thái ban đầu theo đèn vật lý: {physical_state}")
            

    def predict_density(self, motorcycle_count: int, car_count: int, stopped_vehicles: int, pcu_load: float, average_speed: float) -> int:
        """
        Dự đoán lớp mật độ giao thông sử dụng SVM.
        Trả về:
            0: Low / Empty
            1: Medium / Normal
            2: High / Congested
        """
        features = [motorcycle_count, car_count, stopped_vehicles, pcu_load, average_speed]
        
        if self.svm and self.scaler:
            try:
                features_scaled = self.scaler.transform([features])
                pred = self.svm.predict(features_scaled)[0]
                if hasattr(self.svm, "predict_proba"):
                    self.last_prediction_probs = self.svm.predict_proba(features_scaled)[0]
                self.last_predicted_density = int(pred)
                return self.last_predicted_density
            except Exception as e:
                print(f"[AdaptiveLightController] SVM Prediction error: {e}")
                
        # Phân loại theo luật dự phòng nếu SVM chưa load được
        if pcu_load <= 1.0:
            pred = 0
        elif stopped_vehicles == 0:
            pred = 1
        elif pcu_load <= 4.0:
            pred = 1
        else:
            pred = 2
            
        self.last_predicted_density = pred
        return pred

    def update(self, motorcycle_count: int, car_count: int, stopped_vehicles: int, pcu_load: float, average_speed: float) -> dict:
        """
        Cập nhật trạng thái đèn qua mỗi frame (mỗi bước giảm 1/fps giây) và áp dụng logic tối ưu động linh hoạt.
        """
        dt = 1.0 / self.fps
        
        # Dự đoán mật độ
        density_class = self.predict_density(motorcycle_count, car_count, stopped_vehicles, pcu_load, average_speed)
        
        # Chỉ giảm thời gian nếu không bị khóa giữ xanh do ùn tắc
        is_green_locked = (self.current_state == "GREEN" and self.time_remaining <= 0 and density_class == 2)
        if not is_green_locked:
            self.time_remaining -= dt
        self.time_elapsed += dt
        
        # Logic tối ưu động linh hoạt (Đảm bảo thời gian sáng tối thiểu của mỗi pha là 5.0 giây để ổn định):
        # 1. Nếu đang ĐÈN XANH mà mật độ trống (Low) và đèn đã xanh ít nhất 5s: Rút ngắn thời gian xanh còn lại về 3.0s để đếm ngược
        if self.current_state == "GREEN":
            if density_class == 0 and self.time_elapsed >= 5.0:
                if self.time_remaining > 3.0:
                    print(f"[Controller] Mật độ THẤP và đã xanh được {self.time_elapsed:.1f}s. Rút ngắn thời gian đèn XANH còn 3.0s để đếm ngược sang VÀNG.")
                    self.time_remaining = 3.0
                
        # 2. Nếu đang ĐÈN ĐỎ mà mật độ ùn tắc cao (High): Rút ngắn thời gian đỏ còn lại về 3.0s ngay lập tức để đếm ngược sang XANH
        elif self.current_state == "RED":
            if density_class == 2:
                if self.time_remaining > 3.0:
                    print(f"[Controller] Nhiều xe chờ đèn đỏ (ùn tắc CAO). Rút ngắn thời gian đèn ĐỎ còn 3.0s để đếm ngược sang XANH.")
                    self.time_remaining = 3.0

        # Kiểm tra chuyển trạng thái đếm ngược tự nhiên
        if self.time_remaining <= 0:
            if self.current_state == "GREEN" and density_class == 2:
                # Đang ùn tắc cao, giữ đèn xanh sáng tiếp (Khóa giữ xanh) cho đến khi hết ùn tắc mới chuyển màu
                if not self.green_lock_logged:
                    print(f"[Controller] Đèn xanh đã hết giờ đếm ngược nhưng ngã tư vẫn đang ùn tắc CAO. GIỮ ĐÈN XANH liên tục đến khi hết tắc.")
                    self.green_lock_logged = True
                self.time_remaining = 0.0  # Giữ đồng hồ đếm ngược ở 0
            else:
                if self.green_lock_logged:
                    print(f"[Controller] Đã giải tỏa hết ùn tắc. Kết thúc việc giữ pha xanh.")
                    self.green_lock_logged = False
                self.transition_state()
            
        return {
            "state": self.current_state,
            "time_remaining": max(0.0, self.time_remaining),
            "density_class": density_class,
            "density_label": self.classes[density_class],
            "prolong_next_green": False
        }
        
    def transition_state(self):
        self.time_elapsed = 0.0
        self.green_lock_logged = False
        if self.current_state == "GREEN":
            self.current_state = "YELLOW"
            self.time_remaining = self.default_yellow
            print(f"[Controller] Hết thời gian xanh. Đèn chuyển sang VÀNG. Thời gian: {self.default_yellow:.1f}s.")
        elif self.current_state == "YELLOW":
            self.current_state = "RED"
            self.time_remaining = self.default_red
            print(f"[Controller] Hết thời gian vàng. Đèn chuyển sang ĐỎ. Thời gian: {self.default_red:.1f}s.")
        elif self.current_state == "RED":
            self.current_state = "GREEN"
            self.time_remaining = self.default_green
            print(f"[Controller] Hết thời gian đỏ. Đèn chuyển sang XANH. Thời gian: {self.default_green:.1f}s.")
