import os
import yaml
from typing import Dict, Any, List

class SystemConfig:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_data = self._load_config()
        
        # Parse sections
        self.models: Dict[str, Any] = self.config_data.get("models", {})
        self.thresholds: Dict[str, Any] = self.config_data.get("thresholds", {})
        self.tracking: Dict[str, Any] = self.config_data.get("tracking", {})
        self.scene: Dict[str, Any] = self.config_data.get("scene", {})
        self.output: Dict[str, Any] = self.config_data.get("output", {})
        
        # Properties for easy access
        self.vehicle_model_path: str = self.models.get("vehicle_model", "yolov8n.pt")
        self.traffic_light_model_path: str = self.models.get("traffic_light_model", "yolov8n.pt")
        
        self.vehicle_conf: float = self.thresholds.get("vehicle_conf", 0.35)
        self.traffic_light_conf: float = self.thresholds.get("traffic_light_conf", 0.30)
        self.iou: float = self.thresholds.get("iou", 0.45)
        
        self.tracker_type: str = self.tracking.get("tracker_type", "bytetrack")
        
        self.stop_line: List[int] = self.scene.get("stop_line", [500, 495, 1080, 495])
        self.movement_direction: str = self.scene.get("movement_direction", "down")
        self.right_turn_zone: List[List[int]] = self.scene.get("right_turn_zone", [[950, 200], [1200, 200], [1100, 495], [1080, 495]])
        self.traffic_light_roi: List[int] = self.scene.get("traffic_light_roi", [])
        self.static_traffic_lights: List[List[int]] = self.scene.get("static_traffic_lights", [])
        
        self.save_video: bool = self.output.get("save_video", True)
        self.output_video_path: str = self.output.get("output_video_path", "data/output/result_video.mp4")
        self.save_logs: bool = self.output.get("save_logs", True)
        self.json_log_path: str = self.output.get("json_log_path", "data/output/violations.json")
        self.webhook_url: str = self.output.get("webhook_url", "")
        
        # Ensure directories exist
        self._ensure_directories()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _ensure_directories(self) -> None:
        """Ensure necessary output folders exist."""
        output_dir = os.path.dirname(self.output_video_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        log_dir = os.path.dirname(self.json_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        os.makedirs("data/input", exist_ok=True)
        os.makedirs("data/output", exist_ok=True)
