import yaml
import os
from typing import List, Dict, Any

class ConfigLoader:
    """
    Module hỗ trợ đọc và ghi file YAML cấu hình một cách an toàn.
    Sử dụng PyYAML để load, và thay thế chuỗi để ghi đè nhằm giữ lại các comment có sẵn trong file.
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def update_zones(self, stop_line: List[int], right_turn_zone: List[List[int]]) -> bool:
        """
        Cập nhật tọa độ stop_line và right_turn_zone vào file config.yaml.
        Phương pháp thay thế dòng chữ (string manipulation) giúp không làm mất các comment (#) của user.
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                if line.strip().startswith("stop_line:"):
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(f"{indent}stop_line: {stop_line}\n")
                elif line.strip().startswith("right_turn_zone:"):
                    indent = line[:len(line) - len(line.lstrip())]
                    # Format as JSON-like list of lists without spaces for cleaner YAML inline look
                    rtz_str = str(right_turn_zone).replace(" ", "")
                    new_lines.append(f"{indent}right_turn_zone: {rtz_str}\n")
                else:
                    new_lines.append(line)

            with open(self.config_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
                
            return True
        except Exception as e:
            print(f"[Error] Failed to update config: {e}")
            return False
