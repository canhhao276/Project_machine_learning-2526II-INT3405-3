import cv2
import numpy as np
import os
import csv
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.tracker import VehicleTracker

# Set up paths
input_videos = ["data/input/detection1.mp4", "data/input/detection2.mp4"]
output_csv = "data/traffic_light/density_features.csv"

# Ensure directories exist
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Queue ROI: covering the lanes below the stop line
# x=530 to 1100 at y=486, extending down to y=720
roi_polygon = np.array([[530, 486], [1100, 486], [1280, 720], [380, 720]], dtype=np.int32)

def is_inside_roi(pt):
    return cv2.pointPolygonTest(roi_polygon, (float(pt[0]), float(pt[1])), False) >= 0

def calculate_speed(history, frames_back=5):
    if len(history) < frames_back + 1:
        return 999.0  # Not enough history yet, assume moving
    p1 = np.array(history[-1])
    p2 = np.array(history[-(frames_back + 1)])
    displacement = np.linalg.norm(p1 - p2)
    return displacement / frames_back

def run_collection():
    tracker = VehicleTracker(model_path="yolov8s.pt", conf_threshold=0.15, iou_threshold=0.5)
    
    samples = []
    
    for video_name in input_videos:
        video_path = Path(video_name)
        if not video_path.exists():
            print(f"Skipping {video_name} (not found)")
            continue
            
        print(f"\nProcessing {video_name}...")
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        tracker.reset()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            
            # Tracker
            tracked_vehicles = tracker.track(frame)
            
            # Filter vehicles inside Queue ROI
            vehicles_in_roi = []
            for obj in tracked_vehicles:
                cx, cy = obj["center"]
                if is_inside_roi((cx, cy)):
                    vehicles_in_roi.append(obj)
            
            # Extract features every 10 frames to avoid high correlation
            if frame_idx % 10 == 0:
                motorcycles = 0
                cars = 0
                stopped_vehicles = 0
                speeds = []
                
                for obj in vehicles_in_roi:
                    cls_name = obj["class_name"]
                    v_id = obj["id"]
                    
                    if cls_name == "motorcycle":
                        motorcycles += 1
                    else:
                        cars += 1  # count car, truck, bus as large vehicles
                        
                    # Calculate speed from tracker history
                    history = tracker.track_history.get(v_id, [])
                    speed = calculate_speed(history, frames_back=5)
                    
                    if speed != 999.0:
                        speeds.append(speed)
                        # Speed < 1.2 pixels/frame is considered stopped
                        if speed < 1.2:
                            stopped_vehicles += 1
                            
                avg_speed = np.mean(speeds) if speeds else 15.0 # default moving speed if empty
                
                # Standard Passenger Car Unit (PCU) calculation
                # motorcycle = 0.3, car/bus/truck = 1.0
                pcu_load = motorcycles * 0.3 + cars * 1.0
                
                # Auto-labeling rules (Pseudo-labeling)
                if pcu_load == 0:
                    label = 0  # Low / Empty
                elif stopped_vehicles == 0:
                    label = 1  # Medium (moving traffic)
                elif pcu_load <= 0.6:
                    label = 1  # Medium (1-2 motorcycles, stopped/slow)
                else:
                    label = 2  # High (dense or stopped traffic)
                    
                samples.append({
                    "motorcycle_count": motorcycles,
                    "car_count": cars,
                    "stopped_vehicles": stopped_vehicles,
                    "pcu_load": pcu_load,
                    "average_speed": avg_speed,
                    "label": label
                })
                
                if frame_idx % 200 == 0:
                    print(f"  Frame {frame_idx}: MCs={motorcycles}, Cars={cars}, Stopped={stopped_vehicles}, PCU={pcu_load:.1f}, AvgSpeed={avg_speed:.2f} -> Label: {label}")
                    
        cap.release()

    # Save to CSV
    if samples:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["motorcycle_count", "car_count", "stopped_vehicles", "pcu_load", "average_speed", "label"])
            writer.writeheader()
            writer.writerows(samples)
        print(f"\nSUCCESS: Generated dataset with {len(samples)} samples saved to {output_csv}")
        
        # Check distribution
        labels = [s["label"] for s in samples]
        print("Label distribution:")
        for lbl in sorted(list(set(labels))):
            print(f"  Class {lbl}: {labels.count(lbl)} samples")
    else:
        print("\nERROR: No samples collected.")

if __name__ == "__main__":
    run_collection()
