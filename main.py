import os
import sys
import argparse
import cv2

# Add src folder to sys.path to allow clean imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import SystemConfig
from src.utils.video_reader import VideoStreamHandler
from src.models.traffic_light_detector import TrafficLightDetector
from src.core.tracker import VehicleTracker
from src.core.violation_detector import ViolationDetector
from src.utils.visualization import Visualizer
from src.api.exporter import ViolationExporter

def run_pipeline(video_path: str, show_preview: bool = False):
    """
    Main pipeline function that orchestrates all computer vision and AI modules.
    """
    print("=" * 60)
    print("  INITIALIZING TRAFFIC RED LIGHT VIOLATION DETECTION SYSTEM  ")
    print("=" * 60)
    
    # 1. Load Configurations
    print("[System] Loading configuration file...")
    config = SystemConfig("config.yaml")
    
    # Override video path if supplied via argument
    input_video = video_path if video_path else "data/input/traffic.mp4"
    if not os.path.exists(input_video):
        print(f"[Error] Target video file does not exist: {input_video}")
        print("Please place a traffic video at that path or pass another video file using: python main.py --video <path>")
        return
        
    # 2. Initialize Video Stream Handler
    print(f"[System] Opening video file: {input_video}")
    video_handler = VideoStreamHandler(input_video)
    width, height, fps, total_frames = video_handler.get_properties()
    print(f"[Video Details] Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")
    
    # Initialize Output Video Writer if configured
    if config.save_video:
        print(f"[System] Output will be saved to: {config.output_video_path}")
        video_handler.init_writer(config.output_video_path)

    # 3. Initialize AI Deep Learning Models & Logic Controllers
    print("[Model] Loading YOLOv8 models (Vehicle & Traffic Light Detector)...")
    traffic_light_detector = TrafficLightDetector(
        model_path=config.traffic_light_model_path,
        conf_threshold=config.traffic_light_conf,
        roi=config.traffic_light_roi,
        static_lights=config.static_traffic_lights
    )
    
    vehicle_tracker = VehicleTracker(
        model_path=config.vehicle_model_path,
        conf_threshold=config.vehicle_conf,
        iou_threshold=config.iou
    )
    
    print("[Core] Setting up rule engines & violation bounds...")
    violation_detector = ViolationDetector(
        stop_line=config.stop_line,
        movement_direction=config.movement_direction
    )
    
    visualizer = Visualizer(stop_line=config.stop_line)
    
    exporter = ViolationExporter(
        json_path=config.json_log_path,
        webhook_url=config.webhook_url
    )
    
    print("=" * 60)
    print("  PIPELINE READY: COMMENCING FRAME PROCESSING ")
    print("=" * 60)
    
    if show_preview:
        cv2.namedWindow("Red Light Violation System - Live Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Red Light Violation System - Live Preview", 960, 540)
        
    frame_idx = 0
    try:
        for frame in video_handler.frame_generator():
            frame_idx += 1
            
            # A. Detect Traffic Lights and classify dominant intersection state
            lights = traffic_light_detector.detect_and_classify(frame)
            global_light_state = traffic_light_detector.get_global_traffic_light_state(lights)
            
            # B. Track vehicle bounding boxes across consecutive frames using ByteTrack
            tracked_vehicles = vehicle_tracker.track(frame)
            
            # C. Detect violations at stop line when light is RED
            new_violations = violation_detector.process_frame(
                tracked_vehicles=tracked_vehicles,
                traffic_light_state=global_light_state,
                frame_idx=frame_idx
            )
            
            # D. Export events if new violations detected
            for violation in new_violations:
                print(f"[VIOLATION DETECTED] Vehicle ID #{violation['vehicle_id']} "
                      f"({violation['vehicle_type'].upper()}) ran red light at Frame {frame_idx}!")
                exporter.export_event(violation)
                
            # E. Draw visualization HUD overlays
            annotated_frame = visualizer.draw_scene(
                frame=frame,
                tracked_vehicles=tracked_vehicles,
                traffic_lights=lights,
                global_light_state=global_light_state,
                violated_ids=list(violation_detector.violated_vehicles),
                active_violations=new_violations
            )
            
            # F. Write annotated frame to disk
            if config.save_video:
                video_handler.write_frame(annotated_frame)
                
            # G. Display visual preview if requested
            if show_preview:
                cv2.imshow("Red Light Violation System - Live Preview", annotated_frame)
                # Press 'q' to abort execution
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[System] Execution aborted by user.")
                    break
                    
            if frame_idx % 50 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"[Progress] Frame {frame_idx}/{total_frames} processed ({progress:.1f}%) | "
                      f"Active Violations: {len(violation_detector.violated_vehicles)}")
                      
    except KeyboardInterrupt:
        print("[System] Execution interrupted by manual keyboard signal.")
    finally:
        # Clean resources
        video_handler.release_all()
        if show_preview:
            cv2.destroyAllWindows()
            
        print("\n" + "=" * 60)
        print("  PROCESS COMPLETED SUCCESSFULLY  ")
        print("=" * 60)
        print(f"Total processed frames: {frame_idx}")
        print(f"Total violations logged: {len(violation_detector.violated_vehicles)}")
        print(f"Violation report exported to: {config.json_log_path}")
        print(f"Visualized video output saved to: {config.output_video_path}")
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Red Light Violation Detection Pipeline")
    parser.add_argument("--video", type=str, default="", help="Path to input traffic video file")
    parser.add_argument("--preview", action="store_true", help="Display real-time visual output window")
    args = parser.parse_args()
    
    run_pipeline(video_path=args.video, show_preview=args.preview)
