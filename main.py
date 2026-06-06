import os
import sys
import argparse
import cv2
import time

# Add src folder to sys.path to allow clean imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import SystemConfig
from src.utils.video_reader import VideoStreamHandler
from src.models.traffic_light_detector import TrafficLightDetector
from src.core.tracker import VehicleTracker
from src.core.violation_detector import ViolationDetector
from src.utils.visualization import Visualizer
from src.api.exporter import ViolationExporter


def run_pipeline(video_path: str, show_preview: bool = False,
                 total_actual: int = None, true_positives: int = None,
                 false_positives: int = None):
    """
    Main pipeline. Sau khi chạy xong, truyền kết quả đối chiếu thủ công
    (total_actual, true_positives, false_positives) để in Precision/Recall.
    """
    print("=" * 60)
    print("  INITIALIZING TRAFFIC RED LIGHT VIOLATION DETECTION SYSTEM  ")
    print("=" * 60)
    
    config = SystemConfig("config.yaml")
    
    # Override video path if supplied via argument
    input_video = video_path if video_path else "data/input/traffic.mp4"
    if not os.path.exists(input_video):
        print(f"[Error] Target video file does not exist: {input_video}")
        print("Please place a traffic video at that path or pass another video file using: python main.py --video <path>")
        return
        
    # 2. Initialize Video Stream Handler
    print(f"[System] Opening video file: {input_video}")
    target_size = tuple(config.target_resolution) if config.target_resolution else None
    video_handler = VideoStreamHandler(input_video, target_size=target_size)
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
        right_turn_zone=config.right_turn_zone,
        movement_direction=config.movement_direction
    )
    
    visualizer = Visualizer(
        stop_line=config.stop_line,
        right_turn_zone=config.right_turn_zone
    )
    
    # ── THÀNH VIÊN 3: khởi tạo exporter ─────────────────────
    exporter = ViolationExporter(
        json_path=config.json_log_path,
        webhook_url=config.webhook_url,
        root_dir="Luutru_Vipham",
        save_crops=True,
    )
    
    print("=" * 60)
    print("  PIPELINE READY — PROCESSING FRAMES")
    print("=" * 60)
    
    if show_preview:
        cv2.namedWindow("Red Light Violation System - Live Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Red Light Violation System - Live Preview", 960, 540)
        
    pipeline_start_time = time.time()
    frame_idx = 0
    try:
        for frame in video_handler.frame_generator():
            frame_idx += 1
            
            # A. Đèn giao thông
            lights = traffic_light_detector.detect_and_classify(frame)
            global_light_state = traffic_light_detector.get_global_traffic_light_state(lights)
            
            # B. Bám vết xe
            tracked_vehicles = vehicle_tracker.track(frame)
            
            # C. Phát hiện vi phạm
            new_violations = violation_detector.process_frame(
                tracked_vehicles=tracked_vehicles,
                traffic_light_state=global_light_state,
                frame_idx=frame_idx
            )
            
            # D. Xuất vi phạm — TRUYỀN FRAME GỐC để crop ảnh
            for violation in new_violations:
                print(f"[VIOLATION] Vehicle ID #{violation['vehicle_id']} "
                      f"({violation['vehicle_type'].upper()}) at Frame {frame_idx}")
                exporter.export_event(violation, frame=frame)   # ← truyền frame
                
            # E. Vẽ overlay
            annotated_frame = visualizer.draw_scene(
                frame=frame,
                tracked_vehicles=tracked_vehicles,
                traffic_lights=lights,
                global_light_state=global_light_state,
                violated_ids=list(violation_detector.violated_vehicles),
                active_violations=new_violations
            )
            
            # F. Ghi video output
            if config.save_video:
                video_handler.write_frame(annotated_frame)
                
            # G. FPS tracker tick
            exporter.fps_tracker.tick()

            # H. Preview with real-time speed synchronization
            if show_preview:
                elapsed_time = time.time() - pipeline_start_time
                expected_video_time = frame_idx / fps
                
                # If processing is faster than real-time, sleep to maintain 1.0x playback speed
                if elapsed_time < expected_video_time:
                    time.sleep(expected_video_time - elapsed_time)
                    cv2.imshow("Red Light Violation System - Live Preview", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[System] Execution aborted by user.")
                        break
                else:
                    # If processing is slower, skip cv2.imshow for some frames to catch up to real-time speed,
                    # showing at least every 3rd frame to ensure visibility.
                    if (elapsed_time - expected_video_time) < 0.15 or frame_idx % 3 == 0:
                        cv2.imshow("Red Light Violation System - Live Preview", annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("[System] Execution aborted by user.")
                            break
                    
            if frame_idx % 50 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                print(f"[Progress] {frame_idx}/{total_frames} ({progress:.1f}%) | "
                      f"FPS: {exporter.fps_tracker.get_fps():.1f} | "
                      f"Violations: {len(violation_detector.violated_vehicles)}")
                      
    except KeyboardInterrupt:
        print("[System] Execution interrupted by manual keyboard signal.")
    finally:
        # Clean resources
        video_handler.release_all()
        if show_preview:
            cv2.destroyAllWindows()
            
        # ── THÀNH VIÊN 3: in báo cáo tổng kết ───────────────
        exporter.print_summary(
            total_actual=total_actual,
            true_positives=true_positives,
            false_positives=false_positives,
        )

        print("=" * 60)
        print(f"  Frames processed : {frame_idx}")
        print(f"  Violations logged: {exporter.get_total_violations()}")
        print(f"  JSON  → {config.json_log_path}")
        print(f"  Video → {config.output_video_path}")
        print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Red Light Violation Detection")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--preview", action="store_true")
    # ── Thành viên 3: tham số đối chiếu thủ công ──
    parser.add_argument("--total-actual", type=int, default=None,
                        help="Tổng số vi phạm thực tế (đếm tay khi xem video)")
    parser.add_argument("--true-positives", type=int, default=None,
                        help="Số vi phạm hệ thống bắt đúng")
    parser.add_argument("--false-positives", type=int, default=None,
                        help="Số vi phạm hệ thống báo nhầm")
    args = parser.parse_args()
    
    run_pipeline(
        video_path=args.video,
        show_preview=args.preview,
        total_actual=args.total_actual,
        true_positives=args.true_positives,
        false_positives=args.false_positives,
    )
