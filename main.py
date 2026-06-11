import os
import sys
import argparse
import cv2
import time
import numpy as np

# Add src folder to sys.path to allow clean imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import SystemConfig
from src.utils.video_reader import VideoStreamHandler
from src.core.tracker import VehicleTracker
from src.core.violation_detector import ViolationDetector
from src.utils.visualization import Visualizer
from src.api.exporter import ViolationExporter
from src.storage.video_clip_extractor import ViolationClipExtractor
from src.core.adaptive_light_controller import AdaptiveLightController
from src.models.traffic_light_detector import TrafficLightDetector

# Đa giác Queue ROI cho phân tích mật độ xe
roi_polygon = np.array([[530, 486], [1100, 486], [1280, 720], [380, 720]], dtype=np.int32)

def is_inside_roi(pt):
    return cv2.pointPolygonTest(roi_polygon, (float(pt[0]), float(pt[1])), False) >= 0

def calculate_speed(history, frames_back=5):
    if len(history) < frames_back + 1:
        return 999.0  # Chưa đủ lịch sử vết, giả định đang chuyển động
    p1 = np.array(history[-1])
    p2 = np.array(history[-(frames_back + 1)])
    displacement = np.linalg.norm(p1 - p2)
    return displacement / frames_back


def run_pipeline(video_path: str, mode: str = "violation", show_preview: bool = False,
                 total_actual: int = None, true_positives: int = None,
                 false_positives: int = None):
    """
    Main pipeline hỗ trợ 3 chế độ chạy độc lập/so sánh song song:
    1. 'violation': Chỉ phát hiện vượt đèn đỏ thực tế bằng AI dò đèn vật lý.
    2. 'adaptive': Giả lập đèn xanh/vàng/đỏ thích ứng thông minh dựa trên mật độ xe (SVM).
    3. 'compare': So sánh trực quan song song (Side-by-Side) cả 2 chế độ trên cùng khung hình.
    """
    print("=" * 60)
    print(f"  INITIALIZING TRAFFIC SYSTEM - MODE: {mode.upper()}  ")
    print("=" * 60)
    
    config = SystemConfig("config.yaml")
    
    # Ghi đè đường dẫn video nếu cung cấp qua tham số dòng lệnh
    input_video = video_path if video_path else "data/input/detection2.mp4"
    if not os.path.exists(input_video):
        print(f"[Error] Target video file does not exist: {input_video}")
        print("Please place a traffic video at that path or pass another video file using: python main.py --video <path>")
        return
        
    # 2. Khởi tạo Video Stream Handler
    print(f"[System] Opening video file: {input_video}")
    target_size = tuple(config.target_resolution) if config.target_resolution else None
    video_handler = VideoStreamHandler(input_video, target_size=target_size)
    width, height, fps, total_frames = video_handler.get_properties()
    print(f"[Video Details] Resolution: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")
    
    # Khởi tạo ghi video output (Điều chỉnh kích thước nếu chạy compare song song)
    if config.save_video:
        if mode == "compare":
            video_handler.width = 1920
            video_handler.height = 540
        print(f"[System] Output will be saved to: {config.output_video_path} at resolution {video_handler.width}x{video_handler.height}")
        video_handler.init_writer(config.output_video_path)

    # 3. Khởi tạo mô hình phát hiện xe (YOLOv8 + ByteTrack)
    print("[Model] Loading YOLOv8 model (Vehicle Detector)...")
    vehicle_tracker = VehicleTracker(
        model_path=config.vehicle_model_path,
        conf_threshold=config.vehicle_conf,
        iou_threshold=config.iou
    )
    
    # Khởi tạo bộ dò đèn giao thông vật lý (nếu chạy violation hoặc compare)
    traffic_light_detector = None
    if mode in ["violation", "compare"]:
        print("[Model] Loading YOLOv8 & SVM (Traffic Light Detector)...")
        traffic_light_detector = TrafficLightDetector(
            model_path=config.traffic_light_model_path,
            conf_threshold=config.traffic_light_conf,
            roi=config.traffic_light_roi,
            static_lights=config.static_traffic_lights
        )
        
    # Khởi tạo bộ điều khiển đèn giao thông thích ứng (nếu chạy adaptive hoặc compare)
    adaptive_controller = None
    if mode in ["adaptive", "compare"]:
        print("[Core] Setting up adaptive traffic signal controller...")
        adaptive_controller = AdaptiveLightController(fps=fps)
    
    # Khởi tạo luật phát hiện vi phạm
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
    
    # Khởi tạo exporter ghi log
    exporter = ViolationExporter(
        json_path=config.json_log_path,
        webhook_url=config.webhook_url,
        root_dir=config.storage_root_dir,
        save_crops=False,
        save_scene_frame=config.save_scene_frame,
    )
    
    # Khởi tạo clip extractor (điều chỉnh kích thước video clip)
    clip_extractor = None
    if config.save_clips:
        clip_size = (1920, 540) if mode == "compare" else (width, height)
        clip_extractor = ViolationClipExtractor(
            fps=fps,
            clip_before_sec=config.clip_before_sec,
            clip_after_sec=config.clip_after_sec,
            root_dir=config.storage_root_dir,
            frame_size=clip_size,
        )
    
    print("=" * 60)
    print("  PIPELINE READY — PROCESSING FRAMES")
    print("=" * 60)
    
    win_name = f"Adaptive Traffic System - Mode: {mode.upper()}"
    if show_preview:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if mode == "compare":
            cv2.resizeWindow(win_name, 1280, 360)  # Tỷ lệ 16:3 (1920x540)
        else:
            cv2.resizeWindow(win_name, 960, 540)
        
    pipeline_start_time = time.time()
    frame_idx = 0
    try:
        for frame in video_handler.frame_generator():
            frame_idx += 1
            
            # 1. Bám vết xe (Dùng chung cho cả 2 nhánh)
            tracked_vehicles = vehicle_tracker.track(frame)
            
            # Mở rộng bounding box lên trên cho xe máy để lấy cả người lái
            # Thực hiện ở đây để ảnh crop lưu trữ và vẽ UI đều nhận box đã mở rộng
            for v in tracked_vehicles:
                if v["class_name"] == "motorcycle":
                    x1, y1, x2, y2 = v["box"]
                    box_h = y2 - y1
                    new_y1 = max(int(y1 - box_h * 0.6), 0)
                    v["box"] = [x1, new_y1, x2, y2]
            
            # ---------------------------------------------
            # NHÁNH A: PHÁT HIỆN VƯỢT ĐÈN ĐỎ VẬT LÝ (VIOLATION)
            # ---------------------------------------------
            frame_violation = None
            if mode in ["violation", "compare"]:
                # Dò tìm và phân loại đèn giao thông thực tế
                lights = traffic_light_detector.detect_and_classify(frame.copy())
                physical_light_state = traffic_light_detector.get_global_traffic_light_state(lights)
                
                # Phát hiện vi phạm vượt đèn đỏ
                new_violations, cancelled_violations = violation_detector.process_frame(
                    tracked_vehicles=tracked_vehicles,
                    traffic_light_state=physical_light_state,
                    frame_idx=frame_idx
                )
                
                # Xuất vi phạm
                for violation in new_violations:
                    print(f"[VIOLATION] Vehicle ID #{violation['vehicle_id']} "
                          f"({violation['vehicle_type'].upper()}) at Frame {frame_idx}")
                    exporter.export_event(violation, frame=frame)
                    if clip_extractor:
                        clip_extractor.trigger_violation(violation)
     
                # Hủy vi phạm nếu rẽ phải
                for vehicle_id in cancelled_violations:
                    print(f"[CANCEL] Vehicle ID #{vehicle_id} entered right-turn zone, removing previous violation")
                    exporter.remove_violation(vehicle_id)
                    if clip_extractor:
                        clip_extractor.cancel_clip(vehicle_id)
                
                # Vẽ ảnh nhánh Violation
                frame_violation = visualizer.draw_scene(
                    frame=frame,
                    tracked_vehicles=tracked_vehicles,
                    traffic_lights=lights,
                    global_light_state=physical_light_state,
                    violated_ids=list(violation_detector.violated_vehicles),
                    active_violations=new_violations,
                    mode="violation"
                )
                
            # ---------------------------------------------
            # NHÁNH B: ĐÈN GIAO THÔNG THÍCH ỨNG DỰA TRÊN SVM (ADAPTIVE)
            # ---------------------------------------------
            frame_adaptive = None
            if mode in ["adaptive", "compare"]:
                # Tính đặc trưng mật độ trong Queue ROI
                vehicles_in_roi = []
                for obj in tracked_vehicles:
                    cx, cy = obj["center"]
                    if is_inside_roi((cx, cy)):
                        vehicles_in_roi.append(obj)
                
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
                        cars += 1
                    
                    history = vehicle_tracker.track_history.get(v_id, [])
                    speed = calculate_speed(history, frames_back=5)
                    if speed != 999.0:
                        speeds.append(speed)
                        if speed < 1.2:
                            stopped_vehicles += 1
                
                avg_speed = np.mean(speeds) if speeds else 15.0
                pcu_load = motorcycles * 0.3 + cars * 1.0
                
                # Dự đoán mật độ và cập nhật bộ điều khiển đèn giao thông thích ứng
                control_state = adaptive_controller.update(
                    motorcycle_count=motorcycles,
                    car_count=cars,
                    stopped_vehicles=stopped_vehicles,
                    pcu_load=pcu_load,
                    average_speed=avg_speed
                )
                
                virtual_light_state = control_state["state"]
                countdown = control_state["time_remaining"]
                density_class = control_state["density_class"]
                density_label = control_state["density_label"]
                prolong_next_green = control_state["prolong_next_green"]
                
                # Vẽ ảnh nhánh Adaptive (Không hiện vi phạm ảo)
                frame_adaptive = visualizer.draw_scene(
                    frame=frame,
                    tracked_vehicles=tracked_vehicles,
                    traffic_lights=[],
                    global_light_state=virtual_light_state,
                    violated_ids=[],
                    active_violations=[],
                    mode="adaptive",
                    density_label=density_label,
                    density_class=density_class,
                    countdown=countdown,
                    prolong_next_green=prolong_next_green,
                    pcu_load=pcu_load,
                    stopped_count=stopped_vehicles
                )
            
            # ---------------------------------------------
            # TỔNG HỢP FRAME ĐẦU RA (COMPOSITING)
            # ---------------------------------------------
            if mode == "violation":
                annotated_frame = frame_violation
            elif mode == "adaptive":
                annotated_frame = frame_adaptive
            elif mode == "compare":
                # Resize cả 2 khung hình gốc (1280x720) về 960x540
                v_res = cv2.resize(frame_violation, (960, 540))
                a_res = cv2.resize(frame_adaptive, (960, 540))
                
                # Ghép song song ngang (Horizontal Concat) thành kích thước 1920x540
                annotated_frame = cv2.hconcat([v_res, a_res])
            
            # Ghi frame kết quả ra file
            if config.save_video:
                video_handler.write_frame(annotated_frame)
 
            if clip_extractor:
                clip_extractor.push_frame(annotated_frame)
                
            exporter.fps_tracker.tick()
 
            # Preview trực tiếp
            if show_preview:
                elapsed_time = time.time() - pipeline_start_time
                expected_video_time = frame_idx / fps
                
                if elapsed_time < expected_video_time:
                    time.sleep(expected_video_time - elapsed_time)
                    cv2.imshow(win_name, annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[System] Execution aborted by user.")
                        break
                else:
                    if (elapsed_time - expected_video_time) < 0.15 or frame_idx % 3 == 0:
                        cv2.imshow(win_name, annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("[System] Execution aborted by user.")
                            break
                    
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                adaptive_info = ""
                if mode == "compare":
                    adaptive_info = f" | [Adaptive] Light: {virtual_light_state} ({countdown:.1f}s) | Density: {density_label}"
                print(f"[Progress] {frame_idx}/{total_frames} ({progress:.1f}%) | "
                      f"FPS: {exporter.fps_tracker.get_fps():.1f} | "
                      f"Violations: {len(violation_detector.violated_vehicles)}{adaptive_info}")
                      
    except KeyboardInterrupt:
        print("[System] Execution interrupted by manual keyboard signal.")
    finally:
        # Giải phóng tài nguyên
        video_handler.release_all()
        if show_preview:
            cv2.destroyAllWindows()
        
        if clip_extractor:
            clip_extractor.finalize()
            
        if mode in ["violation", "compare"]:
            exporter.print_summary(
                total_actual=total_actual,
                true_positives=true_positives,
                false_positives=false_positives,
            )
 
        print("=" * 60)
        print(f"  Frames processed : {frame_idx}")
        if mode in ["violation", "compare"]:
            print(f"  Violations logged: {exporter.get_total_violations()}")
            print(f"  JSON  → {config.json_log_path}")
        print(f"  Video → {config.output_video_path}")
        print("=" * 60)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Violation Detection & Adaptive Signal Control")
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--mode", type=str, default="violation", choices=["violation", "compare"],
                        help="Chế độ chạy: 'violation' (chỉ bắt vượt đèn đỏ thực tế), 'compare' (so sánh song song)")
    parser.add_argument("--preview", action="store_true")
    # Tham số đối chiếu thủ công
    parser.add_argument("--total-actual", type=int, default=None,
                        help="Tổng số vi phạm thực tế")
    parser.add_argument("--true-positives", type=int, default=None,
                        help="Số vi phạm hệ thống bắt đúng")
    parser.add_argument("--false-positives", type=int, default=None,
                        help="Số vi phạm hệ thống báo nhầm")
    args = parser.parse_args()
    
    run_pipeline(
        video_path=args.video,
        mode=args.mode,
        show_preview=args.preview,
        total_actual=args.total_actual,
        true_positives=args.true_positives,
        false_positives=args.false_positives,
    )
