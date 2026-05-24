#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLOv8 Traffic Light Fine-Tuning Script
This script provides a complete and professional interface to train/fine-tune
a YOLOv8 model for traffic light detection (RED, GREEN, YELLOW states).

Usage:
    python train.py --data data/traffic_light/dataset.yaml --epochs 50 --batch 16 --model yolov8s.pt
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

def print_banner():
    print("=" * 70)
    print("        [TRAIN] YOLOv8 TRAFFIC LIGHT FINE-TUNING UTILITY        ")
    print("=" * 70)

def print_dataset_instructions():
    instructions = """
[INFO] YOLO DATASET FORMAT INSTRUCTIONS:
----------------------------------------------------------------------
Your dataset folder must be structured as follows:

  data/traffic_light/
  |-- dataset.yaml                 <-- Dataset configuration file
  |-- train/
  |   |-- images/                  <-- Training images (.jpg, .png)
  |   |-- labels/                  <-- Training labels (.txt)
  |-- val/
  |   |-- images/                  <-- Validation images (.jpg, .png)
  |   |-- labels/                  <-- Validation labels (.txt)

Label file format (.txt):
- Create a text file with the EXACT same name as each image (e.g., frame_001.jpg -> frame_001.txt).
- Each line in the file represents one object in the image:
  <class_id> <x_center> <y_center> <width> <height>

  * class_id  : 0 (red), 1 (green), 2 (yellow)
  * x_center  : Normalized X coordinate of the bounding box center (0.0 to 1.0)
  * y_center  : Normalized Y coordinate of the bounding box center (0.0 to 1.0)
  * width     : Normalized width of the bounding box (0.0 to 1.0)
  * height    : Normalized height of the bounding box (0.0 to 1.0)

Example label line for a RED light in the center:
  0 0.512 0.345 0.024 0.065
----------------------------------------------------------------------
"""
    print(instructions)

def create_sample_dataset_yaml(yaml_path: str):
    """
    Creates a sample dataset.yaml file if it doesn't exist.
    """
    yaml_path = Path(yaml_path)
    if yaml_path.exists() and yaml_path.stat().st_size > 0:
        return

    print(f"[Setup] Generating sample configuration file at: {yaml_path}")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define absolute or relative paths
    content = {
        'path': str(yaml_path.parent.absolute().as_posix()),  # dataset root dir
        'train': 'train/images',                              # train images (relative to path)
        'val': 'val/images',                                  # val images (relative to path)
        'names': {
            0: 'red_light',
            1: 'green_light',
            2: 'yellow_light'
        }
    }
    
    try:
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(content, f, default_flow_style=False, sort_keys=False)
        print(f"[Done] Created dummy dataset.yaml template. Feel free to edit it!")
        
        # Create empty placeholder folders to help the user get started
        (yaml_path.parent / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (yaml_path.parent / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (yaml_path.parent / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (yaml_path.parent / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        print(f"[Done] Created empty folders under: {yaml_path.parent}")
    except Exception as e:
        print(f"[Error] Failed to create template dataset.yaml: {e}")

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on Custom Traffic Light Dataset")
    parser.add_argument(
        "--data", 
        type=str, 
        default="data/traffic_light/dataset.yaml", 
        help="Path to dataset.yaml file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8s.pt", 
        help="Pretrained YOLOv8 model to start with (yolov8n.pt, yolov8s.pt, yolov8m.pt)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch", 
        type=int, 
        default=16, 
        help="Batch size (16 is recommended for 6GB/8GB VRAM mid-range GPUs)"
    )
    parser.add_argument(
        "--imgsz", 
        type=int, 
        default=640, 
        help="Input image resolution size (640 is standard)"
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="", 
        help="Hardware device to run training on ('0', '1' for GPUs, 'cpu', or leave empty for auto-detect)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4, 
        help="Number of worker threads for data loading"
    )
    parser.add_argument(
        "--project", 
        type=str, 
        default="runs/traffic_light", 
        help="Project directory name to save logs and weights"
    )
    parser.add_argument(
        "--name", 
        type=str, 
        default="train", 
        help="Training run name"
    )
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Only display formatting guidelines and exit"
    )
    
    args = parser.parse_args()
    
    # 1. Print guidelines
    print_dataset_instructions()
    if args.info:
        sys.exit(0)
        
    # 2. Check or create dataset config
    data_path = Path(args.data)
    if not data_path.exists() or data_path.stat().st_size == 0:
        print(f"[Warning] Specified config '{args.data}' does not exist or is empty.")
        create_sample_dataset_yaml(args.data)
        print("\n[Notice] Please put your real traffic light images and labels into the created folders,")
        print("         then run this script again to begin the actual training process.")
        print("=" * 70)
        sys.exit(0)

    # 3. Check for training files (ensure folders are not completely empty)
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        dataset_path = Path(data_config.get('path', '.'))
        train_img_dir = dataset_path / data_config.get('train', '')
        
        if not train_img_dir.exists() or len(list(train_img_dir.glob('*'))) == 0:
            print(f"[Warning] Training images directory '{train_img_dir}' is empty or does not exist!")
            print("Please place your training images inside before running training.")
            print("=" * 70)
            sys.exit(0)
    except Exception as e:
        print(f"[Warning] Could not verify training images: {e}. Attempting to proceed anyway...")

    # 4. Device auto-detection
    if not args.device:
        if torch.cuda.is_available():
            device = "0"
            device_name = f"GPU: {torch.cuda.get_device_name(0)}"
        else:
            device = "cpu"
            device_name = "CPU (Note: Training on CPU will be extremely slow!)"
    else:
        device = args.device
        device_name = device
        
    print(f"[Device] Running training on: {device_name}")
    print(f"[Hyperparams] Model: {args.model} | Epochs: {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz} | lr0: {args.lr0}")
    print(f"[Output] Saving runs to: {args.project}/{args.name}")
    print("-" * 70)
    print("[Wait] Loading YOLO model & starting fine-tuning...")

    try:
        # Load the model
        model = YOLO(args.model)
        
        # Start training with mid-range GPU optimizations
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr0,
            device=device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            pretrained=True,
            val=True,           # Perform validation after each epoch
            save=True,          # Save model checkpoints
            plots=True,         # Generate training metric plots
            amp=True,           # ENABLE MIXED PRECISION (AMP): Reduces VRAM usage, speeds up mid-range GPUs
            patience=15,        # EARLY STOPPING: Stops training if validation doesn't improve for 15 epochs
        )
        
        print("\n" + "=" * 70)
        print("          SUCCESS: FINE-TUNING COMPLETED SUCCESSFULLY          ")
        print("=" * 70)
        best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
        print(f"[Done] Best model weights saved to: {best_model_path.absolute()}")
        print(f"[Done] To use this model in your pipeline, copy it to your models folder:")
        print(f"       copy {best_model_path} c:\\Project_uni\\ML_project\\src\\models\\traffic_light_best.pt")
        print(f"       And update config.yaml: traffic_light_model: \"src/models/traffic_light_best.pt\"")
        
        # Draw elegant evaluation metrics table if available
        print("\n" + "-" * 70)
        print("📊 FINAL VALIDATION PERFORMANCE METRICS SUMMARY:")
        print("-" * 70)
        
        # Safely extract metrics from training results
        if results and hasattr(results, 'results_dict') and results.results_dict:
            for k, v in results.results_dict.items():
                clean_name = k.replace("metrics/", "").replace("(B)", "").strip()
                print(f"  * {clean_name:<25}: {v:.5f}")
        else:
            print("  Metrics (Loss, Precision, Recall, mAP) are logged successfully.")
            print(f"  You can view them in the plots and CSV log at: {Path(args.project) / args.name}")
        print("-" * 70)
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n[Warning] Training interrupted by user.")
    except Exception as e:
        print(f"\n[Error] Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
