import os
import cv2
import csv
import yaml
import numpy as np
from pathlib import Path

def extract_features_from_crop(crop):
    """
    Extracts 27-dimensional spatial and global color features from a traffic light crop.
    - Global features (9): mean RGB, mean HSV, std HSV
    - Spatial features (18): mean RGB/HSV for Top/Left, Middle, Bottom/Right segments
    """
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return None
        
    h, w = crop.shape[:2]
    
    # 1. Global features
    mean_b, mean_g, mean_r = cv2.mean(crop)[:3]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mean_hsv, std_hsv = cv2.meanStdDev(hsv)
    
    mean_h = mean_hsv[0][0]
    mean_s = mean_hsv[1][0]
    mean_v = mean_hsv[2][0]
    
    std_h = std_hsv[0][0]
    std_s = std_hsv[1][0]
    std_v = std_hsv[2][0]
    
    global_feats = [mean_r, mean_g, mean_b, mean_h, mean_s, mean_v, std_h, std_s, std_v]
    
    # 2. Spatial features (Split into 3 segments based on orientation)
    if h >= w:
        # Vertical split (standard)
        h3 = h // 3
        part1 = crop[:h3, :]
        part2 = crop[h3:2*h3, :]
        part3 = crop[2*h3:, :]
    else:
        # Horizontal split
        w3 = w // 3
        part1 = crop[:, :w3]
        part2 = crop[:, w3:2*w3]
        part3 = crop[:, 2*w3:]
        
    def get_part_means(part):
        if part.size == 0:
            return [0.0] * 6
        p_mean_b, p_mean_g, p_mean_r = cv2.mean(part)[:3]
        p_hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        p_mean_h, p_mean_s, p_mean_v = cv2.mean(p_hsv)[:3]
        return [p_mean_r, p_mean_g, p_mean_b, p_mean_h, p_mean_s, p_mean_v]
        
    part1_feats = get_part_means(part1)
    part2_feats = get_part_means(part2)
    part3_feats = get_part_means(part3)
    
    return global_feats + part1_feats + part2_feats + part3_feats

def process_subset(images_dir, labels_dir, output_csv):
    """
    Processes all bounding boxes in a subset and saves spatial features to CSV.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    
    print(f"\nProcessing directory: {images_dir}")
    print(f"Labels directory: {labels_dir}")
    
    if not labels_dir.exists():
        print(f"[Error] Labels directory does not exist: {labels_dir}")
        return
        
    label_files = list(labels_dir.glob("*.txt"))
    print(f"Found {len(label_files)} label files.")
    
    count_dict = {}
    total_crops = 0
    
    # Generate header
    header = ["label"]
    # Global headers
    header += ["g_mean_r", "g_mean_g", "g_mean_b", "g_mean_h", "g_mean_s", "g_mean_v", "g_std_h", "g_std_s", "g_std_v"]
    # Part headers
    for p in [1, 2, 3]:
        header += [f"p{p}_mean_r", f"p{p}_mean_g", f"p{p}_mean_b", f"p{p}_mean_h", f"p{p}_mean_s", f"p{p}_mean_v"]
        
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for lf in label_files:
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                temp = images_dir / f"{lf.stem}{ext}"
                if temp.exists():
                    img_file = temp
                    break
                    
            if img_file is None:
                continue
                
            image = cv2.imread(str(img_file))
            if image is None:
                continue
                
            h_img, w_img = image.shape[:2]
            
            with open(lf, 'r') as label_f:
                lines = label_f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                x_c, y_c, w_box, h_box = map(float, parts[1:5])
                
                x1 = int((x_c - w_box / 2.0) * w_img)
                y1 = int((y_c - h_box / 2.0) * h_img)
                x2 = int((x_c + w_box / 2.0) * w_img)
                y2 = int((y_c + h_box / 2.0) * h_img)
                
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_img, x2)
                y2 = min(h_img, y2)
                
                crop = image[y1:y2, x1:x2]
                features = extract_features_from_crop(crop)
                
                if features is not None:
                    writer.writerow([class_id] + features)
                    count_dict[class_id] = count_dict.get(class_id, 0) + 1
                    total_crops += 1
                    
    print(f"[Done] Finished writing features to: {output_csv}")
    print(f"Total crop samples extracted: {total_crops}")
    for cid, cnt in sorted(count_dict.items()):
        print(f"  * Class {cid}: {cnt} samples")

def main():
    yaml_path = Path("data/traffic_light/dataset.yaml")
    if not yaml_path.exists():
        print(f"[Error] dataset.yaml not found at: {yaml_path}")
        return
        
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    dataset_path = Path(config.get('path', '.'))
    
    # Process train
    train_rel = config.get('train', '')
    train_img_dir = dataset_path / train_rel
    train_lbl_dir = dataset_path / train_rel.replace('images', 'labels')
    train_csv = "data/traffic_light/train_features.csv"
    process_subset(train_img_dir, train_lbl_dir, train_csv)
    
    # Process val
    val_rel = config.get('val', '')
    val_img_dir = dataset_path / val_rel
    val_lbl_dir = dataset_path / val_rel.replace('images', 'labels')
    val_csv = "data/traffic_light/val_features.csv"
    process_subset(val_img_dir, val_lbl_dir, val_csv)

if __name__ == "__main__":
    main()
