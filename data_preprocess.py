"""
AquaDetect-AI-Sonar-GPS Data Preprocessing Pipeline
--------------------------------------------------
1. Converts Forward-Looking Sonar (FLS) segmentation mask annotations into YOLO bounding box format.
2. Performs 80-10-10 train, validation, and test split.
3. Generates realistic 4-satellite GPS dummy data (latitude, longitude, depth, direction) for each sonar frame.
4. Exports dataset structure, data.yaml configuration, and gps_data.csv.
"""

import os
import glob
import shutil
import random
import cv2
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta

# Configuration & Constants
SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# 11 Object Classes defined in FLS Water Tank Marine Debris dataset (0 = Background)
CLASS_NAMES = [
    "bottle",          # 0 (mask value 1)
    "can",             # 1 (mask value 2)
    "chain",           # 2 (mask value 3)
    "drink-carton",    # 3 (mask value 4)
    "hook",            # 4 (mask value 5)
    "propeller",       # 5 (mask value 6)
    "shampoo-bottle",  # 6 (mask value 7)
    "standing-bottle", # 7 (mask value 8)
    "tire",            # 8 (mask value 9)
    "valve",           # 9 (mask value 10)
    "wall"             # 10 (mask value 11)
]

def extract_yolo_bboxes_from_mask(mask_path, min_area=10):
    """
    Extracts YOLO normalized bounding boxes from a multi-class semantic segmentation mask.
    Returns:
        List of tuples: (class_id, x_center, y_center, width, height, class_name)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return []
    
    h, w = mask.shape[:2]
    bboxes = []
    
    # Pixel values 1 to 11 correspond to CLASS_NAMES indices 0 to 10
    for mask_val in range(1, len(CLASS_NAMES) + 1):
        class_id = mask_val - 1
        class_name = CLASS_NAMES[class_id]
        class_mask = (mask == mask_val).astype(np.uint8)
        
        if np.count_nonzero(class_mask) == 0:
            continue
            
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # Convert to YOLO normalized format [0, 1]
            x_center = (bx + bw / 2.0) / float(w)
            y_center = (by + bh / 2.0) / float(h)
            norm_w = bw / float(w)
            norm_h = bh / float(h)
            
            # Clip bounds to [0, 1] for YOLO stability
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            bboxes.append((class_id, x_center, y_center, norm_w, norm_h, class_name))
            
    return bboxes

def generate_satellite_gps_data(total_samples, base_lat=26.5, base_lon=79.9, seed=42):
    """
    Simulates a 4-satellite marine GPS survey trajectory with realistic
    latitude, longitude, depth (5-30 meters), direction/heading (degrees), and satellite metadata.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    gps_records = []
    start_time = datetime(2026, 9, 6, 8, 30, 0)
    
    for i in range(total_samples):
        # lat = 26.5 + random/100, lon = 79.9 + random/100
        curr_lat = round(base_lat + random.uniform(0.0, 1.0) / 100.0, 6)
        curr_lon = round(base_lon + random.uniform(0.0, 1.0) / 100.0, 6)
        curr_depth = round(random.uniform(5.0, 30.0), 2)
        curr_direction = round(random.uniform(0.0, 360.0), 1)
        
        timestamp = start_time + timedelta(seconds=i * 2)
        
        gps_records.append({
            "lat": round(curr_lat, 7),
            "lon": round(curr_lon, 7),
            "depth": round(curr_depth, 2),
            "direction": round(curr_direction, 2),
            "timestamp": timestamp.isoformat(),
            "satellites_locked": 4,
            "satellite_ids": "SAT_GPS_12,SAT_GPS_18,SAT_GLONASS_05,SAT_GALILEO_24",
            "hdop": round(np.random.uniform(0.7, 1.2), 2)  # Horizontal Dilution of Precision
        })
        
    return gps_records

def run_preprocessing(
    workspace_dir=r"c:\Users\sonam\Desktop\AquaDetect-AI-Sonar-GPS",
    output_dir=None
):
    print("=" * 65)
    print(" AquaDetect-AI-Sonar-GPS : Dataset Preprocessing & YOLO Formatting ")
    print("=" * 65)
    
    if output_dir is None:
        output_dir = os.path.join(workspace_dir, "data", "processed")
        
    images_src_dir = os.path.join(workspace_dir, "data", "sonar_images", "watertank-segmentation", "Images")
    masks_src_dir = os.path.join(workspace_dir, "data", "sonar_images", "watertank-segmentation", "Masks")
    
    if not os.path.exists(images_src_dir) or not os.path.exists(masks_src_dir):
        raise FileNotFoundError(f"Source image/mask directories not found at:\n{images_src_dir}\n{masks_src_dir}")
        
    img_files = sorted(glob.glob(os.path.join(images_src_dir, "*.png")))
    total_images = len(img_files)
    print(f"[*] Total Sonar Images Found: {total_images}")
    
    # Create YOLO directory hierarchy
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
        
    # Deterministic Shuffle for Split
    random.seed(SEED)
    indices = list(range(total_images))
    random.shuffle(indices)
    
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)
    test_count = total_images - train_count - val_count
    
    train_indices = set(indices[:train_count])
    val_indices = set(indices[train_count:train_count + val_count])
    test_indices = set(indices[train_count + val_count:])
    
    print(f"[*] Train / Val / Test Split: {train_count} ({TRAIN_RATIO*100:.0f}%) / {val_count} ({VAL_RATIO*100:.0f}%) / {test_count} ({TEST_RATIO*100:.0f}%)")
    
    # Generate 4-satellite dummy GPS data for all images
    print("[*] Generating 4-Satellite GPS Dummy Data...")
    gps_data_list = generate_satellite_gps_data(total_images, seed=SEED)
    
    processed_records = []
    class_stats = {name: 0 for name in CLASS_NAMES}
    
    print("[*] Processing images, generating YOLO labels & linking GPS metadata...")
    for idx, img_path in enumerate(img_files):
        img_name = os.path.basename(img_path)
        base_name, _ = os.path.splitext(img_name)
        mask_path = os.path.join(masks_src_dir, img_name)
        
        # Determine split
        if idx in train_indices:
            split = "train"
        elif idx in val_indices:
            split = "val"
        else:
            split = "test"
            
        # Destination Paths
        dest_img_path = os.path.join(output_dir, "images", split, img_name)
        dest_label_path = os.path.join(output_dir, "labels", split, f"{base_name}.txt")
        
        # Copy image
        shutil.copy2(img_path, dest_img_path)
        
        # Extract bounding boxes
        bboxes = extract_yolo_bboxes_from_mask(mask_path)
        detected_classes = []
        
        # Write YOLO label text file
        with open(dest_label_path, "w") as label_file:
            for bbox in bboxes:
                cls_id, xc, yc, nw, nh, cls_name = bbox
                label_file.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")
                detected_classes.append(cls_name)
                class_stats[cls_name] += 1
                
        # Link GPS data record
        gps_rec = gps_data_list[idx]
        record = {
            "image_id": base_name,
            "filename": img_name,
            "split": split,
            "image_path": os.path.relpath(dest_img_path, workspace_dir),
            "label_path": os.path.relpath(dest_label_path, workspace_dir),
            "lat": gps_rec["lat"],
            "lon": gps_rec["lon"],
            "depth": gps_rec["depth"],
            "direction": gps_rec["direction"],
            "satellites_locked": gps_rec["satellites_locked"],
            "satellite_ids": gps_rec["satellite_ids"],
            "hdop": gps_rec["hdop"],
            "timestamp": gps_rec["timestamp"],
            "num_objects": len(bboxes),
            "detected_classes": ";".join(sorted(list(set(detected_classes)))) if detected_classes else "none"
        }
        processed_records.append(record)
        
        if (idx + 1) % 300 == 0 or (idx + 1) == total_images:
            print(f"    - Processed {idx + 1}/{total_images} frames...")
            
    # Save GPS CSV files
    df_gps = pd.DataFrame(processed_records)
    
    # Save in root workspace and in data/ directory
    root_csv_path = os.path.join(workspace_dir, "gps_data.csv")
    data_csv_path = os.path.join(workspace_dir, "data", "gps_data.csv")
    df_gps.to_csv(root_csv_path, index=False)
    df_gps.to_csv(data_csv_path, index=False)
    print(f"\n[+] Saved GPS metadata CSV to:")
    print(f"    - {root_csv_path}")
    print(f"    - {data_csv_path}")
    
    # Create YOLO data.yaml configuration
    data_yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    
    data_yaml_path = os.path.join(output_dir, "data.yaml")
    root_yaml_path = os.path.join(workspace_dir, "data.yaml")
    
    with open(data_yaml_path, "w") as yf:
        yaml.dump(data_yaml_content, yf, sort_keys=False)
    with open(root_yaml_path, "w") as yf:
        yaml.dump(data_yaml_content, yf, sort_keys=False)
        
    print(f"[+] Saved YOLO data.yaml config to:")
    print(f"    - {data_yaml_path}")
    print(f"    - {root_yaml_path}")
    
    print("\n" + "=" * 65)
    print(" Preprocessing Complete Summary ")
    print("=" * 65)
    print(f"Total Images Processed: {total_images}")
    print(f"Train Set: {train_count} images")
    print(f"Val Set:   {val_count} images")
    print(f"Test Set:  {test_count} images")
    print("\nClass Object Detections:")
    for cls_name, count in class_stats.items():
        print(f"  {cls_name.ljust(16)}: {count} bboxes")
    print("=" * 65)

if __name__ == "__main__":
    run_preprocessing()
