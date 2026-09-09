"""
AquaDetect-AI-Sonar-GPS : YOLOv8 Training Script
------------------------------------------------
Trains Ultralytics YOLOv8 on Forward-Looking Sonar (FLS) Marine Debris dataset
to detect marine debris (plastic, metal, chain, tire, hook, bottles, cans, etc.).
Saves the resulting best weights to `models/best.pt`.
"""

import os
import sys
import glob
import shutil
import argparse
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Sonar Marine Debris Dataset")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data.yaml"),
        help="Path to data.yaml dataset configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pretrained YOLO model checkpoint or YAML config (e.g. yolov8n.pt, yolov8s.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Input image size (default: 320)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on ('0', 'cpu', or None for auto-detection)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Target folder to save best.pt (default: models/)"
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default="best.pt",
        help="Filename for best model weights (default: best.pt)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Ultralytics project directory"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="sonar_debris_yolov8",
        help="Ultralytics experiment run name"
    )
    return parser.parse_args()

def train_sonar_yolo(args):
    print("=" * 65)
    print(" BLUE GUARD AI : YOLOv8 Model Training ")
    print("=" * 65)

    # 1. Validate Dataset Configuration
    if not os.path.exists(args.data):
        alt_data = os.path.join(os.path.dirname(__file__), "data", "processed", "data.yaml")
        if os.path.exists(alt_data):
            args.data = alt_data
        else:
            print(f"[!] Error: data.yaml not found at '{args.data}'. Please run data_preprocess.py first.")
            sys.exit(1)

    print(f"[*] Dataset Config: {os.path.abspath(args.data)}")
    
    # 2. Configure Device
    if args.device is None:
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[*] Training Device: {device} (CUDA Available: {torch.cuda.is_available()})")

    # 3. Ensure Target Models Directory Exists
    os.makedirs(args.save_dir, exist_ok=True)

    # 4. Initialize YOLOv8 Model
    print(f"[*] Initializing YOLOv8 model: {args.model}...")
    model = YOLO(args.model)

    # 5. Train Model
    print(f"\n[*] Starting training for {args.epochs} epochs (Batch Size: {args.batch}, ImgSz: {args.imgsz})...")
    
    train_results = model.train(
        data=os.path.abspath(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        plots=True
    )

    # 6. Locate Best Trained Weights
    trained_best_pt = None
    trained_last_pt = None

    if hasattr(model, "trainer") and model.trainer is not None:
        if hasattr(model.trainer, "best") and model.trainer.best and os.path.exists(str(model.trainer.best)):
            trained_best_pt = str(model.trainer.best)
        if hasattr(model.trainer, "last") and model.trainer.last and os.path.exists(str(model.trainer.last)):
            trained_last_pt = str(model.trainer.last)

    if trained_best_pt is None:
        # Fallback to search inside project directory
        candidates = glob.glob(os.path.join(args.project, "**", "weights", "best.pt"), recursive=True)
        if candidates:
            trained_best_pt = candidates[-1]

    if trained_last_pt is None:
        candidates = glob.glob(os.path.join(args.project, "**", "weights", "last.pt"), recursive=True)
        if candidates:
            trained_last_pt = candidates[-1]

    target_best_pt = os.path.join(args.save_dir, args.save_name)
    target_last_pt = os.path.join(args.save_dir, "last.pt")

    if trained_best_pt and os.path.exists(trained_best_pt):
        shutil.copy2(trained_best_pt, target_best_pt)
        print(f"\n[+] Successfully saved best model to: {os.path.abspath(target_best_pt)}")
    elif trained_last_pt and os.path.exists(trained_last_pt):
        shutil.copy2(trained_last_pt, target_best_pt)
        print(f"\n[+] Saved last model checkpoint as best to: {os.path.abspath(target_best_pt)}")
    else:
        print(f"[!] Warning: Could not locate weights in '{args.project}'")

    if trained_last_pt and os.path.exists(trained_last_pt):
        shutil.copy2(trained_last_pt, target_last_pt)
        print(f"[+] Successfully saved last checkpoint to: {os.path.abspath(target_last_pt)}")

    # 7. Evaluate on Validation Set
    print("\n" + "=" * 65)
    print(" Validation Metrics Evaluation ")
    print("=" * 65)
    
    try:
        val_model = YOLO(target_best_pt if os.path.exists(target_best_pt) else trained_best_pt)
        metrics = val_model.val(data=os.path.abspath(args.data), split="val", device=device, verbose=False)
        
        print(f"mAP50:        {metrics.box.map50:.4f}")
        print(f"mAP50-95:     {metrics.box.map:.4f}")
        print(f"Precision:    {metrics.box.mp:.4f}")
        print(f"Recall:       {metrics.box.mr:.4f}")
    except Exception as e:
        print(f"[!] Note: Validation summary evaluation encountered: {e}")

    print("\n" + "=" * 65)
    print(f" Training Complete! Best model stored at: {os.path.abspath(target_best_pt)}")
    print("=" * 65)
    return target_best_pt

if __name__ == "__main__":
    args = parse_args()
    train_sonar_yolo(args)
