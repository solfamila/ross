#!/usr/bin/env python3
"""
Training script for YOLOv10 Trading Window Detection

Usage:
    python train.py                    # Train with default settings
    python train.py --epochs 100       # Train for 100 epochs
    python train.py --resume           # Resume from last checkpoint
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv10 for trading windows")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--model", default="yolov10n.pt", help="Base model")
    args = parser.parse_args()

    # Check for images
    images_dir = Path("images")
    labels_dir = Path("labels")
    
    if not images_dir.exists():
        images_dir.mkdir()
        print(f"Created {images_dir}/")
        print("Please copy your screenshots to the 'images' folder and run annotate.py first.")
        return
    
    images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    labels = list(labels_dir.glob("*.txt"))
    
    print(f"Found {len(images)} images, {len(labels)} annotations")
    
    if len(labels) == 0:
        print("\nNo annotations found! Run the annotation tool first:")
        print("  python annotate.py images")
        return
    
    if len(labels) < len(images):
        print(f"\nWarning: Only {len(labels)}/{len(images)} images have annotations.")
        print("Run 'python annotate.py images' to annotate remaining images.")
    
    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.run(["pip", "install", "ultralytics", "-q"])
        from ultralytics import YOLO
    
    # Load model
    print(f"\nLoading base model: {args.model}")
    model = YOLO(args.model)
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    
    results = model.train(
        data="trading_windows.yaml",
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=20,  # Early stopping
        save=True,
        plots=True,
        resume=args.resume,
        project="runs",
        name="trading_windows",
        exist_ok=True,
    )
    
    # Export best model to ONNX
    print("\nExporting best model to ONNX...")
    best_pt = Path("runs/trading_windows/weights/best.pt")
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        best_model.export(format="onnx", opset=17, simplify=True, imgsz=args.imgsz)
        
        # Copy to cpp_realtime_ocr/models
        onnx_path = best_pt.with_suffix(".onnx")
        dest = Path("../cpp_realtime_ocr/models/trading_windows.onnx")
        if onnx_path.exists():
            shutil.copy(onnx_path, dest)
            print(f"Exported: {dest}")
            print("\nTo use in trading_monitor:")
            print(f"  .\\trading_monitor.exe --detect-onnx ..\\models\\trading_windows.onnx")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

