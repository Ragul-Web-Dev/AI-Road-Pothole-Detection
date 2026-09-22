"""
Baseline YOLO Training Script - Phase 1 Research Pipeline
Uses transfer learning with a lightweight YOLO nano model for baseline pothole detection.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def train_model(data_yaml="data.yaml", model_name="yolov8n.pt", epochs=20, imgsz=640, batch=16, project="runs", name="pothole_baseline"):
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"[ERROR] Data configuration file '{data_yaml}' not found.")
        print("Ensure data.yaml exists in the working directory.")
        return

    print("=" * 60)
    print("PHASE 1 BASELINE POTHOLE DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"Base Model:       {model_name}")
    print(f"Data Config:      {data_yaml}")
    print(f"Target Epochs:    {epochs}")
    print(f"Image Size:       {imgsz}")
    print(f"Batch Size:       {batch}")
    print(f"Save Directory:   {project}/{name}")
    print("=" * 60)
    
    # Load pretrained model (Transfer Learning)
    model = YOLO(model_name)
    
    # Execute training
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        exist_ok=True,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print(f"Trained weights saved to: {project}/{name}/weights/best.pt")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline YOLO pothole detection model.")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLO weights (e.g. yolov8n.pt, yolo11n.pt)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--project", type=str, default="runs", help="Project output directory")
    parser.add_argument("--name", type=str, default="pothole_baseline", help="Experiment name")
    
    args = parser.parse_args()
    
    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name
    )
