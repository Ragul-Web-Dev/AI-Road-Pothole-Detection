"""
Baseline YOLO Model Training Script
AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System

Trains Ultralytics YOLO nano model using transfer learning on road hazard dataset.
Supports:
- Automatic CPU / GPU hardware detection
- RAM caching for faster training throughput
- Configurable epochs, batch size, imgsz, patience, and seed
- Safe checkpoint handling (avoids duplicate runs unless --force is specified)
- Synchronizes output weights to runs/detect/pothole_training/ and runs/pothole_baseline/
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def detect_device(preferred_device="auto"):
    """Determines optimal compute device (CUDA GPU or multi-core CPU)."""
    if preferred_device != "auto":
        return preferred_device
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[DEVICE] CUDA GPU detected: {gpu_name}")
        return 0
    else:
        cpu_count = os.cpu_count() or 1
        print(f"[DEVICE] Using CPU with {cpu_count} logical cores (PyTorch multi-threading).")
        return "cpu"

def train_model(
    data_yaml="data.yaml",
    model_name="yolov8n.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    seed=42,
    device="auto",
    project="runs/detect",
    name="pothole_training",
    force=False,
    resume=False,
    cache=True
):
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"[ERROR] Data configuration file '{data_yaml}' not found.")
        print("Ensure data.yaml exists in the working directory.")
        sys.exit(1)

    target_weights = Path(project) / name / "weights" / "best.pt"

    # Avoid duplicate training run if completed valid model already exists
    if target_weights.exists() and target_weights.stat().st_size > 1000000 and not force and not resume:
        print("=" * 65)
        print("EXISTING TRAINED MODEL FOUND")
        print("=" * 65)
        print(f"A valid trained model checkpoint already exists at:")
        print(f"  {target_weights.resolve()} ({target_weights.stat().st_size / (1024*1024):.2f} MB)")
        print("\nTo evaluate this model, run:")
        print(f"  python scripts/validate.py --weights {target_weights}")
        print("\nTo force a new training run from scratch, use:")
        print(f"  python scripts/train.py --force --epochs {epochs}")
        print("=" * 65)
        return str(target_weights)

    selected_device = detect_device(device)

    print("=" * 65)
    print("ROAD HAZARD DETECTION - YOLO TRANSFER LEARNING TRAINING")
    print("=" * 65)
    print(f"Pretrained Model:     {model_name}")
    print(f"Dataset Config:       {data_path.resolve()}")
    print(f"Target Epochs:        {epochs}")
    print(f"Early Stopping:       patience={patience}")
    print(f"Image Resolution:     {imgsz}x{imgsz}")
    print(f"Batch Size:           {batch}")
    print(f"Random Seed:          {seed}")
    print(f"Compute Device:       {selected_device}")
    print(f"RAM Caching:          {cache}")
    print(f"Output Project/Name:  {project}/{name}")
    print("=" * 65)

    # Load pretrained model
    try:
        model = YOLO(model_name)
        print(f"[INFO] Loaded pretrained backbone weights: {model_name}")
    except Exception as e:
        print(f"[ERROR] Failed to load/download pretrained model '{model_name}': {e}")
        print("Please check internet connection or provide local weights.")
        sys.exit(1)

    # Resolve project to absolute path to guarantee exact output structure
    abs_project = str(Path(project).resolve())
    target_weights = Path(project) / name / "weights" / "best.pt"

    # Execute training
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        seed=seed,
        device=selected_device,
        workers=0 if os.name == "nt" else 4,  # Safe workers on Windows
        cache=cache,
        project=abs_project,
        name=name,
        exist_ok=True,
        resume=resume,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 65)

    if target_weights.exists():
        print(f"Best checkpoint successfully saved to: {target_weights.resolve()}")
        # Backward compatibility sync: ensure runs/pothole_baseline/weights/best.pt also points to best.pt
        legacy_path = Path("runs/pothole_baseline/weights")
        legacy_path.mkdir(parents=True, exist_ok=True)
        legacy_best = legacy_path / "best.pt"
        shutil.copy2(target_weights, legacy_best)
        print(f"Synchronized baseline checkpoint to:  {legacy_best.resolve()}")
    else:
        print(f"[WARNING] Expected best.pt at {target_weights} not found. Check training logs.")

    save_dir = Path(project) / name
    print(f"Metrics, plots, and training curves saved in: {save_dir.resolve()}")
    print("=" * 65)
    return str(target_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline YOLO pothole detection model.")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Pretrained YOLO weights (e.g. yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Initial target epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, 0)")
    parser.add_argument("--project", type=str, default="runs/detect", help="Output project directory")
    parser.add_argument("--name", type=str, default="pothole_training", help="Experiment name")
    parser.add_argument("--force", action="store_true", help="Force new training run if best.pt exists")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training run")
    parser.add_argument("--no-cache", action="store_true", help="Disable RAM caching")

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        project=args.project,
        name=args.name,
        force=args.force,
        resume=args.resume,
        cache=not args.no_cache
    )
