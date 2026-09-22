"""
Baseline YOLO Validation Script - Phase 1 Research Pipeline
Evaluates trained model weights on validation split and logs actual experimental metrics.
"""

import argparse
import csv
from pathlib import Path
from ultralytics import YOLO

def validate_model(weights="runs/pothole_baseline/weights/best.pt", data_yaml="data.yaml", output_csv="results/baseline_results.csv"):
    weights_path = Path(weights)
    if not weights_path.exists():
        print(f"[ERROR] Weights file '{weights}' not found.")
        print("Please train the model first using 'python scripts/train.py' or provide existing weights.")
        return

    print("=" * 60)
    print("AI-BASED POTHOLE DETECTION - BASELINE MODEL EVALUATION")
    print("=" * 60)
    print(f"Model Weights: {weights}")
    print(f"Data Config:   {data_yaml}")
    print("=" * 60)

    model = YOLO(str(weights_path))
    metrics = model.val(data=data_yaml, split="val")

    # Extract actual calculated evaluation metrics
    precision = metrics.box.mp   # Mean precision
    recall = metrics.box.mr      # Mean recall
    map50 = metrics.box.map50    # mAP @ 0.50
    map50_95 = metrics.box.map   # mAP @ 0.50:0.95
    speed = metrics.speed        # dict with preprocess, inference, loss, postprocess times

    fps_val = ""
    if speed and "inference" in speed and speed["inference"] > 0:
        total_time_ms = sum(speed.values())
        if total_time_ms > 0:
            fps_val = f"{1000.0 / total_time_ms:.1f}"

    print("\n--- Actual Experimental Validation Results ---")
    print(f"Precision:         {precision:.4f} ({precision * 100:.2f}%)")
    print(f"Recall:            {recall:.4f} ({recall * 100:.2f}%)")
    print(f"mAP@0.50:          {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"mAP@0.50:0.95:     {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    if fps_val:
        print(f"Estimated Speed:   {fps_val} FPS")
    else:
        print("Estimated Speed:   (Requires dedicated benchmark run)")

    # Save to baseline_results.csv
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Dataset", "Precision", "Recall", "mAP50", "mAP50_95", "FPS"])
        writer.writerow([
            weights_path.stem,
            Path(data_yaml).stem,
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{map50:.4f}",
            f"{map50_95:.4f}",
            fps_val
        ])
    
    print(f"\nResults successfully logged to: {output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate baseline YOLO model on pothole validation dataset.")
    parser.add_argument("--weights", type=str, default="runs/pothole_baseline/weights/best.pt", help="Path to trained .pt weights")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--csv", type=str, default="results/baseline_results.csv", help="Path to results CSV log")
    args = parser.parse_args()

    validate_model(weights=args.weights, data_yaml=args.data, output_csv=args.csv)
