"""
Prediction Testing and Qualitative Inspection Script
AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System

Runs inference on:
1. Selected test road images with pothole hazards.
2. Normal road images (to verify false-positive suppression).
3. Any custom media inside test_images/ if present.
Saves annotated visual predictions to results/predictions/.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def find_best_weights(preferred_path=None):
    candidates = [
        preferred_path,
        "runs/detect/pothole_training/weights/best.pt",
        "runs/pothole_baseline/weights/best.pt",
        "yolov8n.pt"
    ]
    for c in candidates:
        if c and Path(c).exists() and Path(c).stat().st_size > 1000:
            return Path(c)
    return Path("yolov8n.pt")

def run_prediction_tests(
    weights="runs/detect/pothole_training/weights/best.pt",
    test_dir="dataset/images/test",
    normal_roads_dir="dataset/raw/normal_roads",
    user_test_dir="test_images",
    output_dir="results/predictions",
    conf=0.25,
    num_hazard_samples=6,
    num_normal_samples=3
):
    weights_path = find_best_weights(weights)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("ROAD HAZARD PREDICTION & QUALITATIVE VERIFICATION")
    print("=" * 65)
    print(f"Model Checkpoint: {weights_path.resolve()}")
    print(f"Confidence:       {conf}")
    print(f"Output Directory: {out_path.resolve()}")
    print("=" * 65)

    model = YOLO(str(weights_path))

    # 1. Gather test hazard images
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    test_path = Path(test_dir)
    hazard_images = []
    if test_path.exists():
        # Prefer images with non-empty labels (actual hazards)
        lbl_dir = test_path.parent.parent / "labels" / "test"
        all_imgs = [p for p in test_path.iterdir() if p.suffix.lower() in valid_exts]
        for img in all_imgs:
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists() and lbl.stat().st_size > 0:
                hazard_images.append(img)
            elif not lbl.exists():
                hazard_images.append(img)

    selected_hazards = hazard_images[:num_hazard_samples]

    # 2. Gather normal road samples (background)
    normal_images = []
    norm_path = Path(normal_roads_dir)
    if norm_path.exists():
        normal_images = [p for p in norm_path.iterdir() if p.suffix.lower() in valid_exts][:num_normal_samples]
    else:
        # Check test set for background normal roads
        lbl_dir = test_path.parent.parent / "labels" / "test"
        if test_path.exists():
            for img in test_path.iterdir():
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists() and lbl.stat().st_size == 0:
                    normal_images.append(img)
                    if len(normal_images) >= num_normal_samples:
                        break

    # 3. Gather user test images
    user_images = []
    user_path = Path(user_test_dir)
    if user_path.exists():
        user_images = [p for p in user_path.iterdir() if p.suffix.lower() in valid_exts]

    all_targets = [("hazard", img) for img in selected_hazards] + \
                  [("normal_road", img) for img in normal_images] + \
                  [("user_test", img) for img in user_images]

    print(f"\nProcessing {len(all_targets)} qualitative test images...")
    results_summary = []

    for category, img_path in all_targets:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        results = model.predict(img, conf=conf, verbose=False)
        det_count = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                det_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = model.names.get(cls_id, "hazard").upper()

                # Visual Styling: Red bounding box with high-visibility label header
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

                label = f"{cls_name} {score*100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(img, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, max(0, y1)), (0, 0, 255), -1)
                cv2.putText(img, label, (x1 + 4, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Status badge overlay on top left
        badge_text = f"Category: {category.upper()} | Detections: {det_count}"
        badge_color = (0, 165, 255) if det_count > 0 else (0, 180, 0)
        cv2.rectangle(img, (10, 10), (10 + len(badge_text) * 11, 40), (20, 20, 20), -1)
        cv2.putText(img, badge_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2)

        out_file = out_path / f"pred_{category}_{img_path.name}"
        cv2.imwrite(str(out_file), img)

        status_msg = f"{det_count} hazard(s)" if det_count > 0 else "clean (no false alarm)"
        print(f"  [{category.upper()}] {img_path.name:<24} -> {status_msg:<20} saved to {out_file.name}")
        results_summary.append({
            "category": category,
            "file": img_path.name,
            "output": out_file.name,
            "detections": det_count
        })

    print("-" * 65)
    print(f"All {len(results_summary)} predictions successfully generated and saved to:")
    print(f"  {out_path.resolve()}")
    print("=" * 65)
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run qualitative prediction tests.")
    parser.add_argument("--weights", type=str, default="runs/detect/pothole_training/weights/best.pt", help="Path to weights")
    parser.add_argument("--test-dir", type=str, default="dataset/images/test", help="Test images directory")
    parser.add_argument("--output-dir", type=str, default="results/predictions", help="Predictions output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--hazard-samples", type=int, default=6, help="Number of hazard test samples")
    parser.add_argument("--normal-samples", type=int, default=3, help="Number of normal road test samples")

    args = parser.parse_args()

    run_prediction_tests(
        weights=args.weights,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        conf=args.conf,
        num_hazard_samples=args.hazard_samples,
        num_normal_samples=args.normal_samples
    )
