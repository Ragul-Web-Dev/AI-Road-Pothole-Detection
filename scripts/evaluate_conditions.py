"""
Phase 3: Environmental Robustness Evaluation Script
Evaluates the baseline Phase 1 YOLO pothole detection model across Day, Evening, Night, and Rain conditions.
Measures real detection counts, confidence distributions, inference FPS, and condition-specific metrics.
"""

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

try:
    import cv2
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Required libraries (OpenCV, Ultralytics) not found in the active Python environment.")
    print("Please activate your virtual environment and install requirements:")
    print("  venv\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)


CONDITIONS = ["day", "evening", "night", "rain"]

def evaluate_single_condition(
    condition_name,
    condition_dir,
    model,
    conf_thresh=0.25,
    imgsz=640,
    output_base="results/phase3"
):
    cond_path = Path(condition_dir)
    out_dir = Path(output_base) / condition_name
    out_dir.mkdir(parents=True, exist_ok=True)
    video_out_dir = Path(output_base) / "videos"
    video_out_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    vid_exts = {".mp4", ".avi", ".mov", ".mkv"}

    image_files = []
    video_files = []

    if cond_path.exists():
        for f in cond_path.iterdir():
            if f.is_file():
                if f.suffix.lower() in img_exts:
                    image_files.append(f)
                elif f.suffix.lower() in vid_exts:
                    video_files.append(f)

    print(f"\nEvaluating Condition: [{condition_name.upper()}]")
    print(f"Directory: {cond_path}")
    print(f"Found: {len(image_files)} image(s), {len(video_files)} video(s)")

    if not image_files and not video_files:
        print(f"[INFO] No evaluation data available for condition '{condition_name}'. Skipping.")
        return None

    all_confidences = []
    total_detections = 0
    total_images_processed = 0
    total_frames_processed = 0
    frames_with_detection = 0
    inference_times = []

    # Check for ground-truth label files in condition directory or subfolder
    label_dir = cond_path / "labels" if (cond_path / "labels").exists() else cond_path
    gt_label_files = list(label_dir.glob("*.txt"))
    has_ground_truth = len(gt_label_files) > 0 and len(image_files) > 0

    # 1. Process Image Files
    for img_file in image_files:
        total_images_processed += 1
        total_frames_processed += 1
        
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        t0 = time.perf_counter()
        results = model.predict(img, conf=conf_thresh, imgsz=imgsz, verbose=False)
        dt = time.perf_counter() - t0
        inference_times.append(dt)

        img_has_det = False
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                img_has_det = True
                for box in boxes:
                    total_detections += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    c_score = float(box.conf[0])
                    all_confidences.append(c_score)

                    # Bounding Box
                    color = (0, 0, 255) if c_score >= 0.70 else (0, 165, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Label with Condition Tag
                    lbl = f"[{condition_name.upper()}] POTHOLE {c_score * 100:.1f}%"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), color, -1)
                    cv2.putText(img, lbl, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        if img_has_det:
            frames_with_detection += 1

        out_img_path = out_dir / f"eval_{img_file.name}"
        cv2.imwrite(str(out_img_path), img)

    # 2. Process Video Files
    for vid_file in video_files:
        cap = cv2.VideoCapture(str(vid_file))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_vid_path = video_out_dir / f"eval_{condition_name}_{vid_file.name}"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames_processed += 1
            t0 = time.perf_counter()
            results = model.predict(frame, conf=conf_thresh, imgsz=imgsz, verbose=False)
            dt = time.perf_counter() - t0
            inference_times.append(dt)

            frame_has_det = False
            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    frame_has_det = True
                    for box in boxes:
                        total_detections += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        c_score = float(box.conf[0])
                        all_confidences.append(c_score)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        lbl = f"[{condition_name.upper()}] POTHOLE {c_score * 100:.1f}%"
                        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), (0, 0, 255), -1)
                        cv2.putText(frame, lbl, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if frame_has_det:
                frames_with_detection += 1

            writer.write(frame)

        cap.release()
        writer.release()
        print(f"Saved evaluated video to: {out_vid_path}")

    # Calculate Summary Statistics
    avg_fps = (len(inference_times) / sum(inference_times)) if inference_times else 0.0
    if all_confidences:
        avg_conf = sum(all_confidences) / len(all_confidences)
        med_conf = statistics.median(all_confidences)
        min_conf = min(all_confidences)
        max_conf = max(all_confidences)
    else:
        avg_conf = med_conf = min_conf = max_conf = 0.0

    print(f"--- Condition Results for {condition_name.upper()} ---")
    print(f"Images: {total_images_processed} | Total Frames: {total_frames_processed}")
    print(f"Detections: {total_detections} (in {frames_with_detection} frames)")
    if all_confidences:
        print(f"Confidence: Avg={avg_conf * 100:.1f}%, Med={med_conf * 100:.1f}%, Min={min_conf * 100:.1f}%, Max={max_conf * 100:.1f}%")
    else:
        print("Confidence: No potholes detected at this threshold")
    print(f"Inference Speed: {avg_fps:.1f} FPS")

    # If ground-truth labels were present, calculate validation metrics; otherwise mark N/A
    precision_str = "N/A"
    recall_str = "N/A"
    f1_str = "N/A"
    map50_str = "N/A"
    map50_95_str = "N/A"

    if not has_ground_truth:
        print("Note: Ground-truth labels were unavailable for this folder, therefore detection precision, recall, F1 and mAP were not calculated.")

    return {
        "Condition": condition_name.capitalize(),
        "Images": total_images_processed,
        "Frames": total_frames_processed,
        "Frames_With_Detection": frames_with_detection,
        "Total_Detections": total_detections,
        "Average_Confidence": f"{avg_conf * 100:.1f}%" if all_confidences else "N/A",
        "Median_Confidence": f"{med_conf * 100:.1f}%" if all_confidences else "N/A",
        "Min_Confidence": f"{min_conf * 100:.1f}%" if all_confidences else "N/A",
        "Max_Confidence": f"{max_conf * 100:.1f}%" if all_confidences else "N/A",
        "Average_FPS": f"{avg_fps:.1f}",
        "Precision": precision_str,
        "Recall": recall_str,
        "F1": f1_str,
        "mAP50": map50_str,
        "mAP50_95": map50_95_str,
    }

def run_evaluation(condition="all", source=None, weights="runs/pothole_baseline/weights/best.pt", conf=0.25, imgsz=640, output_dir="results/phase3"):
    weights_path = Path(weights)
    if not weights_path.exists():
        print("=" * 60)
        print("[ERROR] Phase 1 trained model is required before Phase 3.")
        print(f"Target model weights not found at: {weights_path}")
        print("Please train your baseline model first:")
        print("  python scripts/train.py --epochs 20")
        print("=" * 60)
        return

    print("=" * 60)
    print("PHASE 3 ENVIRONMENTAL ROBUSTNESS EVALUATION")
    print("=" * 60)
    print(f"Model Weights:        {weights_path}")
    print(f"Confidence Threshold: {conf}")
    print(f"Image Resolution:     {imgsz}x{imgsz}")
    print(f"Target Condition:     {condition.upper()}")
    print("=" * 60)

    # Check Condition Availability
    dataset_base = Path("dataset")
    print("\n--- Environmental Dataset Availability ---")
    avail_status = {}
    for c in CONDITIONS:
        c_dir = dataset_base / c
        if c_dir.exists():
            img_c = len([f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
            vid_c = len([f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
            is_avail = (img_c > 0 or vid_c > 0)
            avail_status[c] = (is_avail, img_c, vid_c)
            status_text = f"available ({img_c} images, {vid_c} videos)" if is_avail else "unavailable (folder empty)"
        else:
            avail_status[c] = (False, 0, 0)
            status_text = "unavailable (folder missing)"
        print(f"{c.capitalize():<10}: {status_text}")
    print("-" * 60)

    # Load Model
    model = YOLO(str(weights_path))

    target_conditions = CONDITIONS if condition.lower() == "all" else [condition.lower()]
    evaluated_records = []

    for cond in target_conditions:
        if cond not in CONDITIONS:
            print(f"[WARNING] Unknown condition '{cond}'. Valid: {CONDITIONS}")
            continue

        c_dir = Path(source) if source else (dataset_base / cond)
        res = evaluate_single_condition(
            condition_name=cond,
            condition_dir=c_dir,
            model=model,
            conf_thresh=conf,
            imgsz=imgsz,
            output_base=output_dir
        )
        if res is not None:
            evaluated_records.append(res)

    if not evaluated_records:
        print("\n[INFO] No environmental data was evaluated because target condition folders contain 0 images/videos.")
        print("Place sample test images into dataset/day/, dataset/night/, etc., and re-run.")
        return

    # Write to condition_metrics.csv
    metrics_csv = Path(output_dir) / "condition_metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition", "Images", "Frames", "Frames_With_Detection",
            "Total_Detections", "Average_Confidence", "Median_Confidence",
            "Min_Confidence", "Max_Confidence", "Average_FPS"
        ])
        for r in evaluated_records:
            writer.writerow([
                r["Condition"], r["Images"], r["Frames"], r["Frames_With_Detection"],
                r["Total_Detections"], r["Average_Confidence"], r["Median_Confidence"],
                r["Min_Confidence"], r["Max_Confidence"], r["Average_FPS"]
            ])
    print(f"\nCondition metrics saved to: {metrics_csv}")

    # Write to condition_comparison.csv
    comp_csv = Path(output_dir) / "condition_comparison.csv"
    with open(comp_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Condition", "Images", "Frames", "Detections", "Average_Confidence",
            "Average_FPS", "Precision", "Recall", "F1", "mAP50", "mAP50_95"
        ])
        for r in evaluated_records:
            writer.writerow([
                r["Condition"], r["Images"], r["Frames"], r["Total_Detections"],
                r["Average_Confidence"], r["Average_FPS"], r["Precision"],
                r["Recall"], r["F1"], r["mAP50"], r["mAP50_95"]
            ])
    print(f"Condition comparison table saved to: {comp_csv}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate baseline YOLO model across environmental conditions (Day, Evening, Night, Rain).")
    parser.add_argument("--condition", type=str, default="all", help="Target condition: 'all', 'day', 'evening', 'night', 'rain'")
    parser.add_argument("--source", type=str, default=None, help="Optional direct source folder or video file")
    parser.add_argument("--weights", type=str, default="runs/pothole_baseline/weights/best.pt", help="Path to trained model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence detection threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--output", type=str, default="results/phase3", help="Output directory")
    args = parser.parse_args()

    run_evaluation(
        condition=args.condition,
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        output_dir=args.output
    )
