"""
Comprehensive YOLO Model Evaluation and Validation Script
AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System

Evaluates trained checkpoint on Validation and Test datasets.
Computes and exports:
- Precision, Recall, F1-Score
- mAP@0.50, mAP@0.50:0.95
- Confusion Matrix (TP, FP, FN)
- Per-image latency (preprocess, inference, postprocess) and FPS
- False-Positive and False-Negative error analysis
- Exports to results/baseline_results.csv, results/evaluation_metrics.json, and results/evaluation_report.md
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from ultralytics import YOLO

def find_best_weights(preferred_path=None):
    """Finds best.pt from standard or baseline locations."""
    candidates = [
        preferred_path,
        "runs/detect/pothole_training/weights/best.pt",
        "runs/pothole_baseline/weights/best.pt",
        "runs/detect/pothole_baseline/weights/best.pt"
    ]
    for c in candidates:
        if c and Path(c).exists() and Path(c).stat().st_size > 1000:
            return Path(c)
    return None

def analyze_fp_fn_samples(model, data_yaml_path, split="test", conf=0.25, max_samples=5):
    """
    Identifies False Positive (detections on background normal roads) and
    False Negative (hazard present in ground truth but zero detections) cases.
    """
    import yaml
    with open(data_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = Path(data_yaml_path).parent
    root = base_dir / config.get("path", "dataset")
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    if not img_dir.exists():
        return [], []

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts]

    false_positives = []
    false_negatives = []

    for img_p in images:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        has_gt = False
        if lbl_p.exists():
            with open(lbl_p, "r") as lf:
                gt_lines = [l.strip() for l in lf if l.strip()]
            has_gt = len(gt_lines) > 0

        # Predict
        results = model.predict(str(img_p), conf=conf, verbose=False)
        det_count = len(results[0].boxes) if results else 0

        if not has_gt and det_count > 0:
            # False Positive on normal road
            false_positives.append({
                "image": img_p.name,
                "detected_count": det_count,
                "max_confidence": float(results[0].boxes.conf.max()) if det_count else 0.0
            })
        elif has_gt and det_count == 0:
            # False Negative (missed pothole)
            false_negatives.append({
                "image": img_p.name,
                "gt_boxes": len(gt_lines)
            })

    return false_positives[:max_samples], false_negatives[:max_samples]

def evaluate_split(model, data_yaml, split="val", imgsz=640, conf=0.25, iou=0.60):
    """Evaluates YOLO model on a specific split and extracts formal metrics."""
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        plots=True,
        verbose=False
    )

    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    f1 = float(2 * (p * r) / (p + r)) if (p + r) > 0 else 0.0
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    speed = metrics.speed or {}

    total_latency_ms = sum(speed.values()) if speed else 0.0
    fps = (1000.0 / total_latency_ms) if total_latency_ms > 0 else 0.0

    # Extract confusion matrix if available
    cm_data = {}
    try:
        if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
            mat = metrics.confusion_matrix.matrix
            cm_data = {"matrix": mat.tolist()}
    except Exception:
        pass

    return {
        "split": split,
        "precision": p,
        "recall": r,
        "f1_score": f1,
        "map50": map50,
        "map50_95": map50_95,
        "latency_ms": {
            "preprocess": speed.get("preprocess", 0.0),
            "inference": speed.get("inference", 0.0),
            "loss": speed.get("loss", 0.0),
            "postprocess": speed.get("postprocess", 0.0),
            "total": total_latency_ms
        },
        "fps": fps,
        "confusion_matrix": cm_data
    }

def run_evaluation(
    weights=None,
    data_yaml="data.yaml",
    split="both",
    imgsz=640,
    conf=0.25,
    iou=0.60,
    output_dir="results",
    csv_file="results/baseline_results.csv"
):
    weights_path = find_best_weights(weights)
    if not weights_path:
        print(f"[ERROR] Trained model weights not found at '{weights}'.")
        print("Please train the model first using 'python scripts/train.py'.")
        sys.exit(1)

    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"[ERROR] Data YAML file '{data_yaml}' not found.")
        sys.exit(1)

    print("=" * 70)
    print("ROAD HAZARD MODEL COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Model Checkpoint:  {weights_path.resolve()}")
    print(f"Data Config:       {data_path.resolve()}")
    print(f"Evaluation Target: Split = {split} | ImgSz = {imgsz} | Conf = {conf}")
    print("=" * 70)

    model = YOLO(str(weights_path))

    splits_to_run = ["val", "test"] if split == "both" else [split]
    results_by_split = {}

    for s in splits_to_run:
        print(f"\n[EVALUATING] Split: '{s}' ...")
        res = evaluate_split(model, data_path, split=s, imgsz=imgsz, conf=conf, iou=iou)
        results_by_split[s] = res

    # Analyze False Positives / False Negatives on Test Split
    print("\n[ERROR ANALYSIS] Scanning test set for False Positives / Negatives...")
    fps_samples, fns_samples = analyze_fp_fn_samples(model, data_path, split="test", conf=conf)

    # Display Metrics Table
    print("\n" + "=" * 70)
    print("OFFICIAL EVALUATION METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Split':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'mAP@0.50':<10} | {'mAP50-95':<10} | {'FPS':<6}")
    print("-" * 75)
    for s, m in results_by_split.items():
        print(f"{s:<8} | {m['precision']*100:<9.2f}% | {m['recall']*100:<9.2f}% | {m['f1_score']*100:<9.2f}% | {m['map50']*100:<9.2f}% | {m['map50_95']*100:<9.2f}% | {m['fps']:<6.1f}")
    print("-" * 75)

    print(f"\nError Analysis on Test Set (conf={conf}):")
    print(f"  False Positives Identified: {len(fps_samples)}")
    for fp in fps_samples:
        print(f"    - {fp['image']}: {fp['detected_count']} detection(s), max_conf={fp['max_confidence']:.2f}")
    print(f"  False Negatives Identified: {len(fns_samples)}")
    for fn in fns_samples:
        print(f"    - {fn['image']}: missed {fn['gt_boxes']} ground-truth pothole(s)")
    print("=" * 70)

    # Save to CSV
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Split", "Precision", "Recall", "F1", "mAP50", "mAP50_95", "Inference_ms", "FPS"])
        for s, m in results_by_split.items():
            writer.writerow([
                weights_path.stem,
                s,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['map50']:.4f}",
                f"{m['map50_95']:.4f}",
                f"{m['latency_ms']['inference']:.2f}",
                f"{m['fps']:.1f}"
            ])

    # Save JSON Report
    json_path = out_dir_path / "evaluation_metrics.json"
    full_report = {
        "weights": str(weights_path),
        "data_yaml": str(data_path),
        "splits": results_by_split,
        "error_analysis": {
            "false_positives": fps_samples,
            "false_negatives": fns_samples
        }
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(full_report, jf, indent=2)

    # Save Markdown Report
    md_path = out_dir_path / "evaluation_report.md"
    generate_markdown_eval_report(md_path, full_report)

    print(f"Reports successfully saved to:")
    print(f"  - CSV:  {csv_path.resolve()}")
    print(f"  - JSON: {json_path.resolve()}")
    print(f"  - MD:   {md_path.resolve()}")
    print("=" * 70)

def generate_markdown_eval_report(md_path, report):
    """Creates GitHub-style Markdown evaluation report."""
    md = []
    md.append("# Road Hazard Model Validation and Evaluation Report")
    md.append("**Project**: AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System\n")
    md.append(f"- **Evaluated Checkpoint**: `{report['weights']}`")
    md.append(f"- **Data Configuration**: `{report['data_yaml']}`\n")
    md.append("## 1. Quantitative Performance Metrics")
    md.append("| Dataset Split | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 | Latency (Inference) | Real-Time FPS |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for s, m in report["splits"].items():
        md.append(f"| **{s.capitalize()}** | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1_score']*100:.2f}% | {m['map50']*100:.2f}% | {m['map50_95']*100:.2f}% | {m['latency_ms']['inference']:.1f} ms | **{m['fps']:.1f} FPS** |")
    md.append("")

    md.append("## 2. Speed and Latency Breakdown")
    for s, m in report["splits"].items():
        lat = m["latency_ms"]
        md.append(f"### {s.capitalize()} Split Latency")
        md.append(f"- **Preprocess**: {lat['preprocess']:.2f} ms")
        md.append(f"- **Inference**: {lat['inference']:.2f} ms")
        md.append(f"- **Postprocess**: {lat['postprocess']:.2f} ms")
        md.append(f"- **Total Frame Processing Time**: {lat['total']:.2f} ms (~{m['fps']:.1f} FPS)\n")

    md.append("## 3. Error Analysis (Test Split)")
    fps = report["error_analysis"]["false_positives"]
    fns = report["error_analysis"]["false_negatives"]
    md.append(f"### False Positives (Normal Road / False Alarm Rate)")
    if fps:
        for item in fps:
            md.append(f"- `{item['image']}`: {item['detected_count']} detection(s) with max confidence {item['max_confidence']:.2f}")
    else:
        md.append("- No significant false positives observed on evaluated normal road test samples.")

    md.append(f"\n### False Negatives (Missed Hazards)")
    if fns:
        for item in fns:
            md.append(f"- `{item['image']}`: Missed {item['gt_boxes']} ground-truth hazard bounding box(es)")
    else:
        md.append("- No complete false negative failures observed on evaluated test samples.")

    md_path.write_text("\n".join(md), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on validation and test splits.")
    parser.add_argument("--weights", type=str, default="runs/detect/pothole_training/weights/best.pt", help="Path to weights")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"], help="Split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.60, help="NMS IoU threshold")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output reports")
    parser.add_argument("--csv", type=str, default="results/baseline_results.csv", help="CSV log file")

    args = parser.parse_args()

    run_evaluation(
        weights=args.weights,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        output_dir=args.output_dir,
        csv_file=args.csv
    )
