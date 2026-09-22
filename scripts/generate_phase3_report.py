"""
Phase 3 Environmental Robustness Report Generator
Compiles measured metrics from condition_comparison.csv, hardware specs, and generates a structured research report.
"""

import csv
import sys
from pathlib import Path

def generate_report(
    comparison_csv="results/phase3/condition_comparison.csv",
    hardware_file="results/hardware_info.txt",
    output_report="results/phase3/phase3_report.txt"
):
    comp_path = Path(comparison_csv)
    hw_path = Path(hardware_file)
    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Read Hardware Specs if present
    hw_info_text = "Not profiled (Run scripts/record_hardware_info.py to capture host hardware specs)"
    if hw_path.exists() and hw_path.stat().st_size > 0:
        with open(hw_path, "r", encoding="utf-8") as f:
            hw_info_text = f.read().strip()

    # 2. Read Comparison Data
    records = []
    if comp_path.exists() and comp_path.stat().st_size > 0:
        with open(comp_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row:
                    records.append(row)

    report_lines = [
        "=" * 70,
        "PHASE 3 ENVIRONMENTAL ROBUSTNESS REPORT",
        "AI-Based Road Pothole Detection and Safe Path Identification System",
        "=" * 70,
        "",
        "1. OBJECTIVE",
        "----------------------------------------------------------------------",
        "To evaluate the detection robustness and frame-processing stability",
        "of the Phase 1 trained YOLO pothole detection baseline across distinct",
        "environmental illumination and weather conditions (Day, Evening, Night, Rain)",
        "without condition-specific fine-tuning or artificial bias.",
        "",
        "2. DATASET CONDITIONS EVALUATED",
        "----------------------------------------------------------------------",
        "• DAY:     Direct daylight illumination (standard road contrast).",
        "• EVENING: Low-angle sunlight, long shadows, and potential optical glare.",
        "• NIGHT:   Artificial streetlight and vehicular headlight illumination.",
        "• RAIN:    Wet asphalt reflectivity, water puddle distortions, and reduced contrast.",
        "",
        "3. EXPERIMENTAL SETUP",
        "----------------------------------------------------------------------",
        "Model Evaluated:       Phase 1 Pretrained Baseline (runs/pothole_baseline/weights/best.pt)",
        "Inference Resolution:  640 x 640 px",
        "Confidence Threshold:  0.25 (Default baseline)",
        "Fairness Protocol:     Identical model weights and hyperparameters applied across all conditions.",
        "",
        "Host Hardware Environment:",
        hw_info_text,
        "",
        "4. EXPERIMENTAL RESULTS",
        "----------------------------------------------------------------------",
    ]

    if records:
        header = f"{'Condition':<10} | {'Images':<7} | {'Frames':<7} | {'Detections':<11} | {'Avg Conf':<10} | {'Avg FPS':<8} | {'mAP50':<7}"
        sep = "-" * len(header)
        report_lines.append(header)
        report_lines.append(sep)
        for r in records:
            c_name = r.get("Condition", "Unknown")
            imgs = r.get("Images", "0")
            frames = r.get("Frames", "0")
            dets = r.get("Detections", "0")
            conf = r.get("Average_Confidence", "N/A")
            fps = r.get("Average_FPS", "N/A")
            map50 = r.get("mAP50", "N/A")
            report_lines.append(f"{c_name:<10} | {imgs:<7} | {frames:<7} | {dets:<11} | {conf:<10} | {fps:<8} | {map50:<7}")
        report_lines.append(sep)
    else:
        report_lines.append("[INFO] No condition comparison metrics currently logged in condition_comparison.csv.")
        report_lines.append("Run 'python scripts/evaluate_conditions.py' after placing test samples in dataset/ folders.")

    report_lines.extend([
        "",
        "5. FACTUAL OBSERVATIONS",
        "----------------------------------------------------------------------",
    ])

    if records:
        for r in records:
            cond = r["Condition"]
            dets = int(r["Detections"]) if r["Detections"].isdigit() else 0
            conf = r["Average_Confidence"]
            fps = r["Average_FPS"]
            report_lines.append(f"• {cond}: Recorded {dets} detections with average confidence of {conf} at {fps} FPS.")
    else:
        report_lines.append("• Observations will be automatically generated upon completing condition evaluations.")

    report_lines.extend([
        "",
        "6. EXPERIMENTAL LIMITATIONS",
        "----------------------------------------------------------------------",
        "• Sample Size: Quantitative comparisons reflect only the specific test images/videos evaluated.",
        "• Weather & Illumination Diversity: Rain droplet intensity, street lighting coverage, and headlight angle vary by dataset sample.",
        "• Monocular Scope: 2D detection confidence does not measure 3D pothole severity or water depth in puddles.",
        "• Ground-Truth Dependency: Precision and mAP require explicit YOLO .txt annotations per condition folder.",
        "",
        "=" * 70,
        "End of Report"
    ])

    report_content = "\n".join(report_lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(report_content)
    print(f"\nReport written to: {output_report}")

if __name__ == "__main__":
    generate_report()
