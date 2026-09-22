"""
Phase 3 Plotting & Chart Generation Script
Reads condition evaluation results and plots empirical comparisons (Confidence, FPS, Detections, Metrics).
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] Matplotlib is not installed in the active Python environment.")
    print("Please activate your virtual environment: venv\\Scripts\\activate")
    sys.exit(1)


def plot_results(csv_path="results/phase3/condition_comparison.csv", output_dir="results/phase3/plots"):
    in_file = Path(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3 ENVIRONMENTAL ROBUSTNESS PLOTTING UTILITY")
    print("=" * 60)

    if not in_file.exists() or in_file.stat().st_size == 0:
        print(f"[INFO] Results file '{csv_path}' is empty or does not exist.")
        print("Run 'python scripts/evaluate_conditions.py' first to generate empirical evaluation metrics.")
        return

    records = []
    with open(in_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                records.append(row)

    if not records:
        print("[INFO] No rows found in condition comparison CSV.")
        return

    conditions = [r["Condition"] for r in records]
    
    # 1. Plot Average Confidence by Condition
    conf_values = []
    for r in records:
        c_str = r.get("Average_Confidence", "").replace("%", "").strip()
        try:
            conf_values.append(float(c_str))
        except ValueError:
            conf_values.append(0.0)

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(conditions, conf_values, color="#0284c7", edgecolor="#0369a1", width=0.5)
    plt.title("Average Pothole Detection Confidence by Condition", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Environmental Condition", fontsize=10, labelpad=8)
    plt.ylabel("Confidence (%)", fontsize=10, labelpad=8)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 1.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    conf_plot_path = out_dir / "confidence_by_condition.png"
    plt.savefig(conf_plot_path, dpi=300)
    plt.close()
    print(f"Generated chart: {conf_plot_path}")

    # 2. Plot Average FPS by Condition
    fps_values = []
    for r in records:
        f_str = r.get("Average_FPS", "").strip()
        try:
            fps_values.append(float(f_str))
        except ValueError:
            fps_values.append(0.0)

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(conditions, fps_values, color="#10b981", edgecolor="#059669", width=0.5)
    plt.title("Inference Throughput (FPS) Across Environmental Conditions", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Environmental Condition", fontsize=10, labelpad=8)
    plt.ylabel("Frames Per Second (FPS)", fontsize=10, labelpad=8)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fps_plot_path = out_dir / "fps_by_condition.png"
    plt.savefig(fps_plot_path, dpi=300)
    plt.close()
    print(f"Generated chart: {fps_plot_path}")

    # 3. Check for Precision / Recall / mAP50 if available (not 'N/A')
    has_map = any(r.get("mAP50", "N/A") != "N/A" for r in records)
    if has_map:
        prec_vals, rec_vals, map_vals = [], [], []
        for r in records:
            p = float(r.get("Precision", 0)) if r.get("Precision", "N/A") != "N/A" else 0.0
            rc = float(r.get("Recall", 0)) if r.get("Recall", "N/A") != "N/A" else 0.0
            m = float(r.get("mAP50", 0)) if r.get("mAP50", "N/A") != "N/A" else 0.0
            prec_vals.append(p)
            rec_vals.append(rc)
            map_vals.append(m)

        import numpy as np
        x = np.arange(len(conditions))
        w = 0.25

        plt.figure(figsize=(8, 4.5))
        plt.bar(x - w, prec_vals, width=w, label="Precision", color="#3b82f6")
        plt.bar(x, rec_vals, width=w, label="Recall", color="#f59e0b")
        plt.bar(x + w, map_vals, width=w, label="mAP@0.50", color="#10b981")
        plt.xticks(x, conditions)
        plt.title("Object Detection Metrics by Environmental Condition", fontsize=12, fontweight="bold")
        plt.xlabel("Condition", fontsize=10)
        plt.ylabel("Score (0.0 - 1.0)", fontsize=10)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        metrics_plot_path = out_dir / "detection_metrics_by_condition.png"
        plt.savefig(metrics_plot_path, dpi=300)
        plt.close()
        print(f"Generated chart: {metrics_plot_path}")

    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance and confidence comparison plots for Phase 3.")
    parser.add_argument("--csv", type=str, default="results/phase3/condition_comparison.csv", help="Input comparison CSV path")
    parser.add_argument("--output", type=str, default="results/phase3/plots", help="Directory to save generated charts")
    args = parser.parse_args()

    plot_results(csv_path=args.csv, output_dir=args.output)
