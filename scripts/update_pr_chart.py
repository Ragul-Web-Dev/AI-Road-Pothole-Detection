"""
Update PR Curve Script
Generates:
1. Title-free PR vs Recall Chart (Clean IEEE format, no top title) -> precision_recall_curve_ieee.png & pr_vs_recall_chart_notitle.png
2. Clean concise-titled PR vs Recall Chart (Title: "PR vs Recall Chart") -> pr_vs_recall_chart.png
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

def main():
    base_dir = Path(r"e:\React\Real\Code\MERN-Projects\final_year\pothole-research")
    root_dir = base_dir.parent
    weights_path = base_dir / "runs" / "detect" / "pothole_training" / "weights" / "best.pt"
    data_yaml_path = base_dir / "data.yaml"
    eval_dir_pothole = base_dir / "results" / "evaluation"
    eval_dir_root = root_dir / "results" / "evaluation"
    artifact_dir = Path(r"C:\Users\balaj\.gemini\antigravity\brain\c3ae4518-0bef-4843-9334-1647b335d4e9")

    os.chdir(str(base_dir))
    model = YOLO(str(weights_path))

    # Evaluate test split for exact PR curve data
    val_res = model.val(
        data="data.yaml",
        split="test",
        imgsz=640,
        device="cpu",
        plots=False,
        verbose=False
    )

    box = val_res.box
    map50 = float(box.map50)
    map50_95 = float(box.map)

    pr_curve_data = box.curves_results[0]
    f1_curve_data = box.curves_results[1]
    p_curve_data = box.curves_results[2]
    r_curve_data = box.curves_results[3]

    recall_arr = pr_curve_data[0]
    precision_arr = pr_curve_data[1][0]
    conf_arr = f1_curve_data[0]
    f1_arr = f1_curve_data[1][0]
    p_conf_arr = p_curve_data[1][0]
    r_conf_arr = r_curve_data[1][0]

    max_f1_idx = int(np.argmax(f1_arr))
    opt_conf = float(conf_arr[max_f1_idx])
    max_f1 = float(f1_arr[max_f1_idx])
    opt_precision = float(p_conf_arr[max_f1_idx])
    opt_recall = float(r_conf_arr[max_f1_idx])

    # Operational conf=0.25 point
    r_25 = 0.6959
    p_25 = 0.6959

    # Styling
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.edgecolor"] = "#222222"
    plt.rcParams["axes.linewidth"] = 1.2

    summary_text = (
        f"Evaluation Summary:\n"
        f"• Test Images: 74\n"
        f"• GT Instances: 148\n"
        f"• mAP@0.50: {map50:.4f}\n"
        f"• mAP@0.50:0.95: {map50_95:.4f}\n"
        f"• Peak F1: {max_f1:.4f} (@ conf {opt_conf:.2f})"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#b0bec5", alpha=0.95)

    # =========================================================================
    # VERSION 1: TITLE-FREE (Pure IEEE Conference Standard, No Top Title)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=300)
    ax.plot(
        recall_arr,
        precision_arr,
        color="#0052cc",
        linewidth=2.6,
        label=f"Pothole (mAP@0.50 = {map50*100:.2f}%)"
    )
    ax.fill_between(recall_arr, precision_arr, alpha=0.15, color="#0052cc")

    ax.scatter([opt_recall], [opt_precision], color="#d32f2f", s=80, zorder=5,
               label=f"Optimal F1 ({opt_precision:.2f}, {opt_recall:.2f} @ conf={opt_conf:.2f})")
    ax.scatter([r_25], [p_25], color="#2e7d32", s=80, marker="s", zorder=5,
               label=f"Operational conf=0.25 ({p_25:.2f}, {r_25:.2f})")

    # NO ax.set_title call -> Title completely removed!
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6, color="#bbbbbb")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")

    # Shifted slightly right (0.62) so box does not overlap the red dot marker
    ax.text(0.62, 0.96, summary_text, transform=ax.transAxes, fontsize=9.5, verticalalignment="top", bbox=props)

    plt.tight_layout()

    # Save to all target paths
    p1 = eval_dir_pothole / "precision_recall_curve_ieee.png"
    p2 = eval_dir_root / "precision_recall_curve_ieee.png"
    p3 = eval_dir_pothole / "pr_vs_recall_chart_notitle.png"
    p4 = eval_dir_root / "pr_vs_recall_chart_notitle.png"
    for p in [p1, p2, p3, p4]:
        fig.savefig(p, dpi=300)
    if artifact_dir.exists():
        fig.savefig(artifact_dir / "precision_recall_curve_ieee.png", dpi=300)
        fig.savefig(artifact_dir / "pr_vs_recall_chart_notitle.png", dpi=300)
    plt.close(fig)
    print(f"[+] Successfully saved Title-Free PR vs Recall Chart: {p1}")

    # =========================================================================
    # VERSION 2: CONCISE TITLE ("PR vs Recall Chart")
    # =========================================================================
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=300)
    ax.plot(
        recall_arr,
        precision_arr,
        color="#0052cc",
        linewidth=2.6,
        label=f"Pothole (mAP@0.50 = {map50*100:.2f}%)"
    )
    ax.fill_between(recall_arr, precision_arr, alpha=0.15, color="#0052cc")

    ax.scatter([opt_recall], [opt_precision], color="#d32f2f", s=80, zorder=5,
               label=f"Optimal F1 ({opt_precision:.2f}, {opt_recall:.2f} @ conf={opt_conf:.2f})")
    ax.scatter([r_25], [p_25], color="#2e7d32", s=80, marker="s", zorder=5,
               label=f"Operational conf=0.25 ({p_25:.2f}, {r_25:.2f})")

    # Concise Title: "PR vs Recall Chart"
    ax.set_title("PR vs Recall Chart", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Recall", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=12, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks(np.arange(0.0, 1.01, 0.1))
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6, color="#bbbbbb")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.95, edgecolor="#cccccc")

    ax.text(0.62, 0.96, summary_text, transform=ax.transAxes, fontsize=9.5, verticalalignment="top", bbox=props)

    plt.tight_layout()

    t1 = eval_dir_pothole / "pr_vs_recall_chart.png"
    t2 = eval_dir_root / "pr_vs_recall_chart.png"
    for t in [t1, t2]:
        fig.savefig(t, dpi=300)
    if artifact_dir.exists():
        fig.savefig(artifact_dir / "pr_vs_recall_chart.png", dpi=300)
    plt.close(fig)
    print(f"[+] Successfully saved Titled 'PR vs Recall Chart': {t1}")

    # Also update standard precision_recall_curve.png with clean title
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=300)
    ax.plot(recall_arr, precision_arr, color="#1976d2", linewidth=2.2, label=f"pothole {map50:.3f}")
    ax.set_title("PR vs Recall Chart", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    fig.savefig(eval_dir_pothole / "precision_recall_curve.png", dpi=300)
    fig.savefig(eval_dir_root / "precision_recall_curve.png", dpi=300)
    if artifact_dir.exists():
        fig.savefig(artifact_dir / "precision_recall_curve.png", dpi=300)
    plt.close(fig)
    print("[+] Successfully updated standard precision_recall_curve.png")

if __name__ == "__main__":
    main()

