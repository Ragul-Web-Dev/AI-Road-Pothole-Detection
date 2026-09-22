"""
Condition Comparison Visualizer - Phase 3 Research Pipeline
Gathers representative detection results across Day, Evening, Night, and Rain conditions
and builds side-by-side comparative research figures.
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("[ERROR] Required libraries (OpenCV, NumPy) not found in the active Python environment.")
    print("Please activate your virtual environment: venv\\Scripts\\activate")
    sys.exit(1)


CONDITIONS = ["day", "evening", "night", "rain"]

def create_comparison(input_dir="results/phase3", output_dir="results/phase3/comparison"):
    in_base = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    condition_samples = {}

    for cond in CONDITIONS:
        cond_dir = in_base / cond
        if cond_dir.exists():
            imgs = [f for f in cond_dir.iterdir() if f.is_file() and f.suffix.lower() in img_exts]
            if imgs:
                condition_samples[cond] = imgs[0]  # Pick representative image

    print("=" * 60)
    print("PHASE 3 ENVIRONMENTAL CONDITION COMPARISON GENERATOR")
    print("=" * 60)
    print(f"Available Conditions for Comparison: {list(condition_samples.keys())}")

    if not condition_samples:
        print("[INFO] No condition evaluation images found in 'results/phase3/'.")
        print("Run 'python scripts/evaluate_conditions.py' first after adding condition test images.")
        return

    # Resize all samples to uniform dimensions for comparison panel
    target_w, target_h = 640, 360
    panels = []

    for cond in CONDITIONS:
        if cond in condition_samples:
            img = cv2.imread(str(condition_samples[cond]))
            if img is not None:
                panel = cv2.resize(img, (target_w, target_h))
                # Add Banner Header
                cv2.rectangle(panel, (0, 0), (target_w, 35), (15, 23, 42), -1)
                title_text = f"CONDITION: {cond.upper()} (EVALUATION SAMPLE)"
                cv2.putText(panel, title_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                panels.append((cond, panel))
        else:
            # Create a placeholder empty tile indicating data unavailable
            empty_panel = np.zeros((target_h, target_w, 3), dtype=np.uint8) + 40
            cv2.rectangle(empty_panel, (0, 0), (target_w, 35), (15, 23, 42), -1)
            cv2.putText(empty_panel, f"CONDITION: {cond.upper()}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
            cv2.putText(empty_panel, "No Dataset Available", (target_w // 4, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)
            panels.append((cond, empty_panel))

    # Arrange 2x2 Grid
    if len(panels) == 4:
        top_row = np.hstack([panels[0][1], panels[1][1]])
        bot_row = np.hstack([panels[2][1], panels[3][1]])
        grid = np.vstack([top_row, bot_row])

        out_grid_path = out_path / "environmental_comparison_grid.jpg"
        cv2.imwrite(str(out_grid_path), grid)
        print(f"\nSaved 4-way comparison grid to: {out_grid_path}")

    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create side-by-side environmental condition comparison figures.")
    parser.add_argument("--input", type=str, default="results/phase3", help="Base directory of Phase 3 results")
    parser.add_argument("--output", type=str, default="results/phase3/comparison", help="Output directory for comparison figures")
    args = parser.parse_args()

    create_comparison(input_dir=args.input, output_dir=args.output)
