"""
Optional Low-Light & Weather Preprocessing Module - Phase 3 Research Pipeline
Provides isolated image enhancements (CLAHE, Gamma Correction, Brightness Normalization)
for separate comparative evaluation without altering original baseline test data.
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


def apply_clahe(img):
    """Contrast Limited Adaptive Histogram Equalization in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def apply_gamma(img, gamma=1.5):
    """Gamma correction for non-linear dynamic range adjustment."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_brightness_contrast(img, brightness=30, contrast=30):
    """Linear brightness & contrast scaling."""
    alpha = (contrast + 127) / 127.0
    beta = brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def process_folder(input_dir, output_dir, method="clahe"):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in in_path.iterdir() if f.is_file() and f.suffix.lower() in img_exts]

    print("=" * 60)
    print("OPTIONAL LOW-LIGHT PREPROCESSING EXPERIMENT")
    print("=" * 60)
    print(f"Source Folder:     {in_path}")
    print(f"Enhancement Type:  {method.upper()}")
    print(f"Output Directory:  {out_path}")
    print(f"Images Found:      {len(images)}")
    print("=" * 60)

    for img_file in images:
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        if method == "clahe":
            enhanced = apply_clahe(img)
        elif method == "gamma":
            enhanced = apply_gamma(img, gamma=1.6)
        elif method == "linear":
            enhanced = apply_brightness_contrast(img, brightness=25, contrast=20)
        else:
            enhanced = apply_clahe(img)

        out_file = out_path / f"enhanced_{img_file.name}"
        cv2.imwrite(str(out_file), enhanced)

    print(f"\nEnhanced images saved separately in: {out_path}")
    print("Note: Run 'python scripts/evaluate_conditions.py --source " + str(out_path) + "' to evaluate enhancement impact.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimental low-light preprocessing for road images.")
    parser.add_argument("--input", type=str, default="dataset/night", help="Input folder containing low-light images")
    parser.add_argument("--output", type=str, default="dataset/night_enhanced", help="Output folder for enhanced images")
    parser.add_argument("--method", type=str, default="clahe", choices=["clahe", "gamma", "linear"], help="Enhancement algorithm")
    args = parser.parse_args()

    process_folder(args.input, args.output, method=args.method)
