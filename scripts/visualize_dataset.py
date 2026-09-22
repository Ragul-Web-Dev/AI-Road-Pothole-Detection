"""
Dataset Visualization Script - Phase 1 Research Pipeline
Selects sample images, parses YOLO bounding boxes, and exports visual verification samples.
"""

import os
import sys
import random
from pathlib import Path
import cv2

def visualize_samples(dataset_dir="dataset", split="train", num_samples=5, output_dir="results/dataset_samples"):
    img_dir = Path(dataset_dir) / "images" / split
    lbl_dir = Path(dataset_dir) / "labels" / split
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not img_dir.exists():
        print(f"[ERROR] Image directory '{img_dir}' does not exist.")
        return
        
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts]
    
    if not images:
        print(f"[WARNING] No images found in '{img_dir}'. Place images and labels to visualize.")
        return
        
    selected_images = random.sample(images, min(num_samples, len(images)))
    print(f"Visualizing {len(selected_images)} samples from '{split}' split...")
    
    for img_file in selected_images:
        lbl_file = lbl_dir / f"{img_file.stem}.txt"
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"[WARNING] Could not read image {img_file.name}")
            continue
            
        h, w = img.shape[:2]
        
        if lbl_file.exists():
            with open(lbl_file, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
                
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    cls_id, cx, cy, bw, bh = map(float, parts)
                    
                    # Convert normalized coords to pixel bounding box
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    
                    # Draw bounding box (Red/Cyan for clarity)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Label text
                    label = "POTHOLE"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), (0, 0, 255), -1)
                    cv2.putText(img, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            print(f"[INFO] No label file found for {img_file.name}")
            
        out_file = out_path / f"annotated_{img_file.name}"
        cv2.imwrite(str(out_file), img)
        print(f"Saved visualization: {out_file}")

    print(f"\nAll sample visual verification images saved in: {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize ground-truth YOLO pothole dataset annotations.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train, val, test)")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to visualize")
    parser.add_argument("--output", type=str, default="results/dataset_samples", help="Output directory")
    args = parser.parse_args()
    
    visualize_samples(args.dataset, args.split, args.samples, args.output)
