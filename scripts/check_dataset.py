"""
Dataset Validation Script - Phase 1 Research Pipeline
Checks YOLO dataset formatting, missing labels, invalid annotations, and split statistics.
"""

import os
import sys
import glob
from pathlib import Path

def check_dataset(dataset_dir="dataset"):
    base_path = Path(dataset_dir)
    splits = ["train", "val", "test"]
    
    print("=" * 60)
    print("AI-BASED ROAD POTHOLE DETECTION - DATASET VALIDATION REPORT")
    print("=" * 60)
    
    if not base_path.exists():
        print(f"[ERROR] Dataset directory '{dataset_dir}' not found.")
        print("Please create the 'dataset/' folder or place your YOLO dataset inside.")
        return
        
    total_images = 0
    total_labels = 0
    total_annotations = 0
    missing_labels_count = 0
    invalid_labels_count = 0
    
    split_stats = {}
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for split in splits:
        img_dir = base_path / "images" / split
        lbl_dir = base_path / "labels" / split
        
        images = []
        if img_dir.exists():
            images = [f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts]
            
        labels = []
        if lbl_dir.exists():
            labels = list(lbl_dir.glob("*.txt"))
            
        split_images = len(images)
        split_labels = len(labels)
        split_missing = 0
        split_invalid = 0
        split_annotations = 0
        
        # Check label pairing and annotation validity
        for img in images:
            expected_lbl = lbl_dir / f"{img.stem}.txt"
            if not expected_lbl.exists():
                split_missing += 1
            else:
                try:
                    with open(expected_lbl, "r") as f:
                        lines = [line.strip() for line in f if line.strip()]
                        for line in lines:
                            parts = line.split()
                            if len(parts) != 5:
                                split_invalid += 1
                                continue
                            cls_id, cx, cy, w, h = parts
                            # Validate values
                            c = int(cls_id)
                            cx_f, cy_f, w_f, h_f = float(cx), float(cy), float(w), float(h)
                            if not (0 <= cx_f <= 1 and 0 <= cy_f <= 1 and 0 < w_f <= 1 and 0 < h_f <= 1):
                                split_invalid += 1
                            else:
                                split_annotations += 1
                except Exception:
                    split_invalid += 1

        split_stats[split] = {
            "images": split_images,
            "labels": split_labels,
            "annotations": split_annotations,
            "missing": split_missing,
            "invalid": split_invalid,
        }
        
        total_images += split_images
        total_labels += split_labels
        total_annotations += split_annotations
        missing_labels_count += split_missing
        invalid_labels_count += split_invalid

    # Display Report
    print(f"\n--- Dataset Split Statistics ---")
    print(f"{'Split':<12} | {'Images':<8} | {'Labels':<8} | {'Pothole Annotations':<20} | {'Missing':<8} | {'Invalid':<8}")
    print("-" * 75)
    for split in splits:
        s = split_stats[split]
        print(f"{split:<12} | {s['images']:<8} | {s['labels']:<8} | {s['annotations']:<20} | {s['missing']:<8} | {s['invalid']:<8}")
    print("-" * 75)
    
    print("\n--- Summary ---")
    print(f"Total Images:             {total_images}")
    print(f"Training Images:          {split_stats['train']['images']}")
    print(f"Validation Images:        {split_stats['val']['images']}")
    print(f"Testing Images:           {split_stats['test']['images']}")
    print(f"Pothole Annotations:      {total_annotations}")
    print(f"Missing Labels:           {missing_labels_count}")
    print(f"Invalid Labels:           {invalid_labels_count}")
    
    # Check environmental folders structure
    env_dirs = ["day", "evening", "night", "rain"]
    print("\n--- Environmental Folders Check (For Future Robustness Testing) ---")
    for env in env_dirs:
        env_path = base_path / env
        status = "Present" if env_path.exists() else "Missing (Can be created)"
        print(f"dataset/{env:<10}: {status}")

    print("\nValidation complete. Note: All figures reflect actual filesystem contents.")
    print("=" * 60)

if __name__ == "__main__":
    check_dataset()
