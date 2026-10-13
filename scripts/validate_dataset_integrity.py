"""
Dataset Integrity and Data Leakage Validation Script
AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System

Validates:
1. File decodability and image integrity (Pillow/OpenCV).
2. Image and label 1-to-1 pairings (including background normal road images).
3. YOLO bounding box coordinates strictly in [0, 1] range.
4. Class IDs valid against data.yaml classes.
5. Exact duplicate detection and Cross-Split Data Leakage (Train vs Val vs Test).
6. Generates full audit report to results/dataset_validation_report.txt and .json.
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from PIL import Image
import yaml

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def compute_file_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def validate_dataset(data_yaml_path: str = "data.yaml", report_dir: str = "results"):
    yaml_file = Path(data_yaml_path)
    if not yaml_file.exists():
        print(f"[ERROR] Data YAML file '{data_yaml_path}' not found.")
        sys.exit(1)

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    # Base path
    base_dir = yaml_file.parent
    if "path" in config and config["path"]:
        dataset_root = base_dir / config["path"]
    else:
        dataset_root = base_dir / "dataset"

    class_names = config.get("names", {})
    if isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    splits = ["train", "val", "test"]

    print("=" * 75)
    print("ROAD HAZARD DATASET INTEGRITY & DATA LEAKAGE AUDIT")
    print("=" * 75)
    print(f"Data Config:     {yaml_file.resolve()}")
    print(f"Dataset Root:    {dataset_root.resolve()}")
    print(f"Hazard Classes:  {class_names}")
    print("=" * 75)

    split_hashes = {s: {} for s in splits}  # hash -> file_path
    split_stats = {}
    corrupted_images = []
    missing_labels = []
    orphaned_labels = []
    invalid_annotations = []
    class_totals = {cname: 0 for cname in class_names.values()}

    for split in splits:
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split

        if not img_dir.exists():
            print(f"[WARNING] Image directory missing for split '{split}': {img_dir}")
            split_stats[split] = {"images": 0, "labels": 0, "background": 0, "bboxes": 0}
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_IMG_EXTS]
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        img_stems = {img.stem: img for img in images}
        lbl_stems = {lbl.stem: lbl for lbl in labels}

        # Check for orphaned labels
        for stem, lbl in lbl_stems.items():
            if stem not in img_stems:
                orphaned_labels.append({"split": split, "label": lbl.name})

        split_bboxes = 0
        split_background = 0
        split_class_counts = {cname: 0 for cname in class_names.values()}

        for img in images:
            # 1. Validate image integrity
            try:
                with Image.open(img) as im:
                    im.verify()
                with Image.open(img) as im:
                    im.load()
                    w, h = im.size
                if w <= 0 or h <= 0:
                    corrupted_images.append({"split": split, "file": img.name, "reason": "Invalid dimensions"})
                    continue
            except Exception as e:
                corrupted_images.append({"split": split, "file": img.name, "reason": str(e)})
                continue

            # Record hash for leakage check
            fhash = compute_file_hash(img)
            split_hashes[split][fhash] = img

            # 2. Check label pairing
            lbl_file = lbl_dir / f"{img.stem}.txt"
            if not lbl_file.exists():
                missing_labels.append({"split": split, "image": img.name})
                continue

            # 3. Validate annotations
            try:
                with open(lbl_file, "r") as lf:
                    lines = [l.strip() for l in lf if l.strip()]

                if not lines:
                    split_background += 1  # Valid background / normal road image
                else:
                    for line_idx, line in enumerate(lines, 1):
                        parts = line.split()
                        if len(parts) != 5:
                            invalid_annotations.append({
                                "split": split, "file": lbl_file.name, "line": line_idx,
                                "reason": f"Expected 5 tokens, got {len(parts)}"
                            })
                            continue
                        try:
                            cid = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:])
                        except ValueError:
                            invalid_annotations.append({
                                "split": split, "file": lbl_file.name, "line": line_idx,
                                "reason": "Non-numeric token"
                            })
                            continue

                        if cid not in class_names:
                            invalid_annotations.append({
                                "split": split, "file": lbl_file.name, "line": line_idx,
                                "reason": f"Class ID {cid} not in configured classes {list(class_names.keys())}"
                            })
                            continue

                        # Check bounding box bounds [0, 1]
                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                            invalid_annotations.append({
                                "split": split, "file": lbl_file.name, "line": line_idx,
                                "reason": f"Bbox values out of [0, 1] bounds (xc={xc}, yc={yc}, w={bw}, h={bh})"
                            })
                            continue

                        # Check extent
                        if (xc - bw/2.0 < -0.01) or (xc + bw/2.0 > 1.01) or (yc - bh/2.0 < -0.01) or (yc + bh/2.0 > 1.01):
                            invalid_annotations.append({
                                "split": split, "file": lbl_file.name, "line": line_idx,
                                "reason": f"Bbox extends beyond image boundary"
                            })
                            continue

                        split_bboxes += 1
                        cname = class_names[cid]
                        split_class_counts[cname] += 1
                        class_totals[cname] += 1

            except Exception as e:
                invalid_annotations.append({"split": split, "file": lbl_file.name, "reason": str(e)})

        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "background_normal_roads": split_background,
            "bboxes": split_bboxes,
            "class_counts": split_class_counts
        }

    # Cross-Split Data Leakage Check
    print("\n[Cross-Split Data Leakage Audit]")
    train_set = set(split_hashes["train"].keys())
    val_set = set(split_hashes["val"].keys())
    test_set = set(split_hashes["test"].keys())

    leak_train_val = train_set.intersection(val_set)
    leak_train_test = train_set.intersection(test_set)
    leak_val_test = val_set.intersection(test_set)

    leakage_failures = len(leak_train_val) + len(leak_train_test) + len(leak_val_test)
    if leakage_failures == 0:
        leakage_status = "PASSED: Zero Data Leakage Detected (0 hash collisions across all splits)"
        print(f"  [PASS] {leakage_status}")
    else:
        leakage_status = f"FAILED: Data leakage found! (Train/Val: {len(leak_train_val)}, Train/Test: {len(leak_train_test)}, Val/Test: {len(leak_val_test)})"
        print(f"  [FAIL] {leakage_status}")

    # Display Report Table
    print("\n--- Dataset Split Statistics ---")
    print(f"{'Split':<10} | {'Images':<8} | {'Normal Road (Bg)':<18} | {'Hazard BBoxes':<14} | {'Labels':<8}")
    print("-" * 65)
    total_imgs = sum(split_stats[s]["images"] for s in splits)
    total_bg = sum(split_stats[s]["background_normal_roads"] for s in splits)
    total_bboxes = sum(split_stats[s]["bboxes"] for s in splits)
    for s in splits:
        st = split_stats[s]
        print(f"{s:<10} | {st['images']:<8} | {st['background_normal_roads']:<18} | {st['bboxes']:<14} | {st['labels']:<8}")
    print("-" * 65)
    print(f"{'Total':<10} | {total_imgs:<8} | {total_bg:<18} | {total_bboxes:<14} |")

    print("\n--- Hazard Class Counts ---")
    for cid, cname in class_names.items():
        print(f"  Class {cid} ({cname:<16}): {class_totals.get(cname, 0)} bounding boxes")

    print("\n--- Integrity Audit Findings ---")
    print(f"  Corrupted Images:     {len(corrupted_images)}")
    print(f"  Missing Labels:       {len(missing_labels)}")
    print(f"  Orphaned Labels:      {len(orphaned_labels)}")
    print(f"  Invalid Annotations:  {len(invalid_annotations)}")
    print(f"  Data Leakage:         {leakage_status}")
    print("=" * 75)

    # Export Reports
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "status": "PASSED" if (len(corrupted_images) == 0 and len(missing_labels) == 0 and len(invalid_annotations) == 0 and leakage_failures == 0) else "ISSUES_FOUND",
        "data_leakage": leakage_status,
        "splits": split_stats,
        "class_totals": class_totals,
        "issues": {
            "corrupted_images": corrupted_images,
            "missing_labels": missing_labels,
            "orphaned_labels": orphaned_labels,
            "invalid_annotations": invalid_annotations
        }
    }

    with open(out_dir / "dataset_integrity.json", "w", encoding="utf-8") as jf:
        json.dump(report_payload, jf, indent=2)

    with open(out_dir / "validation_report.txt", "w", encoding="utf-8") as tf:
        tf.write(f"ROAD HAZARD DATASET INTEGRITY REPORT\n")
        tf.write(f"Status: {report_payload['status']}\n")
        tf.write(f"Leakage: {leakage_status}\n")
        tf.write(f"Total Images: {total_imgs}\n")
        tf.write(f"Total BBoxes: {total_bboxes}\n")
        tf.write(f"Background Images: {total_bg}\n")
        tf.write(f"Corrupted Images: {len(corrupted_images)}\n")
        tf.write(f"Invalid Annotations: {len(invalid_annotations)}\n")

    print(f"Integrity audit reports saved to: {out_dir.resolve()}")
    return report_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit YOLO dataset integrity and cross-split data leakage.")
    parser.add_argument("--yaml", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--report-dir", type=str, default="results", help="Directory to save audit report")
    args = parser.parse_args()

    validate_dataset(args.yaml, args.report_dir)
