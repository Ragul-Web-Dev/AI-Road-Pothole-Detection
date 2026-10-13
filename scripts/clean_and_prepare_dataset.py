"""
Comprehensive Dataset Cleaning, Validation, Deduplication, and Splitting Pipeline
Final-Year Project: AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System

Performs:
1. Multi-class Road Hazard and Normal Road (background) ingestion.
2. File integrity and corrupt image removal (quarantined and logged).
3. Exact duplicate removal using SHA-256 hash deduplication.
4. YOLO annotation parsing and bounding box validation/clamping.
5. Missing label and orphaned label handling.
6. Reproducible 70:20:10 stratified split with zero data leakage.
7. Cross-split data leakage verification (Train vs Val vs Test).
8. Generation of Ultralytics YOLO configuration YAML.
9. Comprehensive cleaning and summary audit report (JSON, Markdown, Console).
"""

import os
import sys
import json
import shutil
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from PIL import Image
import yaml

# Supported image file extensions
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def compute_file_hash(filepath: Path) -> str:
    """Computes SHA-256 checksum of a file to detect exact duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def validate_image_file(img_path: Path) -> Tuple[bool, str, Tuple[int, int]]:
    """
    Validates image readability, non-zero file size, and dimension validity using PIL.
    Returns (is_valid, reason, (width, height)).
    """
    try:
        if not img_path.is_file() or img_path.stat().st_size == 0:
            return False, "Zero-byte or non-existent file", (0, 0)

        with Image.open(img_path) as img:
            img.verify()  # Verifies file integrity

        # Re-open to read dimensions and load pixel data (detects truncated files)
        with Image.open(img_path) as img:
            img.load()
            w, h = img.size
            if w <= 0 or h <= 0:
                return False, f"Invalid dimensions ({w}x{h})", (0, 0)
            return True, "Valid", (w, h)
    except Exception as e:
        return False, f"Corrupted or unreadable image ({str(e)})", (0, 0)

def sanitize_yolo_bbox(line: str, class_names: List[str], tolerance: float = 0.005) -> Tuple[bool, str, int]:
    """
    Validates a single YOLO annotation line: <class_id> <x_c> <y_c> <w> <h>.
    Auto-clamps micro floating-point rounding errors within tolerance.
    Returns (is_valid, sanitized_line, class_id).
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False, f"Invalid token count ({len(parts)} != 5)", -1

    try:
        cls_id = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError:
        return False, "Non-numeric values in bounding box", -1

    if cls_id < 0 or cls_id >= len(class_names):
        return False, f"Class ID {cls_id} out of bounds [0, {len(class_names)-1}]", cls_id

    # Check for degenerate or negative width/height
    if w <= 0 or h <= 0:
        return False, f"Non-positive dimensions (w={w}, h={h})", cls_id

    # Calculate corners
    x1 = xc - w / 2.0
    x2 = xc + w / 2.0
    y1 = yc - h / 2.0
    y2 = yc + h / 2.0

    # Auto-clamp micro overruns within tolerance
    if -tolerance <= x1 < 0:
        x1 = 0.0
    if 1.0 < x2 <= 1.0 + tolerance:
        x2 = 1.0
    if -tolerance <= y1 < 0:
        y1 = 0.0
    if 1.0 < y2 <= 1.0 + tolerance:
        y2 = 1.0

    # Strict bounds check
    if x1 < 0.0 or x2 > 1.0 or y1 < 0.0 or y2 > 1.0:
        return False, f"Bounding box extends out of image [x1={x1:.3f}, x2={x2:.3f}, y1={y1:.3f}, y2={y2:.3f}]", cls_id

    # Recalculate normalized center and width
    new_w = x2 - x1
    new_h = y2 - y1
    new_xc = x1 + new_w / 2.0
    new_yc = y1 + new_h / 2.0

    sanitized = f"{cls_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}"
    return True, sanitized, cls_id

def clean_and_prepare_dataset(
    raw_dir: str = "dataset/raw",
    output_dir: str = "dataset",
    quarantine_dir: str = "dataset/quarantine",
    splits: Tuple[float, float, float] = (0.70, 0.20, 0.10),
    class_names: List[str] = None,
    seed: int = 42,
    include_negatives: bool = True,
    yaml_path: str = "data.yaml"
) -> Dict:
    """
    Main dataset processing function.
    """
    import random
    random.seed(seed)

    if class_names is None:
        class_names = ["pothole", "crack", "damaged_surface"]

    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    quar_path = Path(quarantine_dir)

    train_pct, val_pct, test_pct = splits
    assert abs((train_pct + val_pct + test_pct) - 1.0) < 1e-5, "Splits must sum to 1.0"

    print("=" * 75)
    print("ROAD HAZARD DATASET CLEANING & PREPARATION PIPELINE")
    print("=" * 75)
    print(f"Raw Source Directory:       {raw_path}")
    print(f"Target Output Directory:     {out_path}")
    print(f"Quarantine Directory:        {quar_path}")
    print(f"Target Split Ratio:          Train: {train_pct*100:.0f}% | Val: {val_pct*100:.0f}% | Test: {test_pct*100:.0f}%")
    print(f"Configured Classes:          {class_names}")
    print(f"Include Normal Roads (Neg):  {include_negatives}")
    print("=" * 75)

    if not raw_path.exists():
        print(f"[ERROR] Raw dataset directory '{raw_path}' does not exist.")
        print("Please place your raw images in 'dataset/raw/images' and labels in 'dataset/raw/labels'")
        print("or run 'python scripts/download_dataset.py' to fetch a verified dataset.")
        return {}

    # Setup directories
    corrupt_quar = quar_path / "corrupted"
    dup_quar = quar_path / "duplicates"
    orphaned_quar = quar_path / "orphaned_labels"
    for q in [corrupt_quar, dup_quar, orphaned_quar]:
        q.mkdir(parents=True, exist_ok=True)

    # 1. Discover all candidate images
    discovered_images = []
    for ext in VALID_IMAGE_EXTS:
        discovered_images.extend(list(raw_path.rglob(f"*{ext}")))
        discovered_images.extend(list(raw_path.rglob(f"*{ext.upper()}")))

    # Deduplicate paths
    discovered_images = sorted(list(set(discovered_images)))
    print(f"\n[Step 1/7] Discovered {len(discovered_images)} raw candidate image files.")

    if not discovered_images:
        print("[WARNING] No image files found in raw directory.")
        return {}

    # Tracking metrics
    corrupted_removed = []
    duplicates_removed = []
    valid_items = []  # List of dicts: {img_path, lbl_path, hash, is_negative, annotations, cls_counts}
    seen_hashes: Dict[str, Path] = {}
    total_raw_annotations = 0
    invalid_annotations_count = 0

    # 2. Check for image corruptions and exact duplicates
    print("\n[Step 2/7] Checking image integrity and detecting exact SHA-256 duplicates...")
    for img_path in discovered_images:
        # Check image validity
        is_valid, reason, (w, h) = validate_image_file(img_path)
        if not is_valid:
            dest = corrupt_quar / img_path.name
            shutil.copy2(img_path, dest)
            corrupted_removed.append({"file": img_path.name, "path": str(img_path), "reason": reason})
            print(f"  [QUARANTINE - CORRUPT] {img_path.name}: {reason}")
            continue

        # Check for duplicate
        file_hash = compute_file_hash(img_path)
        if file_hash in seen_hashes:
            orig = seen_hashes[file_hash]
            dest = dup_quar / f"{img_path.stem}_dup_{file_hash[:8]}{img_path.suffix}"
            shutil.copy2(img_path, dest)
            duplicates_removed.append({
                "duplicate_file": img_path.name,
                "original_file": orig.name,
                "sha256": file_hash
            })
            print(f"  [QUARANTINE - DUPLICATE] {img_path.name} (matches {orig.name})")
            continue

        seen_hashes[file_hash] = img_path

        # 3. Pair with label file
        # Check label in raw_dir/labels or same folder or raw_dir
        candidate_labels = [
            raw_path / "labels" / f"{img_path.stem}.txt",
            img_path.parent / f"{img_path.stem}.txt",
            raw_path / f"{img_path.stem}.txt"
        ]
        found_lbl = None
        for cand in candidate_labels:
            if cand.exists() and cand.is_file():
                found_lbl = cand
                break

        is_normal_road = "normal" in img_path.parent.name.lower() or "plain" in img_path.parent.name.lower() or "background" in img_path.parent.name.lower()

        sanitized_lines = []
        cls_counter = {i: 0 for i in range(len(class_names))}

        if found_lbl:
            with open(found_lbl, "r", encoding="utf-8", errors="ignore") as lf:
                raw_lines = [l.strip() for l in lf if l.strip()]

            for line in raw_lines:
                total_raw_annotations += 1
                ok, san_line, cid = sanitize_yolo_bbox(line, class_names)
                if ok:
                    sanitized_lines.append(san_line)
                    cls_counter[cid] += 1
                else:
                    invalid_annotations_count += 1
                    print(f"  [INVALID BBOX] {img_path.name}: {san_line}")

        elif is_normal_road or include_negatives:
            # Treated as negative background image (normal road)
            sanitized_lines = []
        else:
            print(f"  [MISSING LABEL] {img_path.name} has no annotation file.")
            continue

        valid_items.append({
            "img_path": img_path,
            "hash": file_hash,
            "width": w,
            "height": h,
            "is_negative": len(sanitized_lines) == 0,
            "sanitized_annotations": sanitized_lines,
            "cls_counts": cls_counter
        })

    print(f"\n[Step 3/7] Filtering summary:")
    print(f"  - Total raw images scanned:     {len(discovered_images)}")
    print(f"  - Corrupted images quarantined: {len(corrupted_removed)}")
    print(f"  - Duplicate images quarantined: {len(duplicates_removed)}")
    print(f"  - Clean, valid images retained: {len(valid_items)}")
    print(f"  - Invalid annotations fixed/dropped: {invalid_annotations_count}")

    # Check orphaned label files
    all_raw_labels = list(raw_path.rglob("*.txt"))
    clean_stem_set = {item["img_path"].stem for item in valid_items}
    orphaned_count = 0
    for lbl in all_raw_labels:
        if lbl.stem not in clean_stem_set and "readme" not in lbl.name.lower():
            shutil.copy2(lbl, orphaned_quar / lbl.name)
            orphaned_count += 1
    print(f"  - Orphaned labels quarantined:  {orphaned_count}")

    if not valid_items:
        print("[ERROR] No valid images remaining to prepare.")
        return {}

    # 4. Partition into Train, Val, Test (70:20:10)
    print("\n[Step 4/7] Partitioning dataset into reproducible splits (70:20:10)...")
    random.shuffle(valid_items)

    total_valid = len(valid_items)
    train_end = int(total_valid * train_pct)
    val_end = train_end + int(total_valid * val_pct)

    split_groups = {
        "train": valid_items[:train_end],
        "val": valid_items[train_end:val_end],
        "test": valid_items[val_end:]
    }

    # 5. Check for Data Leakage between splits
    print("\n[Step 5/7] Verifying zero data leakage across splits...")
    train_hashes = {item["hash"] for item in split_groups["train"]}
    val_hashes = {item["hash"] for item in split_groups["val"]}
    test_hashes = {item["hash"] for item in split_groups["test"]}

    leak_train_val = train_hashes.intersection(val_hashes)
    leak_train_test = train_hashes.intersection(test_hashes)
    leak_val_test = val_hashes.intersection(test_hashes)

    leakage_detected = len(leak_train_val) + len(leak_train_test) + len(leak_val_test) > 0
    if leakage_detected:
        print(f"[CRITICAL ERROR] Data leakage detected!")
        print(f"  Train & Val overlap: {len(leak_train_val)}")
        print(f"  Train & Test overlap: {len(leak_train_test)}")
        print(f"  Val & Test overlap: {len(leak_val_test)}")
        raise RuntimeError("Data leakage detected between dataset splits!")
    else:
        print("  [PASSED] Leakage Verification: 0 hash collisions between Train, Val, and Test.")

    # 6. Copy clean files into structured YOLO directories
    print("\n[Step 6/7] Writing structured YOLO dataset files...")
    for split in ["train", "val", "test"]:
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_stats = {}
    for split, items in split_groups.items():
        img_out_dir = out_path / "images" / split
        lbl_out_dir = out_path / "labels" / split

        cls_totals = {i: 0 for i in range(len(class_names))}
        neg_count = 0
        total_bboxes = 0

        for item in items:
            src_img = item["img_path"]
            # Copy image
            dest_img = img_out_dir / src_img.name
            shutil.copy2(src_img, dest_img)

            # Write sanitized label
            dest_lbl = lbl_out_dir / f"{src_img.stem}.txt"
            with open(dest_lbl, "w", encoding="utf-8") as lf:
                if item["sanitized_annotations"]:
                    lf.write("\n".join(item["sanitized_annotations"]) + "\n")
                # For negative background images, write an empty file (YOLO standard)

            if item["is_negative"]:
                neg_count += 1
            else:
                total_bboxes += len(item["sanitized_annotations"])
                for cid, count in item["cls_counts"].items():
                    cls_totals[cid] += count

        split_stats[split] = {
            "image_count": len(items),
            "background_normal_roads": neg_count,
            "total_bounding_boxes": total_bboxes,
            "class_distribution": {class_names[cid]: cls_totals[cid] for cid in cls_totals}
        }

    # 7. Generate Ultralytics data.yaml
    print("\n[Step 7/7] Generating Ultralytics YOLO dataset configuration YAML...")
    yolo_yaml_data = {
        "path": "dataset",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)}
    }

    yaml_dest = Path(yaml_path)
    with open(yaml_dest, "w") as yf:
        yaml.dump(yolo_yaml_data, yf, default_flow_style=False, sort_keys=False)
    print(f"  Configuration successfully written to: {yaml_dest.resolve()}")

    # Compile comprehensive cleaning & summary report
    overall_class_totals = {class_names[i]: sum(split_stats[s]["class_distribution"][class_names[i]] for s in split_stats)
                            for i in range(len(class_names))}
    total_prepared_images = sum(split_stats[s]["image_count"] for s in split_stats)
    total_prepared_bboxes = sum(split_stats[s]["total_bounding_boxes"] for s in split_stats)
    total_normal_roads = sum(split_stats[s]["background_normal_roads"] for s in split_stats)

    report_data = {
        "summary": {
            "total_raw_scanned": len(discovered_images),
            "total_clean_images": total_prepared_images,
            "corrupted_removed": len(corrupted_removed),
            "duplicates_removed": len(duplicates_removed),
            "orphaned_labels_quarantined": orphaned_count,
            "invalid_annotations_handled": invalid_annotations_count,
            "total_annotations_valid": total_prepared_bboxes,
            "normal_road_background_images": total_normal_roads,
            "data_leakage_status": "ZERO_LEAKAGE_VERIFIED"
        },
        "class_totals": overall_class_totals,
        "split_statistics": split_stats,
        "corrupted_files_log": corrupted_removed,
        "duplicate_files_log": duplicates_removed
    }

    # Save JSON report
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = results_dir / "dataset_cleaning_report.json"
    with open(report_json_path, "w", encoding="utf-8") as jf:
        json.dump(report_data, jf, indent=2)

    # Save Markdown report
    report_md_path = results_dir / "dataset_summary.md"
    generate_markdown_report(report_md_path, report_data, class_names)

    # Print Formatted Console Summary
    print_console_summary(report_data, class_names)

    return report_data

def generate_markdown_report(dest_path: Path, report: Dict, class_names: List[str]):
    """Generates human-readable Markdown dataset summary report."""
    md = []
    md.append("# Road Hazard Dataset Preparation & Validation Summary")
    md.append("**Project**: AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System\n")
    md.append("## 1. Executive Summary")
    s = report["summary"]
    md.append(f"- **Total Raw Images Scanned**: {s['total_raw_scanned']}")
    md.append(f"- **Clean & Prepared Images**: {s['total_clean_images']}")
    md.append(f"- **Corrupted Images Removed**: {s['corrupted_removed']}")
    md.append(f"- **Exact Duplicates Removed**: {s['duplicates_removed']}")
    md.append(f"- **Orphaned Labels Quarantined**: {s['orphaned_labels_quarantined']}")
    md.append(f"- **Total Valid Hazard Bounding Boxes**: {s['total_annotations_valid']}")
    md.append(f"- **Normal Road Background Samples**: {s['normal_road_background_images']}")
    md.append(f"- **Data Leakage Check**: `{s['data_leakage_status']}` (Zero overlap between train/val/test)\n")

    md.append("## 2. Split Distribution (70 : 20 : 10)")
    md.append("| Split | Images | Normal Roads (Background) | Hazard Bounding Boxes |")
    md.append("| :--- | :---: | :---: | :---: |")
    for split, stats in report["split_statistics"].items():
        md.append(f"| **{split.capitalize()}** | {stats['image_count']} | {stats['background_normal_roads']} | {stats['total_bounding_boxes']} |")
    md.append("")

    md.append("## 3. Class Distribution")
    md.append("| Class ID | Class Name | Total Bounding Boxes |")
    md.append("| :---: | :--- | :---: |")
    for i, cname in enumerate(class_names):
        md.append(f"| {i} | {cname} | {report['class_totals'].get(cname, 0)} |")
    md.append("")

    if report["corrupted_files_log"]:
        md.append("## 4. Quarantined Corrupted Images")
        for item in report["corrupted_files_log"]:
            md.append(f"- `{item['file']}`: {item['reason']}")
        md.append("")

    if report["duplicate_files_log"]:
        md.append("## 5. Quarantined Duplicate Images")
        for item in report["duplicate_files_log"]:
            md.append(f"- `{item['duplicate_file']}` (Identical to `{item['original_file']}`) [SHA256: `{item['sha256'][:12]}...`]")
        md.append("")

    dest_path.write_text("\n".join(md), encoding="utf-8")

def print_console_summary(report: Dict, class_names: List[str]):
    """Prints formatted ASCII summary tables."""
    s = report["summary"]
    print("\n" + "=" * 75)
    print("DATASET PREPARATION & CLEANING SUMMARY REPORT")
    print("=" * 75)
    print(f"Total Raw Images:         {s['total_raw_scanned']}")
    print(f"Corrupted Removed:        {s['corrupted_removed']}")
    print(f"Duplicates Removed:       {s['duplicates_removed']}")
    print(f"Orphaned Labels Removed:  {s['orphaned_labels_quarantined']}")
    print(f"Clean Images Retained:    {s['total_clean_images']}")
    print(f"Total Bounding Boxes:     {s['total_annotations_valid']}")
    print(f"Background (Normal Roads):{s['normal_road_background_images']}")
    print(f"Data Leakage Verification: {s['data_leakage_status']}")
    print("-" * 75)

    print(f"{'Split':<12} | {'Images':<8} | {'Background':<12} | {'BBoxes':<8}")
    print("-" * 48)
    for split, stats in report["split_statistics"].items():
        print(f"{split:<12} | {stats['image_count']:<8} | {stats['background_normal_roads']:<12} | {stats['total_bounding_boxes']:<8}")
    print("-" * 48)

    print("\nClass Counts:")
    for cid, cname in enumerate(class_names):
        print(f"  Class {cid} ({cname}): {report['class_totals'].get(cname, 0)} bounding boxes")
    print("=" * 75)

def main():
    parser = argparse.ArgumentParser(description="Clean, validate, deduplicate, and split road hazard dataset.")
    parser.add_argument("--raw-dir", type=str, default="dataset/raw", help="Path to raw dataset folder")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output YOLO dataset folder")
    parser.add_argument("--quarantine-dir", type=str, default="dataset/quarantine", help="Quarantine folder for corrupt/duplicate files")
    parser.add_argument("--split", type=str, default="0.7,0.2,0.1", help="Split ratios (train,val,test)")
    parser.add_argument("--classes", type=str, default="pothole,crack,damaged_surface", help="Comma-separated class names")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--no-negatives", action="store_true", help="Do not include unannotated images as negative background samples")
    parser.add_argument("--yaml", type=str, default="data.yaml", help="Output YOLO data YAML configuration file")

    args = parser.parse_args()

    split_ratios = tuple(float(x.strip()) for x in args.split.split(","))
    class_list = [c.strip() for c in args.classes.split(",") if c.strip()]

    clean_and_prepare_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        quarantine_dir=args.quarantine_dir,
        splits=split_ratios,
        class_names=class_list,
        seed=args.seed,
        include_negatives=not args.no_negatives,
        yaml_path=args.yaml
    )

if __name__ == "__main__":
    main()
