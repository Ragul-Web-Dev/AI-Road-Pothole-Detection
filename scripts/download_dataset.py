"""
Dataset Collection and Download Script - Phase 1 ML Pipeline
AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System

Downloads verified publicly available road hazard datasets (Kaggle / Roboflow Universe)
or sets up local raw directories for potholes, cracks, damaged surfaces, and normal roads.
"""

import os
import sys
import shutil
import zipfile
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Standard verified public datasets for road hazard research
VERIFIED_KAGGLE_DATASETS = {
    "andrewmvd-potholes": {
        "dataset_ref": "andrewmvd/pothole-detection",
        "description": "Standard Pothole Detection benchmark dataset (665 annotated images, VOC XML format)",
        "format": "voc_xml",
        "default_class": "pothole"
    },
    "chitholian-potholes": {
        "dataset_ref": "chitholian/annotated-potholes-dataset",
        "description": "Annotated Potholes Image Dataset with road surface images",
        "format": "yolo_or_voc",
        "default_class": "pothole"
    },
    "lorenzoarcioni-road-damage": {
        "dataset_ref": "lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes",
        "description": "Road Damage Dataset: Potholes, Cracks, and Manholes",
        "format": "voc_or_yolo",
        "default_class": "multi"
    }
}

def convert_voc_xml_to_yolo(xml_path: Path, img_width: int, img_height: int, class_map: dict):
    """
    Parses Pascal VOC XML annotation and converts to normalized YOLO format lines.
    [class_id x_center y_center width height]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []

    # Read image size if not provided
    size_elem = root.find("size")
    if size_elem is not None:
        try:
            w_elem = size_elem.find("width")
            h_elem = size_elem.find("height")
            if w_elem is not None and int(w_elem.text) > 0:
                img_width = int(w_elem.text)
            if h_elem is not None and int(h_elem.text) > 0:
                img_height = int(h_elem.text)
        except Exception:
            pass

    if img_width <= 0 or img_height <= 0:
        return []

    for obj in root.findall("object"):
        name = obj.find("name")
        if name is None or not name.text:
            continue
        cls_name = name.text.strip().lower()

        # Map class name to ID
        cls_id = None
        for key, idx in class_map.items():
            if key in cls_name:
                cls_id = idx
                break
        if cls_id is None:
            # Default to 0 (pothole/primary hazard)
            cls_id = 0

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        try:
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
        except (ValueError, AttributeError):
            continue

        # Clamp coordinates to image boundaries
        xmin = max(0.0, min(xmin, float(img_width)))
        xmax = max(0.0, min(xmax, float(img_width)))
        ymin = max(0.0, min(ymin, float(img_height)))
        ymax = max(0.0, min(ymax, float(img_height)))

        box_w = xmax - xmin
        box_h = ymax - ymin

        if box_w <= 0 or box_h <= 0:
            continue

        x_center = (xmin + xmax) / (2.0 * img_width)
        y_center = (ymin + ymax) / (2.0 * img_height)
        norm_w = box_w / float(img_width)
        norm_h = box_h / float(img_height)

        # Micro-clamp float precision
        x_center = min(1.0, max(0.0, x_center))
        y_center = min(1.0, max(0.0, y_center))
        norm_w = min(1.0, max(0.0001, norm_w))
        norm_h = min(1.0, max(0.0001, norm_h))

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return yolo_lines

def download_kaggle_dataset(dataset_key: str, target_dir: Path, class_map: dict):
    """
    Downloads a verified dataset from Kaggle via the official Kaggle API and formats it into raw directory.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[ERROR] kaggle python package is not installed. Run: pip install kaggle")
        return False

    if dataset_key not in VERIFIED_KAGGLE_DATASETS:
        print(f"[ERROR] Unknown dataset key '{dataset_key}'.")
        print(f"Available verified keys: {list(VERIFIED_KAGGLE_DATASETS.keys())}")
        return False

    meta = VERIFIED_KAGGLE_DATASETS[dataset_key]
    dataset_ref = meta["dataset_ref"]

    print("=" * 65)
    print("ROAD HAZARD DATASET DOWNLOAD (KAGGLE VERIFIED SOURCE)")
    print("=" * 65)
    print(f"Dataset Reference: {dataset_ref}")
    print(f"Description:       {meta['description']}")
    print(f"Target Directory:  {target_dir}")
    print("-" * 65)

    download_cache = target_dir / "_download_cache"
    download_cache.mkdir(parents=True, exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()
        print(f"[1/4] Authenticated successfully with Kaggle API.")
    except Exception as e:
        print(f"[ERROR] Kaggle authentication failed: {e}")
        print("Please ensure your 'kaggle.json' API token is present in '~/.kaggle/kaggle.json'.")
        return False

    print(f"[2/4] Downloading '{dataset_ref}' archive...")
    try:
        api.dataset_download_files(dataset_ref, path=str(download_cache), unzip=True, quiet=False)
        print("[3/4] Download and extraction complete.")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset from Kaggle: {e}")
        return False

    # Organize extracted files into raw images and labels
    print("[4/4] Organizing raw images and converting annotations to YOLO format...")
    raw_images_dir = target_dir / "images"
    raw_labels_dir = target_dir / "labels"
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_labels_dir.mkdir(parents=True, exist_ok=True)

    valid_img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    copied_images = 0
    generated_labels = 0

    from PIL import Image

    for file_path in download_cache.rglob("*"):
        if not file_path.is_file():
            continue

        ext = file_path.suffix.lower()
        if ext in valid_img_exts:
            dest_img = raw_images_dir / file_path.name
            shutil.copy2(file_path, dest_img)
            copied_images += 1

            # Check if corresponding XML or TXT annotation exists
            base_name = file_path.stem
            # Check for VOC XML
            xml_candidates = list(download_cache.rglob(f"{base_name}.xml"))
            txt_candidates = list(download_cache.rglob(f"{base_name}.txt"))

            if xml_candidates:
                try:
                    with Image.open(dest_img) as img:
                        w, h = img.size
                    yolo_lines = convert_voc_xml_to_yolo(xml_candidates[0], w, h, class_map)
                    dest_lbl = raw_labels_dir / f"{base_name}.txt"
                    with open(dest_lbl, "w") as lf:
                        lf.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
                    generated_labels += 1
                except Exception as ex:
                    print(f"[WARNING] Could not process XML for {file_path.name}: {ex}")

            elif txt_candidates:
                # Copy existing YOLO TXT
                dest_lbl = raw_labels_dir / f"{base_name}.txt"
                shutil.copy2(txt_candidates[0], dest_lbl)
                generated_labels += 1

    # Cleanup temporary cache
    shutil.rmtree(download_cache, ignore_errors=True)

    print("-" * 65)
    print(f"Downloaded and structured: {copied_images} images, {generated_labels} label files.")
    print(f"Files ready in '{target_dir}'. Next step: run clean_and_prepare_dataset.py.")
    print("=" * 65)
    return True

def download_roboflow_dataset(api_key: str, workspace: str, project: str, version: int, target_dir: Path):
    """
    Downloads a dataset from Roboflow Universe using official Roboflow API.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed. Run: pip install roboflow")
        return False

    print("=" * 65)
    print("ROAD HAZARD DATASET DOWNLOAD (ROBOFLOW UNIVERSE)")
    print("=" * 65)
    print(f"Workspace: {workspace} | Project: {project} | Version: {version}")

    try:
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=str(target_dir))
        print(f"Dataset successfully downloaded into: {dataset.location}")
        return True
    except Exception as e:
        print(f"[ERROR] Roboflow download failed: {e}")
        return False

def setup_raw_template(target_dir: Path):
    """
    Creates ready-to-use template directories for manual data placement.
    """
    subdirs = [
        target_dir / "images",
        target_dir / "labels",
        target_dir / "normal_roads"  # Negative background road images
    ]
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)

    readme_path = target_dir / "RAW_DATASET_INSTRUCTIONS.md"
    readme_content = """# Raw Road Hazard Dataset Instructions

Place your raw dataset files into this directory before running `clean_and_prepare_dataset.py`:

1. `images/`: Put all road hazard images (`.jpg`, `.png`, etc.) containing potholes, cracks, or damaged road surfaces.
2. `labels/`: Put matching YOLO-format `.txt` annotation files with the same base filename (e.g. `road_01.jpg` -> `road_01.txt`).
3. `normal_roads/`: (Optional but recommended) Put images of normal roads without hazards. These will automatically be ingested as negative background samples (with empty `.txt` labels) to suppress false positives in YOLO.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"[INFO] Raw dataset folder template created at: {target_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download and ingest road hazard datasets.")
    parser.add_argument("--source", type=str, choices=["kaggle", "roboflow", "template"], default="kaggle",
                        help="Data source: kaggle, roboflow, or template setup")
    parser.add_argument("--dataset", type=str, default="andrewmvd-potholes",
                        help="Kaggle dataset preset key (andrewmvd-potholes, chitholian-potholes, lorenzoarcioni-road-damage)")
    parser.add_argument("--target-dir", type=str, default="dataset/raw",
                        help="Target raw data directory")
    # Roboflow specific args
    parser.add_argument("--roboflow-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--roboflow-workspace", type=str, default=None, help="Roboflow workspace name")
    parser.add_argument("--roboflow-project", type=str, default=None, help="Roboflow project name")
    parser.add_argument("--roboflow-version", type=int, default=1, help="Roboflow project version")

    args = parser.parse_args()
    target_path = Path(args.target_dir)

    class_map = {
        "pothole": 0,
        "crack": 1,
        "damage": 2
    }

    if args.source == "kaggle":
        success = download_kaggle_dataset(args.dataset, target_path, class_map)
        if not success:
            sys.exit(1)
    elif args.source == "roboflow":
        if not (args.roboflow_key and args.roboflow_workspace and args.roboflow_project):
            print("[ERROR] Roboflow requires --roboflow-key, --roboflow-workspace, and --roboflow-project.")
            sys.exit(1)
        download_roboflow_dataset(
            args.roboflow_key, args.roboflow_workspace, args.roboflow_project,
            args.roboflow_version, target_path
        )
    elif args.source == "template":
        setup_raw_template(target_path)

if __name__ == "__main__":
    main()
