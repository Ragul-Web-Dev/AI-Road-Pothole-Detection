# Dataset Collection, Cleaning, and Preparation Guide
**AI-Based Road Hazard Early Detection and Adaptive Safe-Speed Warning System**

---

## 1. Hazard Taxonomy & Annotation Format

To enable early road hazard detection and speed reduction warnings, the system supports three road hazard classes and negative background (normal road) samples:

| Class ID | Class Name | Description | Annotation Format |
| :---: | :--- | :--- | :--- |
| `0` | **`pothole`** | Depressions, cavities, and broken asphalt pockets | `<0> <x_c> <y_c> <w> <h>` |
| `1` | **`crack`** | Longitudinal, transverse, and alligator cracks | `<1> <x_c> <y_c> <w> <h>` |
| `2` | **`damaged_surface`** | Ravelling, surface distortion, peeling asphalt, bumps | `<2> <x_c> <y_c> <w> <h>` |
| `N/A` | **Normal Road** | Clean asphalt, road markings, shadows (False-Positive Suppressors) | Empty `.txt` file (0 bounding boxes) |

All coordinates are normalized within $[0.0, 1.0]$ relative to image dimensions:
- `x_c`: Center X coordinate divided by image width
- `y_c`: Center Y coordinate divided by image height
- `w`: Bounding box width divided by image width
- `h`: Bounding box height divided by image height

---

## 2. Public Dataset Sources

### A. Kaggle Verified Benchmarks
1. **`chitholian/annotated-potholes-dataset`**: 665 real road photos with annotated potholes.
2. **`andrewmvd/pothole-detection`**: Benchmark dataset (665 images, Pascal VOC XML format, automatically converted to YOLO format by our pipeline).
3. **`lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes`**: Multi-class road damage dataset covering potholes and road cracks.
4. **`virenbr11/pothole-and-plain-rode-images`**: Contains clean, normal road asphalt images used as negative background samples.

### B. Roboflow Universe
Public datasets on Roboflow Universe (e.g. `roboflow-100/road-damage-dataset`, `pothole-computer-vision`) can be downloaded in YOLOv8 format either directly via web export or using the Roboflow API integration in `scripts/download_dataset.py`.

---

## 3. Pipeline Architecture & Directory Layout

```
pothole-research/
├── dataset/
│   ├── raw/                        # Ingested raw datasets
│   │   ├── images/                 # Raw hazard images
│   │   ├── labels/                 # Raw annotation files (.txt or .xml)
│   │   └── normal_roads/           # Clean road images without hazards
│   ├── images/                     # Sanitized, split images
│   │   ├── train/                  # 70% Training split
│   │   ├── val/                    # 20% Validation split
│   │   └── test/                   # 10% Testing split
│   ├── labels/                     # Sanitized YOLO label files (.txt)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── quarantine/                 # Quarantined files
│   │   ├── corrupted/              # Unreadable or truncated images
│   │   ├── duplicates/             # Exact SHA-256 duplicate images
│   │   └── orphaned_labels/        # Labels without corresponding images
│   ├── day/                        # Phase 3 robustness testing
│   ├── evening/
│   ├── night/
│   └── rain/
│
├── scripts/
│   ├── download_dataset.py         # Automated dataset fetcher & VOC XML converter
│   ├── clean_and_prepare_dataset.py# End-to-end cleaning, deduplication, 70:20:10 splitter
│   ├── validate_dataset_integrity.py# Integrity & zero-leakage cross-split auditor
│   ├── check_dataset.py            # Quick dataset statistics checker
│   └── visualize_dataset.py        # Visual ground-truth bounding box inspection
│
├── results/
│   ├── dataset_cleaning_report.json# Machine-readable cleaning log
│   ├── dataset_summary.md          # Human-readable Markdown summary
│   ├── dataset_integrity.json      # Full audit results
│   ├── validation_report.txt       # Split & integrity audit log
│   └── dataset_samples/            # Visualized bounding box samples
│
├── data.yaml                       # YOLO Ultralytics dataset configuration
└── requirements.txt                # Pinned ML dependencies
```

---

## 4. Step-by-Step Execution Workflow

### Step 1: Install Dependencies
```bash
cd pothole-research
pip install -r requirements.txt
```

### Step 2: Download Verified Dataset
To download the verified Kaggle pothole benchmark dataset:
```bash
python scripts/download_dataset.py --source kaggle --dataset chitholian-potholes --target-dir dataset/raw
```

*(Optional: If using Roboflow Universe)*:
```bash
python scripts/download_dataset.py --source roboflow --roboflow-key <YOUR_KEY> --roboflow-workspace <WORKSPACE> --roboflow-project <PROJECT> --roboflow-version 1
```

*(Optional: If supplying manual images)*:
```bash
# Setup template folders
python scripts/download_dataset.py --source template
# Place images into dataset/raw/images, labels into dataset/raw/labels, normal roads into dataset/raw/normal_roads
```

### Step 3: Clean, Deduplicate, Validate, and Split (70:20:10)
Run the automated cleaning and preparation pipeline:
```bash
python scripts/clean_and_prepare_dataset.py --raw-dir dataset/raw --output-dir dataset --classes pothole --split 0.7,0.2,0.1 --seed 42 --yaml data.yaml
```

**What this step performs:**
1. **File Integrity Verification**: Uses Pillow to decode and verify headers and pixel data, catching corrupted or truncated files.
2. **SHA-256 Deduplication**: Computes cryptographic hashes for all images, quarantining identical files and logging their SHA-256 hashes.
3. **Annotation Sanitization**: Verifies bounding boxes, ensures coordinate ranges $[0, 1]$, and clamps micro floating-point rounding errors within $\pm 0.005$.
4. **Negative Sample Ingestion**: Converts normal road images into background samples with empty label files (vital for preventing false alarms during road monitoring).
5. **Leakage-Proof Splitting**: Partitions clean samples into $70\%$ Train, $20\%$ Validation, and $10\%$ Test sets using a fixed seed (`42`).
6. **Cross-Split Collision Verification**: Verifies $Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, and $Val \cap Test = \emptyset$.
7. **Config & Report Generation**: Automatically writes `data.yaml`, `results/dataset_cleaning_report.json`, and `results/dataset_summary.md`.

### Step 4: Audit Dataset Integrity & Cross-Split Leakage
Run the comprehensive verification tool:
```bash
python scripts/validate_dataset_integrity.py --yaml data.yaml --report-dir results
```
This confirms zero data leakage, validates every bounding box, and checks class balance.

### Step 5: Visual Inspection of Bounding Boxes
Inspect sample ground-truth bounding boxes overlaid on road images:
```bash
python scripts/visualize_dataset.py --split train --samples 5 --output results/dataset_samples
```
Review the resulting images in `results/dataset_samples/annotated_*.jpg`.

---

## 5. Dataset Summary & Current Baseline Metrics

From the active verified dataset:
- **Total Images**: 735 (514 Train, 147 Val, 74 Test)
- **Normal Road Background Images**: 70 (prevents false positives)
- **Total Bounding Boxes**: 1,739
- **Data Leakage**: `ZERO_LEAKAGE_VERIFIED` (0 cross-split hash collisions)
- **Corrupted / Invalid Annotations**: 0
