# Machine Learning & Computer Vision Research Subsystem
## Real-Time Road Pothole Detection and Driver Warning System

[![Model](https://img.shields.io/badge/Architecture-YOLOv8n%20(Ultralytics)-blue.svg)](https://github.com/ultralytics/ultralytics)
[![mAP@0.50](https://img.shields.io/badge/Test%20mAP%400.50-71.27%25-brightgreen.svg)](../results/evaluation/precision_recall_curve_ieee.png)
[![Throughput](https://img.shields.io/badge/Inference%20Speed-30.9%20ms%20%7C%2027--32%20FPS%20(CPU)-success.svg)](../results/evaluation/evaluation_report.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14.1%2Bcpu-EE4C2C.svg)](https://pytorch.org/)

This subsystem encapsulates the dataset curation, transfer learning pipeline, model validation, continuous video processing, and FastAPI REST microservice powering the **Real-Time Road Pothole Detection and Driver Warning System**.

---

## 1. Verified Model Specifications

- **Model Variant**: Ultralytics YOLOv8 Nano (`YOLOv8n`)
- **Trained Weights Path**: `runs/detect/pothole_training/weights/best.pt` (6.5 MB)
- **Architecture**: 73 layers, 3,137,700 parameters (3.13M), 8.6 GFLOPs at $640 \times 640$ input
- **Pretrained Initialization**: Pretrained on MS COCO `yolov8n.pt`
- **Class Configuration**: Single-class detection: `0: pothole`
- **Runtime Framework**: Ultralytics YOLO 8.4.175 with PyTorch 2.14.1+cpu
- **Execution Target**: CPU multi-threaded execution (16 logical threads on Intel Core i7-11700F @ 2.50 GHz)

---

## 2. Test Set Evaluation Metrics

Evaluated on the independent **74-image labelled test split** (148 ground-truth annotations and 70 negative road images) using standard IoU threshold $0.50$:

| Metric | Empirical Score | Notes |
| :--- | :---: | :--- |
| **mAP@0.50** | **71.27% (0.7127)** | Primary academic benchmark metric |
| **mAP@0.50:0.95** | **45.63% (0.4563)** | Multi-threshold localization metric |
| **Optimal F1-Score** | **0.6952** | Achieved at confidence threshold $conf = 0.340$ |
| **Operational Precision ($conf=0.25$)** | **69.59% (0.6959)** | True Positives: 103 / 148 |
| **Operational Recall ($conf=0.25$)** | **69.59% (0.6959)** | False Positives: 45 \| False Negatives: 45 |
| **Inference Latency** | **33.3 – 37.4 ms** | Pre: 2.5 ms \| Model: 30.9 ms \| Post (NMS): 2.0 ms |
| **Real-Time Frame Rate** | **26.8 – 32.1 FPS** | Exceeds 25 FPS video real-time threshold on CPU |

All high-resolution evaluation curves (PR curve, F1-confidence, confusion matrix) are saved in [`../results/evaluation/`](file:///e:/React/Real/Code/MERN-Projects/final_year/results/evaluation/).

---

## 3. Dataset Engineering

The dataset combines **665 pothole hazard images** (Kaggle `chitholian/annotated-potholes-dataset`) and **70 plain road images** (Kaggle `virenbr11/pothole-and-plain-rode-images`) to prevent false-positive detections on clear pavement.

- **Total Images**: 735 clean road images
- **Total Annotations**: 1,739 bounding boxes
- **Split Distribution**:
  - `dataset/images/train/`: **514 images** (1,274 boxes) — 70%
  - `dataset/images/val/`: **147 images** (317 boxes) — 20%
  - `dataset/images/test/`: **74 images** (148 boxes) — 10%
- **Data Leakage Verification**: All splits audited via SHA-256 image hashes; zero cross-split overlap verified.

---

## 4. Subsystem Directory Structure

```
pothole-research/
├── data.yaml                               # YOLO dataset configuration
├── runs/
│   └── detect/
│       └── pothole_training/
│           ├── weights/
│           │   ├── best.pt                 # Optimal checkpoint (6.5 MB)
│           │   └── last.pt                 # Epoch 50 final checkpoint
│           ├── results.csv                 # Per-epoch training/val loss logs
│           └── confusion_matrix.png        # Training-phase validation matrix
├── dataset/
│   ├── images/                             # train/ (514), val/ (147), test/ (74)
│   ├── labels/                             # YOLO txt format bounding boxes
│   ├── day/                                # Day illumination test set
│   ├── evening/                            # Low-sun / evening test set
│   ├── night/                              # Night / headlight reflection test set
│   └── rain/                               # Wet / rain condition test set
└── scripts/
    ├── server.py                           # FastAPI REST inference service (:5000)
    ├── dashcam.py                          # Continuous video tracking & FPS benchmark
    ├── evaluate_conditions.py              # Day/Evening/Night/Rain robustness suite
    ├── clean_and_prepare_dataset.py        # Dataset validation & 70/20/10 split generator
    ├── validate_dataset_integrity.py       # SHA-256 cross-split leakage verification
    ├── train.py                            # Transfer learning script
    └── validate.py                         # Test set evaluation logger
```

---

## 5. Execution Commands

### 1. Launch FastAPI Backend Inference Server
```bash
python scripts/server.py --port 5000
```
- Swagger UI Documentation: [http://localhost:5000/docs](http://localhost:5000/docs)
- Health Probe: [http://localhost:5000/api/health](http://localhost:5000/api/health)

### 2. Run Dashcam Video Stream Tracking
```bash
python scripts/dashcam.py --source test_images/road_video.mp4 --weights runs/detect/pothole_training/weights/best.pt
```

### 3. Evaluate Environmental Robustness
```bash
# Evaluate all conditions (Day, Evening, Night, Rain):
python scripts/evaluate_conditions.py

# Evaluate a specific condition:
python scripts/evaluate_conditions.py --condition night --conf 0.25
```

### 4. Verify Dataset Integrity & Zero Cross-Split Leakage
```bash
python scripts/validate_dataset_integrity.py --yaml data.yaml --report-dir ../results
```

---

## 6. Academic References & Master Handbook

For full theoretical derivations, YOLOv8n layer breakdowns, loss functions, and viva voce defense questions, refer to the master documentation in the root directory:
- [Complete Project Technical Handbook (`PROJECT_TECHNICAL_HANDBOOK.md`)](file:///e:/React/Real/Code/MERN-Projects/final_year/PROJECT_TECHNICAL_HANDBOOK.md)
- [Project Viva Voce Examination Guide (`PROJECT_VIVA_QUESTIONS.md`)](file:///e:/React/Real/Code/MERN-Projects/final_year/PROJECT_VIVA_QUESTIONS.md)
- [Project Source Code Map (`PROJECT_SOURCE_CODE_MAP.md`)](file:///e:/React/Real/Code/MERN-Projects/final_year/PROJECT_SOURCE_CODE_MAP.md)
- [Implementation Verification Report (`PROJECT_IMPLEMENTATION_VERIFICATION_REPORT.md`)](file:///e:/React/Real/Code/MERN-Projects/final_year/PROJECT_IMPLEMENTATION_VERIFICATION_REPORT.md)
