# AI-Based Road Pothole Detection and Safe Path Identification System
**Research Phase 3: Environmental Robustness Evaluation (Day / Evening / Night / Rain)**

---

## 1. Project Title
**AI-Based Road Pothole Detection and Safe Path Identification System**  
*Phase 3: Multi-Environmental Illumination and Weather Condition Robustness Evaluation*

---

## 2. Research Objective
- **Phase 1 Objective**: Baseline 2D pothole detection accuracy using lightweight YOLO on standardized road images.
- **Phase 2 Objective**: Continuous dashcam video processing, temporal pothole tracking, and real-time FPS benchmarking.
- **Phase 3 Objective**: Evaluate the environmental robustness of the baseline pothole detection model across diverse real-world illumination and weather conditions:
  > *"How robust is the dashcam-based pothole detection system under different illumination and weather conditions (Day, Evening, Night, Rain) when evaluated with uniform model weights and hyperparameters?"*

---

## 3. Experimental Principle & Fairness Protocol
To ensure scientific validity and unbiased comparisons:
1. **Identical Model**: The exact same Phase 1 trained YOLO weights (`runs/pothole_baseline/weights/best.pt`) are evaluated across all four conditions without condition-specific fine-tuning.
2. **Identical Hyperparameters**: Image resolution ($640\times 640$) and baseline confidence threshold ($\text{conf}=0.25$) remain strictly uniform across all test groups.
3. **Isolated Preprocessing Experiments**: Low-light enhancements (e.g., CLAHE, Gamma) are evaluated in separate experimental pipelines and never substituted for original baseline data.
4. **No Metric Fabrication**: If ground-truth annotations are not provided for a condition, metrics such as precision and mAP are logged as `N/A`, reporting only true measured inference counts, confidence distributions, and FPS.

---

## 4. Directory Layout
```
pothole-research/
├── dataset/
│   ├── images/             # Phase 1 dataset splits (train, val, test)
│   ├── labels/             # Phase 1 YOLO format txt labels
│   ├── day/                # Day condition road test images/videos
│   ├── evening/            # Evening/low-sun condition road test images/videos
│   ├── night/              # Nighttime/headlight road test images/videos
│   └── rain/               # Wet/rainy asphalt road test images/videos
│
├── test_images/            # General test media
├── runs/                   # Model training runs and weights
├── scripts/
│   ├── download_dataset.py         # Phase 1: Automated dataset downloader & VOC to YOLO converter
│   ├── clean_and_prepare_dataset.py# Phase 1: Image validation, SHA-256 deduplication & 70:20:10 splitter
│   ├── validate_dataset_integrity.py# Phase 1: Cross-split leakage auditor & bbox validator
│   ├── check_dataset.py            # Phase 1: Quick dataset integrity & split statistics checker
│   ├── visualize_dataset.py        # Phase 1: Ground-truth visual verification
│   ├── train.py                    # Phase 1: Baseline YOLO transfer learning training
│   ├── validate.py                 # Phase 1: Validation evaluation & metrics logger
│   ├── detect.py           # Phase 1: Basic image/video inference
│   ├── dashcam.py          # Phase 2: Dashcam tracking, FPS benchmarking & HUD warnings
│   ├── phase2_report.py    # Phase 2: Dashcam tracking summary generator
│   ├── record_hardware_info.py # Phase 2: Host hardware profiler for reproducibility
│   ├── evaluate_conditions.py  # Phase 3: Environmental condition evaluator
│   ├── create_condition_comparison.py # Phase 3: Cross-condition figure generator
│   ├── low_light_preprocessing.py     # Phase 3: Optional enhancement experiment
│   ├── plot_phase3_results.py         # Phase 3: Matplotlib chart generator
│   ├── generate_phase3_report.py      # Phase 3: Robustness research report generator
│   └── server.py                      # Phase 3: Live backend API server (POST /api/detect/image)
│
├── results/
│   ├── dataset_samples/    # Bounding-box verification sample outputs
│   ├── detections/         # Static test detections
│   ├── phase2_videos/      # Phase 2 annotated output dashcam videos
│   ├── hardware_info.txt   # Phase 2 host CPU/GPU/OS/framework profiling
│   └── phase3/
│       ├── day/            # Annotated Day detection images
│       ├── evening/        # Annotated Evening detection images
│       ├── night/          # Annotated Night detection images
│       ├── rain/           # Annotated Rain detection images
│       ├── videos/         # Condition-evaluated output videos
│       ├── comparison/     # Multi-condition comparison grid figures
│       ├── plots/          # Matplotlib research charts (Confidence, FPS, Metrics)
│       ├── condition_metrics.csv    # Detailed confidence distributions & FPS
│       ├── condition_comparison.csv # Cross-condition comparison summary table
│       ├── image_results.csv        # Live UI inference research log
│       └── phase3_report.txt        # Full research robustness report
│
├── data.yaml               # YOLO dataset configuration
├── requirements.txt        # Python dependencies
├── DATASET_PREPARATION_GUIDE.md # Detailed data collection, cleaning & audit guide
└── README.md
```

---

## 5. Dataset Preparation Pipeline (Phase 1)

For comprehensive instructions, see [DATASET_PREPARATION_GUIDE.md](file:///e:/React/Real/Code/MERN-Projects/final_year/pothole-research/DATASET_PREPARATION_GUIDE.md).

### Quick Commands:
```bash
# 1. Download verified public dataset (Kaggle benchmark)
python scripts/download_dataset.py --source kaggle --dataset chitholian-potholes --target-dir dataset/raw

# 2. Clean, deduplicate (SHA-256), validate bounding boxes & split (70:20:10)
python scripts/clean_and_prepare_dataset.py --raw-dir dataset/raw --output-dir dataset --classes pothole --split 0.7,0.2,0.1 --seed 42 --yaml data.yaml

# 3. Verify dataset integrity & zero cross-split data leakage
python scripts/validate_dataset_integrity.py --yaml data.yaml --report-dir results

# 4. Visually verify sample bounding boxes
python scripts/visualize_dataset.py --split train --samples 5
```

---

## 6. How to Run Phase 3

### Step 1: Place Test Data in Condition Folders
Add test road images (`.jpg`, `.png`) or video files (`.mp4`) into:
- `dataset/day/`
- `dataset/evening/`
- `dataset/night/`
- `dataset/rain/`

### Step 2: Evaluate Environmental Robustness
```bash
# Evaluate all available conditions:
python scripts/evaluate_conditions.py

# Evaluate a specific condition (e.g., night or rain):
python scripts/evaluate_conditions.py --condition night
python scripts/evaluate_conditions.py --condition rain --conf 0.25

# Evaluate a specific video file under a condition:
python scripts/evaluate_conditions.py --condition night --source dataset/night/night_dashcam.mp4
```

### Step 3: (Optional) Test Low-Light Enhancement Experiment
```bash
# Generate CLAHE-enhanced test images in a separate folder:
python scripts/low_light_preprocessing.py --input dataset/night --output dataset/night_enhanced --method clahe

# Evaluate the enhanced images independently:
python scripts/evaluate_conditions.py --condition night --source dataset/night_enhanced
```

### Step 4: Generate Side-by-Side Comparison Figures
```bash
python scripts/create_condition_comparison.py
```
*Saves 4-way comparison grid to `results/phase3/comparison/`.*

### Step 5: Generate Research Plots & Charts
```bash
python scripts/plot_phase3_results.py
```
*Saves publication-quality plots (Confidence, FPS, Metrics) to `results/phase3/plots/`.*

### Step 6: Generate Robustness Summary Report
```bash
python scripts/generate_phase3_report.py
```
*Generates `results/phase3/phase3_report.txt`.*

### Step 7: Launch Live Image Upload & Result View Web Interface
```bash
# Terminal 1: Start the Python Backend API Server (Port 5000)
cd pothole-research
python scripts/server.py

# Terminal 2: Start the React Frontend Application (Port 5173)
cd ..
npm run dev
```
Open `http://localhost:5173` in your web browser. Upload any road image, select test metadata (`Day`, `Evening`, `Night`, `Rain`), adjust confidence threshold, and click **[ Detect Potholes ]**. Research outputs and latency will be automatically saved to `results/phase3/image_results.csv`.

---

## 6. Current Limitations (Phase 3 Scope)
- ❌ **Depth & Spatial Geometry**: 2D bounding boxes under rain/night do not determine puddle depth or true physical hazard severity.
- ❌ **Safe-Path Recommendation**: Dynamic obstacle avoidance paths belong to Phase 6.
- ❌ **Hardware/Sensor Differences**: Camera ISO settings, headlight quality, and rain wiper interference vary across dashcams.

---

## 7. Research Roadmap

```
PHASE 1
Baseline YOLO Detection
✓ Completed

      ↓

PHASE 2
Dashcam Video + Pothole Tracking
✓ Completed

      ↓

PHASE 3
Environmental Robustness (DAY / EVENING / NIGHT / RAIN)
← CURRENT PHASE

      ↓

PHASE 4
Severity Index & Dynamic Risk Assessment Formulation

      ↓

PHASE 5
Calibrated Dimension and Depth Estimation

      ↓

PHASE 6
Drivable-Area Segmentation & Visible Safe-Region Recommendation

      ↓

PHASE 7
Embedded Edge Hardware & Real-World In-Vehicle Testing

      ↓

PHASE 8
Comprehensive Empirical Evaluation & IEEE Research Paper Submission
```
