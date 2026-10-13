import argparse
import base64
import csv
import datetime
import io
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from ultralytics import YOLO

app = FastAPI(
    title="Road Hazard Early Detection & Safe-Speed Warning API",
    description="Real-time YOLOv8n inference service for pothole detection with preliminary hazard advisories.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CANDIDATE_WEIGHTS = [
    Path("runs/detect/pothole_training/weights/best.pt"),
    Path("runs/pothole_baseline/weights/best.pt"),
    Path("pothole-research/runs/detect/pothole_training/weights/best.pt"),
    Path("pothole-research/runs/pothole_baseline/weights/best.pt"),
    Path(__file__).resolve().parent.parent / "runs" / "detect" / "pothole_training" / "weights" / "best.pt",
    Path(__file__).resolve().parent.parent / "runs" / "pothole_baseline" / "weights" / "best.pt",
]

MODEL_CACHE = {}


def resolve_model_path(custom_path: Optional[str] = None) -> tuple[Optional[Path], Optional[str]]:
    if custom_path:
        cp = Path(custom_path)
        if cp.exists():
            return cp, None
        return None, f"Specified custom weights file not found at: {custom_path}"

    for candidate in CANDIDATE_WEIGHTS:
        try:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 100000:
                return candidate.resolve(), None
        except Exception:
            continue

    return None, "No trained best.pt checkpoint found in candidate paths. Please ensure Phase 1 training has completed."


def get_model(custom_path: Optional[str] = None) -> tuple[Optional[YOLO], Optional[str], Optional[str]]:
    resolved_path, err = resolve_model_path(custom_path)
    if err or resolved_path is None:
        try:
            if "yolov8n.pt" not in MODEL_CACHE:
                MODEL_CACHE["yolov8n.pt"] = YOLO("yolov8n.pt")
            return MODEL_CACHE["yolov8n.pt"], "yolov8n.pt", f"Warning: {err} Falling back to yolov8n.pt"
        except Exception as e:
            return None, None, f"Failed to load YOLO model: {e}"

    path_str = str(resolved_path)
    if path_str not in MODEL_CACHE:
        try:
            MODEL_CACHE[path_str] = YOLO(path_str)
        except Exception as e:
            return None, path_str, f"Error loading model from {path_str}: {e}"

    return MODEL_CACHE[path_str], path_str, None


class InferenceRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image string or Data URL")
    conf: float = Field(default=0.25, ge=0.05, le=0.95, description="Confidence threshold")
    legal_speed_limit: int = Field(default=50, ge=10, le=140, description="Configured posted legal speed limit (km/h)")
    filename: Optional[str] = Field(default="uploaded_image.jpg", description="Original filename")
    condition: Optional[str] = Field(default="Not specified", description="Environmental condition tag")


class BoundingBox(BaseModel):
    id: int
    confidence: str
    confidence_val: float
    bbox: List[int]
    relative_area_pct: float


class HazardAdvisory(BaseModel):
    level: str
    alert_title: str
    alert_type: str
    alert_message: str
    legal_speed_limit_kmh: int
    advisory_disclaimer: str


class InferenceResponse(BaseModel):
    success: bool
    annotated_image: str
    original_filename: str
    confidence_threshold: float
    potholes_detected: int
    average_confidence: str
    highest_confidence: str
    inference_time_ms: float
    image_width: int
    image_height: int
    detections: List[BoundingBox]
    hazard_advisory: HazardAdvisory
    model_path: str
    model_notice: Optional[str] = None
    timestamp: str


@app.get("/")
@app.get("/api/health")
def health_check():
    model, path_str, warn = get_model()
    return {
        "status": "online",
        "service": "AI-Based Road Hazard Detection Inference API",
        "model_loaded": model is not None,
        "model_path": path_str,
        "model_classes": model.names if model else None,
        "model_notice": warn,
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.get("/api/samples")
def get_sample_list():
    sample_files = [
        {"id": "pothole_1", "name": "Test Pothole (img-104)", "rel_path": "dataset/images/test/img-104.jpg", "category": "hazard"},
        {"id": "pothole_2", "name": "Multi-Hazard (img-121)", "rel_path": "dataset/images/test/img-121.jpg", "category": "hazard"},
        {"id": "pothole_3", "name": "Severe Hazard (img-126)", "rel_path": "dataset/images/test/img-126.jpg", "category": "hazard"},
        {"id": "normal_1", "name": "Normal Road (Clean Asphalt 1)", "rel_path": "dataset/images/test/normal_road_001.jpg", "category": "normal"},
        {"id": "normal_2", "name": "Normal Road (Clean Asphalt 2)", "rel_path": "dataset/images/test/normal_road_028.jpg", "category": "normal"},
    ]

    base_dir = Path(__file__).resolve().parent.parent
    available_samples = []

    for s in sample_files:
        candidates = [
            base_dir / s["rel_path"],
            Path(s["rel_path"]),
            Path("pothole-research") / s["rel_path"],
        ]
        found_path = None
        for c in candidates:
            if c.exists() and c.is_file():
                found_path = c
                break

        if found_path:
            try:
                with open(found_path, "rb") as f:
                    b64_data = base64.b64encode(f.read()).decode("utf-8")
                available_samples.append({
                    "id": s["id"],
                    "name": s["name"],
                    "category": s["category"],
                    "filename": found_path.name,
                    "data_url": f"data:image/jpeg;base64,{b64_data}",
                })
            except Exception:
                continue

    return {"samples": available_samples}


@app.post("/api/detect/image", response_model=InferenceResponse)
def detect_image(req: InferenceRequest):
    image_b64 = req.image
    if "," in image_b64:
        _, image_b64 = image_b64.split(",", 1)

    try:
        raw_bytes = base64.b64decode(image_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64-encoded image string.")

    np_arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Please ensure valid JPG, PNG, or WEBP format.")

    h, w = img.shape[:2]

    model, path_str, warn = get_model()
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {warn}")

    t_start = time.perf_counter()
    results = model.predict(img, conf=req.conf, verbose=False)
    inference_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    detections = []
    confidences = []
    annotated_img = img.copy()

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes, 1):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf[0])
                confidences.append(conf_val)

                box_w = max(0, x2 - x1)
                box_h = max(0, y2 - y1)
                box_area = box_w * box_h
                rel_area_pct = round((box_area / float(w * h)) * 100.0, 2)

                if conf_val >= 0.70:
                    box_color = (30, 30, 220)
                elif conf_val >= 0.40:
                    box_color = (0, 140, 255)
                else:
                    box_color = (0, 215, 255)

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)

                label_text = f"POTHOLE {conf_val * 100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                lbl_y1 = max(0, y1 - th - 10)
                lbl_y2 = max(th + 10, y1)
                lbl_x1 = max(0, x1)
                lbl_x2 = min(w, x1 + tw + 10)

                cv2.rectangle(annotated_img, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), box_color, -1)
                cv2.putText(
                    annotated_img,
                    label_text,
                    (lbl_x1 + 5, lbl_y2 - 6),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

                detections.append(
                    BoundingBox(
                        id=idx,
                        confidence=f"{conf_val * 100:.1f}%",
                        confidence_val=round(conf_val, 4),
                        bbox=[x1, y1, x2, y2],
                        relative_area_pct=rel_area_pct,
                    )
                )

    potholes_count = len(detections)
    if confidences:
        avg_conf = f"{(sum(confidences) / len(confidences)) * 100:.1f}%"
        highest_conf = f"{max(confidences) * 100:.1f}%"
    else:
        avg_conf = "0.0%"
        highest_conf = "0.0%"

    if potholes_count > 0:
        hazard_level = "HAZARD_DETECTED"
        alert_title = "POTHOLE DETECTED — SLOW DOWN"
        alert_type = "WARNING"
        alert_msg = (
            f"Visual road hazard identified ({potholes_count} pothole{'s' if potholes_count > 1 else ''} detected). "
            f"Exercise caution and reduce speed below the posted legal limit of {req.legal_speed_limit} km/h."
        )
    else:
        hazard_level = "CLEAR"
        alert_title = "ROAD SURFACE CLEAR — NO HAZARDS DETECTED"
        alert_type = "SAFE"
        alert_msg = (
            f"No pothole hazards detected on the evaluated road surface at confidence threshold {req.conf:.2f}. "
            f"Normal driving speed up to the legal limit of {req.legal_speed_limit} km/h is permitted."
        )

    advisory_disclaimer = (
        "Preliminary visual advisory. Validated scientific safe-speed calculation requires real-time optical/stereo "
        "distance measurement and vehicle dynamics, evaluated during continuous video tracking."
    )

    advisory = HazardAdvisory(
        level=hazard_level,
        alert_title=alert_title,
        alert_type=alert_type,
        alert_message=alert_msg,
        legal_speed_limit_kmh=req.legal_speed_limit,
        advisory_disclaimer=advisory_disclaimer,
    )

    _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")
    annotated_data_url = f"data:image/jpeg;base64,{annotated_b64}"

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = Path("results")
    if not log_dir.exists() and Path("pothole-research/results").exists():
        log_dir = Path("pothole-research/results")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "image_inference_logs.csv"
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    try:
        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Filename", "Condition", "Legal_Speed_Limit_kmh",
                    "Confidence_Threshold", "Potholes_Detected", "Avg_Confidence",
                    "Highest_Confidence", "Inference_Time_ms", "Width", "Height", "Hazard_Alert"
                ])
            writer.writerow([
                timestamp_str,
                req.filename,
                req.condition,
                req.legal_speed_limit,
                req.conf,
                potholes_count,
                avg_conf,
                highest_conf,
                inference_time_ms,
                w,
                h,
                alert_title,
            ])
    except Exception:
        pass

    return InferenceResponse(
        success=True,
        annotated_image=annotated_data_url,
        original_filename=req.filename or "uploaded_image.jpg",
        confidence_threshold=req.conf,
        potholes_detected=potholes_count,
        average_confidence=avg_conf,
        highest_confidence=highest_conf,
        inference_time_ms=inference_time_ms,
        image_width=w,
        image_height=h,
        detections=detections,
        hazard_advisory=advisory,
        model_path=str(path_str),
        model_notice=warn,
        timestamp=timestamp_str,
    )


def run():
    parser = argparse.ArgumentParser(description="Start FastAPI Pothole Inference Server.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind server (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print("=" * 70)
    print("  ROAD HAZARD DETECTION FASTAPI INFERENCE SERVER (YOLOv8n)")
    print("=" * 70)
    print(f"  Server URL:    http://{args.host}:{args.port}")
    print(f"  API Docs:      http://{args.host}:{args.port}/docs")
    print(f"  Inference API: POST http://{args.host}:{args.port}/api/detect/image")
    print(f"  Health Check:  GET  http://{args.host}:{args.port}/api/health")
    print(f"  Sample Images: GET  http://{args.host}:{args.port}/api/samples")
    print("=" * 70)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    run()
