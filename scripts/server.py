"""
Phase 3 Backend API Server for Road Pothole Detection
Provides POST /api/detect/image endpoint for live web UI inference.
Runs actual YOLO model on uploaded images, measures inference latency,
and logs empirical research results into results/phase3/image_results.csv.
"""

import argparse
import base64
import csv
import datetime
import http.server
import io
import json
import socketserver
import sys
import time
from pathlib import Path

# Optional / lazy imports for image handling
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ImportError:
    cv2 = None
    np = None
    YOLO = None

DEFAULT_WEIGHTS = "runs/pothole_baseline/weights/best.pt"
MODEL_CACHE = {}

def get_model(weights_path=DEFAULT_WEIGHTS):
    if YOLO is None:
        return None, "Ultralytics YOLO is not installed in the active Python environment."

    p = Path(weights_path)
    if not p.exists():
        # Fallback to pretrained nano model if custom weights are not yet generated
        return YOLO("yolov8n.pt"), "Warning: Using base yolov8n.pt because Phase 1 trained weights 'best.pt' were not found."

    if weights_path not in MODEL_CACHE:
        MODEL_CACHE[weights_path] = YOLO(str(p))
    return MODEL_CACHE[weights_path], None

class PotholeAPIRequestHandler(http.server.BaseHTTPRequestHandler):
    def _set_cors_headers(self, status=200, content_type="application/json"):
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Content-Type", content_type)
        self.end_headers()

    def do_OPTIONS(self):
        self._set_cors_headers(200)

    def do_GET(self):
        if self.path == "/api/health" or self.path == "/api/":
            model, warn = get_model()
            resp = {
                "status": "online",
                "service": "Pothole Detection Research API (Phase 3)",
                "model_loaded": model is not None,
                "model_notice": warn,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self._set_cors_headers(200)
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self._set_cors_headers(404)
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode("utf-8"))

    def do_POST(self):
        if self.path != "/api/detect/image":
            self._set_cors_headers(404)
            self.wfile.write(json.dumps({"error": "Unknown API endpoint"}).encode("utf-8"))
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._set_cors_headers(400)
            self.wfile.write(json.dumps({"error": "Empty request body"}).encode("utf-8"))
            return

        post_data = self.rfile.read(content_length)

        try:
            req_json = json.loads(post_data.decode("utf-8"))
        except Exception:
            self._set_cors_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON format in request."}).encode("utf-8"))
            return

        image_data_b64 = req_json.get("image", "")
        condition = req_json.get("condition", "Not specified")
        confidence_thresh = float(req_json.get("conf", 0.25))
        filename = req_json.get("filename", "uploaded_image.jpg")

        if not image_data_b64:
            self._set_cors_headers(400)
            self.wfile.write(json.dumps({"error": "No image payload provided."}).encode("utf-8"))
            return

        # Strip Data URL prefix if present (e.g. data:image/png;base64,...)
        if "," in image_data_b64:
            header, image_data_b64 = image_data_b64.split(",", 1)

        try:
            raw_bytes = base64.b64decode(image_data_b64)
        except Exception:
            self._set_cors_headers(400)
            self.wfile.write(json.dumps({"error": "Unable to decode base64 image data."}).encode("utf-8"))
            return

        if cv2 is None or np is None:
            self._set_cors_headers(500)
            self.wfile.write(json.dumps({"error": "OpenCV / NumPy dependencies missing on server."}).encode("utf-8"))
            return

        # Decode image array
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            self._set_cors_headers(400)
            self.wfile.write(json.dumps({"error": "Unable to process this image. Please upload a valid JPG, PNG, or WEBP road image."}).encode("utf-8"))
            return

        h, w = img.shape[:2]

        # Load YOLO Model
        model, model_warn = get_model()
        if model is None:
            self._set_cors_headers(500)
            self.wfile.write(json.dumps({"error": "YOLO model could not be initialized on backend."}).encode("utf-8"))
            return

        # Run Real YOLO Inference
        t_start = time.perf_counter()
        results = model.predict(img, conf=confidence_thresh, verbose=False)
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

                    # Bounding box color (Red for high conf, Orange for low)
                    box_color = (0, 0, 255) if conf_val >= 0.70 else (0, 165, 255)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)

                    # Label
                    label = f"POTHOLE {conf_val * 100:.1f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(annotated_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), box_color, -1)
                    cv2.putText(annotated_img, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    detections.append({
                        "id": idx,
                        "confidence": f"{conf_val * 100:.1f}%",
                        "confidence_val": round(conf_val, 4),
                        "bbox": [x1, y1, x2, y2]
                    })

        potholes_count = len(detections)
        if confidences:
            avg_conf = f"{(sum(confidences) / len(confidences)) * 100:.1f}%"
            highest_conf = f"{max(confidences) * 100:.1f}%"
        else:
            avg_conf = "N/A"
            highest_conf = "N/A"

        # Encode annotated image to JPEG base64 Data URL
        _, buffer = cv2.imencode(".jpg", annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")
        annotated_data_url = f"data:image/jpeg;base64,{annotated_b64}"

        # Log actual metrics to results/phase3/image_results.csv
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_path = Path("results/phase3/image_results.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = csv_path.exists() and csv_path.stat().st_size > 0

        with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Image", "Condition", "Confidence_Threshold",
                    "Potholes_Detected", "Average_Confidence", "Highest_Confidence",
                    "Inference_Time_ms", "Width", "Height"
                ])
            writer.writerow([
                timestamp_str,
                filename,
                condition,
                confidence_thresh,
                potholes_count,
                avg_conf,
                highest_conf,
                inference_time_ms,
                w,
                h
            ])

        response_payload = {
            "success": True,
            "annotated_image": annotated_data_url,
            "original_filename": filename,
            "condition": condition,
            "confidence_threshold": confidence_thresh,
            "potholes_detected": potholes_count,
            "average_confidence": avg_conf,
            "highest_confidence": highest_conf,
            "inference_time_ms": inference_time_ms,
            "image_width": w,
            "image_height": h,
            "detections": detections,
            "model_notice": model_warn
        }

        self._set_cors_headers(200)
        self.wfile.write(json.dumps(response_payload).encode("utf-8"))

def run_server(port=5000, host="127.0.0.1"):
    server_address = (host, port)
    httpd = socketserver.TCPServer(server_address, PotholeAPIRequestHandler)
    httpd.allow_reuse_address = True
    print("=" * 60)
    print("PHASE 3 ROAD POTHOLE DETECTION BACKEND API SERVER")
    print("=" * 60)
    print(f"Server Running At: http://{host}:{port}")
    print(f"Inference API:     POST http://{host}:{port}/api/detect/image")
    print(f"Health Check:      GET  http://{host}:{port}/api/health")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Backend server stopped.")
    finally:
        httpd.server_close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Phase 3 Pothole Detection Backend API Server.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind server (default: 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    args = parser.parse_args()

    run_server(port=args.port, host=args.host)
