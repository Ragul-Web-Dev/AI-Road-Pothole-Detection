"""
Inference & Detection Script - Phase 1 Research Pipeline
Processes test images or road dashcam video, performs YOLO pothole detection,
and renders visual bounding boxes with confidence scores.
"""

import argparse
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

def run_detection(source="test_images", weights="runs/pothole_baseline/weights/best.pt", conf=0.25, output_dir="results/detections"):
    source_path = Path(source)
    weights_path = Path(weights)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        # Fallback to pretrained nano model if custom weights are not yet trained
        print(f"[INFO] Custom weights '{weights}' not found. Falling back to base 'yolov8n.pt' for syntax/pipeline verification.")
        weights_path = "yolov8n.pt"

    print("=" * 60)
    print("AI-BASED POTHOLE DETECTION INFERENCE")
    print("=" * 60)
    print(f"Source:        {source}")
    print(f"Model:         {weights_path}")
    print(f"Confidence:    {conf}")
    print(f"Output Path:   {out_dir}")
    print("=" * 60)

    model = YOLO(str(weights_path))

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Case 1: Video File Input
    if source_path.is_file() and source_path.suffix.lower() in video_exts:
        print(f"Processing video: {source_path.name} ...")
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            print(f"[ERROR] Unable to open video source: {source_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_path = out_dir / f"detected_{source_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            # Run inference on frame
            results = model.predict(frame, conf=conf, verbose=False)
            
            # Draw detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    score = float(box.conf[0])
                    
                    # Bounding Box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Label
                    label = f"POTHOLE {score * 100:.1f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            out.write(frame)
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                print(f"Processed frame {frame_idx}/{total_frames}...")

        cap.release()
        out.release()
        print(f"\nAnnotated video saved to: {output_video_path}")
        return

    # Case 2: Directory or Single Image File
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = [f for f in source_path.iterdir() if f.suffix.lower() in image_exts]
    else:
        print(f"[ERROR] Source '{source}' is not a valid file or directory.")
        return

    if not image_files:
        print(f"[WARNING] No valid test images found in '{source}'.")
        print("Add .jpg or .png road images into 'test_images/' and re-run.")
        return

    print(f"Running detection on {len(image_files)} image(s)...")

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(img, conf=conf, verbose=False)
        potholes_found = 0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                potholes_found += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                score = float(box.conf[0])

                # Draw Bounding Box (Red)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Overlay Label Box
                label = f"POTHOLE {score * 100:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), (0, 0, 255), -1)
                cv2.putText(img, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out_file = out_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_file), img)
        print(f"Detected {potholes_found} pothole(s) in {img_path.name} -> Saved to {out_file}")

    print(f"\nAll detection outputs saved in: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO pothole detection on test images or video.")
    parser.add_argument("--source", type=str, default="test_images", help="Path to test images folder, single image, or .mp4 video")
    parser.add_argument("--weights", type=str, default="runs/pothole_baseline/weights/best.pt", help="Trained YOLO weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=str, default="results/detections", help="Output directory")
    args = parser.parse_args()

    run_detection(source=args.source, weights=args.weights, conf=args.conf, output_dir=args.output)
