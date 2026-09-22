"""
Phase 2: Dashcam Video Processing & Pothole Tracking Pipeline
Integrates the Phase 1 trained YOLO model with dashcam video/webcam feeds,
performs frame-by-frame pothole tracking, real-time FPS benchmarking, and visual hazard warnings.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

def process_dashcam(
    source="test_images/road_video.mp4",
    weights="runs/pothole_baseline/weights/best.pt",
    conf=0.25,
    skip=0,
    output_dir="results/phase2_videos",
    metrics_csv="results/phase2_video_metrics.csv",
    tracking_csv="results/phase2_tracking_results.csv",
    show_preview=False
):
    weights_path = Path(weights)
    if not weights_path.exists():
        print("=" * 60)
        print("[ERROR] Phase 1 trained model is required before Phase 2.")
        print(f"Target model weights not found at: {weights_path}")
        print("Please train your baseline model first:")
        print("  python scripts/train.py --epochs 20")
        print("=" * 60)
        return

    # Check source: Webcam integer or video file
    is_webcam = False
    source_str = str(source).strip()
    if source_str.isdigit() or source == 0:
        cap_source = int(source_str)
        is_webcam = True
        video_name = f"webcam_{cap_source}"
    else:
        source_file = Path(source)
        if not source_file.exists():
            print(f"[ERROR] Source video file '{source}' does not exist.")
            print("Place a test road video (.mp4) in 'test_images/' and re-run.")
            return
        cap_source = str(source_file)
        video_name = source_file.name

    print("=" * 60)
    print("PHASE 2 DASHCAM VIDEO PROCESSING & POTHOLE TRACKING")
    print("=" * 60)
    print(f"Video Source:       {source}")
    print(f"Model Weights:      {weights_path}")
    print(f"Confidence Thresh:  {conf}")
    print(f"Frame Skipping:     {skip} (Processes 1 in every {skip + 1} frames)")
    print(f"Output Directory:   {output_dir}")
    print("=" * 60)

    # Initialize YOLO model with built-in tracking
    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video stream: {cap_source}")
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_video_path = out_dir_path / f"tracked_{Path(video_name).stem}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_video_path), fourcc, orig_fps, (width, height))

    # In-memory tracking history registry
    # id -> { "first_frame": int, "last_frame": int, "frames_tracked": int, "confidences": [] }
    tracked_potholes = {}
    
    frame_idx = 0
    processed_count = 0
    start_total_time = time.perf_counter()
    processing_times = []

    print(f"Beginning video analysis on {video_name} ({width}x{height} @ {orig_fps:.1f} FPS)...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1

            # Handle frame skipping
            if skip > 0 and (frame_idx - 1) % (skip + 1) != 0:
                # Write unannotated frame or skip write based on pipeline
                out_writer.write(frame)
                continue

            processed_count += 1
            frame_start = time.perf_counter()

            # Execute YOLO tracking on current frame
            # Uses built-in ByteTrack tracker
            results = model.track(frame, persist=True, conf=conf, verbose=False)
            
            frame_elapsed = time.perf_counter() - frame_start
            processing_times.append(frame_elapsed)
            current_fps = 1.0 / frame_elapsed if frame_elapsed > 0 else 0.0

            active_detections_in_frame = 0
            max_conf_in_frame = 0.0

            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        active_detections_in_frame += 1
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf_score = float(box.conf[0])
                        if conf_score > max_conf_in_frame:
                            max_conf_in_frame = conf_score

                        # Retrieve or assign Track ID
                        track_id = int(box.id[0]) if (box.id is not None) else (active_detections_in_frame)

                        # Update tracking registry
                        if track_id not in tracked_potholes:
                            tracked_potholes[track_id] = {
                                "first_frame": frame_idx,
                                "last_frame": frame_idx,
                                "frames_tracked": 1,
                                "confidences": [conf_score],
                                "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                            }
                        else:
                            tracked_potholes[track_id]["last_frame"] = frame_idx
                            tracked_potholes[track_id]["frames_tracked"] += 1
                            tracked_potholes[track_id]["confidences"].append(conf_score)
                            tracked_potholes[track_id]["center"] = ((x1 + x2) // 2, (y1 + y2) // 2)

                        # Draw Pothole Bounding Box
                        box_color = (0, 0, 255) if conf_score >= 0.70 else (0, 165, 255) # Red for high conf, Orange for moderate
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                        # Bounding Box Label
                        label = f"Pothole #{track_id} | {conf_score * 100:.1f}%"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(0, y1)), box_color, -1)
                        cv2.putText(frame, label, (x1 + 3, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Draw HUD Overlay Banner
            # Top dark translucent panel
            cv2.rectangle(frame, (0, 0), (width, 48), (15, 23, 42), -1)

            # 1. FPS Display (Left)
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(frame, fps_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # 2. Real-Time Warning Banner (Center)
            if active_detections_in_frame > 0:
                if max_conf_in_frame >= 0.70:
                    warning_text = "[!] WARNING: POTHOLE DETECTED (HIGH CONFIDENCE)"
                    warn_color = (0, 0, 255) # Red
                else:
                    warning_text = "[!] WARNING: POTHOLE DETECTED (LOW CONFIDENCE)"
                    warn_color = (0, 165, 255) # Orange
            else:
                warning_text = "[OK] ROAD CLEAR - NO POTHOLE HAZARD"
                warn_color = (0, 255, 128) # Green

            (w_w, _), _ = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, warning_text, ((width - w_w) // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, warn_color, 2)

            # 3. Frame & Track Counter (Right)
            count_text = f"Frame: {frame_idx}" + (f"/{total_frames}" if total_frames > 0 else "") + f" | Tracks: {len(tracked_potholes)}"
            (c_w, _), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(frame, count_text, (width - c_w - 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            out_writer.write(frame)

            if show_preview:
                cv2.imshow("Phase 2 Dashcam Pothole Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] Stopped by user (q pressed).")
                    break

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                print(f"Processed frame {frame_idx}" + (f"/{total_frames}" if total_frames > 0 else "") + f" | Tracks: {len(tracked_potholes)} | Speed: {current_fps:.1f} FPS")

    finally:
        cap.release()
        out_writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    total_processing_time = time.perf_counter() - start_total_time
    avg_fps = (processed_count / sum(processing_times)) if processing_times else 0.0

    print("\n" + "=" * 60)
    print("PHASE 2 DASHCAM PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Source Video:             {video_name}")
    print(f"Total Video Frames:       {frame_idx}")
    print(f"Processed Frames:         {processed_count}")
    print(f"Total Processing Time:    {total_processing_time:.2f} s")
    print(f"Average Processing Speed: {avg_fps:.2f} FPS")
    print(f"Unique Potholes Tracked:  {len(tracked_potholes)}")
    print(f"Output Video Saved:       {out_video_path}")
    print("=" * 60)

    # Save results to Video Metrics CSV
    metrics_path = Path(metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_exists = metrics_path.exists() and metrics_path.stat().st_size > 0

    with open(metrics_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not metrics_exists:
            writer.writerow(["Video", "Total Frames", "Processed Frames", "Processing Time", "Average FPS", "Potholes Detected"])
        writer.writerow([
            video_name,
            frame_idx,
            processed_count,
            f"{total_processing_time:.2f}",
            f"{avg_fps:.2f}",
            len(tracked_potholes)
        ])
    print(f"Logged video metrics to: {metrics_csv}")

    # Save results to Tracking Results CSV
    tracking_path = Path(tracking_csv)
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_exists = tracking_path.exists() and tracking_path.stat().st_size > 0

    with open(tracking_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not tracking_exists:
            writer.writerow(["Video", "Pothole_ID", "First_Frame", "Last_Frame", "Frames_Tracked", "Average_Confidence", "Max_Confidence"])
        
        for pid, pdata in tracked_potholes.items():
            avg_c = sum(pdata["confidences"]) / len(pdata["confidences"])
            max_c = max(pdata["confidences"])
            writer.writerow([
                video_name,
                pid,
                pdata["first_frame"],
                pdata["last_frame"],
                pdata["frames_tracked"],
                f"{avg_c * 100:.1f}%",
                f"{max_c * 100:.1f}%"
            ])
    print(f"Logged individual tracking records to: {tracking_csv}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: Dashcam Video Pothole Tracking and Real-Time Hazard Warning.")
    parser.add_argument("--source", type=str, default="test_images/road_video.mp4", help="Path to road dashcam video or '0' for webcam")
    parser.add_argument("--weights", type=str, default="runs/pothole_baseline/weights/best.pt", help="Phase 1 trained model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0.0 - 1.0)")
    parser.add_argument("--skip", type=int, default=0, help="Frame skipping interval (0 = every frame, 1 = every 2nd frame)")
    parser.add_argument("--output", type=str, default="results/phase2_videos", help="Directory to save output video")
    parser.add_argument("--metrics-csv", type=str, default="results/phase2_video_metrics.csv", help="CSV path for video summary metrics")
    parser.add_argument("--tracking-csv", type=str, default="results/phase2_tracking_results.csv", help="CSV path for per-track metrics")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window during processing")
    
    args = parser.parse_args()

    process_dashcam(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        skip=args.skip,
        output_dir=args.output,
        metrics_csv=args.metrics_csv,
        tracking_csv=args.tracking_csv,
        show_preview=args.show
    )
