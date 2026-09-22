"""
Phase 2 Report Generator Script
Reads results/phase2_video_metrics.csv and results/phase2_tracking_results.csv
to generate an empirical summary of dashcam performance and pothole tracking consistency.
"""

import csv
import sys
from pathlib import Path

def generate_report(metrics_csv="results/phase2_video_metrics.csv", tracking_csv="results/phase2_tracking_results.csv"):
    metrics_path = Path(metrics_csv)
    tracking_path = Path(tracking_csv)

    print("=" * 60)
    print("PHASE 2 DASHCAM RESULTS SUMMARY")
    print("=" * 60)

    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        print(f"[INFO] No video performance metrics found at '{metrics_csv}'.")
        print("Run 'python scripts/dashcam.py --source test_images/road_video.mp4' first to log actual experimental data.")
        print("=" * 60)
        return

    # Read Video Metrics
    video_metrics = []
    with open(metrics_path, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row:
                video_metrics.append(row)

    if not video_metrics:
        print("[INFO] Video metrics CSV is empty. Execute dashcam processing to populate real data.")
        print("=" * 60)
        return

    # Read Tracking Results
    tracking_records = []
    if tracking_path.exists() and tracking_path.stat().st_size > 0:
        with open(tracking_path, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row:
                    tracking_records.append(row)

    print(f"\nTotal Dashcam Video Runs Logged: {len(video_metrics)}\n")

    for v_idx, vm in enumerate(video_metrics, 1):
        v_name = vm.get("Video", "Unknown")
        avg_fps = vm.get("Average FPS", "N/A")
        total_frames = vm.get("Total Frames", "N/A")
        proc_frames = vm.get("Processed Frames", "N/A")
        proc_time = vm.get("Processing Time", "N/A")
        potholes_count = vm.get("Potholes Detected", "0")

        # Compute tracking metrics for this video
        v_tracks = [t for t in tracking_records if t.get("Video") == v_name]
        
        longest_track_frames = 0
        conf_values = []
        for t in v_tracks:
            try:
                frames_t = int(t.get("Frames_Tracked", 0))
                if frames_t > longest_track_frames:
                    longest_track_frames = frames_t
            except ValueError:
                pass
            
            avg_c_str = t.get("Average_Confidence", "").replace("%", "").strip()
            try:
                if avg_c_str:
                    conf_values.append(float(avg_c_str))
            except ValueError:
                pass

        overall_avg_conf = (sum(conf_values) / len(conf_values)) if conf_values else 0.0

        print(f"--- Run #{v_idx}: {v_name} ---")
        print(f"Video:                     {v_name}")
        print(f"Average FPS:               {avg_fps}")
        print(f"Total Video Frames:        {total_frames} (Processed: {proc_frames})")
        print(f"Total Processing Time:     {proc_time} s")
        print(f"Total Pothole Tracks:      {len(v_tracks) if v_tracks else potholes_count}")
        if conf_values:
            print(f"Average Confidence:        {overall_avg_conf:.1f}%")
        else:
            print(f"Average Confidence:        N/A")
        if longest_track_frames > 0:
            print(f"Longest Tracked Pothole:   {longest_track_frames} frames")
        else:
            print(f"Longest Tracked Pothole:   N/A")
        print("-" * 60)

    print("Note: All reported figures represent actual measured execution data.")
    print("=" * 60)

if __name__ == "__main__":
    generate_report()
