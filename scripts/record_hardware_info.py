"""
Hardware & Environment Profiler - Phase 2 Research Pipeline
Inspects the actual host system hardware and software versions for reproducible research reporting.
"""

import platform
import sys
import os
from pathlib import Path

def capture_hardware_info(output_file="results/hardware_info.txt"):
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    os_info = f"{platform.system()} {platform.release()} ({platform.version()})"
    cpu_info = platform.processor() or "Unknown CPU Architecture"
    cpu_count = os.cpu_count() or 1
    py_version = sys.version.replace("\n", " ")

    # Check ultralytics version
    try:
        import ultralytics
        ultralytics_ver = ultralytics.__version__
    except ImportError:
        ultralytics_ver = "Not installed in active Python environment"

    # Check OpenCV version
    try:
        import cv2
        opencv_ver = cv2.__version__
    except ImportError:
        opencv_ver = "Not installed in active Python environment"

    # Check PyTorch & GPU
    try:
        import torch
        torch_ver = torch.__version__
        gpu_avail = torch.cuda.is_available()
        if gpu_avail:
            gpu_info = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        else:
            gpu_info = "None (CPU Inference Only)"
    except ImportError:
        torch_ver = "Not installed"
        gpu_info = "Unknown (PyTorch not detected)"

    # Total RAM approximation if psutil is present
    try:
        import psutil
        ram_gb = f"{psutil.virtual_memory().total / (1024 ** 3):.2f} GB"
    except ImportError:
        ram_gb = "Available via systeminfo / Task Manager"

    content = f"""============================================================
RESEARCH HARDWARE & ENVIRONMENT PROFILE
============================================================
Operating System:    {os_info}
CPU Architecture:    {cpu_info}
Logical CPU Cores:   {cpu_count}
System Memory (RAM): {ram_gb}
GPU Acceleration:    {gpu_info}
------------------------------------------------------------
Python Version:      {py_version}
PyTorch Version:     {torch_ver}
Ultralytics Version: {ultralytics_ver}
OpenCV Version:      {opencv_ver}
============================================================
* This document ensures experimental transparency and reproducibility.
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(content)
    print(f"Hardware specifications recorded to: {output_file}")

if __name__ == "__main__":
    capture_hardware_info()
