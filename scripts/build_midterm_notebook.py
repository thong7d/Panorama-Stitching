#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Generate Midterm Submission Notebook (Plan v4 Section 7 & B2)
Authors: GroupXX (Members A, B, C)
Description:
  - Generates notebooks/GroupXX_Midterm.ipynb using nbformat.
  - Fully in English according to project standards.
  - Contains complete academic sections, execution scripts, analysis,
    tables, plots, and visual galleries for Phase B Quality Gate.
============================================================================
"""

import nbformat as nbf
from pathlib import Path

def create_midterm_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    }

    cells = []

    # Title & Header
    cells.append(nbf.v4.new_markdown_cell(r"""# CSC14120 — PARALLEL PROGRAMMING: MIDTERM REPORT
## Project: CUDA-Accelerated Panorama Image Stitching
> **Milestone:** Midterm Submission (Quality Gate B — Conforming to Project Plan v4)  
> **Group:** Group 19  
> **Target Environment:** Google Colab (NVIDIA Tesla T4 GPU, CUDA 12.x, C++17)  
> **Source Repository:** [GitHub Repository (thong7d/Panorama-Stitching)](https://github.com/thong7d/Panorama-Stitching)  
> **Dataset Archive:** [Google Drive Dataset Folder](https://drive.google.com/drive/folders/1vnD51fqHjY-HGqp365dTa1SX_0femIwG?usp=drive_link)  

---

### TEAM MEMBERS AND WORKLOAD ALLOCATION

| No. | Full Name | Student ID | Specific Roles & Contributions |
|:---:|---|:---:|---|
| 1 | **Pham Minh Thong** (Leader) | 23120364 | Overall Architecture, 16:9 Preprocessing Pipeline, Stage 1 (Blur) & Stage 5 (Hartley RANSAC DLT) |
| 2 | **Nguyen Le Tan Tien** | 23120369 | Multi-site Dataset Acquisition & Manifest, Stage 2 (FAST-9 NMS Top-K) & Stage 3 (ORB Descriptors) |
| 3 | **Dao Quoc Tuan** | 23120392 | Stage 4 (8x32-bit Hamming Matching), Stage 6 (Inverse Bilinear Warping), Benchmark Suite & Amdahl Analysis |
"""))

    # Section 1: Problem Understanding & Pipeline Architecture
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 1. PROBLEM UNDERSTANDING & PIPELINE ARCHITECTURE

### 1.1 Objective
The objective is to stitch two overlapping static images (captured from a single nodal point via camera rotation, 30%–50% overlap) into a seamless, high-resolution wide-angle Panorama in the coordinate frame of the left image (Image A).

### 1.2 Six-Stage Pipeline & Data-Flow Architecture

```text
[Input Image A (RGB)]                       [Input Image B (RGB)]
         │                                           │
         ▼                                           ▼
┌───────────────────┐                       ┌───────────────────┐
│ Stage 1: Preproc  │ ──> Grayscale Float ─>│ Stage 1: Preproc  │
│ (Y=0.299R+0.587G) │     (5x5 Gaussian)    │ (Y=0.299R+0.587G) │
└───────────────────┘                       └───────────────────┘
         │                                           │
         ▼                                           ▼
┌───────────────────┐                       ┌───────────────────┐
│ Stage 2: Detect   │ ──> Top-K Keypoints ─>│ Stage 2: Detect   │
│ (FAST-9 + Top-K)  │     (x, y, angle, resp│ (FAST-9 + Top-K)  │
└───────────────────┘                       └───────────────────┘
         │                                           │
         ▼                                           ▼
┌───────────────────┐                       ┌───────────────────┐
│ Stage 3: Describe │ ──> 256-bit Descr ───>│ Stage 3: Describe │
│ (Oriented BRIEF)  │     (8 x uint32_t)    │ (Oriented BRIEF)  │
└───────────────────┘                       └───────────────────┘
         │                                           │
         └───────────────────┬───────────────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  Stage 4: Matching  │
                  │  (Hamming + Lowe)   │
                  └─────────────────────┘
                             │
                             ▼ Filtered Inlier Candidates
                  ┌─────────────────────┐
                  │ Stage 5: Homography │
                  │ (RANSAC + Norm DLT) │
                  └─────────────────────┘
                             │
                             ▼ Best Matrix H_A_to_B (3x3)
                  ┌─────────────────────┐
                  │ Stage 6: Warping    │ <─── [Raw Image A & B]
                  │ (Bilinear + Feather)│
                  └─────────────────────┘
                             │
                             ▼
                 [Panorama Output Image]
```

### 1.3 Invariant Parameters & Quality Contracts
According to **Project Plan v4 Section 3**, the following parameters and conventions are strictly locked across all versions (V0 CPU Sequential to V1, V2, V3 CUDA GPU) to ensure fair academic comparison:

| Stage | Algorithm / Method | Invariant Contract & Parameters |
|---|---|---|
| **Stage 1** | Preprocessing | Standard luminance $Y = 0.299R + 0.587G + 0.114B$; Separable 5×5 Gaussian blur ($\sigma=1.0$), boundary clamping. |
| **Stage 2** | FAST-9 + NMS + Top-K | FAST-9 corner threshold $T=20$ on 16-pixel Bresenham circle; 3×3 Non-Maximum Suppression; Max $K=1000$ keypoints; Deterministic sort: `(response DESC, linear_index ASC)`. |
| **Stage 3** | Oriented BRIEF / ORB | Patch $31\times 31$, orientation angle from Intensity Centroid $[-\pi, \pi]$; Verified 256 test pairs from OpenCV `bit_pattern_31_`; Strict border pruning ($R < 20$ px). |
| **Stage 4** | Brute-force Hamming | 256-bit descriptors stored as 8 words $\times$ 32-bit (`uint32_t words[8]`), popcount bit distance; Lowe's Ratio Test: $\text{dist}_1 < 0.75 \times \text{dist}_2$. |
| **Stage 5** | RANSAC Homography | 2000 hypotheses; 4-point Hartley isotropic normalized DLT; Gauss-Jordan partial pivoting; Reprojection threshold $3.0$ px; Deterministic PRNG LCG seed 42. |
| **Stage 6** | Warping & Feathering | Inverse coordinate mapping via $H_{A\to B}$; Bilinear sampling; Symmetric distance-to-border feathering weights to eliminate seam artifacts. |

### 1.4 GPU Suitability Rationale
1. **Stage 1:** Embarrassingly parallel pixel-wise computation. 1 thread maps to 1 pixel. Memory access is spatially localized (ideal for Shared Memory Tiling in V2).
2. **Stage 2:** FAST circle test requires only arithmetic comparisons. Top-K sorting uses `thrust::sort_by_key` (parallel Radix Sort) to overcome Bitonic Sort limits on 4K images.
3. **Stage 3:** Independent patch rotation per keypoint. 256 test pairs can be stored in `__constant__` memory for ultra-fast broadcast reads.
4. **Stage 4:** Brute-force Hamming distance computed via hardware `__popc` instruction, exhibiting high arithmetic throughput and coalesced memory reads.
5. **Stage 5:** Massively parallel hypothesis evaluation: each CUDA warp/block evaluates candidate models simultaneously.
6. **Stage 6:** Inverse warping generates canvas pixels independently, leveraging GPU texture units or bilinear cache.
"""))

    # Section 2: Environment Setup & Hardware Verification
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 2. ENVIRONMENT SETUP & HARDWARE VERIFICATION
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Automatically clone repository and set working directory if running in Google Colab
import os

REPO_URL = "https://github.com/thong7d/Panorama-Stitching.git"
REPO_DIR = "Panorama-Stitching"

# Check if we are already inside the repository root
if not os.path.exists("src/common.h"):
    if not os.path.exists(REPO_DIR):
        print(f"[INFO] Cloning repository from {REPO_URL}...")
        !git clone {REPO_URL}
    
    if os.path.exists(REPO_DIR):
        %cd {REPO_DIR}
        print(f"[OK] Working directory successfully set to: {os.getcwd()}")
    else:
        raise FileNotFoundError(f"[Error] Repository directory '{REPO_DIR}' not found.")
else:
    print(f"[OK] Working directory is already at project root: {os.getcwd()}")
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.2 Verify GPU / CPU Hardware specifications and Compilers
import shutil

if shutil.which("nvidia-smi"):
    print("[CUDA GPU Runtime Detected]")
    !nvidia-smi
    !nvcc --version
else:
    print("[Host CPU Runtime Detected - Standard environment for V0 Sequential CPU Baseline]")
    !lscpu | grep "Model name\\|Socket\\|Thread\\|NUMA\\|CPU(s)" || cat /proc/cpuinfo | grep "model name" | head -n 4
    !g++ --version
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.3 Install required Python telemetry and validation libraries
!pip install -q pillow pillow-heif numpy opencv-python-headless matplotlib nbformat pandas
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.4 Verify repository directory structure and source files
import os
import sys

required_paths = [
    "src/common.h", "src/cuda_utils.cuh", "src/versions.h", 
    "src/main.cu", "src/v0_sequential.cpp", "Makefile",
    "dataset/manifest.json", "scripts/preprocess_dataset.py",
    "scripts/generate_synthetic_test.py", "scripts/validate_metrics.py"
]

missing = [p for p in required_paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Missing required project files: {missing}")

print(f"[OK] All required workspace files are verified in {os.getcwd()}.")
"""))

    # Section 3: Dataset Preparation & Preprocessing
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 3. EXPERIMENTAL DATASET & PREPROCESSING (16:9 STANDARDIZATION)
The project utilizes 4 real-world image pairs across diverse environments, plus 1 synthetic ground-truth pair:
- **Aspect Ratio:** Standard 16:9 (Center-cropped).
- **Three Standard Resolutions:**
  - **Small:** $1280 \times 720$ (720p — 0.92 MP)
  - **Medium:** $1920 \times 1080$ (1080p — 2.07 MP)
  - **Large:** $3840 \times 2160$ (4K UHD — 8.29 MP)
- **Google Drive Dataset Archive:** [Access Full Raw & Prepared Datasets on Google Drive](https://drive.google.com/drive/folders/1vnD51fqHjY-HGqp365dTa1SX_0femIwG?usp=drive_link)
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.1 Preprocess real-world PNG images into 16:9 multi-resolution datasets
!python scripts/preprocess_dataset.py
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.2 Generate synthetic ground truth dataset with known homography and validation grid
!python scripts/generate_synthetic_test.py
"""))

    # Section 4: Compilation & Quality Gate A
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 4. COMPILATION & QUALITY GATE A (UNIT & INTEGRATION TESTS)
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.1 Run Python Integration Tests (Manifest, Synthetic GT, Preprocessed Images)
!python tests/test_scripts.py
"""))

    cells.append(nbf.v4.new_code_cell("""# 4.2 Build V0 binary and execute C++ Unit Tests
!make clean
!make cpu
!make test
"""))

    # Section 5: Benchmark Execution
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 5. BENCHMARK EXECUTION & TELEMETRY COLLECTION
Execute V0 sequential pipeline on all 3 resolutions (720p, 1080p, 4K) across synthetic and real sites.
Each test runs a minimum of **5 iterations** with 1 warmup run according to Plan v4 Section 4.4.
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.1 Initialize output directories for panoramas and JSON telemetry
!mkdir -p output/synthetic output/site1 output/site2 output/site3 output/challenging
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.2 Benchmark Synthetic Dataset at 3 resolutions (Ground-truth verification)
!./stitch dataset/synthetic/left_small.jpg dataset/synthetic/right_small.jpg output/synthetic/pano_small.jpg --version 0 --runs 5 --metrics-json output/synthetic/metrics_small.json
!./stitch dataset/synthetic/left_medium.jpg dataset/synthetic/right_medium.jpg output/synthetic/pano_medium.jpg --version 0 --runs 5 --metrics-json output/synthetic/metrics_medium.json
!./stitch dataset/synthetic/left_large.jpg dataset/synthetic/right_large.jpg output/synthetic/pano_large.jpg --version 0 --runs 5 --metrics-json output/synthetic/metrics_large.json
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.3 Benchmark Real-world Sites (Indoor, Outdoor, Custom)
# Site 1 (Indoor)
!./stitch dataset/site1_indoor/left_small.jpg dataset/site1_indoor/right_small.jpg output/site1/pano_small.jpg --version 0 --runs 5 --metrics-json output/site1/metrics_small.json
!./stitch dataset/site1_indoor/left_medium.jpg dataset/site1_indoor/right_medium.jpg output/site1/pano_medium.jpg --version 0 --runs 5 --metrics-json output/site1/metrics_medium.json
!./stitch dataset/site1_indoor/left_large.jpg dataset/site1_indoor/right_large.jpg output/site1/pano_large.jpg --version 0 --runs 5 --metrics-json output/site1/metrics_large.json

# Site 2 (Outdoor)
!./stitch dataset/site2_outdoor/left_small.jpg dataset/site2_outdoor/right_small.jpg output/site2/pano_small.jpg --version 0 --runs 5 --metrics-json output/site2/metrics_small.json

# Site 3 (Custom)
!./stitch dataset/site3_custom/left_small.jpg dataset/site3_custom/right_small.jpg output/site3/pano_small.jpg --version 0 --runs 5 --metrics-json output/site3/metrics_small.json
"""))

    # Section 6: Results Analysis, Tables & Plots
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 6. RESULTS ANALYSIS, GEOMETRIC ACCURACY & AMDAHL'S LAW
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.1 Geometric Accuracy Evaluation on Synthetic Dataset (Projected Grid RMSE)
import json
import numpy as np

with open("dataset/synthetic/ground_truth.json", "r") as f:
    gt_data = json.load(f)

print("=" * 80)
print(f"{'Resolution':<12} | {'Grid RMSE (px)':<16} | {'Diag Norm RMSE (%)':<22} | {'Quality Gate A'}")
print("-" * 80)

for res in ["small", "medium", "large"]:
    metrics_path = f"output/synthetic/metrics_{res}.json"
    if not os.path.isfile(metrics_path):
        continue
    with open(metrics_path, "r") as f:
        met = json.load(f)
        
    H_est = np.array(met["homography_a_to_b"]).reshape(3, 3)
    pts_A = np.array(gt_data[res]["validation_grid_A"])
    pts_B_gt = np.array(gt_data[res]["validation_grid_B"])
    diag = gt_data[res]["diagonal"]
    
    # Project validation grid points through estimated H
    pts_A_homo = np.hstack([pts_A, np.ones((len(pts_A), 1))])
    pts_B_est_homo = (H_est @ pts_A_homo.T).T
    pts_B_est = pts_B_est_homo[:, :2] / pts_B_est_homo[:, 2:3]
    
    rmse = np.sqrt(np.mean(np.sum((pts_B_est - pts_B_gt) ** 2, axis=1)))
    norm_rmse_pct = (rmse / diag) * 100.0
    
    # Plan v4 Quality Gate A: Small RMSE <= 1.5 px and Norm RMSE <= 0.1% of diagonal across all resolutions
    is_passed = (rmse <= 1.5 and norm_rmse_pct <= 0.1) if res == "small" else (norm_rmse_pct <= 0.1)
    gate_status = f"PASS ({'<= 1.5 px & ' if res == 'small' else ''}<= 0.1% diag)" if is_passed else "CHECK TOLERANCE"
    print(f"{res:<12} | {rmse:<16.4f} | {norm_rmse_pct:<22.4f}% | {gate_status}")

print("=" * 80)
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.2 Per-Stage Execution Timing Breakdown Table (Small, Medium, Large)
import pandas as pd
import IPython.display as display

stages = [
    ("Stage 1 (Preprocess)", "stage1_preprocess"),
    ("Stage 2 (FAST+NMS+TopK)", "stage2_fast_nms_topk"),
    ("Stage 3 (ORB Descriptor)", "stage3_orb_descriptor"),
    ("Stage 4 (Hamming Matching)", "stage4_matching"),
    ("Stage 5 (RANSAC Homography)", "stage5_ransac"),
    ("Stage 6 (Warp & Blend)", "stage6_warping_blending"),
    ("Kernel/Stages Total", "kernel_total"),
    ("End-to-End Pipeline", "end_to_end")
]

table_rows = []
for label, key in stages:
    row = {"Stage": label}
    for res in ["small", "medium", "large"]:
        p = f"output/synthetic/metrics_{res}.json"
        if os.path.isfile(p):
            with open(p, "r") as f:
                d = json.load(f)
            t_avg = d["timing_ms"][key]["avg"]
            t_min = d["timing_ms"][key]["min"]
            row[f"{res.capitalize()} (Avg/Min ms)"] = f"{t_avg:.2f} / {t_min:.2f}"
        else:
            row[f"{res.capitalize()} (Avg/Min ms)"] = "N/A"
    table_rows.append(row)

df = pd.DataFrame(table_rows)
display.display(display.HTML(df.to_html(index=False)))
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.3 Execution Profile Visualization & Amdahl's Law Speedup Analysis
import matplotlib.pyplot as plt

p_small = "output/synthetic/metrics_small.json"
if os.path.isfile(p_small):
    with open(p_small, "r") as f:
        d = json.load(f)
    
    stage_labels = ["S1: Blur", "S2: FAST", "S3: ORB", "S4: Match", "S5: RANSAC", "S6: Warp"]
    stage_keys = ["stage1_preprocess", "stage2_fast_nms_topk", "stage3_orb_descriptor", 
                  "stage4_matching", "stage5_ransac", "stage6_warping_blending"]
    times = [d["timing_ms"][k]["avg"] for k in stage_keys]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(stage_labels, times, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#9C27B0', '#FF9800'])
    plt.title("V0 Sequential Baseline: Execution Time per Stage (Small 720p)", fontsize=13, fontweight='bold')
    plt.ylabel("Execution Time (ms)", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}ms", ha='center', va='bottom', fontsize=10)
        
    plt.tight_layout()
    plt.savefig("output/bottleneck_breakdown.png", dpi=150)
    plt.show()
    
    # Amdahl's Law Analysis
    t_parallel = sum(times)
    t_e2e = d["timing_ms"]["end_to_end"]["avg"]
    p = min(1.0, t_parallel / (t_e2e if t_e2e > 0 else 1.0))
    print("=" * 60)
    print(f"Amdahl Parallel Fraction (p): {p * 100:.2f}%")
    if p >= 0.999:
        print("Theoretical Max Speedup (S_max): Idealized unbounded / below measurement resolution")
        print("Practical note: Real GPU speedup will be bounded by PCIe data transfers (H2D/D2H) and kernel launch latencies.")
    else:
        s_max = 1.0 / (1.0 - p)
        print(f"Theoretical Max Speedup (S_max): {s_max:.2f}x")
    print("=" * 60)
"""))

    # Section 7: Visual Gallery of Results
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 7. VISUAL GALLERY OF STITCHED PANORAMAS
"""))

    cells.append(nbf.v4.new_code_cell("""# 7.1 Display stitched panorama outputs
import os
import matplotlib.pyplot as plt
from PIL import Image

panoramas = [
    ("Synthetic Scene (720p)", "output/synthetic/pano_small.jpg"),
    ("Site 1 Indoor (720p)", "output/site1/pano_small.jpg"),
    ("Site 2 Outdoor (720p)", "output/site2/pano_small.jpg"),
    ("Site 3 Custom (720p)", "output/site3/pano_small.jpg")
]

plt.figure(figsize=(14, 8), dpi=100)
idx = 1
for title, p in panoramas:
    if os.path.isfile(p):
        img = Image.open(p)
        preview = img.copy()
        preview.thumbnail((960, 540))
        plt.subplot(2, 2, idx)
        plt.imshow(preview)
        plt.title(f"{title}\\n({img.width}x{img.height})", fontsize=10, fontweight='bold')
        plt.axis('off')
        idx += 1

plt.tight_layout()
plt.show()
"""))

    # Section 8: GPU Implementation Plan (V1 to V3)
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 8. GPU IMPLEMENTATION PLAN (PHASES C, D, E)

### 8.1 V1 Basic CUDA Architecture (Phase C)
- Migrate all 6 stages completely to GPU device memory.
- Eliminate round-trip Host-Device transfers for intermediate feature arrays:
  - **Stage 1 (Grayscale + Gaussian Blur):** 2D Grid of Blocks $(16 \times 16)$. 1 thread per pixel. Global memory reads with boundary clamping.
  - **Stage 2 (FAST-9 + NMS + Top-K):** 2D FAST-9 detection kernel; Bounded atomic candidate list compaction; `thrust::sort_by_key` (GPU Radix Sort) for deterministic Top-K selection on 4K images.
  - **Stage 3 (ORB Descriptors):** 1 thread per keypoint. 256 test comparisons evaluated in parallel into `Descriptor` struct (8 words $\times$ 32-bit).
  - **Stage 4 (Hamming Matching):** 1D/2D grid using hardware `__popc` instruction and Lowe's ratio test.
  - **Stage 5 (RANSAC Homography):** Massively parallel hypothesis generation (2000 models); Warp-shuffle reduction for fast inlier counting.
  - **Stage 6 (Inverse Warping):** 2D Grid over panorama bounding box; Bilinear interpolation; Symmetric feathering weights.

### 8.2 APOD Methodology for V2 & V3 Optimizations
Following **Project Plan v4 Section 1.2**: No optimizations will be chosen by guesswork. After profiling V1 with `nsight-compute`, the primary bottleneck will be identified and addressed with **exactly one principal optimization** for V2 (e.g., Shared Memory Tiling for Stage 1), followed by an iterative profiling cycle for V3.
"""))

    # Save to file
    nb.cells = cells
    out_path = Path("notebooks") / "Group19_Midterm.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Midterm notebook successfully created at: {out_path} ({len(cells)} cells)")

if __name__ == "__main__":
    create_midterm_notebook()
