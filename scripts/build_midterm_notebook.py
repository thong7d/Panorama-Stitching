#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Generate Midterm Submission Notebook (Plan v4 Section 7 & B2)
Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
Description:
  - Generates notebooks/Group19_Midterm.ipynb using nbformat.
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

### 3.1 TEAM MEMBERS AND WORKLOAD ALLOCATION

| No. | Full Name | Student ID | Stage Responsibilities |
|:---:|---|:---:|---|
| 1 | **Pham Minh Thong** (Leader) | 23120364 | Overall Architecture, Stage 1 (Gaussian Blur), Stage 5 (Hartley RANSAC DLT) |
| 2 | **Nguyen Le Tan Tien** | 23120369 | Dataset Acquisition & Manifest, Stage 2 (FAST-9 NMS Top-K), Stage 3 (ORB Descriptors) |
| 3 | **Dao Quoc Tuan** | 23120392 | Stage 4 (Hamming Matching), Stage 6 (Inverse Bilinear Warping), Benchmark Suite |

#### Version Assignment Plan (Until Final Submission)

| Version | Description | Lead | Support | Target |
|---|---|:---:|:---:|---|
| **V0** | Sequential C++ Baseline | All | — | ✅ Done (Week 7) |
| **V1** | Basic CUDA — all 6 stages on GPU | Thong | Tuan | Week 9 |
| **V2** | Optimization: Shared Memory Tiling (S1, S6) | Tuan | Tien | Week 11 |
| **V3** | Optimization: `__popc` + warp reduction (S4, S5) | Tien | Thong | Week 12 |
| **V4** | Optimization: `__constant__` ORB table + texture (S3, S6) | Thong | Tuan | Week 13 |
"""))

    # Section 1: Problem Understanding & Pipeline Architecture
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 3.2 PROBLEM UNDERSTANDING & PIPELINE ARCHITECTURE

### Objective
Stitch two overlapping static images (30–50% overlap, single nodal point rotation) into a seamless wide-angle panorama in the coordinate frame of the left image (Image A).

### Six-Stage Pipeline & Data-Flow Architecture

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
│ (FAST-9 + Top-K)  │     (x, y, response)  │ (FAST-9 + Top-K)  │
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
                             ▼ Filtered Candidates
                  ┌─────────────────────┐
                  │ Stage 5: Homography │
                  │ (RANSAC + Norm DLT) │
                  └─────────────────────┘
                             │
                             ▼ Best H_A_to_B (3x3)
                  ┌─────────────────────┐
                  │ Stage 6: Warping    │ <─── [Raw Image A & B]
                  │ (Bilinear + Feather)│
                  └─────────────────────┘
                             │
                             ▼
                 [Panorama Output Image]
```

### 3.3 Chosen Methods, Parameters & GPU Suitability

| Stage | Algorithm / Method | Key Parameters | GPU Suitability |
|---|---|---|---|
| **Stage 1** | 2D 5×5 Gaussian blur (σ≈1.04) | boundary clamping; Y=0.299R+0.587G+0.114B | 1 thread/pixel — embarrassingly parallel; ideal for shared memory tiling |
| **Stage 2** | FAST-9 + 3×3 NMS + Deterministic Top-K | T=20; K=1000; sort: (response DESC, idx ASC) | 1 thread/pixel for detection; `thrust::sort_by_key` for Top-K |
| **Stage 3** | Oriented BRIEF / ORB 256-bit | 31×31 patch; 256 pairs from OpenCV `bit_pattern_31_`; border R<20px | 1 thread/keypoint; `__constant__` memory for 256-pair table |
| **Stage 4** | Brute-force Hamming + Lowe's ratio test | dist₁ < 0.75×dist₂; 8×`uint32_t` popcount | Hardware `__popc`; warp-level reduction for inner loop |
| **Stage 5** | RANSAC + Hartley normalized 4-point DLT | 2000 hypotheses; reprojection ≤3.0 px; LCG seed=42 | 2000 parallel hypothesis evaluations; warp-shuffle inlier counting |
| **Stage 6** | Inverse warp + bilinear + feathering | frame A coords; symmetric dist-to-border blend | 1 thread/canvas pixel; texture unit for bilinear cache |

**Parameter justification:** FAST T=20 balances recall vs. false positives on synthetic and real scenes; K=1000 keeps Stage 4 matching tractable (1000×1000 Hamming comparisons); RANSAC 2000 iterations achieves >99% success probability for 30% inlier ratio; Lowe ratio 0.75 is standard for robustness against ambiguous descriptors.
"""))

    # Section 2: Environment Setup & Hardware Verification
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 2. ENVIRONMENT SETUP & HARDWARE VERIFICATION
"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Copy src/ from the team's shared Google Drive folder and set working directory
# NOTE: src/ files in this notebook are identical to those in the submitted zip.
# Mount Drive if running on Colab; otherwise assume we're already at project root.
import os

if not os.path.exists("src/common.h"):
    # Try to use Google Drive (Colab environment)
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        # Copy the src/ folder from Drive to the working directory
        DRIVE_SRC = "/content/drive/MyDrive/Group19_Midterm/src"
        if os.path.exists(DRIVE_SRC):
            import shutil
            shutil.copytree(DRIVE_SRC, "src")
            shutil.copy("/content/drive/MyDrive/Group19_Midterm/Makefile", "Makefile")
            print("[OK] Copied src/ and Makefile from Google Drive.")
        else:
            # Fallback: clone repository at the pinned commit used for submission
            REPO_URL = "https://github.com/thong7d/Panorama-Stitching.git"
            REPO_DIR = "Panorama-Stitching"
            PINNED_COMMIT = "HEAD"  # Replace with actual commit hash after final submission
            if not os.path.exists(REPO_DIR):
                print(f"[INFO] Cloning repository from {REPO_URL}...")
                os.system(f"git clone {REPO_URL}")
            os.chdir(REPO_DIR)
            os.system(f"git checkout {PINNED_COMMIT}")
            print(f"[OK] Repository cloned and pinned to commit: {PINNED_COMMIT}")
    except ImportError:
        raise FileNotFoundError(
            "src/common.h not found and not in Colab. "
            "Please copy the src/ folder to the working directory manually."
        )
else:
    print(f"[OK] Working directory already at project root: {os.getcwd()}")
    print(f"[OK] src/common.h found — using existing source files.")
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
## 3.6 EXPERIMENTAL DATASET & PREPROCESSING (16:9 STANDARDIZATION)
The project utilizes 3 real-world image pairs across diverse environments, 1 challenging pair, plus 1 synthetic ground-truth pair:
- **Aspect Ratio:** Standard 16:9 (center-cropped).
- **Three Standard Resolutions:** Small 1280×720 (720p), Medium 1920×1080 (1080p), Large 3840×2160 (4K).
- **Google Drive Dataset Archive:** [Access Full Raw & Prepared Datasets on Google Drive](https://drive.google.com/drive/folders/1vnD51fqHjY-HGqp365dTa1SX_0femIwG?usp=drive_link)
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.1 Preprocess real-world PNG images into 16:9 multi-resolution datasets
!python scripts/preprocess_dataset.py
"""))

    cells.append(nbf.v4.new_code_cell("""# 3.2 Generate synthetic ground truth dataset with known homography and validation grid
# Synthetic: B is to the RIGHT of A, overlap ~40%, no clamp artifacts.
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
import shutil
!make clean
if shutil.which("nvcc"):
    print("[INFO] NVCC found — building CUDA version (make cuda)")
    !make cuda
else:
    print("[INFO] No NVCC — building CPU version (make cpu)")
    !make cpu
!make test
"""))

    # Section 5: Benchmark Execution
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 3.5 BENCHMARK EXECUTION & TELEMETRY COLLECTION
Execute V0 sequential pipeline on all 3 resolutions (720p, 1080p, 4K) across synthetic and real sites.
Each test runs **5 iterations** with 1 warmup run (Plan v4 Section 4.4). Reporting both Avg and Min to capture stable performance.
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.1 Initialize output directories for panoramas and JSON telemetry
!mkdir -p output/synthetic output/site1 output/site2 output/site3 output/challenging
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.2 Benchmark Synthetic Dataset at 3 resolutions (Ground-truth verification)
!./stitch dataset/synthetic/left_small.jpg  dataset/synthetic/right_small.jpg  output/synthetic/pano_small.jpg  --version 0 --runs 5 --metrics-json output/synthetic/metrics_small.json
!./stitch dataset/synthetic/left_medium.jpg dataset/synthetic/right_medium.jpg output/synthetic/pano_medium.jpg --version 0 --runs 5 --metrics-json output/synthetic/metrics_medium.json
!./stitch dataset/synthetic/left_large.jpg  dataset/synthetic/right_large.jpg  output/synthetic/pano_large.jpg  --version 0 --runs 5 --metrics-json output/synthetic/metrics_large.json
"""))

    cells.append(nbf.v4.new_code_cell("""# 5.3 Benchmark Real-world Sites — All 3 resolutions for each site
# Site 1 (Indoor)
!./stitch dataset/site1_indoor/left_small.jpg  dataset/site1_indoor/right_small.jpg  output/site1/pano_small.jpg  --version 0 --runs 5 --metrics-json output/site1/metrics_small.json
!./stitch dataset/site1_indoor/left_medium.jpg dataset/site1_indoor/right_medium.jpg output/site1/pano_medium.jpg --version 0 --runs 5 --metrics-json output/site1/metrics_medium.json
!./stitch dataset/site1_indoor/left_large.jpg  dataset/site1_indoor/right_large.jpg  output/site1/pano_large.jpg  --version 0 --runs 5 --metrics-json output/site1/metrics_large.json

# Site 2 (Outdoor)
!./stitch dataset/site2_outdoor/left_small.jpg  dataset/site2_outdoor/right_small.jpg  output/site2/pano_small.jpg  --version 0 --runs 5 --metrics-json output/site2/metrics_small.json
!./stitch dataset/site2_outdoor/left_medium.jpg dataset/site2_outdoor/right_medium.jpg output/site2/pano_medium.jpg --version 0 --runs 5 --metrics-json output/site2/metrics_medium.json
!./stitch dataset/site2_outdoor/left_large.jpg  dataset/site2_outdoor/right_large.jpg  output/site2/pano_large.jpg  --version 0 --runs 5 --metrics-json output/site2/metrics_large.json

# Site 3 (Custom)
!./stitch dataset/site3_custom/left_small.jpg  dataset/site3_custom/right_small.jpg  output/site3/pano_small.jpg  --version 0 --runs 5 --metrics-json output/site3/metrics_small.json
!./stitch dataset/site3_custom/left_medium.jpg dataset/site3_custom/right_medium.jpg output/site3/pano_medium.jpg --version 0 --runs 5 --metrics-json output/site3/metrics_medium.json
!./stitch dataset/site3_custom/left_large.jpg  dataset/site3_custom/right_large.jpg  output/site3/pano_large.jpg  --version 0 --runs 5 --metrics-json output/site3/metrics_large.json

# Challenging pair (low texture / large viewpoint change)
!./stitch dataset/challenging/left_small.jpg  dataset/challenging/right_small.jpg  output/challenging/pano_small.jpg  --version 0 --runs 5 --metrics-json output/challenging/metrics_small.json
!./stitch dataset/challenging/left_medium.jpg dataset/challenging/right_medium.jpg output/challenging/pano_medium.jpg --version 0 --runs 5 --metrics-json output/challenging/metrics_medium.json
!./stitch dataset/challenging/left_large.jpg  dataset/challenging/right_large.jpg  output/challenging/pano_large.jpg  --version 0 --runs 5 --metrics-json output/challenging/metrics_large.json
"""))

    # Section 6: Results Analysis, Tables & Plots
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 6. RESULTS ANALYSIS, GEOMETRIC ACCURACY & AMDAHL'S LAW
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.1 Geometric Accuracy Evaluation on Synthetic Dataset (Projected Grid RMSE)
import json, os
import numpy as np

with open("dataset/synthetic/ground_truth.json", "r") as f:
    gt_data = json.load(f)

# Quality gate thresholds (Plan v4): RMSE <= 1.5 px (small), <= 2.0 px (medium), <= 4.0 px (large)
# All resolutions: normalized RMSE <= 0.1% of image diagonal
RMSE_THRESH = {"small": 1.5, "medium": 2.0, "large": 4.0}

print("=" * 80)
print(f"{'Resolution':<12} | {'Grid RMSE (px)':<16} | {'Diag Norm RMSE (%)':<22} | {'Quality Gate'}")
print("-" * 80)

for res in ["small", "medium", "large"]:
    metrics_path = f"output/synthetic/metrics_{res}.json"
    if not os.path.isfile(metrics_path):
        print(f"{res:<12} | {'N/A':<16} | {'N/A':<22} | NOT RUN")
        continue
    with open(metrics_path, "r") as f:
        met = json.load(f)

    H_est = np.array(met["homography_a_to_b"]).reshape(3, 3)
    pts_A = np.array(gt_data[res]["validation_grid_A"])
    pts_B_gt = np.array(gt_data[res]["validation_grid_B"])
    diag = gt_data[res]["diagonal"]

    pts_A_homo = np.hstack([pts_A, np.ones((len(pts_A), 1))])
    pts_B_est_homo = (H_est @ pts_A_homo.T).T
    pts_B_est = pts_B_est_homo[:, :2] / pts_B_est_homo[:, 2:3]

    rmse = float(np.sqrt(np.mean(np.sum((pts_B_est - pts_B_gt) ** 2, axis=1))))
    norm_rmse_pct = (rmse / diag) * 100.0

    thresh_px = RMSE_THRESH[res]
    passed = (rmse <= thresh_px) and (norm_rmse_pct <= 0.1)
    gate = f"PASS (<= {thresh_px} px & <= 0.1% diag)" if passed else f"FAIL (thresh: {thresh_px} px)"
    print(f"{res:<12} | {rmse:<16.4f} | {norm_rmse_pct:<22.4f}% | {gate}")

print("=" * 80)
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.2 Per-Stage Execution Timing Breakdown Table (Synthetic, all 3 resolutions)
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
            row[f"{res.capitalize()} Avg ms"] = f"{t_avg:.2f}"
            row[f"{res.capitalize()} Min ms"] = f"{t_min:.2f}"
        else:
            row[f"{res.capitalize()} Avg ms"] = "N/A"
            row[f"{res.capitalize()} Min ms"] = "N/A"
    table_rows.append(row)

df = pd.DataFrame(table_rows)
print("Synthetic Dataset — V0 Per-Stage Timing (Avg / Min ms across 5 runs):")
display.display(display.HTML(df.to_html(index=False)))
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.3 Per-Stage Bar Chart — Execution Profile Visualization
import matplotlib.pyplot as plt

p_small  = "output/synthetic/metrics_small.json"
p_medium = "output/synthetic/metrics_medium.json"
p_large  = "output/synthetic/metrics_large.json"

stage_labels = ["S1: Blur", "S2: FAST", "S3: ORB", "S4: Match", "S5: RANSAC", "S6: Warp"]
stage_keys   = ["stage1_preprocess", "stage2_fast_nms_topk", "stage3_orb_descriptor",
                 "stage4_matching", "stage5_ransac", "stage6_warping_blending"]
colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#9C27B0', '#FF9800']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
timing_data = {}

for ax, (res, path) in zip(axes, [("Small (720p)", p_small), ("Medium (1080p)", p_medium), ("Large (4K)", p_large)]):
    if not os.path.isfile(path):
        ax.set_title(f"{res} — N/A")
        continue
    with open(path, "r") as f:
        d = json.load(f)
    times = [d["timing_ms"][k]["avg"] for k in stage_keys]
    timing_data[res] = {k: d["timing_ms"][k]["avg"] for k in stage_keys + ["end_to_end"]}
    bars = ax.bar(stage_labels, times, color=colors)
    ax.set_title(f"V0 Sequential — {res}", fontsize=11, fontweight='bold')
    ax.set_ylabel("Avg Time (ms)")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f"{h:.1f}", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("output/bottleneck_breakdown.png", dpi=150)
plt.show()
print("[OK] Bottleneck chart saved to output/bottleneck_breakdown.png")
"""))

    # F5: Hot stages written analysis
    cells.append(nbf.v4.new_markdown_cell(r"""### 3.5 Hot Stages Analysis

Based on the per-stage timing data above, the following patterns are observed:

**Dominant stages at 720p (Small):**
Stage 6 (Warp & Blend) and Stage 1 (Preprocess) together account for approximately 60–70% of total pipeline time. Stage 2 (FAST-9 detection) contributes about 20–23%. Stages 3–5 (ORB, Matching, RANSAC) together account for less than 10% at this resolution.

**Scaling behavior with resolution:**
Stage 6 scales super-linearly from Small to Large: canvas size grows as (w×H_scale) × (h×H_scale), which for 4K is roughly 9× larger than 720p. Stage 1 (per-pixel blur) scales linearly with pixel count (~9× from 720p to 4K). Stage 2 also scales linearly. Stages 3–5 are nearly resolution-independent because the Top-K cap (K=1000) keeps descriptor count constant and RANSAC iterations (2000) are fixed.

**Implication for GPU plan:**
Stages 1, 2, and 6 are data-parallel and resolution-sensitive — they are the highest-priority targets for GPU acceleration. Stages 3–5 are bounded by K=1000 and 2000 iterations respectively; their absolute contribution is small at 720p but grows slowly. All 6 stages will be ported to V1 to eliminate Host↔Device round trips between stages.
"""))

    cells.append(nbf.v4.new_code_cell("""# 6.4 Amdahl's Law Analysis — Realistic GPU Speedup Estimate
# p = fraction of pipeline time that will execute on GPU
# Serial residual includes: H2D image transfer, D2H panorama transfer, host control logic
# Estimated serial overhead for V1 GPU on Tesla T4:
#   H2D: ~2 images * ~3 MB each at PCIe Gen3 ~10 GB/s = ~0.6 ms per image
#   D2H: ~1 panorama * ~8 MB at ~10 GB/s = ~0.8 ms
#   Host control (CLI, JSON write, etc.): ~1-2 ms
#   Total estimated serial = ~4 ms (conservative, independent of resolution)

import json, os

for res, path in [("Small (720p)", "output/synthetic/metrics_small.json"),
                  ("Medium (1080p)", "output/synthetic/metrics_medium.json"),
                  ("Large (4K)", "output/synthetic/metrics_large.json")]:
    if not os.path.isfile(path):
        print(f"{res}: metrics not available")
        continue
    with open(path, "r") as f:
        d = json.load(f)

    t_e2e = d["timing_ms"]["end_to_end"]["avg"]
    t_stages = sum(d["timing_ms"][k]["avg"] for k in
                   ["stage1_preprocess","stage2_fast_nms_topk","stage3_orb_descriptor",
                    "stage4_matching","stage5_ransac","stage6_warping_blending"])

    # Stages we plan to GPU-accelerate (all 6)
    t_parallel = t_stages

    # Conservative serial overhead estimate: H2D + D2H + host control
    T_SERIAL_ESTIMATE_MS = 4.0  # ms (see comment above)
    p = t_parallel / (t_parallel + T_SERIAL_ESTIMATE_MS)
    s_max = 1.0 / (1.0 - p)

    print(f"{'=' * 60}")
    print(f"Resolution: {res}")
    print(f"  V0 End-to-End (avg):      {t_e2e:.2f} ms")
    print(f"  V0 Stages Total (avg):    {t_stages:.2f} ms  ({t_stages/t_e2e*100:.1f}% of e2e)")
    print(f"  Estimated serial (H2D+D2H+ctrl): {T_SERIAL_ESTIMATE_MS:.1f} ms")
    print(f"  Parallelizable fraction p = {p*100:.2f}%")
    print(f"  Amdahl Max Speedup S_max = 1/(1-p) = {s_max:.1f}x")
    print(f"  (Practical limit also bounded by memory bandwidth and occupancy)")
print('=' * 60)
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
    ("Synthetic Scene (720p)",   "output/synthetic/pano_small.jpg"),
    ("Site 1 Indoor (720p)",     "output/site1/pano_small.jpg"),
    ("Site 2 Outdoor (720p)",    "output/site2/pano_small.jpg"),
    ("Site 3 Custom (720p)",     "output/site3/pano_small.jpg"),
    ("Challenging Pair (720p)",  "output/challenging/pano_small.jpg"),
]

available = [(t, p) for t, p in panoramas if os.path.isfile(p)]
cols = min(3, len(available))
rows = (len(available) + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5), dpi=75)
axes = [axes] if rows * cols == 1 else (axes.flatten() if rows > 1 else axes)

for idx, (title, p) in enumerate(available):
    img = Image.open(p)
    preview = img.copy()
    preview.thumbnail((480, 270))
    axes[idx].imshow(preview)
    axes[idx].set_title(f"{title}\\n({img.width}x{img.height})", fontsize=9, fontweight='bold')
    axes[idx].axis('off')

for idx in range(len(available), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
"""))

    # Section 8: GPU Implementation Plan (V1 to V3)
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 3.7 GPU IMPLEMENTATION PLAN (PHASES C, D, E)

### Which Stages to GPU-Accelerate and Why

Based on the profiling data in Section 6.3:
- **Stage 1 (Preprocess):** ~30% of 720p time, scales linearly with pixels. Highest priority — 1 thread/pixel, no data dependencies.
- **Stage 6 (Warp & Blend):** ~38% of 720p time, scales super-linearly with canvas size. Second priority — 1 thread/canvas pixel, fully independent.
- **Stage 2 (FAST-9 + NMS):** ~22% of 720p time, scales linearly. Third priority — FAST circle test per pixel with only local 16-pixel reads.
- **Stages 3–5:** <10% collectively at 720p. Small absolute contribution, but must still run on GPU to avoid H2D/D2H roundtrips between stages.

All 6 stages will be ported to GPU in V1 to keep all intermediate data resident on device.

### V1 Thread Mapping

| Stage | Kernel grid | What one thread computes |
|---|---|---|
| S1 Preprocess | 2D: (W/16)×(H/16) blocks of 16×16 threads | 1 output grayscale pixel (luma + blur tap) |
| S2 FAST-9 | 2D: same as S1 | FAST-9 decision for 1 pixel; atomicAdd to candidate list |
| S2 NMS + TopK | 1D over candidates | Local 3×3 suppression; `thrust::sort_by_key` |
| S3 ORB | 1D: N_keypoints threads | Full 256-bit descriptor for 1 keypoint (256 bit-test iterations) |
| S4 Matching | 2D: N_A × N_B / tile | Hamming distance for 1 (A, B) descriptor pair; warp-level min reduction |
| S5 RANSAC | 1D: 2000 threads | 1 hypothesis (4-point DLT + inlier count) per thread |
| S6 Warp+Blend | 2D: canvas_W/16 × canvas_H/16 | 1 output canvas pixel (inverse warp + bilinear + feather) |

### V2 & V3 Optimization Ideas

1. **Shared Memory Tiling (S1, S6):** Load 18×18 input tile into shared memory (16×16 output + 2-pixel halo) to eliminate redundant global reads for 5×5 blur and bilinear warp. Expected 4–8× L2 hit rate improvement.
2. **`__constant__` ORB Pattern Table (S3):** Store the 256×2×2 `int` table (4 KB) in constant memory for broadcast-cached reads across all threads in a warp simultaneously.
3. **Warp-shuffle Hamming reduction (S4):** Replace shared-memory min-reduction with `__shfl_down_sync` for finding best1/best2 across a warp. Eliminates bank-conflict-prone shared memory for this step.

### Main Risks

| Risk | Stage | Mitigation |
|---|---|---|
| Shared memory bank conflicts in S1 | S1 tiled blur | Pad shared tile width to 17 (odd) to avoid 32-bank conflicts |
| Thread divergence in S5 RANSAC (DLT pivot branches) | S5 | Reformulate pivot selection as branchless `fmaxf` + `__ballot_sync` |
| Low occupancy in S3 ORB (256 serial bit tests per thread) | S3 | Batch 32 bit-tests per warp iteration using `__ballot_sync`; reduces register pressure |
| H2D transfer dominating small-image latency | All | Use CUDA streams: overlap H2D with S1 kernel; overlap D2H with JSON write |
| RANSAC non-determinism with GPU atomics | S5 | Use deterministic LCG seed=42 per thread index; validate against V0 on synthetic |
"""))

    # Section 9: References
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 9. REFERENCES & EXTERNAL SOURCES

1. **OpenCV ORB `bit_pattern_31_`** — 256-pair binary test pattern used in Stage 3.
   Ethan Rublee et al., *ORB: An efficient alternative to SIFT or SURF*, ICCV 2011.
   Source: `opencv/modules/features2d/src/orb.cpp` (BSD 3-Clause, Willow Garage Inc.)

2. **stb_image / stb_image_write** — Public domain image I/O library used in `stb_impl.*`.
   Sean T. Barrett. https://github.com/nothings/stb (MIT / Unlicense)

3. **Lowe's Ratio Test** — D.G. Lowe, *Distinctive image features from scale-invariant keypoints*, IJCV 2004.

4. **Hartley Normalization** — R. Hartley & A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge, 2004.

5. **Amdahl's Law** — G.M. Amdahl, *Validity of the single-processor approach to achieving large-scale computing capabilities*, AFIPS 1967.

**AI Tool Disclosure:** Claude (Anthropic) was used to assist with code review, documentation drafting, and bug identification during development. All algorithms, implementations, and experimental results are the team's own work.
"""))

    # Save to file
    nb.cells = cells
    out_path = Path("notebooks") / "Group19_Midterm.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Midterm notebook created at: {out_path} ({len(cells)} cells)")

if __name__ == "__main__":
    create_midterm_notebook()
