#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Generate Midterm Submission Notebook (Plan v4 Section 7 & B2)
Authors: Group 19 - Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
External code: none
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
| **Stage 1** | Preprocessing | Standard luminance $Y = 0.299R + 0.587G + 0.114B$; Separable 5×5 Gaussian blur ($\sigma=1.0$), boundary clamping [9]. |
| **Stage 2** | FAST-9 + NMS + Top-K | FAST-9 corner threshold $T=20$ on 16-pixel Bresenham circle; 3×3 Non-Maximum Suppression; Max $K=1000$ keypoints; Deterministic sort: `(response DESC, linear_index ASC)` [1]. |
| **Stage 3** | Oriented BRIEF / ORB | Patch $31\times 31$, orientation angle from Intensity Centroid $[-\pi, \pi]$; Verified 256 test pairs from OpenCV `bit_pattern_31_` [7]; Strict border pruning ($R < 20$ px) [2, 3]. |
| **Stage 4** | Brute-force Hamming | 256-bit descriptors stored as 8 words $\times$ 32-bit (`uint32_t words[8]`), popcount bit distance; Lowe's Ratio Test: $\text{dist}_1 < 0.75 \times \text{dist}_2$ [4]. |
| **Stage 5** | RANSAC Homography | 2000 hypotheses; 4-point Hartley isotropic normalized DLT; Gauss-Jordan partial pivoting; Reprojection threshold $3.0$ px; Deterministic PRNG LCG seed 42; **least-squares refit on all inliers of the best hypothesis** (3 rounds, Hartley-normalised normal equations in double precision, accepted only if the inlier count does not decrease) [5, 6]. |
| **Stage 6** | Warping & Feathering | Inverse coordinate mapping via $H_{A\to B}$; Bilinear sampling; Symmetric distance-to-border feathering weights to eliminate seam artifacts [9, 10]. |

### 1.4 GPU Suitability Rationale
1. **Stage 1:** Embarrassingly parallel pixel-wise computation. 1 thread maps to 1 pixel. Memory access is spatially localized (ideal for Shared Memory Tiling in V2).
2. **Stage 2:** FAST circle test requires only arithmetic comparisons. Top-K sorting uses `thrust::sort_by_key` (parallel Radix Sort) to overcome Bitonic Sort limits on 4K images.
3. **Stage 3:** Independent patch rotation per keypoint. 256 test pairs can be stored in `__constant__` memory for ultra-fast broadcast reads.
4. **Stage 4:** Brute-force Hamming distance computed via hardware `__popc` instruction, exhibiting high arithmetic throughput and coalesced memory reads.
5. **Stage 5:** Massively parallel hypothesis evaluation: each CUDA warp/block evaluates candidate models simultaneously.
6. **Stage 6:** Inverse warping generates canvas pixels independently, leveraging GPU texture units or bilinear cache.

### 1.5 Why these methods: output quality and parallel efficiency

| Stage | Output quality | Parallel efficiency |
|---|---|---|
| **1. Gray + separable Gaussian** | Smoothing removes sensor noise that would create false FAST corners, while $\sigma=1$ keeps corner positions precise. | Separable filtering costs $2k$ instead of $k^2$ taps per pixel; a pure stencil, so shared-memory tiling with halos applies directly. |
| **2. FAST-9 + NMS + top-$K$** | Corners are repeatable under the small rotation and viewpoint change of the capture protocol [1]; NMS removes clustered duplicates; top-$K$ by response keeps the strongest corners and bounds the cost of Stages 3-5. | The per-pixel test is independent. The variable-size output needs compaction and a sort, which makes this the hardest stage to port. |
| **3. ORB (oriented BRIEF, 256 bit)** | Rotation invariance from the intensity centroid [2] and cheap binary tests [3]. There is no scale invariance, which is acceptable here: the pictures are taken by rotating in place without zooming (Section 1.2 of the assignment). SIFT-like descriptors would be more robust to scale but heavier. | One independent thread per keypoint; a 256-bit descriptor is 8 words, compared with XOR and popcount. |
| **4. Brute-force Hamming + ratio test** | The ratio test [4] rejects ambiguous matches (repetitive texture) before RANSAC sees them. | Regular all-pairs computation, similar to a matrix product; $K$ is capped, so exhaustive search stays affordable. |
| **5. RANSAC + normalised DLT + refit** | Robust to wrong matches [5]; Hartley normalisation keeps the DLT well conditioned [6]; refitting on all inliers removes the noise of a 4-point minimal sample (evidence in Section 6.5). | Hypotheses are independent and inlier counting is a reduction. The refit works on a handful of equations and can stay on the host. |
| **6. Inverse warp + bilinear + feathering** | Inverse mapping leaves no holes; distance-to-border feathering hides the seam [9, 10]. Multi-band blending would handle exposure differences better but adds pyramid stages. | A gather per output pixel; texture units provide hardware bilinear interpolation. |
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

    cells.append(nbf.v4.new_code_cell(r"""# 3.3 Dataset overview: registered sites, prepared pairs and input thumbnails
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import IPython.display as display

with open("dataset/manifest.json", encoding="utf-8") as f:
    manifest = json.load(f)

# Free-text descriptions (scene, location, challenge type, ...) are shown when the manifest provides them.
DESCRIPTION_KEYS = ("site", "location", "scene", "description", "challenge", "notes")
SIZES = ("small", "medium", "large")

rows, ready = [], []
for p in manifest.get("pairs", []):
    d = p.get("directory") or ""
    files = [os.path.join(d, f"{side}_{s}.jpg") for side in ("left", "right") for s in SIZES]
    prepared = bool(d) and all(os.path.isfile(x) for x in files)
    row = {"Pair": p.get("id"), "Kind": p.get("kind"), "Directory": d,
           "Manifest status": p.get("status", ""), "3 sizes prepared": "yes" if prepared else "no"}
    for k in DESCRIPTION_KEYS:
        if k in p:
            row[k.capitalize()] = p[k]
    rows.append(row)
    if prepared:
        ready.append(p)

display.display(display.HTML(pd.DataFrame(rows).to_html(index=False)))
kinds = {}
for p in ready:
    kinds[p.get("kind")] = kinds.get(p.get("kind"), 0) + 1
print(f"{len(ready)} of {len(rows)} registered pairs are prepared at small, medium and large size; prepared pairs by kind: {kinds}")

if ready:
    fig, axes = plt.subplots(len(ready), 2, figsize=(9, 2.6 * len(ready)), squeeze=False)
    for i, p in enumerate(ready):
        for j, side in enumerate(("left", "right")):
            img = Image.open(os.path.join(p["directory"], f"{side}_small.jpg"))
            img.thumbnail((480, 270))
            axes[i][j].imshow(img)
            axes[i][j].set_title(f"{p['id']} - {side}", fontsize=9, fontweight="bold")
            axes[i][j].axis("off")
    plt.tight_layout()
    plt.show()
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

    cells.append(nbf.v4.new_code_cell(r"""# 5.2 Benchmark V0 on every pair at all 3 resolutions (1 warm-up + 5 timed runs inside ./stitch)
import os
import json
import subprocess

RUNS = 5
BENCH = [
    ("synthetic",     "dataset/synthetic",     "output/synthetic"),
    ("site1_indoor",  "dataset/site1_indoor",  "output/site1"),
    ("site2_outdoor", "dataset/site2_outdoor", "output/site2"),
    ("site3_custom",  "dataset/site3_custom",  "output/site3"),
    ("challenging",   "dataset/challenging",   "output/challenging"),
]

print(f"{'Pair':<14}{'Res':<8}{'Device':<36}{'E2E avg (ms)':>13}{'E2E min (ms)':>13}{'Matches':>9}{'Inliers':>9}  Status")
print("-" * 118)
for name, ddir, odir in BENCH:
    os.makedirs(odir, exist_ok=True)
    for res in ["small", "medium", "large"]:
        metrics = f"{odir}/metrics_{res}.json"
        cmd = ["./stitch", f"{ddir}/left_{res}.jpg", f"{ddir}/right_{res}.jpg",
               f"{odir}/pano_{res}.jpg", "--version", "0", "--runs", str(RUNS),
               "--metrics-json", metrics]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        with open(f"{odir}/log_{res}.txt", "w") as f:      # full console output of every run is kept on disk
            f.write(proc.stdout + proc.stderr)
        if os.path.isfile(metrics):
            with open(metrics) as f:
                d = json.load(f)
            e2e = d["timing_ms"]["end_to_end"]
            print(f"{name:<14}{res:<8}{d['device_name']:<36}{e2e['avg']:>13.2f}{e2e['min']:>13.2f}"
                  f"{d['matches']:>9}{d['inliers']:>9}  {d['status']} (exit {proc.returncode})")
        else:
            print(f"{name:<14}{res:<8}{'-':<36}{'-':>13}{'-':>13}{'-':>9}{'-':>9}  NO METRICS (exit {proc.returncode}): "
                  f"{(proc.stderr.strip().splitlines() or ['?'])[-1]}")
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

    cells.append(nbf.v4.new_code_cell(r"""# 6.2 Per-stage V0 timing tables (Avg / Min over the timed runs) for every pair and resolution
import os
import json
import pandas as pd
import IPython.display as display

PAIRS = [
    ("synthetic",     "output/synthetic"),
    ("site1_indoor",  "output/site1"),
    ("site2_outdoor", "output/site2"),
    ("site3_custom",  "output/site3"),
    ("challenging",   "output/challenging"),
]
RESOLUTIONS = ["small", "medium", "large"]
STAGES = [
    ("Stage 1 (Preprocess)", "stage1_preprocess"),
    ("Stage 2 (FAST+NMS+TopK)", "stage2_fast_nms_topk"),
    ("Stage 3 (ORB Descriptor)", "stage3_orb_descriptor"),
    ("Stage 4 (Hamming Matching)", "stage4_matching"),
    ("Stage 5 (RANSAC Homography)", "stage5_ransac"),
    ("Stage 6 (Warp & Blend)", "stage6_warping_blending"),
    ("Kernel/Stages Total", "kernel_total"),
    ("End-to-End Pipeline", "end_to_end"),
]

for name, odir in PAIRS:
    metrics = {}
    for res in RESOLUTIONS:
        p = f"{odir}/metrics_{res}.json"
        if os.path.isfile(p):
            with open(p) as f:
                metrics[res] = json.load(f)
    if not metrics:
        print(f"\n=== {name}: no metrics found ===")
        continue

    device = next(iter(metrics.values()))["device_name"]
    status = ", ".join(f"{r}={metrics[r]['status']}" for r in RESOLUTIONS if r in metrics)
    print(f"\n=== {name} | device: {device} | status: {status} ===")

    rows = []
    for label, key in STAGES:
        row = {"Stage": label}
        for res in RESOLUTIONS:
            if res in metrics:
                t = metrics[res]["timing_ms"][key]
                row[f"{res.capitalize()} (Avg / Min ms)"] = f"{t['avg']:.2f} / {t['min']:.2f}"
            else:
                row[f"{res.capitalize()} (Avg / Min ms)"] = "N/A"
        rows.append(row)
    display.display(display.HTML(pd.DataFrame(rows).to_html(index=False)))
"""))

    cells.append(nbf.v4.new_code_cell(r"""# 6.3 V0 execution profile (hot stages vs. resolution) and Amdahl's Law limit
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import IPython.display as display

STAGES = [
    ("S1", "S1: Preprocess", "stage1_preprocess"),
    ("S2", "S2: FAST+NMS+TopK", "stage2_fast_nms_topk"),
    ("S3", "S3: ORB", "stage3_orb_descriptor"),
    ("S4", "S4: Match", "stage4_matching"),
    ("S5", "S5: RANSAC", "stage5_ransac"),
    ("S6", "S6: Warp+Blend", "stage6_warping_blending"),
]
COLORS = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#9C27B0", "#FF9800"]

# Which stages are moved to the GPU in each plan
PLANS = {
    "A: all 6 stages (our plan)": ["S1", "S2", "S3", "S4", "S5", "S6"],
    "B: all except S5 (RANSAC stays on CPU)": ["S1", "S2", "S3", "S4", "S6"],
    "C: hot trio only (S1+S2+S6)": ["S1", "S2", "S6"],
}

DATASETS = [("synthetic", "output/synthetic"), ("site1_indoor", "output/site1")]
RESOLUTIONS = ["small", "medium", "large"]
NOISE_LIMIT = 1.5  # avg/min of end-to-end above this => run is dominated by outliers

# ---------------------------------------------------------------- load V0 metrics
records = []  # one dict per (dataset, resolution)
for ds_name, ds_dir in DATASETS:
    for res in RESOLUTIONS:
        path = f"{ds_dir}/metrics_{res}.json"
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            t = json.load(f)["timing_ms"]
        records.append({
            "dataset": ds_name,
            "res": res,
            "stage_avg": {s: t[k]["avg"] for s, _, k in STAGES},
            "e2e_avg": t["end_to_end"]["avg"],
            "e2e_min": t["end_to_end"]["min"],
        })

if not records:
    raise FileNotFoundError("No V0 metrics found: run section 5 (benchmark cells) first.")

# ---------------------------------------------------------------- 1) per-stage share vs. resolution
share_rows = []
for r in records:
    row = {"Dataset": r["dataset"], "Res": r["res"], "V0 e2e (ms)": round(r["e2e_avg"], 2)}
    for s, label, _ in STAGES:
        row[f"{s} (%)"] = round(100 * r["stage_avg"][s] / r["e2e_avg"], 1)
    row["Top-3 stages"] = ", ".join(
        s for s, _ in sorted(r["stage_avg"].items(), key=lambda kv: -kv[1])[:3])
    row["Noise avg/min"] = round(r["e2e_avg"] / r["e2e_min"], 2)
    share_rows.append(row)

df_share = pd.DataFrame(share_rows)
print("Share of V0 end-to-end time per stage (average of timed runs):")
display.display(display.HTML(df_share.to_html(index=False)))

noisy = df_share[df_share["Noise avg/min"] > NOISE_LIMIT]
if not noisy.empty:
    print("\n[WARNING] High run-to-run variance (avg/min > "
          f"{NOISE_LIMIT}) in: " + "; ".join(f"{r.Dataset}/{r.Res} ({r['Noise avg/min']}x)"
                                              for _, r in noisy.iterrows()))
    print("          Stage shares of these rows are unreliable; prefer the stable rows or re-run the benchmark.")

# ---------------------------------------------------------------- 2) stacked chart (share vs. resolution)
datasets_present = [d for d, _ in DATASETS if any(r["dataset"] == d for r in records)]
fig, axes = plt.subplots(1, len(datasets_present), figsize=(6.5 * len(datasets_present), 5), squeeze=False)
for ax, ds in zip(axes[0], datasets_present):
    recs = [r for r in records if r["dataset"] == ds]
    xs = [r["res"] for r in recs]
    bottoms = [0.0] * len(recs)
    for (s, label, _), color in zip(STAGES, COLORS):
        vals = [100 * r["stage_avg"][s] / r["e2e_avg"] for r in recs]
        ax.bar(xs, vals, bottom=bottoms, color=color, label=label, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(i, bottoms[i] + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="black")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title(f"V0 stage share vs. resolution: {ds}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Share of end-to-end time (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
axes[0][0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig("output/bottleneck_breakdown.png", dpi=150, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------- 3) Amdahl's Law
# p = fraction of V0 time spent in the stages moved to the GPU; S_max = 1 / (1 - p).
# Averages are used because per-stage averages add up to the total (per-stage minima do not).
amdahl_rows = []
for r in records:
    for plan_name, plan_stages in PLANS.items():
        t_acc = sum(r["stage_avg"][s] for s in plan_stages)
        t_ser = r["e2e_avg"] - t_acc
        p = t_acc / r["e2e_avg"]
        s_max = r["e2e_avg"] / t_ser if t_ser > 0 else float("inf")
        amdahl_rows.append({
            "Dataset": r["dataset"],
            "Res": r["res"],
            "Plan": plan_name,
            "Accelerated (ms)": round(t_acc, 2),
            "Serial left (ms)": round(t_ser, 3),
            "p (%)": round(100 * p, 4),
            "S_max = 1/(1-p)": round(s_max, 1),
        })

df_amdahl = pd.DataFrame(amdahl_rows)
print("\nAmdahl's Law: upper bound on end-to-end speedup for each GPU plan")
display.display(display.HTML(df_amdahl.to_html(index=False)))

print("\nSummary for our plan (A):")
for _, row in df_amdahl[df_amdahl["Plan"].str.startswith("A")].iterrows():
    print(f"  {row['Dataset']:<13} {row['Res']:<7} p = {row['p (%)']:.4f}%  ->  S_max = {row['S_max = 1/(1-p)']:.1f}x")
print("\nNote: S_max ignores host-device transfers and kernel-launch overhead, which V0 does not measure.")
print("      They add to the serial part once V1 is timed, so the real limit is lower than shown here.")
"""))

    cells.append(nbf.v4.new_code_cell(r"""# 6.4 Hot-stage analysis. The text below is generated from the measurements: every number is computed here.
import os
import json
from IPython.display import Markdown, display

STAGES = [("S1", "preprocess", "stage1_preprocess"), ("S2", "FAST+NMS+Top-K", "stage2_fast_nms_topk"),
          ("S3", "ORB descriptor", "stage3_orb_descriptor"), ("S4", "Hamming matching", "stage4_matching"),
          ("S5", "RANSAC", "stage5_ransac"), ("S6", "warp+blend", "stage6_warping_blending")]
PAIRS = [("synthetic", "output/synthetic"), ("site1_indoor", "output/site1"), ("site2_outdoor", "output/site2"),
         ("site3_custom", "output/site3"), ("challenging", "output/challenging")]
RES = ["small", "medium", "large"]
PIXELS = {"small": 1280 * 720, "medium": 1920 * 1080, "large": 3840 * 2160}
NOISE_LIMIT = 1.5  # avg/min of end-to-end above this: the run is dominated by outliers

def load(odir, res):
    p = f"{odir}/metrics_{res}.json"
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        t = json.load(f)["timing_ms"]
    return {"stage": {s: t[k]["avg"] for s, _, k in STAGES},
            "e2e": t["end_to_end"]["avg"], "e2e_min": t["end_to_end"]["min"]}

def shares(r):
    return {s: 100 * v / r["e2e"] for s, v in r["stage"].items()}

lines = []
mean_share = {s: 0.0 for s, _, _ in STAGES}
n_used = 0
for name, odir in PAIRS:
    r = {res: load(odir, res) for res in RES}
    if any(v is None for v in r.values()):
        lines.append(f"- **{name}**: skipped (metrics missing for at least one resolution).")
        continue
    noise = {res: r[res]["e2e"] / r[res]["e2e_min"] for res in RES}
    if max(noise.values()) > NOISE_LIMIT:
        lines.append(f"- **{name}**: skipped, run-to-run variance too high (max avg/min = {max(noise.values()):.2f}); re-run the benchmark.")
        continue

    n_used += 1
    sh = {res: shares(r[res]) for res in RES}
    for res in RES:
        for s in mean_share:
            mean_share[s] += sh[res][s]

    e = {res: r[res]["e2e"] for res in RES}
    lines.append(f"- **{name}**: end-to-end time is {e['small']:.1f} ms (small), {e['medium']:.1f} ms (medium) and {e['large']:.1f} ms (large), "
                 f"i.e. x{e['medium'] / e['small']:.2f} and x{e['large'] / e['small']:.2f} for a pixel count that grows x{PIXELS['medium'] / PIXELS['small']:.2f} and x{PIXELS['large'] / PIXELS['small']:.2f}.")
    for res in RES:
        top = sorted(sh[res].items(), key=lambda kv: -kv[1])[:3]
        lines.append(f"    - {res}: top-3 = " + ", ".join(f"{s} {v:.1f}%" for s, v in top) + f" (together {sum(v for _, v in top):.1f}%)")
    delta = {s: sh["large"][s] - sh["small"][s] for s in sh["small"]}
    up = max(delta, key=delta.get)
    down = min(delta, key=delta.get)
    growth = {s: r["large"]["stage"][s] / r["small"]["stage"][s] for s in sh["small"]}
    lines.append(f"    - share change small to large: largest gain {up} ({delta[up]:+.1f} points), largest loss {down} ({delta[down]:+.1f} points)")
    lines.append("    - time growth small to large per stage: " + ", ".join(f"{s} x{g:.1f}" for s, g in growth.items())
                 + f" (pixels x{PIXELS['large'] / PIXELS['small']:.1f})")

if n_used:
    ranking = sorted(((s, v / (3 * n_used)) for s, v in mean_share.items()), key=lambda kv: -kv[1])
    lines.append("")
    lines.append(f"**Mean share over all {3 * n_used} usable runs (pairs x resolutions):** "
                 + ", ".join(f"{s} {v:.1f}%" for s, v in ranking)
                 + f". The three most expensive stages ({', '.join(s for s, _ in ranking[:3])}) account for {sum(v for _, v in ranking[:3]):.1f}% of the pipeline, "
                 + "so they are the first candidates for the GPU and for later optimization.")
display(Markdown("\n".join(lines)))
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""### 6.5 Is the estimated homography trustworthy?
The entries of $H$ cannot be judged one by one: the translation grows with the image size, the perspective terms shrink, and
scale, perspective and translation can trade off against each other, so two matrices with different entries may describe almost the same
mapping in the overlap. The check below therefore compares **where each matrix sends the points of image A** (a $40\times24$ grid,
in 720p coordinates), in two ways: against the estimate obtained at another resolution of the same pair, and against an independent
estimate from OpenCV (ORB + `findHomography` with RANSAC) on the same images. Stage 5 refits the model on all inliers of the best
RANSAC hypothesis (Section 1.3); the cell reports the agreement obtained with that version of V0.
"""))

    cells.append(nbf.v4.new_code_cell(r"""# 6.5 Homography sanity check: (a) consistency across resolutions, (b) agreement with an independent OpenCV estimate
import os
import json
import numpy as np
import pandas as pd
import cv2
import IPython.display as display

PAIRS = [
    ("synthetic",     "output/synthetic",     "dataset/synthetic"),
    ("site1_indoor",  "output/site1",         "dataset/site1_indoor"),
    ("site2_outdoor", "output/site2",         "dataset/site2_outdoor"),
    ("site3_custom",  "output/site3",         "dataset/site3_custom"),
    ("challenging",   "output/challenging",   "dataset/challenging"),
]
SIZE = {"small": (1280, 720), "medium": (1920, 1080), "large": (3840, 2160)}
W0, H0 = SIZE["small"]
DIAG0 = float(np.hypot(W0, H0))
TOL_PCT = 0.5  # our tolerance for "same geometry": point-transfer difference <= 0.5% of the 720p diagonal

# Raw matrix entries cannot be compared across resolutions or judged on their own (translation scales with the
# image, perspective terms shrink, entries trade off against each other). Instead we compare where the matrix sends points.
gx, gy = np.meshgrid(np.linspace(0.02 * W0, 0.98 * W0, 40), np.linspace(0.02 * H0, 0.98 * H0, 24))
GRID = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])  # points of image A, in 720p coordinates

def to_small(H, res):
    # Express a homography estimated at resolution `res` in 720p coordinates: H_s = S^-1 H S
    s = SIZE[res][0] / W0
    S = np.diag([s, s, 1.0])
    Hs = np.linalg.inv(S) @ H @ S
    return Hs / Hs[2, 2]

def transfer(H, P):
    Q = H @ P
    ok = Q[2] > 1e-6
    w = np.where(ok, Q[2], 1.0)
    return Q[0] / w, Q[1] / w, ok

def inside(x, y):
    return (x >= 0) & (x < W0) & (y >= 0) & (y < H0)

def overlap_pct(H):
    x, y, ok = transfer(H, GRID)
    return 100.0 * float(np.mean(ok & inside(x, y)))

def discrepancy(H1, H2):
    # RMS distance (px, 720p coordinates) between where H1 and H2 send the points of A that land inside B
    x1, y1, o1 = transfer(H1, GRID)
    x2, y2, o2 = transfer(H2, GRID)
    valid = o1 & o2 & inside(x1, y1) & inside(x2, y2)
    if valid.sum() < 10:
        return np.nan
    return float(np.sqrt(np.mean(np.hypot(x1 - x2, y1 - y2)[valid] ** 2)))

cv2.setRNGSeed(42)
orb = cv2.ORB_create(nfeatures=1000, fastThreshold=20)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

def opencv_homography(pair_dir, res):
    a = cv2.imread(f"{pair_dir}/left_{res}.jpg", cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(f"{pair_dir}/right_{res}.jpg", cv2.IMREAD_GRAYSCALE)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return None, 0
    pairs = [p for p in matcher.knnMatch(da, db, k=2) if len(p) == 2]
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return None, len(good)
    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
    return H, (int(mask.sum()) if mask is not None else 0)

rows = []
for name, odir, ddir in PAIRS:
    H_small = None
    for res in ["small", "medium", "large"]:
        p = f"{odir}/metrics_{res}.json"
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            d = json.load(f)
        if d["status"] == "FAILURE":
            rows.append({"Pair": name, "Res": res, "Status": "FAILURE", "Matches": d["matches"], "Inliers": d["inliers"]})
            continue
        Hs = to_small(np.array(d["homography_a_to_b"], dtype=np.float64).reshape(3, 3), res)
        if res == "small":
            H_small = Hs
        d_res = discrepancy(Hs, H_small) if (res != "small" and H_small is not None) else np.nan
        Hcv, cv_inl = opencv_homography(ddir, res)
        d_cv = discrepancy(Hs, to_small(Hcv, res)) if Hcv is not None else np.nan
        checks = [v for v in (d_res, d_cv) if not np.isnan(v)]
        verdict = "OK" if checks and max(checks) <= TOL_PCT / 100 * DIAG0 else ("CHECK" if checks else "n/a")
        rows.append({
            "Pair": name, "Res": res, "Status": d["status"],
            "Matches": d["matches"], "Inliers": d["inliers"],
            "Inlier %": round(100.0 * d["inliers"] / max(d["matches"], 1), 1),
            "Overlap % (ours)": round(overlap_pct(Hs), 1),
            "vs small-res H (px)": None if np.isnan(d_res) else round(d_res, 2),
            "vs OpenCV H (px)": None if np.isnan(d_cv) else round(d_cv, 2),
            "OpenCV inliers": cv_inl,
            "Verdict": verdict,
        })

df_h = pd.DataFrame(rows)
display.display(display.HTML(df_h.to_html(index=False, na_rep="-")))
print(f"\nTolerance: point-transfer difference <= {TOL_PCT}% of the 720p diagonal = {TOL_PCT / 100 * DIAG0:.2f} px.")
print("'vs small-res H': our H at this resolution against our H at 720p, both expressed in 720p coordinates.")
print("'vs OpenCV H'   : our H against cv2.ORB + cv2.findHomography(RANSAC, 3 px) on the same images.")
print("'Overlap %'     : share of image A that lands inside image B under our H (assignment asks for 30-50% overlap).")
print("A large difference means at least one of the two estimates is wrong; OpenCV is a reference, not ground truth.")
"""))

    # Section 7: Visual Gallery of Results
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 7. VISUAL GALLERY OF STITCHED PANORAMAS
"""))

    cells.append(nbf.v4.new_code_cell(r"""# 7.1 Display stitched panorama outputs (720p)
import os
import matplotlib.pyplot as plt
from PIL import Image

panoramas = [
    ("Synthetic scene", "output/synthetic/pano_small.jpg"),
    ("Site 1 Indoor", "output/site1/pano_small.jpg"),
    ("Site 2 Outdoor", "output/site2/pano_small.jpg"),
    ("Site 3 Custom", "output/site3/pano_small.jpg"),
    ("Challenging pair", "output/challenging/pano_small.jpg"),
]

plt.figure(figsize=(12, 6), dpi=75)
shown = 0
for title, path in panoramas:
    if not os.path.isfile(path):
        continue
    img = Image.open(path)
    preview = img.copy()
    preview.thumbnail((480, 270))
    shown += 1
    plt.subplot(2, 3, shown)
    plt.imshow(preview)
    plt.title(f"{title}\n({img.width}x{img.height})", fontsize=9, fontweight="bold")
    plt.axis("off")
plt.tight_layout()
plt.show()

missing = [t for t, path in panoramas if not os.path.isfile(path)]
if missing:
    print("No panorama was written for:", ", ".join(missing), "(see the status column of Section 5.2)")
"""))

    # Section 8: GPU Implementation Plan (V1 to V3)
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 8. GPU IMPLEMENTATION PLAN (PHASES C, D, E)

### 8.0 Which stages go to the GPU, and why
All six stages move to the GPU in V1. The profile (Sections 6.3 and 6.4) shows which stages are the most expensive and how their share changes with the resolution; these are where most of the speedup comes from. The remaining stages are cheaper individually, but if they stayed on the CPU every intermediate result (keypoints, descriptors, matches) would have to cross the PCIe bus between GPU stages, and they would become the serial fraction that bounds the speedup (plans B and C in Section 6.3). Keeping all data on the device is therefore part of the V1 design.

### 8.1 V1 Basic CUDA: thread mapping
- **Stage 1 (Grayscale + Gaussian blur):** 2D grid of $16 \times 16$ blocks, one thread per pixel, global memory reads with boundary clamping.
- **Stage 2 (FAST-9 + NMS + Top-K):** one thread per pixel for FAST and NMS; candidates are appended to a bounded list with an atomic counter; `thrust::sort_by_key` gives the deterministic Top-K order (`response` descending, linear index ascending), as in V0.
- **Stage 3 (ORB descriptors):** one thread per keypoint evaluates the 256 oriented tests and writes one `Descriptor` (8 words $\times$ 32 bit).
- **Stage 4 (Hamming matching):** one thread per keypoint of image A scans all descriptors of image B with `__popc`, keeping the best and second-best distance for the ratio test.
- **Stage 5 (RANSAC):** one warp per hypothesis: lanes split the match list and a warp-shuffle reduction counts inliers; the best hypothesis is selected with the same tie-break as V0 (smallest hypothesis ID); the inlier refit runs on the host.
- **Stage 6 (Inverse warp + feathering):** 2D grid over the panorama canvas, one thread per output pixel, bilinear sampling of both images and symmetric feather weights.

### 8.2 APOD methodology for V2 and V3
Following Project Plan v4 Section 1.2 and the CUDA C++ Best Practices Guide [11], no optimization is chosen by guesswork: V1 is profiled (per-stage timing, Nsight Systems and Nsight Compute), the dominant bottleneck is addressed with exactly one main optimization per version, and every version is verified against V0 before it is timed.

### 8.3 Optimization ideas we expect to try (the final choice follows the V1 profile)
1. **Stage 1 and the start of the pipeline:** separable Gaussian with the weights in constant memory, a shared-memory tile with halo, and fusion of grayscale and blur into one kernel, to cut the global-memory traffic of a stencil.
2. **Stage 6:** texture-object bilinear sampling and a coalesced canvas layout, since warping and blending is one of the most expensive stages and grows with the resolution (Section 6.4).
3. **Pipeline level:** pinned host memory and one CUDA stream per image, so that Stages 1-3 of image A and image B overlap, plus keeping every intermediate buffer on the device.

Fallback if Stage 2 dominates in V1: replace atomic candidate compaction by a prefix-sum compaction.

### 8.4 Main risks
- **Consistency with V0.** Floating-point differences in the blur or in the FAST threshold test can change which keypoints survive and therefore which matches exist. Each version is verified by the homography error on the synthetic pair and by the transfer difference to V0 (Section 6.5), not by bit-exact equality.
- **RANSAC reproducibility.** V0 draws samples from a sequential LCG; the GPU needs the same samples per hypothesis (pre-generated indices) and the same tie-break, otherwise a different but equally valid model may be returned.
- **Variable-size outputs (Stage 2).** Atomic compaction can become a contention point on large images and needs a capacity bound; the Top-K sort must reproduce the ordering of V0.
- **Transfers and launch overhead.** At the smallest resolution, host-device copies and kernel launches may cancel part of the gain; the Amdahl bound of Section 6.3 ignores them, so V1 must time them separately (`h2d`, `d2h`).
- **Warp divergence.** FAST-9 and its early exits diverge between neighbouring threads.
- **Timing fairness.** Colab may assign a different GPU between sessions, so all versions of one comparison table are measured in the same session and the GPU name is printed with every result.

### 8.5 Work plan until the final submission
| Phase | Scope | Lead | Support |
|---|---|---|---|
| V1 Basic CUDA | Stages 1 and 5 kernels | Pham Minh Thong | Stage owners review each other's kernels |
| V1 Basic CUDA | Stages 2 and 3 kernels | Nguyen Le Tan Tien | |
| V1 Basic CUDA | Stages 4 and 6 kernels, `--version 1` integration, timing harness | Dao Quoc Tuan | |
| V1 profiling | Nsight Systems / Nsight Compute on V1, choice of the V2 and V3 targets | Dao Quoc Tuan | All members |
| V2 | First optimization from Section 8.3 | Owner of the dominant stage in the V1 profile | |
| V3 | Second optimization from Section 8.3 | Owner of the next stage in the profile | |
| Final | Speedup table and charts, gallery with a failure case, report, slides, video | Dao Quoc Tuan (benchmark) | All members |
"""))

    cells.append(nbf.v4.new_markdown_cell(r"""---
## 9. REFERENCES AND EXTERNAL CODE

**Papers and books**
1. E. Rosten and T. Drummond, "Machine learning for high-speed corner detection", ECCV 2006 (FAST).
2. E. Rublee, V. Rabaud, K. Konolige and G. Bradski, "ORB: An efficient alternative to SIFT or SURF", ICCV 2011.
3. M. Calonder, V. Lepetit, C. Strecha and P. Fua, "BRIEF: Binary Robust Independent Elementary Features", ECCV 2010.
4. D. G. Lowe, "Distinctive image features from scale-invariant keypoints", International Journal of Computer Vision 60(2), 2004 (ratio test).
5. M. A. Fischler and R. C. Bolles, "Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography", Communications of the ACM 24(6), 1981.
6. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed., Cambridge University Press, 2004 (DLT, normalisation and robust estimation: Chapter 4).
7. OpenCV, `modules/features2d/src/orb.cpp`, table `bit_pattern_31_` (BSD license, Willow Garage / OpenCV team), https://github.com/opencv/opencv
8. S. Barrett, `stb_image.h` v2.30 and `stb_image_write.h` v1.16, public domain, https://github.com/nothings/stb
9. R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., Springer, 2022 (feature detection and matching, image alignment, stitching).
10. S. Nayar, "Image Stitching", First Principles of Computer Vision monograph FPCV-2-4, 2025.
11. NVIDIA, *CUDA C++ Best Practices Guide* (APOD design cycle).

**External code used**

| Item | Used in | Source | License |
|---|---|---|---|
| ORB sampling pattern (256 pairs) | `src/v0_sequential.cpp`, `scripts/orb_pattern_data.py` | [7] | BSD |
| stb_image, stb_image_write (unmodified) | `src/stb_image.h`, `src/stb_image_write.h`, `src/stb_impl.*` | [8] | Public domain |
| OpenCV (`cv2`), independent reference only | notebook cell 6.5 | https://opencv.org | OpenCV license |

The other source files carry the author header of the team members. Libraries used by the notebook: NumPy, pandas, Matplotlib, Pillow.
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
