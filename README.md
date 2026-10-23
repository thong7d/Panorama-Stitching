# CSC14120 — CUDA-Accelerated Panorama Image Stitching
## Course Project: Parallel Programming

This repository implements and parallelizes an end-to-end **Panorama Image Stitching** pipeline using CUDA and modern C++17, comprising:
1. **Stage 1 (Preprocessing):** Luminance grayscale conversion ($Y = 0.299R + 0.587G + 0.114B$) and 5×5 Gaussian convolution ($\sigma = 1.0$).
2. **Stage 2 (Detection):** FAST-9 Corner Detector ($T=20$), 3×3 Non-Maximum Suppression (NMS), and deterministic Top-$K$ selection ($K=1000$) ordered by `(response DESC, linear_index ASC)`.
3. **Stage 3 (Description):** Oriented BRIEF / ORB 256-bit descriptor (8 words $\times$ 32-bit `uint32_t`), 256 OpenCV `bit_pattern_31_` test pairs, orientation angle via Intensity Centroid ($31\times 31$ patch), and border margin filtering ($R < 20$ px).
4. **Stage 4 (Matching):** Brute-force Hamming distance matching (hardware popcount) combined with Lowe's Ratio Test ($\text{dist}_1 < 0.75 \times \text{dist}_2$).
5. **Stage 5 (Homography Estimation):** RANSAC (2000 hypotheses), 4-point Normalized DLT (Hartley isotropic scaling), Gauss-Jordan elimination with relative partial pivoting, finite value checks, and deterministic LCG PRNG (seed 42).
6. **Stage 6 (Warping & Blending):** Inverse perspective warping in Frame A coordinates, bilinear interpolation sampling, and symmetric distance-to-border feathering blending.

---

## 📁 Repository Directory Structure (Conforming to Project Plan v4)

```text
Prj/
├── src/
│   ├── common.h               # RAII move-only data structures (Image, GrayImage, Keypoint, Match, H, Diagnostics, Metrics)
│   ├── cuda_utils.cuh         # CHECK_CUDA macro (exit code 5) and GPU/CPU timer wrappers
│   ├── versions.h             # Pipeline dispatcher interface (V0 to V5)
│   ├── stb_image.h            # Header-only image loading library (stb)
│   ├── stb_image_write.h      # Header-only image writing library (stb)
│   ├── stb_impl.h / .cpp      # Image I/O helper implementations
│   ├── v0_sequential.h / .cpp # 6-stage single-thread C++ pipeline (Golden Reference Baseline)
│   └── main.cu                # CLI entrypoint, telemetry metrics exporter, Amdahl speedup analysis
├── scripts/
│   ├── preprocess_dataset.py  # PNG/JPEG/HEIC loader, EXIF orientation, 16:9 center crop, multi-resolution export
│   ├── generate_synthetic_test.py # Synthetic scene generator with exact projective homography and grid points
│   ├── validate_metrics.py    # JSON metrics schema validator (Plan v4 Section 4.3)
│   ├── build_midterm_notebook.py # Automated notebook generator using nbformat
│   └── package_midterm.py     # Midterm zip packaging script (Plan v4 Section B3)
├── tests/
│   ├── test_cpu.cpp           # C++ unit tests: ORB pattern, RAII contracts, Hamming popcount, Homography
│   └── test_scripts.py        # Python integration tests: manifest, synthetic ground truth, preprocessor
├── dataset/
│   ├── manifest.json          # Dataset manifest catalog (indoor, outdoor, custom, challenging, provided)
│   ├── synthetic/             # Synthetic images with ground_truth.json across 3 resolutions
│   ├── site1_indoor/          # Indoor classroom/hallway scene (Small, Medium, Large)
│   ├── site2_outdoor/         # Outdoor campus plaza scene (Small, Medium, Large)
│   ├── site3_custom/          # Custom architectural scene (Small, Medium, Large)
│   ├── challenging/           # Challenging failure analysis scene (Small, Medium, Large)
│   └── provided/              # Slot for instructor-provided dataset (external blocker)
├── notebooks/
│   └── GroupXX_Midterm.ipynb  # Self-contained Jupyter Notebook for Midterm Submission (Week 7)
├── output/                    # Destination directory for stitched panoramas and telemetry JSON files
├── Makefile                   # Automated build system: make cuda, make cpu, make test, make clean
├── requirements.txt           # Python dependency specifications
└── project_plan_v4.md         # Single authoritative project execution plan
```

---

## ⚙️ Compilation & Execution Guide

### 1. On Google Colab (NVIDIA Tesla T4 GPU — Recommended Standard)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build CUDA binary with NVCC (sm_75 architecture for Turing T4)
make cuda

# 3. Execute test suites to verify V0 correctness
make test
python tests/test_scripts.py

# 4. Run V0 sequential baseline (5 repeated runs) and export JSON metrics
./stitch dataset/synthetic/left_small.jpg \
         dataset/synthetic/right_small.jpg \
         output/synthetic_pano_small.jpg \
         --version 0 --runs 5 --metrics-json output/synthetic_metrics_small.json
```

### 2. On Local Machine (Host-only CPU — Linux / macOS / WSL)
```bash
make cpu
./stitch <left.jpg> <right.jpg> <output.jpg> --version 0 --runs 5
```

---

## 📐 Aspect Ratio Standardization (16:9) & Resolutions

1. **Small (720p):** $1280 \times 720$ (0.92 MP) — Rapid correctness verification and pipeline regression testing.
2. **Medium (1080p):** $1920 \times 1080$ (2.07 MP) — Standard resolution for primary benchmarks and speedup evaluation.
3. **Large (4K UHD):** $3840 \times 2160$ ($8,294,400\text{ pixels}$) — Stress testing compute throughput, memory scaling, and sorting capacity.

---

## 🌐 Dataset & Google Drive Storage

The complete raw image dataset (including raw camera PNG files and high-resolution preprocessed images) is hosted on Google Drive:
- **Google Drive Dataset Folder:** [CSC14120 Panorama Dataset](https://drive.google.com/drive/folders/1vnD51fqHjY-HGqp365dTa1SX_0femIwG?usp=drive_link)
- **Included Content:** Raw captures for `site1_indoor`, `site2_outdoor`, `site3_custom`, `challenging`, and `synthetic` ground truth.

---

## 🐙 GitHub Repository Setup

To clone this repository and run locally:
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
make test
```

To push local code to a newly created GitHub remote:
```bash
git remote add origin https://github.com/<username>/<repo-name>.git
git branch -M main
git push -u origin main
```

---

## 📦 Midterm Submission Guide (Phase B Quality Gate)

- **Submission Timing:** Completed following **Phase B** (Stage B4: Midterm Submission Preparation).
- **Archive Format (`GroupXX_Midterm.zip`):**
  1. `GroupXX_Midterm.ipynb`: Notebook executed via "Runtime -> Run all" on Google Colab T4 with all outputs saved (GPU device info, per-stage Avg/Min timing tables, bottleneck plots, Amdahl analysis, and stitched panoramas with synthetic grid RMSE).
  2. `src/`: Clean C++/CUDA source files (`common.h`, `cuda_utils.cuh`, `versions.h`, `main.cu`, `stb_impl.*`, `v0_sequential.*`).
  3. `Makefile` and `README.md`.
  4. Google Drive link containing original raw images and preprocessed 16:9 images (placed in the first cell of the notebook).

To generate the submission archive automatically:
```bash
python scripts/package_midterm.py --group GroupXX
```
