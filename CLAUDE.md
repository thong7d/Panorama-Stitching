# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

All commands run from `src/`:

```bash
make cuda     # NVCC build for Tesla T4 (sm_75), output: stitch
make cpu      # g++ CPU-only build, output: stitch
make test     # Build + run C++ unit tests (test_cpu binary)
make clean    # Remove build artifacts
```

Run pipeline:
```bash
./stitch <left.jpg> <right.jpg> <output.jpg> --version 0 --runs 5 [--metrics-json <path>]
```

Run Python integration tests (from repo root):
```bash
pytest tests/test_scripts.py
```

## Exit Codes

- `0` Success
- `2` CLI/input error
- `3` Pipeline failure (< 4 inliers, singular H, invalid canvas)
- `4` Output write error
- `5` CUDA runtime error

## Architecture

Six-stage sequential pipeline (V0 baseline complete; V1–V5 CUDA versions pending):

| Stage | What |
|-------|------|
| 1 Preprocess | RGB→grayscale (Y=0.299R+0.587G+0.114B) + 5×5 Gaussian blur (σ=1.0) |
| 2 Detect | FAST-9 (threshold=20) + 3×3 NMS + deterministic Top-K (K≤1000) |
| 3 Describe | Oriented BRIEF/ORB 256-bit (8×32-bit words), OpenCV `bit_pattern_31_` table |
| 4 Match | Brute-force Hamming + Lowe's ratio test (best1 < 0.75×best2) |
| 5 Homography | RANSAC (2000 hypotheses) + normalized 4-point DLT, LCG PRNG seed=42 |
| 6 Warp/Blend | Inverse perspective warp in Frame A coords, bilinear interp, feathering |

**Determinism invariant:** RANSAC seed=42, Top-K breaks ties by linear index. V0 must produce identical output across runs.

## Key Source Files

- `src/common.h` — RAII move-only structs: `Image`, `GrayImage`, `Keypoint`, `Descriptor`, `Match`, `Homography`, `PipelineDiagnostics`, `TimingSummary`, `PipelineResult`
- `src/v0_sequential.h/.cpp` — 6-stage pipeline with per-stage CPU timing
- `src/main.cu` — CLI entrypoint, device detection, pipeline dispatch by `--version`
- `src/versions.h` — Forward declarations for V0–V5
- `src/cuda_utils.cuh` — `CHECK_CUDA` macro, `GpuTimer`/`CpuTimer`
- `src/stb_impl.h/.cpp` — Image I/O via stb_image/stb_image_write

## Data Structures

All structs are **move-only** (deleted copy constructor/assignment). Never shallow-copy image buffers. `Image` stores raw `uint8_t*` RGB, `GrayImage` stores `uint8_t*` grayscale.

## Dataset

`dataset/manifest.json` catalogs 5 pairs: `site1_indoor`, `site2_outdoor`, `site3_custom`, `challenging`, `provided`. Each has Small (720p), Medium (1080p), Large (4K) variants — all 16:9 center-cropped.

Synthetic ground truth at `dataset/synthetic/ground_truth.json` contains exact `H_A_to_B` homography and validation grids.

## Pipeline Versions

- **V0** — Sequential C++ baseline (complete, golden reference)
- **V1** — Basic CUDA port (Phase C, pending)
- **V2/V3** — Optimized CUDA (shared memory, tiling, etc.)
- **V4/V5** — Future advanced optimizations

Version selected via `--version N` flag in CLI.

## Python Scripts (`scripts/`)

- `preprocess_dataset.py` — PNG/JPEG/HEIC loader, EXIF correction, 16:9 crop, multi-res export
- `generate_synthetic_test.py` — Synthetic scene with exact projective homography
- `validate_metrics.py` — JSON metrics schema validator
- `build_midterm_notebook.py` — Jupyter notebook generator for Colab submission
