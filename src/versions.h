// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Pipeline Version Declarations (Plan v4 Section 4.2)
// Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// ============================================================================

#pragma once

#include "common.h"
#include "v0_sequential.h"

// V0 Sequential C++ Baseline (Golden Reference)
// run_v0_pipeline(const Image& imgA, const Image& imgB, int runs = DEFAULT_RUNS);

// V1 Basic CUDA Pipeline (Implemented in Phase C)
PipelineResult run_v1_basic_cuda(const Image& imgA, const Image& imgB, int runs = DEFAULT_RUNS);

// V2 & V3 Optimization Versions (Names & order locked after profiling V1)
// PipelineResult run_v2_opt1(...);
// PipelineResult run_v3_opt2(...);
