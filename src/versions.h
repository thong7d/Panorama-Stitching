// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama (Panorama Image Stitching)
// Module: Khai báo các phiên bản Pipeline (Plan v4 Mục 4.2)
// Tác giả: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
//          Dao Quoc Tuan (23120392)
// External sources: Không có; giao diện là mã gốc của đồ án.
// ============================================================================

#pragma once

#include "common.h"
#include "v0_sequential.h"

// V0: Baseline tuần tự C++ (Chuẩn tham chiếu độ chính xác)
// run_v0_pipeline(const Image& imgA, const Image& imgB, int runs = DEFAULT_RUNS);

// V1: Pipeline CUDA cơ bản (Triển khai trong Phase C)
// Điều phối viên và buffer GPU: cuda_pipeline.cu/.cuh (Thông)
// Stage 1 kernels: cuda_stage1_preprocess.cu (Thông)
// Stage 2 kernels: cuda_stage2_fast_nms_topk.cu (Thông)
// Stage 3-4 kernels: TODO Tiến (cuda_stage3_orb.cu, cuda_stage4_matching.cu)
// Stage 5-6 kernels: TODO Tuấn (cuda_stage5_ransac.cu, cuda_stage6_warp.cu)
#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
#include "cuda_pipeline.cuh"
// run_v1_basic_cuda đã được khai báo trong cuda_pipeline.cuh
#endif

// V2 & V3: Các phiên bản tối ưu hóa (Tên và thứ tự được khóa sau khi profile V1)
// PipelineResult run_v2_opt1(...);
// PipelineResult run_v3_opt2(...);

