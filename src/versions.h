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

// V1: Pipeline CUDA cơ bản (Triển khai trong Phase C, Plan v4 Mục 18)
// Điều phối viên, Workspace & Bounds: cuda_pipeline.cu/.cuh (Phạm Minh Thông - 23120364)
// Stage 1 kernels (Preprocess): cuda_stage1_preprocess.cu (Phạm Minh Thông - 23120364)
// Stage 2 kernels (FAST+NMS+TopK): cuda_stage2_fast_nms_topk.cu (Nguyễn Lê Tấn Tiến - 23120369)
// Stage 3 kernels (ORB Descriptors): cuda_stage3_orb.cu (Nguyễn Lê Tấn Tiến - 23120369)
// Stage 4 kernels (Hamming Matching): cuda_stage4_matching.cu (Đào Quốc Tuấn - 23120392)
// Stage 5 kernels (RANSAC DLT + Fallback): cuda_stage5_ransac.cu (Phạm Minh Thông - 23120364)
// Stage 6 kernels (Warping & Blending): cuda_stage6_warp.cu (Đào Quốc Tuấn - 23120392)
#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
#include "cuda_pipeline.cuh"
// run_v1_basic_cuda đã được khai báo trong cuda_pipeline.cuh
#endif

// V2 & V3: Các phiên bản tối ưu hóa (Tên và thứ tự được khóa sau khi profile V1)
// PipelineResult run_v2_opt1(...);
// PipelineResult run_v3_opt2(...);

