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
PipelineResult run_v1_basic_cuda(const Image& imgA, const Image& imgB, int runs = DEFAULT_RUNS);

// V2 & V3: Các phiên bản tối ưu hóa (Tên và thứ tự được khóa sau khi profile V1)
// PipelineResult run_v2_opt1(...);
// PipelineResult run_v3_opt2(...);

