// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: CudaWorkspace Implementation + V1 Pipeline Orchestrator (Phase C)
// Phụ trách & Trưởng nhóm: Pham Minh Thong (23120364)
//   - CudaWorkspace RAII, điều phối viên run_v1_basic_cuda, H2D/D2H
//   - Tích hợp Stage 1 (Preprocess GPU) và Stage 5 (RANSAC DLT GPU)
//   - Stubs CPU cho Stage 3-6 để kiểm tra tích hợp trong lúc bàn giao
// Phụ trách Stage 2 & 3: Nguyen Le Tan Tien (23120369)
// Phụ trách Stage 4 & 6: Dao Quoc Tuan (23120392)
// External sources: NVIDIA CUDA Best Practices Guide (developer.nvidia.com);
//   Plan v4 Mục 3.1, Mục 8 và Mục 18 (CudaWorkspace RAII, bounds validation).
// ============================================================================

#include "cuda_pipeline.cuh"
#include "v0_sequential.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <cub/cub.cuh>

// ============================================================================
// compute_canvas_bounds (Plan v4 Mục 3.1 & 18.2)
// Tính toán kích thước canvas trong hệ tọa độ frame A.
// Ánh xạ 4 góc của ảnh B về frame A thông qua inverse(H_A_to_B).
// Kiểm tra nghiêm ngặt: pole, finite, diện tích tối đa 4× mỗi chiều và 16× số pixel.
// ============================================================================
CanvasBounds compute_canvas_bounds(int wA, int hA, int wB, int hB, const Homography& H_A_to_B) {
    CanvasBounds b{};
    if (wA <= 0 || hA <= 0 || wB <= 0 || hB <= 0) {
        return b;
    }
    if (!H_A_to_B.is_valid || !H_A_to_B.check_finite()) {
        return b;
    }

    const float* h = H_A_to_B.h;

    // 1. Kiểm tra pole của H_A_to_B trên 4 góc của ảnh A
    // Mẫu số phép chiếu A -> B: den_A = h[6]*x + h[7]*y + h[8]
    // Nếu mẫu số đổi dấu hoặc xấp xỉ 0, đường chân trời/pole cắt qua ảnh A
    const double corners_A[4][2] = {
        {0.0, 0.0},
        {static_cast<double>(wA), 0.0},
        {static_cast<double>(wA), static_cast<double>(hA)},
        {0.0, static_cast<double>(hA)}
    };
    double first_den_A = 0.0;
    for (int i = 0; i < 4; i++) {
        double x = corners_A[i][0];
        double y = corners_A[i][1];
        double den = (double)h[6] * x + (double)h[7] * y + (double)h[8];
        if (!std::isfinite(den) || std::abs(den) < 1e-5) {
            return b; // Điểm kỳ dị hoặc pole cắt ngay tại góc
        }
        if (i == 0) first_den_A = den;
        if (den * first_den_A <= 0.0) {
            return b; // Pole cắt ngang qua ảnh A
        }
    }

    // 2. Tính ma trận nghịch đảo invH = H^{-1} để chiếu B về frame A
    double det = (double)h[0] * ((double)h[4] * h[8] - (double)h[5] * h[7])
               - (double)h[1] * ((double)h[3] * h[8] - (double)h[5] * h[6])
               + (double)h[2] * ((double)h[3] * h[7] - (double)h[4] * h[6]);

    if (std::abs(det) < 1e-12 || !std::isfinite(det)) {
        return b; // Singular matrix
    }

    double inv_det = 1.0 / det;
    double invH[9];
    invH[0] =  ((double)h[4] * h[8] - (double)h[5] * h[7]) * inv_det;
    invH[1] = -((double)h[1] * h[8] - (double)h[2] * h[7]) * inv_det;
    invH[2] =  ((double)h[1] * h[5] - (double)h[2] * h[4]) * inv_det;
    invH[3] = -((double)h[3] * h[8] - (double)h[5] * h[6]) * inv_det;
    invH[4] =  ((double)h[0] * h[8] - (double)h[2] * h[6]) * inv_det;
    invH[5] = -((double)h[0] * h[5] - (double)h[2] * h[3]) * inv_det;
    invH[6] =  ((double)h[3] * h[7] - (double)h[4] * h[6]) * inv_det;
    invH[7] = -((double)h[0] * h[7] - (double)h[1] * h[6]) * inv_det;
    invH[8] =  ((double)h[0] * h[4] - (double)h[1] * h[3]) * inv_det;

    for (int i = 0; i < 9; i++) {
        if (!std::isfinite(invH[i])) return b;
    }

    // 3. Kiểm tra pole của invH trên 4 góc của ảnh B
    // Mẫu số phép chiếu B -> A: den_B = invH[6]*bx + invH[7]*by + invH[8]
    const double corners_B[4][2] = {
        {0.0, 0.0},
        {static_cast<double>(wB), 0.0},
        {static_cast<double>(wB), static_cast<double>(hB)},
        {0.0, static_cast<double>(hB)}
    };

    double first_den_B = 0.0;
    for (int i = 0; i < 4; i++) {
        double bx = corners_B[i][0];
        double by = corners_B[i][1];
        double den = invH[6] * bx + invH[7] * by + invH[8];
        if (!std::isfinite(den) || std::abs(den) < 1e-5) {
            return b;
        }
        if (i == 0) first_den_B = den;
        if (den * first_den_B <= 0.0) {
            return b; // Pole cắt ngang qua ảnh B (đổi dấu mẫu số)
        }
    }

    // 4. Chiếu 4 góc của ảnh B về frame A
    double min_x = 0.0, max_x = static_cast<double>(wA);
    double min_y = 0.0, max_y = static_cast<double>(hA);

    for (int i = 0; i < 4; i++) {
        double bx = corners_B[i][0];
        double by = corners_B[i][1];
        double w = invH[6] * bx + invH[7] * by + invH[8];
        double ax = (invH[0] * bx + invH[1] * by + invH[2]) / w;
        double ay = (invH[3] * bx + invH[4] * by + invH[5]) / w;
        if (!std::isfinite(ax) || !std::isfinite(ay)) {
            return b;
        }
        // Giới hạn giá trị tọa độ trước khi ép kiểu để tránh undefined behavior
        if (std::abs(ax) > 1e5 || std::abs(ay) > 1e5) {
            return b;
        }
        min_x = std::min(min_x, ax);
        max_x = std::max(max_x, ax);
        min_y = std::min(min_y, ay);
        max_y = std::max(max_y, ay);
    }

    // Kiểm tra tính hữu hạn và phạm vi trước khi làm tròn sang int
    if (!std::isfinite(min_x) || !std::isfinite(max_x) ||
        !std::isfinite(min_y) || !std::isfinite(max_y)) {
        return b;
    }

    double floor_min_x = std::floor(min_x);
    double ceil_max_x  = std::ceil(max_x);
    double floor_min_y = std::floor(min_y);
    double ceil_max_y  = std::ceil(max_y);

    if (floor_min_x < -1e5 || ceil_max_x > 1e5 ||
        floor_min_y < -1e5 || ceil_max_y > 1e5) {
        return b;
    }

    b.x_min = static_cast<int>(floor_min_x);
    b.x_max = static_cast<int>(ceil_max_x);
    b.y_min = static_cast<int>(floor_min_y);
    b.y_max = static_cast<int>(ceil_max_y);
    b.width = b.x_max - b.x_min;
    b.height = b.y_max - b.y_min;
    b.total_pixels = static_cast<size_t>(b.width) * b.height;

    // Kiểm tra giới hạn an toàn Plan v4 Mục 3.1 & Mục 18.4:
    // "Tối đa 4× mỗi chiều và 16× số pixel lớn nhất của input"
    int max_allowed_w = wA * MAX_CANVAS_DIM_FACTOR;
    int max_allowed_h = hA * MAX_CANVAS_DIM_FACTOR;
    size_t max_allowed_pixels = static_cast<size_t>(wA * hA) * MAX_CANVAS_PIXELS_FACTOR;

    if (b.width <= 0 || b.height <= 0 ||
        b.width > max_allowed_w || b.height > max_allowed_h ||
        b.total_pixels > max_allowed_pixels) {
        b.is_valid = false;
        return b;
    }

    b.is_valid = true;
    return b;
}

// ============================================================================
// Move Constructor & Move Assignment cho CudaWorkspace
// Quản lý chuyển quyền sở hữu VRAM an toàn, không rò rỉ và không double-free
// ============================================================================
CudaWorkspace::CudaWorkspace(CudaWorkspace&& other) noexcept {
    *this = std::move(other);
}

CudaWorkspace& CudaWorkspace::operator=(CudaWorkspace&& other) noexcept {
    if (this != &other) {
        free_all(); // Giải phóng tài nguyên đích trước nếu đã có

        d_imgA = other.d_imgA;
        d_imgB = other.d_imgB;
        d_grayA = other.d_grayA;
        d_grayB = other.d_grayB;
        d_blurA = other.d_blurA;
        d_blurB = other.d_blurB;
        d_respA = other.d_respA;
        d_respB = other.d_respB;
        d_kpA = other.d_kpA;
        d_kpB = other.d_kpB;
        d_nkpA = other.d_nkpA;
        d_nkpB = other.d_nkpB;
        d_candA = other.d_candA;
        d_candB = other.d_candB;
        d_ncandA = other.d_ncandA;
        d_ncandB = other.d_ncandB;
        d_sort_tmp = other.d_sort_tmp;
        sort_tmp_bytes = other.sort_tmp_bytes;
        d_sort_keysA = other.d_sort_keysA;
        d_sort_keysB = other.d_sort_keysB;
        d_sort_valsA = other.d_sort_valsA;
        d_sort_valsB = other.d_sort_valsB;
        d_descA = other.d_descA;
        d_descB = other.d_descB;
        d_matches = other.d_matches;
        d_nmatches = other.d_nmatches;
        d_sample_indices = other.d_sample_indices;
        d_inlier_counts = other.d_inlier_counts;
        d_all_H = other.d_all_H;
        d_best_hyp = other.d_best_hyp;
        d_ransac_result = other.d_ransac_result;
        d_pano = other.d_pano;
        d_pano_mask = other.d_pano_mask;
        d_canvas_bounds = other.d_canvas_bounds;
        pano_capacity = other.pano_capacity;
        img_w = other.img_w;
        img_h = other.img_h;
        canvas_w = other.canvas_w;
        canvas_h = other.canvas_h;

        // Reset đối tượng nguồn về trạng thái rỗng
        other.d_imgA = nullptr;
        other.d_imgB = nullptr;
        other.d_grayA = nullptr;
        other.d_grayB = nullptr;
        other.d_blurA = nullptr;
        other.d_blurB = nullptr;
        other.d_respA = nullptr;
        other.d_respB = nullptr;
        other.d_kpA = nullptr;
        other.d_kpB = nullptr;
        other.d_nkpA = nullptr;
        other.d_nkpB = nullptr;
        other.d_candA = nullptr;
        other.d_candB = nullptr;
        other.d_ncandA = nullptr;
        other.d_ncandB = nullptr;
        other.d_sort_tmp = nullptr;
        other.sort_tmp_bytes = 0;
        other.d_sort_keysA = nullptr;
        other.d_sort_keysB = nullptr;
        other.d_sort_valsA = nullptr;
        other.d_sort_valsB = nullptr;
        other.d_descA = nullptr;
        other.d_descB = nullptr;
        other.d_matches = nullptr;
        other.d_nmatches = nullptr;
        other.d_sample_indices = nullptr;
        other.d_inlier_counts = nullptr;
        other.d_all_H = nullptr;
        other.d_best_hyp = nullptr;
        other.d_ransac_result = nullptr;
        other.d_pano = nullptr;
        other.d_pano_mask = nullptr;
        other.d_canvas_bounds = nullptr;
        other.pano_capacity = 0;
        other.img_w = 0;
        other.img_h = 0;
        other.canvas_w = 0;
        other.canvas_h = 0;
    }
    return *this;
}

// ============================================================================
// CudaWorkspace::allocate
// Cấp phát toàn bộ GPU buffer theo kích thước ảnh đầu vào.
// Phải gọi DUY NHẤT MỘT LẦN trước warm-up run.
// Plan v4 C.3: "CudaWorkspace RAII, allocation trước warm-up, reuse."
// ============================================================================
void CudaWorkspace::allocate(int w, int h) {
    if (w <= 0 || h <= 0 || w > MAX_IMG_W || h > MAX_IMG_H) {
        std::cerr << "[Lỗi CudaWorkspace] Kích thước ảnh ngoài phạm vi hỗ trợ: " << w << "x" << h << "\n";
        return;
    }

    img_w    = w;
    img_h    = h;
    canvas_w = 2 * w;
    canvas_h = h;

    const size_t N     = static_cast<size_t>(w) * h;

    // Kiểm tra VRAM khả dụng trước khi cấp phát (Plan v4 Section 18.2)
    size_t free_vram = 0, total_vram = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_vram, &total_vram));

    // Dự trù dung lượng canvas tĩnh: PANO_CAPACITY_FACTOR × N (đủ cho mọi H hợp lệ)
    pano_capacity = N * PANO_CAPACITY_FACTOR;

    // ---- Input images ----
    CHECK_CUDA(cudaMalloc(&d_imgA,  N * sizeof(uchar4)));
    CHECK_CUDA(cudaMalloc(&d_imgB,  N * sizeof(uchar4)));

    // ---- Stage 1: Grayscale + Blur ----
    CHECK_CUDA(cudaMalloc(&d_grayA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grayB, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_blurA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_blurB, N * sizeof(float)));

    // ---- Stage 2: Response maps ----
    CHECK_CUDA(cudaMalloc(&d_respA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_respB, N * sizeof(float)));

    // ---- Stage 2: Keypoints sau Top-K ----
    CHECK_CUDA(cudaMalloc(&d_kpA,   MAX_KEYPOINTS * sizeof(Keypoint)));
    CHECK_CUDA(cudaMalloc(&d_kpB,   MAX_KEYPOINTS * sizeof(Keypoint)));
    CHECK_CUDA(cudaMalloc(&d_nkpA,  sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_nkpB,  sizeof(int)));

    // ---- Stage 2: Candidate buffers (trước sort) ----
    CHECK_CUDA(cudaMalloc(&d_candA, MAX_CANDIDATES * sizeof(Keypoint)));
    CHECK_CUDA(cudaMalloc(&d_candB, MAX_CANDIDATES * sizeof(Keypoint)));
    CHECK_CUDA(cudaMalloc(&d_ncandA, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ncandB, sizeof(int)));

    // ---- Sort keys/vals: nhân đôi để có in/out buffer cho CUB ----
    CHECK_CUDA(cudaMalloc(&d_sort_keysA, 2 * MAX_CANDIDATES * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_sort_keysB, 2 * MAX_CANDIDATES * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_sort_valsA, 2 * MAX_CANDIDATES * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_sort_valsB, 2 * MAX_CANDIDATES * sizeof(uint32_t)));

    // ---- CUB sort scratch: Query chính xác dung lượng cần thiết một lần duy nhất ----
    size_t cub_req = 0;
    cudaError_t cub_err = cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
        nullptr, cub_req, nullptr, nullptr, nullptr, nullptr, MAX_CANDIDATES);
    CHECK_CUDA(cub_err);
    sort_tmp_bytes = cub_req;
    CHECK_CUDA(cudaMalloc(&d_sort_tmp, sort_tmp_bytes));

    // ---- Stage 3: ORB Descriptors (Tiến phụ trách) ----
    CHECK_CUDA(cudaMalloc(&d_descA, MAX_DESCRIPTORS * sizeof(Descriptor)));
    CHECK_CUDA(cudaMalloc(&d_descB, MAX_DESCRIPTORS * sizeof(Descriptor)));

    // ---- Stage 4: Matches (Tuấn phụ trách) ----
    CHECK_CUDA(cudaMalloc(&d_matches,  MAX_MATCHES * sizeof(Match)));
    CHECK_CUDA(cudaMalloc(&d_nmatches, sizeof(int)));

    // ---- Stage 5: RANSAC (Thông phụ trách - Plan v4 Mục 18.2) ----
    CHECK_CUDA(cudaMalloc(&d_sample_indices, MAX_RANSAC_HYPOTHESES * 4 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_inlier_counts,  MAX_RANSAC_HYPOTHESES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_all_H,          MAX_RANSAC_HYPOTHESES * 9 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_best_hyp,       sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ransac_result,  sizeof(GpuRansacResult)));

    // ---- Stage 6: Panorama output & Mask (Tuấn phụ trách) ----
    CHECK_CUDA(cudaMalloc(&d_pano,          pano_capacity * sizeof(uchar4)));
    CHECK_CUDA(cudaMalloc(&d_pano_mask,     pano_capacity * sizeof(uint8_t)));
    CHECK_CUDA(cudaMalloc(&d_canvas_bounds, 6 * sizeof(int))); // x_min, y_min, x_max, y_max, w, h

    std::cout << "[CudaWorkspace] Đã cấp phát tĩnh GPU memory: "
              << img_w << "x" << img_h << " px | Pano capacity: " << pano_capacity << " px"
              << " | VRAM còn trống: " << (free_vram / (1024 * 1024)) << " MB\n";
}

// ============================================================================
// CudaWorkspace::reset_counters
// Reset tất cả atomic counter và kết quả device về 0 trước mỗi run (kể cả warm-up).
// ============================================================================
void CudaWorkspace::reset_counters() {
    if (d_nkpA)          CHECK_CUDA(cudaMemset(d_nkpA, 0, sizeof(int)));
    if (d_nkpB)          CHECK_CUDA(cudaMemset(d_nkpB, 0, sizeof(int)));
    if (d_ncandA)        CHECK_CUDA(cudaMemset(d_ncandA, 0, sizeof(int)));
    if (d_ncandB)        CHECK_CUDA(cudaMemset(d_ncandB, 0, sizeof(int)));
    if (d_nmatches)      CHECK_CUDA(cudaMemset(d_nmatches, 0, sizeof(int)));
    if (d_best_hyp)      CHECK_CUDA(cudaMemset(d_best_hyp, 0, sizeof(int)));
    if (d_ransac_result) CHECK_CUDA(cudaMemset(d_ransac_result, 0, sizeof(GpuRansacResult)));
}

// ============================================================================
// CudaWorkspace::free_all
// Giải phóng toàn bộ GPU memory — gọi từ destructor.
// Mọi pointer được null sau khi free để đảm bảo an toàn tuyệt đối khi gọi lại.
// ============================================================================
void CudaWorkspace::free_all() {
    auto safe_free = [](void*& ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    };
    safe_free(reinterpret_cast<void*&>(d_imgA));
    safe_free(reinterpret_cast<void*&>(d_imgB));
    safe_free(reinterpret_cast<void*&>(d_grayA));
    safe_free(reinterpret_cast<void*&>(d_grayB));
    safe_free(reinterpret_cast<void*&>(d_blurA));
    safe_free(reinterpret_cast<void*&>(d_blurB));
    safe_free(reinterpret_cast<void*&>(d_respA));
    safe_free(reinterpret_cast<void*&>(d_respB));
    safe_free(reinterpret_cast<void*&>(d_kpA));
    safe_free(reinterpret_cast<void*&>(d_kpB));
    safe_free(reinterpret_cast<void*&>(d_nkpA));
    safe_free(reinterpret_cast<void*&>(d_nkpB));
    safe_free(reinterpret_cast<void*&>(d_candA));
    safe_free(reinterpret_cast<void*&>(d_candB));
    safe_free(reinterpret_cast<void*&>(d_ncandA));
    safe_free(reinterpret_cast<void*&>(d_ncandB));
    safe_free(reinterpret_cast<void*&>(d_sort_keysA));
    safe_free(reinterpret_cast<void*&>(d_sort_keysB));
    safe_free(reinterpret_cast<void*&>(d_sort_valsA));
    safe_free(reinterpret_cast<void*&>(d_sort_valsB));
    safe_free(d_sort_tmp);
    safe_free(reinterpret_cast<void*&>(d_descA));
    safe_free(reinterpret_cast<void*&>(d_descB));
    safe_free(reinterpret_cast<void*&>(d_matches));
    safe_free(reinterpret_cast<void*&>(d_nmatches));
    safe_free(reinterpret_cast<void*&>(d_sample_indices));
    safe_free(reinterpret_cast<void*&>(d_inlier_counts));
    safe_free(reinterpret_cast<void*&>(d_all_H));
    safe_free(reinterpret_cast<void*&>(d_best_hyp));
    safe_free(reinterpret_cast<void*&>(d_ransac_result));
    safe_free(reinterpret_cast<void*&>(d_pano));
    safe_free(reinterpret_cast<void*&>(d_pano_mask));
    safe_free(reinterpret_cast<void*&>(d_canvas_bounds));
    sort_tmp_bytes = 0;
    pano_capacity = 0;
    img_w = img_h = canvas_w = canvas_h = 0;
}

// ============================================================================
// Helper: tính median của vector float
// ============================================================================
static float compute_median(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) * 0.5f : v[n/2];
}

// ============================================================================
// Helper: điền StageMetric từ vector raw runs
// ============================================================================
static StageMetric make_timing_entry(const std::vector<float>& runs) {
    StageMetric e{};
    if (runs.empty()) return e;
    float sum = 0.0f;
    float mn  = runs[0];
    for (float v : runs) { sum += v; if (v < mn) mn = v; }
    e.avg    = sum / static_cast<float>(runs.size());
    e.median = compute_median(runs);
    e.min    = mn;
    return e;
}

// ============================================================================
// run_v1_basic_cuda — Điều phối viên V1 Pipeline (Scaffold Hybrid)
// ============================================================================
PipelineResult run_v1_basic_cuda(const Image& imgA, const Image& imgB, int runs) {
    PipelineResult result;
    result.version_str = "V1";
    result.optimization_str = "V1 Hybrid Scaffold (S1+S2+S5 GPU, S3+S4+S6 CPU Stub - NOT Full V1 Benchmark)";
    result.runs = runs;
    result.warmups = 1;
    result.image_width = imgA.w;
    result.image_height = imgA.h;

    // Validate kích thước đầu vào và tham số (Plan v4 Mục 18.2)
    if (imgA.empty() || imgB.empty() || imgA.data == nullptr || imgB.data == nullptr ||
        imgA.w <= 0 || imgA.h <= 0 || imgB.w <= 0 || imgB.h <= 0 ||
        imgA.w != imgB.w || imgA.h != imgB.h ||
        imgA.w > MAX_IMG_W || imgA.h > MAX_IMG_H || runs <= 0) {
        std::cerr << "[Lỗi Pipeline V1] Ảnh đầu vào hoặc tham số runs không hợp lệ!\n";
        result.diagnostics.status = PipelineStatus::FAILURE;
        result.diagnostics.warning = "Invalid input dimensions, null image data, or runs <= 0";
        return result;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        result.device_name = prop.name;
    } else {
        result.device_name = "CUDA GPU";
    }

    const int w = imgA.w;
    const int h = imgA.h;
    const size_t N = static_cast<size_t>(w) * h;

    // ---- Cấp phát GPU workspace (Setup một lần duy nhất, nằm ngoài timed benchmark loop) ----
    CpuTimer setup_timer;
    setup_timer.Start();
    CudaWorkspace ws;
    ws.allocate(w, h);
    float t_setup = setup_timer.Stop();
    result.timings.setup.avg    = t_setup;
    result.timings.setup.median = t_setup;
    result.timings.setup.min    = t_setup;

    if (ws.d_imgA == nullptr || ws.pano_capacity == 0) {
        std::cerr << "[Lỗi Pipeline V1] Cấp phát CudaWorkspace thất bại (không đủ VRAM)!\n";
        result.diagnostics.status = PipelineStatus::FAILURE;
        result.diagnostics.warning = "CudaWorkspace allocation failed: insufficient GPU memory";
        return result;
    }

    // ============================================================
    // Lambda: thực thi một lần đầy đủ pipeline và điền timing vào
    // các vector tương ứng. Dùng cho cả warm-up và benchmark runs.
    // ============================================================
    auto run_once = [&](
        std::vector<float>& t_h2d_runs,
        std::vector<float>& t_s1_runs,
        std::vector<float>& t_s2_runs,
        std::vector<float>& t_s3_runs,
        std::vector<float>& t_s4_runs,
        std::vector<float>& t_s5_runs,
        std::vector<float>& t_s6_runs,
        std::vector<float>& t_ctrl_runs,
        std::vector<float>& t_d2h_runs,
        std::vector<float>& t_e2e_runs,
        Image& out_pano,
        PipelineDiagnostics& diag,
        bool record
    ) -> bool {
        ws.reset_counters();

        CpuTimer e2e_timer;
        e2e_timer.Start();

        // ---- H2D: Copy ảnh lên GPU ----
        GpuTimer h2d_timer;
        h2d_timer.Start();
        CHECK_CUDA(cudaMemcpy(ws.d_imgA, imgA.data, N * sizeof(uchar4), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_imgB, imgB.data, N * sizeof(uchar4), cudaMemcpyHostToDevice));
        float t_h2d = h2d_timer.Stop();

        // ---- Stage 1: Grayscale + Gaussian Blur (GPU) ----
        float t_s1A = stage1_preprocess_gpu(ws.d_imgA, ws.d_grayA, ws.d_blurA, w, h);
        float t_s1B = stage1_preprocess_gpu(ws.d_imgB, ws.d_grayB, ws.d_blurB, w, h);
        float t_s1  = t_s1A + t_s1B;

        // ---- Stage 2: FAST-9 + NMS + Top-K (GPU) ----
        float t_s2A = stage2_fast_nms_topk_gpu(
            ws.d_blurA, ws.d_respA,
            ws.d_candA, ws.d_ncandA,
            ws.d_kpA,   ws.d_nkpA,
            ws.d_sort_keysA, ws.d_sort_valsA,
            ws.d_sort_tmp, ws.sort_tmp_bytes,
            w, h, DEFAULT_TOP_K);
        float t_s2B = stage2_fast_nms_topk_gpu(
            ws.d_blurB, ws.d_respB,
            ws.d_candB, ws.d_ncandB,
            ws.d_kpB,   ws.d_nkpB,
            ws.d_sort_keysB, ws.d_sort_valsB,
            ws.d_sort_tmp, ws.sort_tmp_bytes,
            w, h, DEFAULT_TOP_K);
        float t_s2 = t_s2A + t_s2B;

        if (t_s1 < 0.0f || t_s2A < 0.0f || t_s2B < 0.0f) {
            diag.status  = PipelineStatus::FAILURE;
            diag.warning = "GPU Stage 1 or Stage 2 execution failed (kernel error or scratch overflow)";
            return false;
        }

        // ---- D2H: Lấy kết quả Stage 2 về host cho CPU stub Stage 3 ----
        CpuTimer ctrl_timer;
        ctrl_timer.Start();

        int nkpA = 0, nkpB = 0;
        CHECK_CUDA(cudaMemcpy(&nkpA, ws.d_nkpA, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&nkpB, ws.d_nkpB, sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<Keypoint> h_kpA(nkpA), h_kpB(nkpB);
        if (nkpA > 0) CHECK_CUDA(cudaMemcpy(h_kpA.data(), ws.d_kpA, nkpA * sizeof(Keypoint), cudaMemcpyDeviceToHost));
        if (nkpB > 0) CHECK_CUDA(cudaMemcpy(h_kpB.data(), ws.d_kpB, nkpB * sizeof(Keypoint), cudaMemcpyDeviceToHost));

        GrayImage blurA_host(w, h);
        GrayImage blurB_host(w, h);
        CHECK_CUDA(cudaMemcpy(blurA_host.data, ws.d_blurA, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(blurB_host.data, ws.d_blurB, N * sizeof(float), cudaMemcpyDeviceToHost));
        float t_ctrl_s2_s3 = ctrl_timer.Stop();

        // ================================================================
        // Stage 3: ORB Descriptor — CPU STUB
        // TODO(Tiến): Thay bằng stage3_orb_descriptors_gpu()
        // ================================================================
        float t_s3 = 0.0f;
        std::vector<Descriptor> descA, descB;
        {
            CpuTimer s3_timer;
            s3_timer.Start();
            descA = stage3_orb_descriptors_cpu(blurA_host, h_kpA, t_s3);
            float dummy3b = 0.0f;
            descB = stage3_orb_descriptors_cpu(blurB_host, h_kpB, dummy3b);
            t_s3  = s3_timer.Stop();
        }

        // ================================================================
        // Stage 4: Hamming Matching — CPU STUB
        // TODO(Tuấn): Thay bằng stage4_matching_gpu()
        // ================================================================
        float t_s4 = 0.0f;
        std::vector<Match> matches;
        {
            matches = stage4_matching_cpu(descA, descB, t_s4);
        }

        // ================================================================
        // Stage 5: RANSAC Homography — GPU (Thông phụ trách - Plan v4 Mục 18.2)
        // ================================================================
        float t_s5 = 0.0f;
        Homography best_H{};
        int inliers = 0;
        bool homo_ok = false;
        {
            ctrl_timer.Start();
            int n_matches = static_cast<int>(matches.size());
            if (n_matches > 0) {
                CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), n_matches * sizeof(Match), cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaMemcpy(ws.d_nmatches, &n_matches, sizeof(int), cudaMemcpyHostToDevice));
            float t_ctrl_s4_s5 = ctrl_timer.Stop();
            t_ctrl_s2_s3 += t_ctrl_s4_s5;

            homo_ok = stage5_ransac_gpu(
                ws.d_matches, n_matches, ws.d_kpA, ws.d_kpB,
                ws, w, h, best_H, inliers, t_s5);
        }

        // Điền diagnostics và đánh giá trạng thái đúng chuẩn V0 (Plan v4 Section 18.2)
        diag.keypoints_a  = nkpA;
        diag.keypoints_b  = nkpB;
        diag.matches      = static_cast<int>(matches.size());
        diag.inliers      = inliers;
        diag.inlier_ratio = (matches.size() > 0)
                            ? static_cast<float>(inliers) / matches.size()
                            : 0.0f;
        diag.H = best_H;

        if (!homo_ok || inliers < 4) {
            diag.status  = PipelineStatus::FAILURE;
            diag.warning = "RANSAC failed: less than 4 inliers or singular system";
        } else if (inliers < std::max(12, static_cast<int>(0.15f * matches.size()))) {
            diag.status  = PipelineStatus::LOW_CONFIDENCE;
            diag.warning = best_H.used_translation_fallback
                ? "Unstable homography rejected; translation fallback has low consensus"
                : "Low inlier ratio or count; output may have alignment defects";
        } else if (best_H.used_translation_fallback) {
            diag.status  = PipelineStatus::SUCCESS;
            diag.warning = "Unstable homography rejected; robust translation fallback used";
        } else {
            diag.status  = PipelineStatus::SUCCESS;
            diag.warning = "None";
        }

        // ================================================================
        // Stage 6: Warp + Blend (CPU STUB trong bản scaffold)
        // Kiểm tra hợp lệ hình học của Canvas Bounds trước khi warp
        // ================================================================
        float t_s6 = 0.0f;
        bool warp_ok = false;
        if (diag.status != PipelineStatus::FAILURE) {
            CanvasBounds bounds = compute_canvas_bounds(w, h, w, h, best_H);
            if (!bounds.is_valid) {
                diag.status  = PipelineStatus::FAILURE;
                diag.warning = "Canvas bounds validation failed: exceeds safety threshold or singular projection";
            } else if (bounds.total_pixels > ws.pano_capacity) {
                diag.status  = PipelineStatus::FAILURE;
                diag.warning = "Canvas bounds validation failed: total pixels exceed preallocated workspace capacity";
            } else {
                // [STUB] Gọi V0 CPU warp+blend
                out_pano = stage6_warp_blend_cpu(imgA, imgB, best_H, warp_ok, t_s6);
                if (!warp_ok) {
                    diag.status  = PipelineStatus::FAILURE;
                    diag.warning = "Warping and blending failed: invalid canvas or interpolation error";
                }
            }
        }

        // D2H: trong bản scaffold hybrid, panorama được tạo bởi CPU stub nên không có D2H thực sự
        float t_d2h = 0.0f;

        float t_e2e = e2e_timer.Stop();

        if (record) {
            t_h2d_runs .push_back(t_h2d);
            t_s1_runs  .push_back(t_s1);
            t_s2_runs  .push_back(t_s2);
            t_s3_runs  .push_back(t_s3);
            t_s4_runs  .push_back(t_s4);
            t_s5_runs  .push_back(t_s5);
            t_s6_runs  .push_back(t_s6);
            t_ctrl_runs.push_back(t_ctrl_s2_s3);
            t_d2h_runs .push_back(t_d2h);
            t_e2e_runs .push_back(t_e2e);
        }
        return (diag.status != PipelineStatus::FAILURE);
    };

    // ---- Vectơ lưu raw timing ----
    std::vector<float> t_h2d, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6, t_ctrl, t_d2h, t_e2e;

    // ---- Warm-up: 1 lần đầy đủ, không ghi timing ----
    PipelineDiagnostics warmup_diag{};
    Image warmup_pano;
    std::cout << "[V1 Hybrid Scaffold] Warm-up run...\n";
    run_once(t_h2d, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6, t_ctrl, t_d2h, t_e2e,
             warmup_pano, warmup_diag, /*record=*/false);

    // ---- Benchmark runs ----
    std::cout << "[V1 Hybrid Scaffold] Bắt đầu " << runs << " benchmark run(s)...\n";
    std::cout << "  [LƯU Ý ĐỘ TRUNG THỰC]: S1, S2, S5 chạy thuần GPU; S3, S4, S6 là CPU stub phục vụ tích hợp.\n";
    PipelineDiagnostics last_diag{};
    Image last_pano;
    bool any_failure = false;
    std::string failure_warning;

    for (int r = 0; r < runs; r++) {
        PipelineDiagnostics run_diag{};
        bool ok = run_once(t_h2d, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6, t_ctrl, t_d2h, t_e2e,
                           last_pano, run_diag, /*record=*/true);
        last_diag = run_diag;
        if (!ok) {
            any_failure = true;
            failure_warning = run_diag.warning;
        }
        std::printf("[V1 Run %d/%d] S1(GPU): %.2f ms | S2(GPU): %.2f ms | S5(GPU): %.2f ms | Status: %s\n",
                    r + 1, runs,
                    t_s1.back(), t_s2.back(), t_s5.back(),
                    status_to_string(run_diag.status));
    }

    // ---- Tổng hợp timing & diagnostics ----
    result.diagnostics = last_diag;
    if (any_failure && result.diagnostics.status != PipelineStatus::FAILURE) {
        result.diagnostics.status = PipelineStatus::FAILURE;
        result.diagnostics.warning = "At least one run failed during benchmark: " + failure_warning;
    }
    result.panorama = std::move(last_pano);

    StageMetric h2d_entry  = make_timing_entry(t_h2d);
    StageMetric d2h_entry  = make_timing_entry(t_d2h);
    StageMetric ctrl_entry = make_timing_entry(t_ctrl);

    result.timings.h2d     = h2d_entry;
    result.timings.d2h     = d2h_entry;
    result.timings.control = ctrl_entry;

    result.timings.stage1_preprocess      = make_timing_entry(t_s1);
    result.timings.stage2_fast_nms_topk   = make_timing_entry(t_s2);
    result.timings.stage3_orb_descriptor  = make_timing_entry(t_s3);
    result.timings.stage4_matching        = make_timing_entry(t_s4);
    result.timings.stage5_ransac          = make_timing_entry(t_s5);
    result.timings.stage6_warping_blending = make_timing_entry(t_s6);

    // kernel_total: CHỈ TÍNH THỜI GIAN GPU KERNEL THỰC SỰ (S1 + S2 + S5), KHÔNG CỘNG CPU STUBS
    // Plan v4 Section 20.4: Tính trung thực của đo lường
    std::vector<float> t_gpu_kernels(runs);
    for (int r = 0; r < runs; r++) {
        t_gpu_kernels[r] = t_s1[r] + t_s2[r] + t_s5[r];
    }
    result.timings.kernel_total = make_timing_entry(t_gpu_kernels);
    result.timings.end_to_end   = make_timing_entry(t_e2e);

    std::printf("[V1 GPU Kernels S1+S2+S5] TB: %.3f ms | Trung vị: %.3f ms | Min: %.3f ms\n",
                result.timings.kernel_total.avg,
                result.timings.kernel_total.median,
                result.timings.kernel_total.min);
    std::printf("[V1 Scaffold Transfers] H2D: %.3f ms | Control/Bridge: %.3f ms | D2H: %.3f ms\n",
                h2d_entry.avg, ctrl_entry.avg, d2h_entry.avg);

    // Amdahl analysis cho V1 Scaffold
    if (result.timings.end_to_end.avg > 0.0f) {
        float par = result.timings.stage1_preprocess.avg
                  + result.timings.stage2_fast_nms_topk.avg
                  + result.timings.stage5_ransac.avg;
        result.amdahl_parallel_fraction = par / result.timings.end_to_end.avg;
        float seq_frac = 1.0f - result.amdahl_parallel_fraction;
        result.amdahl_max_speedup = (seq_frac > 1e-6f) ? (1.0f / seq_frac) : 999.0f;
    }

    std::cout << "[V1 Scaffold] Hoàn thành. Keypoints A=" << result.diagnostics.keypoints_a
              << " B=" << result.diagnostics.keypoints_b
              << " Matches=" << result.diagnostics.matches
              << " Inliers=" << result.diagnostics.inliers
              << " Status=" << status_to_string(result.diagnostics.status) << "\n";

    return result;
}
