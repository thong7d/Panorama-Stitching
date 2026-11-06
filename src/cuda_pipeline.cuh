// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: CudaWorkspace + V1 Pipeline Declarations (Phase C, Plan v4 Mục 8)
// Tác giả: Pham Minh Thong (23120364)
// Phụ trách: Thiết kế bộ nhớ GPU tĩnh (CudaWorkspace), Stage 1, Stage 2 kernels
//            và điều phối viên run_v1_basic_cuda
// Phụ trách Stage 3: Nguyen Le Tan Tien (23120369)
// Phụ trách Stage 4: Nguyen Le Tan Tien (23120369)
// Phụ trách Stage 5+6: Dao Quoc Tuan (23120392)
// External sources: Tài liệu NVIDIA CUDA Best Practices Guide, CUDA C++ Programming
//   Guide (developer.nvidia.com). Không sao chép kernel từ nguồn ngoài.
// ============================================================================

#pragma once

#include "common.h"
#include "cuda_utils.cuh"
#include <cstddef>

// ============================================================================
// CAPACITY CONSTANTS (Plan v4 Mục C.3)
// Tất cả capacity được định nghĩa ở đây để Stage 1-6 dùng chung và nhất quán.
// Thay đổi chỉ ở một chỗ, không rải hằng số khắp nơi.
// ============================================================================

// Kích thước ảnh tối đa hỗ trợ (4K = 3840×2160)
constexpr int MAX_IMG_W    = 3840;
constexpr int MAX_IMG_H    = 2160;
constexpr int MAX_PIXELS   = MAX_IMG_W * MAX_IMG_H; // 8.294.400 pixels

// Capacity cho FAST candidate buffer (Mục C.3: bounded w×h, không hơn)
// Thực tế ≪ w×h vì FAST + border constraint lọc rất nhiều
constexpr int MAX_CANDIDATES = MAX_PIXELS;

// Capacity keypoints sau Top-K (Plan v4 Section 3: K=2000)
constexpr int MAX_KEYPOINTS = DEFAULT_TOP_K; // 2000 mỗi ảnh

// Capacity Descriptor (mỗi Descriptor = 32 bytes = 8×uint32_t)
constexpr int MAX_DESCRIPTORS = MAX_KEYPOINTS;

// Capacity Matches (tối đa nA matches - một chiều A→B)
constexpr int MAX_MATCHES = MAX_KEYPOINTS;

// Capacity RANSAC hypotheses
constexpr int MAX_RANSAC_HYPOTHESES = RANSAC_ITERS; // 2000

// ============================================================================
// CUDA WORKSPACE - RAII quản lý toàn bộ bộ nhớ GPU của V1
// Cấp phát một lần trước warm-up, tái sử dụng cho mọi benchmark run.
// Bộ nhớ GPU không được giải phóng và tái cấp phát giữa các run.
// Plan v4 Mục C.3: "CudaWorkspace RAII, allocation trước warm-up, reuse."
// ============================================================================
struct CudaWorkspace {
    // ---- Input Images (H2D buffer) ----
    uchar4* d_imgA      = nullptr;   // Ảnh RGB A trên GPU  [MAX_PIXELS × uchar4]
    uchar4* d_imgB      = nullptr;   // Ảnh RGB B trên GPU  [MAX_PIXELS × uchar4]

    // ---- Stage 1: Grayscale + Gaussian Blur ----
    float*  d_grayA     = nullptr;   // Grayscale A         [MAX_PIXELS × float]
    float*  d_grayB     = nullptr;   // Grayscale B         [MAX_PIXELS × float]
    float*  d_blurA     = nullptr;   // Blurred grayscale A [MAX_PIXELS × float]
    float*  d_blurB     = nullptr;   // Blurred grayscale B [MAX_PIXELS × float]

    // ---- Stage 2: FAST response map + candidates ----
    float*  d_respA     = nullptr;   // FAST response map A [MAX_PIXELS × float]
    float*  d_respB     = nullptr;   // FAST response map B [MAX_PIXELS × float]

    // Keypoints sau NMS+TopK
    Keypoint* d_kpA     = nullptr;   // Keypoints A         [MAX_KEYPOINTS]
    Keypoint* d_kpB     = nullptr;   // Keypoints B         [MAX_KEYPOINTS]
    int*    d_nkpA      = nullptr;   // Số lượng kp A (scalar)
    int*    d_nkpB      = nullptr;   // Số lượng kp B (scalar)

    // Intermediate: candidate buffer (trước Top-K)
    // Layout: [idx (int), response (float)] interleaved để sorting
    // Dùng struct nhỏ gọn hơn Keypoint để sort nhanh hơn
    // Thực chất là Keypoint array — dùng Keypoint struct sẵn có
    Keypoint* d_candA   = nullptr;   // Candidates A        [MAX_CANDIDATES]
    Keypoint* d_candB   = nullptr;   // Candidates B        [MAX_CANDIDATES]
    int*    d_ncandA    = nullptr;   // Số candidates A (atomic counter)
    int*    d_ncandB    = nullptr;   // Số candidates B (atomic counter)

    // Temporary sort buffer (CUB DeviceRadixSort scratch)
    // Stage 2 sẽ cấp phát d_sort_tmp động trong lần đầu tiên nếu cần
    void*   d_sort_tmp  = nullptr;
    size_t  sort_tmp_bytes = 0;

    // Sort keys/values buffer (composite key uint64 cho Top-K tất định)
    uint64_t* d_sort_keysA   = nullptr;  // [MAX_CANDIDATES]
    uint64_t* d_sort_keysB   = nullptr;
    uint32_t* d_sort_valsA   = nullptr;  // index gốc của candidate
    uint32_t* d_sort_valsB   = nullptr;

    // ---- Stage 3: ORB Descriptors (Tiến phụ trách) ----
    // Mỗi Descriptor = 32 bytes = 8 × uint32_t (alignas 16)
    Descriptor* d_descA = nullptr;   // Descriptors A       [MAX_DESCRIPTORS]
    Descriptor* d_descB = nullptr;   // Descriptors B       [MAX_DESCRIPTORS]

    // ---- Stage 4: Matches (Tiến phụ trách) ----
    Match*  d_matches   = nullptr;   // Matches A→B         [MAX_MATCHES]
    int*    d_nmatches  = nullptr;   // Số matches hợp lệ (scalar)

    // ---- Stage 5: RANSAC (Tuấn phụ trách) ----
    // Inlier counts per hypothesis
    int*    d_inlier_counts = nullptr; // [MAX_RANSAC_HYPOTHESES]
    // Best H candidate (9 floats × MAX_RANSAC_HYPOTHESES)
    float*  d_all_H    = nullptr;    // [MAX_RANSAC_HYPOTHESES × 9]
    // Best hypothesis index (scalar, written by argmax kernel)
    int*    d_best_hyp  = nullptr;

    // ---- Stage 6: Panorama output (Tuấn phụ trách) ----
    uchar4* d_pano      = nullptr;   // Output panorama     [4 × MAX_CANVAS]
    // Canvas bounds (4 × int: min_x, max_x, min_y, max_y) — small control copy
    int*    d_canvas_bounds = nullptr;  // [4]

    // ---- Image dimensions (set at allocate time) ----
    int img_w = 0;
    int img_h = 0;
    int canvas_w = 0;
    int canvas_h = 0;

    // ---- Lifecycle ----
    CudaWorkspace() = default;

    // Cấp phát toàn bộ GPU memory theo kích thước ảnh đầu vào.
    // Phải gọi một lần duy nhất trước warm-up run.
    // canvas_w và canvas_h ước lượng (2×img_w × img_h) cho safe bound.
    void allocate(int w, int h);

    // Reset atomic counters giữa các benchmark run (không deallocate).
    // Phải gọi trước mỗi run kể cả warm-up.
    void reset_counters();

    // Giải phóng toàn bộ GPU memory — gọi từ destructor.
    void free_all();

    ~CudaWorkspace() { free_all(); }

    // Cấm copy, cho phép move
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;
    CudaWorkspace(CudaWorkspace&&) = default;
    CudaWorkspace& operator=(CudaWorkspace&&) = default;
};

// ============================================================================
// API DECLARATIONS — STAGE 1 & 2 (Thông phụ trách)
// Các hàm này được định nghĩa trong cuda_stage1_preprocess.cu và
// cuda_stage2_fast_nms_topk.cu tương ứng. main.cu / cuda_pipeline.cu gọi qua API.
// ============================================================================

// Stage 1: Grayscale + Gaussian Blur 5×5
// d_rgb:  input  — d_imgA hoặc d_imgB (uchar4, w×h pixels)
// d_gray: output — grayscale trung gian (float, w×h)
// d_blur: output — blurred grayscale (float, w×h)  ← đây là output chuẩn
// Timing: đo bằng cudaEvent bên trong; trả về elapsed ms
float stage1_preprocess_gpu(const uchar4* d_rgb,
                             float*        d_gray,
                             float*        d_blur,
                             int w, int h);

// Stage 2: FAST-9 + 3×3 NMS + Deterministic Top-K
// d_blur:   input  — blurred grayscale (float, w×h)
// d_resp:   output — FAST response map (float, w×h)
// d_cand:   buffer — candidate keypoints (Keypoint[MAX_CANDIDATES])
// d_ncand:  buffer — số candidates (int*, atomic counter, reset trước khi gọi)
// d_kp:     output — keypoints sau Top-K (Keypoint[MAX_KEYPOINTS])
// d_nkp:    output — số keypoints thực tế (int*, ≤ K)
// d_sort_keys/vals/tmp: temporary sort buffers (pre-allocated trong CudaWorkspace)
// Timing: đo bằng cudaEvent bên trong; trả về elapsed ms
float stage2_fast_nms_topk_gpu(const float*  d_blur,
                                float*        d_resp,
                                Keypoint*     d_cand,
                                int*          d_ncand,
                                Keypoint*     d_kp,
                                int*          d_nkp,
                                uint64_t*     d_sort_keys,
                                uint32_t*     d_sort_vals,
                                void*         d_sort_tmp,
                                size_t        sort_tmp_bytes,
                                int w, int h, int K);

// ============================================================================
// PIPELINE ENTRY POINT — V1 Basic CUDA
// Định nghĩa trong cuda_pipeline.cu; được gọi từ main.cu qua versions.h
// ============================================================================
PipelineResult run_v1_basic_cuda(const Image& imgA, const Image& imgB,
                                  int runs = DEFAULT_RUNS);

