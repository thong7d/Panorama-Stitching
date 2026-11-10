// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: CudaWorkspace + V1 Pipeline Declarations (Phase C, Plan v4 Mục 8 & 18)
// Phụ trách & Trưởng nhóm: Pham Minh Thong (23120364)
//   - Thiết kế CudaWorkspace, Stage 1 (Preprocess), Stage 5 (RANSAC DLT GPU)
//   - Dispatcher run_v1_basic_cuda, Makefile & Tích hợp hệ thống
//   - Khởi tạo nền tảng Stage 2 FAST+NMS+TopK để bàn giao
// Phụ trách Stage 2 & 3: Nguyen Le Tan Tien (23120369)
//   - Tiếp nhận Stage 2 và triển khai Stage 3 (ORB Descriptors GPU)
// Phụ trách Stage 4 & 6: Dao Quoc Tuan (23120392)
//   - Triển khai Stage 4 (Hamming Matching GPU) & Stage 6 (Warp/Blend GPU)
// External sources: Tài liệu NVIDIA CUDA Best Practices Guide, CUDA C++ Programming
//   Guide (developer.nvidia.com). Không sao chép kernel từ nguồn ngoài.
// ============================================================================

#pragma once

#include "common.h"
#include "cuda_utils.cuh"
#include <cstddef>
#include <cstdint>

// ============================================================================
// CAPACITY CONSTANTS (Plan v4 Mục C.3 & Section 3.1)
// Tất cả capacity được định nghĩa ở đây để Stage 1-6 dùng chung và nhất quán.
// Thay đổi chỉ ở một chỗ, không rải hằng số khắp nơi.
// ============================================================================

// Kích thước ảnh tối đa hỗ trợ (4K = 3840×2160)
constexpr int MAX_IMG_W    = 3840;
constexpr int MAX_IMG_H    = 2160;
constexpr int MAX_PIXELS   = MAX_IMG_W * MAX_IMG_H; // 8.294.400 pixels

// Hệ số an toàn và giới hạn kích thước Canvas (Plan v4 Mục 3.1 & 18.2)
// Giới hạn an toàn: tối đa 4× mỗi chiều và 16× số pixel lớn nhất của input
constexpr int    MAX_CANVAS_DIM_FACTOR    = 4;
constexpr size_t MAX_CANVAS_PIXELS_FACTOR = 16;
// Hệ số dung lượng cấp phát tĩnh trước cho canvas panorama (8× w×h)
// Hỗ trợ mọi ma trận H hợp lệ theo tiêu chí hình học Plan v4 (bao gồm fixture 3200x1200 = 4.167N)
constexpr size_t PANO_CAPACITY_FACTOR     = 8;

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

// Kết quả trung gian của Stage 5 trên GPU
struct GpuRansacResult {
    Homography H;
    int inliers;
    bool success;
};

// ============================================================================
// CANVAS BOUNDS (Plan v4 Mục 3.1 & 18.2)
// Tọa độ canvas nằm trong frame A.
// x_min, y_min có thể âm nếu ảnh B nằm dịch sang trái hoặc lên trên frame A.
// ============================================================================
struct CanvasBounds {
    int x_min = 0;
    int y_min = 0;
    int x_max = 0;
    int y_max = 0;
    int width = 0;
    int height = 0;
    size_t total_pixels = 0;
    bool is_valid = false;
};

// Hàm tính toán và kiểm tra tính hợp lệ hình học của Canvas Bounds
// Dùng chung cho cả V0, V1 scaffold và kernel Stage 6 GPU của Tuấn
CanvasBounds compute_canvas_bounds(int wA, int hA, int wB, int hB, const Homography& H_A_to_B);

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
    Keypoint* d_candA   = nullptr;   // Candidates A        [MAX_CANDIDATES]
    Keypoint* d_candB   = nullptr;   // Candidates B        [MAX_CANDIDATES]
    int*    d_ncandA    = nullptr;   // Số candidates A (atomic counter)
    int*    d_ncandB    = nullptr;   // Số candidates B (atomic counter)

    // Temporary sort buffer (CUB DeviceRadixSort scratch - pre-allocated trong allocate)
    void*   d_sort_tmp  = nullptr;
    size_t  sort_tmp_bytes = 0;

    // Sort keys/values buffer (composite key uint64 cho Top-K tất định)
    uint64_t* d_sort_keysA   = nullptr;  // [MAX_CANDIDATES]
    uint64_t* d_sort_keysB   = nullptr;
    uint32_t* d_sort_valsA   = nullptr;  // index gốc của candidate
    uint32_t* d_sort_valsB   = nullptr;

    // ---- Stage 3: ORB Descriptors (Tiến phụ trách) ----
    Descriptor* d_descA = nullptr;   // Descriptors A       [MAX_DESCRIPTORS]
    Descriptor* d_descB = nullptr;   // Descriptors B       [MAX_DESCRIPTORS]

    // ---- Stage 4: Matches (Tuấn phụ trách) ----
    Match*  d_matches   = nullptr;   // Matches A→B         [MAX_MATCHES]
    int*    d_nmatches  = nullptr;   // Số matches hợp lệ (scalar)

    // ---- Stage 5: RANSAC (Thông phụ trách - Plan v4 Mục 18.2) ----
    int*    d_sample_indices = nullptr; // Bảng 2000×4 indices sinh bởi LCG [MAX_RANSAC_HYPOTHESES * 4]
    int*    d_inlier_counts = nullptr;  // Số inliers cho từng hypothesis  [MAX_RANSAC_HYPOTHESES]
    float*  d_all_H         = nullptr;  // Ma trận H cho từng hypothesis   [MAX_RANSAC_HYPOTHESES * 9]
    int*    d_best_hyp      = nullptr;  // Chỉ số hypothesis tối ưu        [1 int]
    GpuRansacResult* d_ransac_result = nullptr; // Kết quả RANSAC trên device [1 struct]

    // ---- Stage 6: Panorama output & Mask (Tuấn phụ trách) ----
    uchar4* d_pano          = nullptr;  // Output panorama buffer  [pano_capacity × uchar4]
    uint8_t* d_pano_mask    = nullptr;  // Validity/blend mask     [pano_capacity × uint8_t]
    int*    d_canvas_bounds = nullptr;  // Canvas bounds device    [6 × int: x_min, y_min, x_max, y_max, w, h]
    size_t  pano_capacity   = 0;        // Số pixel tối đa cấp phát cho panorama

    // ---- Image dimensions & Canvas parameters ----
    int img_w = 0;
    int img_h = 0;
    int canvas_w = 0;
    int canvas_h = 0;

    // ---- Lifecycle ----
    CudaWorkspace() = default;

    // Cấp phát toàn bộ GPU memory theo kích thước ảnh đầu vào.
    // Kiểm tra VRAM thực tế trước khi cấp phát, đảm bảo không vượt quá dung lượng.
    // Phải gọi một lần duy nhất trước warm-up run.
    void allocate(int w, int h);

    // Reset atomic counters và kết quả device giữa các benchmark run (không deallocate).
    // Phải gọi trước mỗi run kể cả warm-up.
    void reset_counters();

    // Giải phóng toàn bộ GPU memory an toàn (idempotent: gọi lặp lại không lỗi).
    void free_all();

    ~CudaWorkspace() { free_all(); }

    // Cấm copy (tránh double-free VRAM)
    CudaWorkspace(const CudaWorkspace&) = delete;
    CudaWorkspace& operator=(const CudaWorkspace&) = delete;

    // Cho phép move có kiểm soát (chuyển quyền sở hữu VRAM an toàn)
    CudaWorkspace(CudaWorkspace&& other) noexcept;
    CudaWorkspace& operator=(CudaWorkspace&& other) noexcept;
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

// Stage 5: RANSAC Homography + Fallback Translation (Thông phụ trách - Plan v4 Mục 18.2)
// d_matches:   input  — mảng Match trên GPU [n_matches]
// n_matches:   input  — số lượng cặp khớp M
// d_kpA, d_kpB:input  — mảng Keypoint A và B trên GPU
// ws:          buffer — tham chiếu CudaWorkspace chứa các GPU buffer trung gian
// image_w, image_h:   — kích thước ảnh phục vụ kiểm tra hình học & fallback
// best_H:      output — kết quả ma trận Homography cuối cùng
// out_inliers: output — số lượng inliers của mô hình tốt nhất
// elapsed_ms:  output — thời gian thực thi thuần trên GPU đo bằng cudaEvent
// Kernel sinh mẫu ngẫu nhiên RANSAC bằng LCG seed 42 trên GPU (Plan v4 Mục 18.2)
__global__ void kernel_ransac_generate_samples(
    int* __restrict__ d_samples,
    int n_matches,
    int iters);

bool stage5_ransac_gpu(const Match*    d_matches,
                       int             n_matches,
                       const Keypoint* d_kpA,
                       const Keypoint* d_kpB,
                       CudaWorkspace&  ws,
                       int image_w,
                       int image_h,
                       Homography&     best_H,
                       int&            out_inliers,
                       float&          elapsed_ms);

// ============================================================================
// API CONTRACT FOR TIẾN (STAGE 3 — ORB DESCRIPTORS)
// File đích: src/cuda_stage3_orb.cu do Nguyễn Lê Tấn Tiến triển khai
// - d_blur: ảnh xám làm mịn trên GPU (float, w*h)
// - d_kp: mảng keypoints trên GPU (Keypoint[MAX_KEYPOINTS])
// - n_kp: số keypoint thực tế (<= MAX_KEYPOINTS)
// - d_desc: mảng descriptors xuất ra trên GPU (Descriptor[MAX_DESCRIPTORS])
// - w, h: kích thước ảnh gốc
// Trả về thời gian thực thi kernel GPU đo bằng cudaEvent (ms).
// ============================================================================
float stage3_orb_descriptors_gpu(const float*    d_blur,
                                 const Keypoint* d_kp,
                                 int             n_kp,
                                 Descriptor*     d_desc,
                                 int w, int h,
                                 cudaStream_t stream = 0);

// ============================================================================
// API CONTRACT FOR TUẤN (STAGE 4 — HAMMING MATCHING)
// File đích: src/cuda_stage4_matching.cu do Đào Quốc Tuấn triển khai
// - d_descA: mảng descriptor A (Descriptor[nA])
// - d_descB: mảng descriptor B (Descriptor[nB])
// - nA, nB: số lượng keypoints ảnh A và B (<= MAX_KEYPOINTS)
// - d_matches: mảng lưu cặp khớp xuất ra trên GPU (Match[MAX_MATCHES])
// - d_nmatches: con trỏ scalar trên device lưu số matches (atomic counter)
// - ratio_threshold: ngưỡng Lowe's ratio test (mặc định MATCH_RATIO = 0.75f)
// Trả về thời gian thực thi kernel GPU đo bằng cudaEvent (ms).
// ============================================================================
float stage4_matching_gpu(const Descriptor* d_descA,
                          int               nA,
                          const Descriptor* d_descB,
                          int               nB,
                          Match*            d_matches,
                          int*              d_nmatches,
                          float             ratio_threshold = RATIO_THRESH,
                          cudaStream_t      stream = 0);

// ============================================================================
// API CONTRACT FOR TUẤN (STAGE 6 — WARPING & BLENDING)
// File đích: src/cuda_stage6_warp.cu do Đào Quốc Tuấn triển khai
// - d_imgA, d_imgB: ảnh gốc RGB trên GPU (uchar4, w*h)
// - img_w, img_h: kích thước ảnh gốc
// - H_A_to_B: ma trận homography ánh xạ A->B trong hệ tọa độ frame A
// - bounds: thông số canvas bounds (x_min, y_min, width, height) đã kiểm tra hợp lệ
// - d_pano: buffer panorama đích trên GPU (uchar4, bounds.total_pixels <= pano_capacity)
// - d_pano_mask: (tùy chọn) buffer mask nhị phân hợp lệ trên GPU (uint8_t)
// Trả về thời gian thực thi kernel GPU đo bằng cudaEvent (ms).
// ============================================================================
float stage6_warp_blend_gpu(const uchar4*       d_imgA,
                            const uchar4*       d_imgB,
                            int                 img_w,
                            int                 img_h,
                            const Homography&   H_A_to_B,
                            const CanvasBounds& bounds,
                            uchar4*             d_pano,
                            uint8_t*            d_pano_mask = nullptr,
                            cudaStream_t        stream = 0);

// ============================================================================
// PIPELINE ENTRY POINT — V1 Basic CUDA
// Định nghĩa trong cuda_pipeline.cu; được gọi từ main.cu qua versions.h
// ============================================================================
PipelineResult run_v1_basic_cuda(const Image& imgA, const Image& imgB,
                                  int runs = DEFAULT_RUNS);

