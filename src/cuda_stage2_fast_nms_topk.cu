// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: Stage 2 CUDA Kernels — FAST-9 Detection + 3×3 NMS + Deterministic Top-K
// Tác giả: Pham Minh Thong (23120364)
// Phụ trách: Stage 2 GPU kernels (phát hiện điểm đặc trưng)
// External sources: Thuật toán FAST-9 theo Rosten & Drummond (2006);
//   offset vòng tròn Bresenham bán kính 3 và shortcut điều kiện từ V0
//   (v0_sequential.cpp Stage 2). CUB DeviceRadixSort từ CUDA Toolkit
//   (developer.nvidia.com/cub). Không sao chép kernel từ nguồn ngoài.
// ============================================================================

#include "cuda_pipeline.cuh"
#include <cub/cub.cuh>

// ============================================================================
// Offset vòng tròn Bresenham bán kính 3 — KHÓA theo V0 dòng 143-144
// 16 pixel trên đường tròn theo thứ tự chiều kim đồng hồ
// ============================================================================
__constant__ int d_circle_x[16] = { 0,  1,  2,  3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };
__constant__ int d_circle_y[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3,  3,  2,  1,  0, -1, -2, -3 };

// ============================================================================
// Helper: chuyển float sang dạng int có thể dùng unsigned sort
// (flip sign bit; nếu âm flip tất cả để sort ascending = response descending)
// ============================================================================
__device__ __forceinline__ uint32_t floatToOrderedInt(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ============================================================================
// Kernel 1: FAST-9 Response Map
// Mỗi thread xử lý một pixel; các pixel trên border bị bỏ qua.
// Kết quả: d_resp[y*w+x] = response score (>0 nếu là corner, 0 otherwise)
//
// Thuật toán khớp chính xác V0 (v0_sequential.cpp dòng 146-200):
//   - Shortcut: brighter>=2 OR darker>=2 trong 4 điểm {0,4,8,12} (FAST-9, không phải FAST-12)
//   - Full arc: i=0..30 (31 bước, wrap modulo 16), đếm run liên tiếp >= 9
//   - Response = sum max(0, |circle[i] - center| - thresh) cho i=0..15
// ============================================================================
__global__ void kernel_fast_response(const float* __restrict__ d_blur,
                                      float* __restrict__ d_resp,
                                      int w, int h) {
    // 1D grid, mỗi thread xử lý một pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % w;
    int y = idx / w;

    if (idx >= w * h) return;

    // Border margin BORDER_MARGIN=20 (Plan v4 Mục 3)
    if (x < BORDER_MARGIN || x >= w - BORDER_MARGIN ||
        y < BORDER_MARGIN || y >= h - BORDER_MARGIN) {
        d_resp[idx] = 0.0f;
        return;
    }

    float center = d_blur[idx];
    const float thresh = FAST_THRESH; // 20.0f (Plan v4 Mục 3)

    // Đọc 16 pixel vòng tròn
    float circle[16];
    for (int i = 0; i < 16; i++) {
        int cx = x + d_circle_x[i];
        int cy = y + d_circle_y[i];
        circle[i] = d_blur[cy * w + cx];
    }

    // Fast rejection: kiểm tra {0,4,8,12} — cần brighter>=2 OR darker>=2
    // (FAST-9 shortcut, V0 dòng 156-167)
    int brighter = (circle[0]  > center + thresh) +
                   (circle[4]  > center + thresh) +
                   (circle[8]  > center + thresh) +
                   (circle[12] > center + thresh);
    int darker   = (circle[0]  < center - thresh) +
                   (circle[4]  < center - thresh) +
                   (circle[8]  < center - thresh) +
                   (circle[12] < center - thresh);
    if (brighter < 2 && darker < 2) {
        d_resp[idx] = 0.0f;
        return;
    }

    // Full arc test: i=0..30 (31 bước), tìm run liên tiếp dài nhất >= 9
    int bright_run = 0, max_bright_run = 0;
    int dark_run   = 0, max_dark_run   = 0;
    for (int i = 0; i < 31; i++) {
        float val = circle[i % 16];
        if (val > center + thresh) {
            bright_run++;
            if (bright_run > max_bright_run) max_bright_run = bright_run;
        } else {
            bright_run = 0;
        }
        if (val < center - thresh) {
            dark_run++;
            if (dark_run > max_dark_run) max_dark_run = dark_run;
        } else {
            dark_run = 0;
        }
    }

    if (max_bright_run >= 9 || max_dark_run >= 9) {
        // Tính response score = sum |circle[i]-center| - thresh (i=0..15, chỉ khi |diff|>thresh)
        float score = 0.0f;
        for (int i = 0; i < 16; i++) {
            float diff = fabsf(circle[i] - center);
            if (diff > thresh) score += (diff - thresh);
        }
        d_resp[idx] = score;
    } else {
        d_resp[idx] = 0.0f;
    }
}

// ============================================================================
// Kernel 2: 3×3 NMS + Atomic Compaction
// Mỗi thread kiểm tra một pixel; nếu là local max trong 3×3 → thêm vào d_cand
// NMS: strict ">" (V0 dòng 234: response_map[neighbor] >= r → not max)
// Atomic: atomicAdd(d_ncand, 1) → chỉ số vị trí; guard nếu vượt MAX_CANDIDATES
// ============================================================================
__global__ void kernel_nms_compact(const float* __restrict__ d_resp,
                                    Keypoint* __restrict__ d_cand,
                                    int* __restrict__ d_ncand,
                                    int w, int h) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx % w;
    int y = idx / w;
    if (idx >= w * h) return;

    // Chỉ xét vùng bên trong border margin (nhất quán với FAST kernel)
    if (x < BORDER_MARGIN || x >= w - BORDER_MARGIN ||
        y < BORDER_MARGIN || y >= h - BORDER_MARGIN) return;

    float r = d_resp[idx];
    if (r <= 0.0f) return;

    // Kiểm tra 3×3 strict max (V0: nếu bất kỳ neighbor >= r thì không phải max)
    bool is_max = true;
    for (int dy = -1; dy <= 1 && is_max; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            if (d_resp[(y + dy) * w + (x + dx)] >= r) {
                is_max = false;
                break;
            }
        }
    }

    if (is_max) {
        int pos = atomicAdd(d_ncand, 1);
        if (pos < MAX_CANDIDATES) {
            Keypoint kp;
            kp.x        = static_cast<float>(x);
            kp.y        = static_cast<float>(y);
            kp.angle    = 0.0f;
            kp.response = r;
            d_cand[pos] = kp;
        }
        // pos >= MAX_CANDIDATES: silent drop (không viết ngoài buffer)
    }
}

// ============================================================================
// Kernel 3: Xây dựng composite sort key cho Top-K tất định
// Key 64-bit: upper 32 = ~floatToOrderedInt(response) (sort ASC = response DESC)
//             lower 32 = linear_index y*w+x (sort ASC = tọa độ nhỏ hơn trước)
// Mục đích: khớp chính xác tie-break của V0 (response DESC, linear_index ASC)
// ============================================================================
__global__ void kernel_build_sort_keys(const Keypoint* __restrict__ d_cand,
                                        uint64_t* __restrict__ d_keys,
                                        uint32_t* __restrict__ d_vals,
                                        int n, int w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Keypoint kp = d_cand[i];
    uint32_t li = static_cast<uint32_t>(static_cast<int>(kp.y) * w + static_cast<int>(kp.x));
    // upper 32: đảo floatToOrderedInt để sort ascending = response descending
    uint32_t resp_key = ~floatToOrderedInt(kp.response);
    uint64_t key = ((uint64_t)resp_key << 32) | (uint64_t)li;

    d_keys[i] = key;
    d_vals[i] = static_cast<uint32_t>(i);
}

// ============================================================================
// Kernel 4: Scatter Top-K candidates vào output keypoint array
// Dùng sorted vals array (index gốc) để đọc từ d_cand
// ============================================================================
__global__ void kernel_scatter_topk(const Keypoint* __restrict__ d_cand,
                                     const uint32_t* __restrict__ d_vals,
                                     Keypoint* __restrict__ d_kp,
                                     int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    d_kp[i] = d_cand[d_vals[i]];
}

// ============================================================================
// stage2_fast_nms_topk_gpu — API công khai (khai báo trong cuda_pipeline.cuh)
//
// Pipeline:
//   1. kernel_fast_response  — tính response map
//   2. kernel_nms_compact    — NMS + compact vào d_cand
//   3. D2H copy d_ncand      — biết n_cand
//   4. kernel_build_sort_keys — tạo composite 64-bit key
//   5. CUB DeviceRadixSort   — sắp xếp theo key (ascending = response DESC)
//   6. kernel_scatter_topk   — copy Top-K vào d_kp
//   7. H2D copy n_actual     — ghi d_nkp
//
// d_sort_tmp/sort_tmp_bytes: pre-allocated trong CudaWorkspace (có thể null)
//   Nếu null, hàm này tự cấp phát (chỉ xảy ra lần đầu cho mỗi image size).
//
// Trả về: elapsed ms của toàn bộ GPU work (cudaEvent bao quanh 4 kernel + sort)
// ============================================================================
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
                                int w, int h, int K) {
    const int total_pixels = w * h;

    // 1D Grid — Plan v4 C.3: "Block 256 1D cho FAST detection"
    const int BLOCK = 256;
    int grid_pixels = (total_pixels + BLOCK - 1) / BLOCK;

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start));

    // --- Pass 1: FAST response map ---
    kernel_fast_response<<<grid_pixels, BLOCK>>>(d_blur, d_resp, w, h);
    CHECK_CUDA(cudaGetLastError());

    // --- Pass 2: NMS + atomic compaction ---
    // d_ncand phải được reset về 0 bởi caller (CudaWorkspace::reset_counters)
    kernel_nms_compact<<<grid_pixels, BLOCK>>>(d_resp, d_cand, d_ncand, w, h);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    // --- D2H: lấy số candidates thực tế ---
    int n_cand = 0;
    CHECK_CUDA(cudaMemcpy(&n_cand, d_ncand, sizeof(int), cudaMemcpyDeviceToHost));

    if (n_cand == 0) {
        // Không có candidate nào — ghi 0 vào d_nkp và trả về
        int zero = 0;
        CHECK_CUDA(cudaMemcpy(d_nkp, &zero, sizeof(int), cudaMemcpyHostToDevice));
        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));
        return elapsed_ms;
    }

    // Giới hạn n_cand theo MAX_CANDIDATES (atomic có thể vượt nếu có nhiều corner)
    if (n_cand > MAX_CANDIDATES) n_cand = MAX_CANDIDATES;

    // Tiếp tục timing bao quanh sort và scatter
    cudaEvent_t ev_sort_start, ev_sort_stop;
    CHECK_CUDA(cudaEventCreate(&ev_sort_start));
    CHECK_CUDA(cudaEventCreate(&ev_sort_stop));
    CHECK_CUDA(cudaEventRecord(ev_sort_start));

    // --- Pass 3: Build composite sort keys ---
    int grid_cand = (n_cand + BLOCK - 1) / BLOCK;
    kernel_build_sort_keys<<<grid_cand, BLOCK>>>(d_cand, d_sort_keys, d_sort_vals, n_cand, w);
    CHECK_CUDA(cudaGetLastError());

    // Cần buffer output riêng cho CUB sort (in-place không được hỗ trợ)
    // Dùng phần sau của d_sort_keys/d_sort_vals làm output buffer (offset n_cand)
    // Đảm bảo workspace được cấp phát đủ 2×MAX_CANDIDATES trong CudaWorkspace
    uint64_t* d_keys_out = d_sort_keys + MAX_CANDIDATES;
    uint32_t* d_vals_out = d_sort_vals + MAX_CANDIDATES;

    // --- Pass 4: CUB DeviceRadixSort (ascending key = response DESC, index ASC) ---
    // Kiểm tra/cấp phát temp storage nếu chưa đủ
    size_t required_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, required_bytes,
                                     d_sort_keys, d_keys_out,
                                     d_sort_vals, d_vals_out,
                                     n_cand);
    void* tmp_storage = d_sort_tmp;
    if (tmp_storage == nullptr || sort_tmp_bytes < required_bytes) {
        // Caller chưa cấp phát (hoặc quá nhỏ) — cấp phát tạm thời
        CHECK_CUDA(cudaMalloc(&tmp_storage, required_bytes));
    }
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs(tmp_storage, required_bytes,
                                                d_sort_keys, d_keys_out,
                                                d_sort_vals, d_vals_out,
                                                n_cand));
    if (tmp_storage != d_sort_tmp) {
        // Giải phóng buffer tạm thời đã cấp phát nội bộ
        CHECK_CUDA(cudaFree(tmp_storage));
    }
    CHECK_CUDA(cudaGetLastError());

    // --- Pass 5: Scatter Top-K vào d_kp ---
    int actual_k = (n_cand < K) ? n_cand : K;
    int grid_topk = (actual_k + BLOCK - 1) / BLOCK;
    kernel_scatter_topk<<<grid_topk, BLOCK>>>(d_cand, d_vals_out, d_kp, actual_k);
    CHECK_CUDA(cudaGetLastError());

    // Ghi n_actual vào d_nkp (host→device copy của scalar)
    CHECK_CUDA(cudaMemcpy(d_nkp, &actual_k, sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(ev_sort_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_sort_stop));

    // Tổng elapsed = fast+nms time + sort+scatter time
    float t1 = 0.0f, t2 = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&t1, ev_start, ev_stop));
    CHECK_CUDA(cudaEventElapsedTime(&t2, ev_sort_start, ev_sort_stop));

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaEventDestroy(ev_sort_start));
    CHECK_CUDA(cudaEventDestroy(ev_sort_stop));

    return t1 + t2;
}
