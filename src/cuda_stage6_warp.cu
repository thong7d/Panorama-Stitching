// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: Stage 6 CUDA Kernels — Inverse Warping + Bilinear Sampling + Feathering Blend
// Phụ trách & Tác giả chính: Dao Quoc Tuan (23120392)
// Nhóm tác giả: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// External sources: Richard Szeliski (2006) "Image Alignment and Stitching: A Tutorial";
//   NVIDIA CUDA C++ Programming Guide (developer.nvidia.com).
//   Bất biến thuật toán khớp 100% V0 CPU (v0_sequential.cpp Stage 6).
// ============================================================================

#include "cuda_pipeline.cuh"

// ============================================================================
// Helper: Lấy mẫu song tuyến tính (Bilinear Interpolation) trên GPU
// Khớp chính xác logic và làm tròn V0 CPU (v0_sequential.cpp dòng 1060-1085).
// Điểm ảnh ngoài miền [0, w-1) x [0, h-1) không được gọi vào hàm này.
// ============================================================================
__device__ __forceinline__ uchar4 sample_bilinear_gpu(const uchar4* __restrict__ img,
                                                       int w,
                                                       float x, float y) {
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = x - static_cast<float>(x0);
    float fy = y - static_cast<float>(y0);

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    uchar4 p00 = img[y0 * w + x0];
    uchar4 p10 = img[y0 * w + x1];
    uchar4 p01 = img[y1 * w + x0];
    uchar4 p11 = img[y1 * w + x1];

    // Làm tròn + 0.5f trước khi ép kiểu uint8_t khớp chính xác V0 CPU
    uint8_t r = static_cast<uint8_t>(w00 * static_cast<float>(p00.x) +
                                     w10 * static_cast<float>(p10.x) +
                                     w01 * static_cast<float>(p01.x) +
                                     w11 * static_cast<float>(p11.x) + 0.5f);
    uint8_t g = static_cast<uint8_t>(w00 * static_cast<float>(p00.y) +
                                     w10 * static_cast<float>(p10.y) +
                                     w01 * static_cast<float>(p01.y) +
                                     w11 * static_cast<float>(p11.y) + 0.5f);
    uint8_t b = static_cast<uint8_t>(w00 * static_cast<float>(p00.z) +
                                     w10 * static_cast<float>(p10.z) +
                                     w01 * static_cast<float>(p01.z) +
                                     w11 * static_cast<float>(p11.z) + 0.5f);

    return make_uchar4(r, g, b, 255);
}

// ============================================================================
// Kernel: Inverse Warping + Symmetric Feathering Blending
// Mỗi thread xử lý 1 pixel (cx, cy) trên canvas.
// Grid 2D, Block 16×16 (Plan v4 Mục C.3: 2D Block 16x16 cho image kernels).
// Canvas nằm trong hệ tọa độ frame A.
// ============================================================================
__global__ void kernel_stage6_warp_blend(const uchar4* __restrict__ d_imgA,
                                         const uchar4* __restrict__ d_imgB,
                                         int img_w,
                                         int img_h,
                                         Homography H,
                                         CanvasBounds bounds,
                                         uchar4* __restrict__ d_pano,
                                         uint8_t* __restrict__ d_pano_mask) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;

    if (cx >= bounds.width || cy >= bounds.height) return;

    int pixel_idx = cy * bounds.width + cx;

    // Tọa độ tương ứng trên frame A
    float ax = static_cast<float>(cx + bounds.x_min);
    float ay = static_cast<float>(cy + bounds.y_min);

    // Kiểm tra điểm có thuộc ảnh A không
    bool in_A = (ax >= 0.0f && ax < static_cast<float>(img_w) &&
                 ay >= 0.0f && ay < static_cast<float>(img_h));
    uchar4 col_A = make_uchar4(0, 0, 0, 0);
    if (in_A) {
        int iax = static_cast<int>(ax);
        int iay = static_cast<int>(ay);
        col_A = d_imgA[iay * img_w + iax];
    }

    // Chiếu tọa độ frame A sang frame B bằng H: p_B = H * p_A
    // Mẫu số tọa độ thuần nhất: w = H[6]*ax + H[7]*ay + H[8]
    float w = H.h[6] * ax + H.h[7] * ay + H.h[8];
    bool in_B = false;
    uchar4 col_B = make_uchar4(0, 0, 0, 0);
    float bx = 0.0f, by = 0.0f;

    if (fabsf(w) > 1e-6f) {
        bx = (H.h[0] * ax + H.h[1] * ay + H.h[2]) / w;
        by = (H.h[3] * ax + H.h[4] * ay + H.h[5]) / w;
        if (bx >= 0.0f && bx < static_cast<float>(img_w - 1) &&
            by >= 0.0f && by < static_cast<float>(img_h - 1)) {
            col_B = sample_bilinear_gpu(d_imgB, img_w, bx, by);
            in_B = (col_B.w > 0);
        }
    }

    uchar4 out_col = make_uchar4(0, 0, 0, 0);

    if (in_A && in_B) {
        // Symmetric Feathering dựa trên khoảng cách tới cả 4 biên của A và B
        float dist_A = fminf(fminf(ax, static_cast<float>(img_w - 1) - ax),
                             fminf(ay, static_cast<float>(img_h - 1) - ay));
        float dist_B = fminf(fminf(bx, static_cast<float>(img_w - 1) - bx),
                             fminf(by, static_cast<float>(img_h - 1) - by));

        float alpha = (dist_A + 1e-6f) / (dist_A + dist_B + 2e-6f);
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);

        out_col.x = static_cast<uint8_t>(alpha * static_cast<float>(col_A.x) +
                                         (1.0f - alpha) * static_cast<float>(col_B.x));
        out_col.y = static_cast<uint8_t>(alpha * static_cast<float>(col_A.y) +
                                         (1.0f - alpha) * static_cast<float>(col_B.y));
        out_col.z = static_cast<uint8_t>(alpha * static_cast<float>(col_A.z) +
                                         (1.0f - alpha) * static_cast<float>(col_B.z));
        out_col.w = 255;
    } else if (in_A) {
        out_col = col_A;
        out_col.w = 255;
    } else if (in_B) {
        out_col = col_B;
        out_col.w = 255;
    }

    d_pano[pixel_idx] = out_col;

    // Mask hợp lệ độc lập: 1 nếu thuộc A hoặc B, 0 nếu là nền ngoài canvas
    if (d_pano_mask != nullptr) {
        d_pano_mask[pixel_idx] = (in_A || in_B) ? 1 : 0;
    }
}

// ============================================================================
// stage6_warp_blend_gpu — API công khai (khai báo trong cuda_pipeline.cuh)
// Đo thời gian thực thi bằng cudaEvent trên stream được chỉ định.
// ============================================================================
float stage6_warp_blend_gpu(const uchar4*       d_imgA,
                            const uchar4*       d_imgB,
                            int                 img_w,
                            int                 img_h,
                            const Homography&   H_A_to_B,
                            const CanvasBounds& bounds,
                            uchar4*             d_pano,
                            uint8_t*            d_pano_mask,
                            cudaStream_t        stream) {
    if (d_imgA == nullptr || d_imgB == nullptr || d_pano == nullptr ||
        img_w <= 0 || img_h <= 0 || !bounds.is_valid ||
        bounds.width <= 0 || bounds.height <= 0 || bounds.total_pixels == 0) {
        std::cerr << "[Lỗi Stage 6 GPU] Tham số đầu vào hoặc bounds canvas không hợp lệ!\n";
        return 0.0f;
    }

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start, stream));

    // Lưới 2D Block 16x16
    dim3 block(16, 16);
    dim3 grid((bounds.width  + block.x - 1) / block.x,
              (bounds.height + block.y - 1) / block.y);

    kernel_stage6_warp_blend<<<grid, block, 0, stream>>>(
        d_imgA, d_imgB, img_w, img_h, H_A_to_B, bounds, d_pano, d_pano_mask);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));

    return elapsed_ms;
}
