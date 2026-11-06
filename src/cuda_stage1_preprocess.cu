// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: Stage 1 CUDA Kernels — Grayscale + Gaussian Blur 5×5
// Tác giả: Pham Minh Thong (23120364)
// Phụ trách: Stage 1 GPU kernels (tiền xử lý ảnh)
// External sources: NVIDIA CUDA C++ Programming Guide (developer.nvidia.com);
//   thuật toán Gaussian blur boundary-clamp và hệ số grayscale ITU-R BT.601
//   lấy từ V0 (v0_sequential.cpp Stage 1). Không sao chép kernel từ nguồn ngoài.
// ============================================================================

#include "cuda_pipeline.cuh"

// ============================================================================
// Gaussian kernel 5×5 trên __constant__ memory
// Giá trị KHÓA theo V0 (v0_sequential.cpp dòng 98-104) — KHÔNG được thay đổi.
// ============================================================================
__constant__ float d_gauss_kernel[25] = {
    0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f,
    0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
    0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f,
    0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
    0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f
};

// ============================================================================
// Kernel 1: RGB → Grayscale (ITU-R BT.601, hệ số khóa theo V0)
// Y = 0.299R + 0.587G + 0.114B
// Grid: 2D, Block 16×16
// d_rgb:  input  — ảnh RGB dạng uchar4 (w×h pixels)
// d_gray: output — grayscale dạng float (w×h, range [0, 255])
// ============================================================================
__global__ void kernel_rgb_to_gray(const uchar4* __restrict__ d_rgb,
                                    float* __restrict__ d_gray,
                                    int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    uchar4 p = d_rgb[y * w + x];
    // Hệ số khóa theo V0 v0_sequential.cpp dòng 117
    d_gray[y * w + x] = 0.299f * (float)p.x + 0.587f * (float)p.y + 0.114f * (float)p.z;
}

// ============================================================================
// Kernel 2: Gaussian Blur 5×5 với boundary clamping (không zero-pad)
// Khớp chính xác V0: std::clamp(x+kx, 0, w-1) cho pixel ngoài biên
// Grid: 2D, Block 16×16
// d_gray: input  — grayscale float (w×h)
// d_blur: output — blurred grayscale float (w×h)
// ============================================================================
__global__ void kernel_gaussian_blur(const float* __restrict__ d_gray,
                                      float* __restrict__ d_blur,
                                      int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float sum = 0.0f;
    // Duyệt kernel 5×5, offset ky in [-2,2], kx in [-2,2]
    for (int ky = -2; ky <= 2; ky++) {
        int py = max(0, min(h - 1, y + ky));       // boundary clamp
        for (int kx = -2; kx <= 2; kx++) {
            int px = max(0, min(w - 1, x + kx));   // boundary clamp
            // d_gauss_kernel row-major: index = (ky+2)*5 + (kx+2)
            sum += d_gray[py * w + px] * d_gauss_kernel[(ky + 2) * 5 + (kx + 2)];
        }
    }
    d_blur[y * w + x] = sum;
}

// ============================================================================
// stage1_preprocess_gpu — API công khai (khai báo trong cuda_pipeline.cuh)
// Thực hiện tuần tự: kernel_rgb_to_gray → kernel_gaussian_blur
// Đo thời gian bằng cudaEvent (Plan v4 C.3: kernel timing bằng cudaEvent)
// Trả về thời gian GPU elapsed ms (chỉ 2 kernel, không bao gồm H2D/D2H)
// ============================================================================
float stage1_preprocess_gpu(const uchar4* d_rgb,
                             float*        d_gray,
                             float*        d_blur,
                             int w, int h) {
    // Lưới 2D 16×16 — Plan v4 C.3: "Block 16×16 cho 2D image kernels"
    dim3 block(16, 16);
    dim3 grid((w + 15) / 16, (h + 15) / 16);

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start));

    // Pass 1: RGB → Grayscale
    kernel_rgb_to_gray<<<grid, block>>>(d_rgb, d_gray, w, h);
    CHECK_CUDA(cudaGetLastError());

    // Pass 2: Gaussian Blur 5×5
    kernel_gaussian_blur<<<grid, block>>>(d_gray, d_blur, w, h);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));

    return elapsed_ms;
}
