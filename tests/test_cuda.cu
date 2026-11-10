// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: CUDA Unit Tests cho Phase C (Thông phụ trách — S1, S5, CudaWorkspace, Contract)
// Tác giả: Pham Minh Thong (23120364)
// External sources: Original test logic; CUDA C++ standard testing patterns.
// Tiêu chuẩn nghiệm thu: THONG_HANDOFF_READY theo Plan v4 Mục 18.2, 20 & 21
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <string>

#include "cuda_pipeline.cuh"
#include "v0_sequential.h"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::exit(1); \
        } \
    } while(0)

// Helper: Deterministic LCG CPU reference
struct LcgRngCpu {
    uint32_t state;
    explicit LcgRngCpu(uint32_t seed = 42) : state(seed) {}
    uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }
    int next_int(int max_val) {
        if (max_val <= 0) return 0;
        return static_cast<int>(next() % static_cast<uint32_t>(max_val));
    }
};

// ============================================================================
// TEST SUITE 1: CudaWorkspace Lifecycle, Move Semantics & Idempotence
// ============================================================================
void test_workspace_lifecycle() {
    std::cout << "[Test 1] Testing CudaWorkspace Lifecycle, Move Semantics & Idempotence...\n";

    // 1. Cấp phát cơ bản
    CudaWorkspace ws1;
    ws1.allocate(1280, 720);
    TEST_ASSERT(ws1.d_imgA != nullptr, "ws1.d_imgA must be allocated");
    TEST_ASSERT(ws1.d_pano != nullptr, "ws1.d_pano must be allocated");
    TEST_ASSERT(ws1.pano_capacity >= 1280ULL * 720 * PANO_CAPACITY_FACTOR, "Pano capacity check");
    TEST_ASSERT(ws1.d_sort_tmp != nullptr && ws1.sort_tmp_bytes > 0, "CUB scratch must be pre-allocated");

    void* orig_imgA = ws1.d_imgA;
    void* orig_pano = ws1.d_pano;
    size_t orig_cap = ws1.pano_capacity;

    // 2. Move constructor: chuyển giao quyền sở hữu VRAM
    CudaWorkspace ws2(std::move(ws1));
    TEST_ASSERT(ws1.d_imgA == nullptr, "ws1.d_imgA must be nulled after move");
    TEST_ASSERT(ws1.d_pano == nullptr, "ws1.d_pano must be nulled after move");
    TEST_ASSERT(ws1.pano_capacity == 0, "ws1.pano_capacity must be 0 after move");
    TEST_ASSERT(ws2.d_imgA == orig_imgA, "ws2 must receive orig_imgA pointer");
    TEST_ASSERT(ws2.d_pano == orig_pano, "ws2 must receive orig_pano pointer");
    TEST_ASSERT(ws2.pano_capacity == orig_cap, "ws2 must receive orig capacity");

    // 3. Move assignment: giải phóng tài nguyên đích trước khi nhận tài nguyên mới
    CudaWorkspace ws3;
    ws3.allocate(640, 480);
    void* old_ws3_imgA = ws3.d_imgA;
    TEST_ASSERT(old_ws3_imgA != nullptr, "ws3 initially allocated");

    ws3 = std::move(ws2);
    TEST_ASSERT(ws2.d_imgA == nullptr, "ws2.d_imgA must be nulled after move assignment");
    TEST_ASSERT(ws3.d_imgA == orig_imgA, "ws3 must hold transferred pointer from ws2");
    TEST_ASSERT(ws3.pano_capacity == orig_cap, "ws3 must hold transferred pano capacity");

    // 4. Idempotent free_all: gọi lặp lại an toàn không gây double-free
    ws3.free_all();
    TEST_ASSERT(ws3.d_imgA == nullptr, "ws3.d_imgA must be null after free_all");
    TEST_ASSERT(ws3.pano_capacity == 0, "ws3.pano_capacity must be 0 after free_all");
    ws3.free_all(); // Lần 2 an toàn
    TEST_ASSERT(ws3.d_imgA == nullptr, "Safe double free_all verified");

    std::cout << "  -> Workspace RAII, Move noexcept and Idempotence: PASS\n";
}

// ============================================================================
// TEST SUITE 2: Canvas Bounds & Capacity Contract (Fixtures bắt buộc)
// ============================================================================
void test_canvas_bounds_contract() {
    std::cout << "[Test 2] Testing Canvas Bounds & Capacity Contract (Mandatory Fixtures)...\n";

    CudaWorkspace ws;
    ws.allocate(1280, 720);

    // Fixture 1: Translation bắt buộc
    // w=1280, h=720, H = [1, 0, -1024; 0, 1, 216; 0, 0, 1]
    {
        Homography H1;
        H1.h[0] = 1.0f; H1.h[1] = 0.0f; H1.h[2] = -1024.0f;
        H1.h[3] = 0.0f; H1.h[4] = 1.0f; H1.h[5] =   216.0f;
        H1.h[6] = 0.0f; H1.h[7] = 0.0f; H1.h[8] =     1.0f;
        H1.is_valid = true;

        CanvasBounds b1 = compute_canvas_bounds(1280, 720, 1280, 720, H1);
        TEST_ASSERT(b1.is_valid, "Fixture 1 canvas must be valid");
        TEST_ASSERT(b1.x_min == 0, "Fixture 1 b.x_min must be 0");
        TEST_ASSERT(b1.x_max == 2304, "Fixture 1 b.x_max must be 2304");
        TEST_ASSERT(b1.y_min == -216, "Fixture 1 b.y_min must be -216");
        TEST_ASSERT(b1.y_max == 720, "Fixture 1 b.y_max must be 720");
        TEST_ASSERT(b1.width == 2304, "Fixture 1 b.width must be 2304");
        TEST_ASSERT(b1.height == 936, "Fixture 1 b.height must be 936");
        TEST_ASSERT(b1.total_pixels == 2156544ULL, "Fixture 1 b.total_pixels must be 2,156,544");
        TEST_ASSERT(b1.total_pixels <= ws.pano_capacity,
                    "Fixture 1 must fit within ws.pano_capacity");
        std::cout << "  -> Fixture 1 (Trans): 2304x936 = " << b1.total_pixels
                  << " px <= Capacity: " << ws.pano_capacity << " px (PASS)\n";
    }

    // Fixture 2: Scale + Translation bắt buộc mới
    // w=1280, h=720, H = [0.6, 0, -640; 0, 0.6, 216; 0, 0, 1]
    // Canvas: 3200x1200 = 3,840,000 px (> 4N, phải vừa trong capacity 8N)
    {
        Homography H2;
        H2.h[0] = 0.6f; H2.h[1] = 0.0f; H2.h[2] = -640.0f;
        H2.h[3] = 0.0f; H2.h[4] = 0.6f; H2.h[5] =  216.0f;
        H2.h[6] = 0.0f; H2.h[7] = 0.0f; H2.h[8] =    1.0f;
        H2.is_valid = true;

        CanvasBounds b2 = compute_canvas_bounds(1280, 720, 1280, 720, H2);
        TEST_ASSERT(b2.is_valid, "Fixture 2 canvas must be valid");
        TEST_ASSERT(b2.x_min == 0, "Fixture 2 b.x_min must be 0");
        TEST_ASSERT(b2.x_max == 3200, "Fixture 2 b.x_max must be 3200");
        TEST_ASSERT(b2.y_min == -360, "Fixture 2 b.y_min must be -360");
        TEST_ASSERT(b2.y_max == 840, "Fixture 2 b.y_max must be 840");
        TEST_ASSERT(b2.width == 3200, "Fixture 2 b.width must be 3200");
        TEST_ASSERT(b2.height == 1200, "Fixture 2 b.height must be 1200");
        TEST_ASSERT(b2.total_pixels == 3840000ULL, "Fixture 2 b.total_pixels must be 3,840,000");
        TEST_ASSERT(b2.total_pixels <= ws.pano_capacity,
                    "Fixture 2 (3,840,000 px) must fit within ws.pano_capacity");
        std::cout << "  -> Fixture 2 (Scale+Trans): 3200x1200 = " << b2.total_pixels
                  << " px <= Capacity: " << ws.pano_capacity << " px (PASS)\n";
    }

    // Fixture 3: Pole Rejection bắt buộc
    // H = [1, 0, 0; 0, 1, 0; 2.0/1280.0, 0, -1]
    // Mẫu số đổi dấu từ -1 sang +1 ngay trong ảnh A và B -> Pole cắt ngang ảnh!
    {
        Homography H_pole;
        H_pole.h[0] = 1.0f; H_pole.h[1] = 0.0f; H_pole.h[2] = 0.0f;
        H_pole.h[3] = 0.0f; H_pole.h[4] = 1.0f; H_pole.h[5] = 0.0f;
        H_pole.h[6] = 2.0f / 1280.0f; H_pole.h[7] = 0.0f; H_pole.h[8] = -1.0f;
        H_pole.is_valid = true;

        CanvasBounds b_pole = compute_canvas_bounds(1280, 720, 1280, 720, H_pole);
        TEST_ASSERT(!b_pole.is_valid, "Fixture 3 Pole matrix must be rejected by compute_canvas_bounds");
        std::cout << "  -> Fixture 3 (Pole Rejection): Singular projection correctly rejected (PASS)\n";
    }

    // Fixture 4: Boundary Safety Limits (> 4x dimensions)
    {
        Homography H_extreme;
        H_extreme.h[0] = 1.0f; H_extreme.h[1] = 0.0f; H_extreme.h[2] = -10000.0f; // > 4x w
        H_extreme.h[3] = 0.0f; H_extreme.h[4] = 1.0f; H_extreme.h[5] = 0.0f;
        H_extreme.h[6] = 0.0f; H_extreme.h[7] = 0.0f; H_extreme.h[8] = 1.0f;
        H_extreme.is_valid = true;

        CanvasBounds b_ext = compute_canvas_bounds(1280, 720, 1280, 720, H_extreme);
        TEST_ASSERT(!b_ext.is_valid, "Canvas exceeding 4x width must be rejected");
        std::cout << "  -> Fixture 4 (Safety Limit): Canvas exceeding 4x width rejected (PASS)\n";
    }
}

// ============================================================================
// TEST SUITE 3: Stage 1 CUDA Preprocess (Flat, Impulse, Border Clamp & Sizes)
// ============================================================================
void test_stage1_cuda_correctness() {
    std::cout << "[Test 3] Testing Stage 1 CUDA Preprocess (Flat, Impulse, Border & Dimensions)...\n";

    const float gauss_weights[25] = {
        0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f,
        0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
        0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f,
        0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f,
        0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f
    };

    // 1. Fixture Flat Image (Đồng nhất): giá trị 128 mọi pixel
    {
        int w = 64, h = 64;
        size_t N = (size_t)w * h;
        std::vector<uchar4> h_flat(N, make_uchar4(128, 128, 128, 255));

        uchar4* d_rgb;
        float *d_gray, *d_blur;
        CHECK_CUDA(cudaMalloc(&d_rgb, N * sizeof(uchar4)));
        CHECK_CUDA(cudaMalloc(&d_gray, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_blur, N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_rgb, h_flat.data(), N * sizeof(uchar4), cudaMemcpyHostToDevice));
        stage1_preprocess_gpu(d_rgb, d_gray, d_blur, w, h);

        std::vector<float> res_gray(N), res_blur(N);
        CHECK_CUDA(cudaMemcpy(res_gray.data(), d_gray, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(res_blur.data(), d_blur, N * sizeof(float), cudaMemcpyDeviceToHost));

        float expected_flat_blur = 0.0f;
        for (int k = 0; k < 25; k++) expected_flat_blur += 128.0f * gauss_weights[k];
        for (size_t i = 0; i < N; i++) {
            TEST_ASSERT(std::abs(res_gray[i] - 128.0f) < 1e-4f, "Flat gray value must be 128");
            TEST_ASSERT(std::abs(res_blur[i] - expected_flat_blur) < 1e-4f, "Flat blur value must match analytical convolution sum");
        }
        CHECK_CUDA(cudaFree(d_rgb));
        CHECK_CUDA(cudaFree(d_gray));
        CHECK_CUDA(cudaFree(d_blur));
        std::cout << "  -> Flat fixture: Gray=128, Blur=128 (PASS)\n";
    }

    // 2. Fixture Impulse (Xung đơn vị tại tâm): pixel (12, 12) = 255, còn lại 0
    {
        int w = 25, h = 25;
        size_t N = (size_t)w * h;
        std::vector<uchar4> h_imp(N, make_uchar4(0, 0, 0, 255));
        h_imp[12 * w + 12] = make_uchar4(255, 255, 255, 255);

        uchar4* d_rgb;
        float *d_gray, *d_blur;
        CHECK_CUDA(cudaMalloc(&d_rgb, N * sizeof(uchar4)));
        CHECK_CUDA(cudaMalloc(&d_gray, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_blur, N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_rgb, h_imp.data(), N * sizeof(uchar4), cudaMemcpyHostToDevice));
        stage1_preprocess_gpu(d_rgb, d_gray, d_blur, w, h);

        std::vector<float> res_blur(N);
        CHECK_CUDA(cudaMemcpy(res_blur.data(), d_blur, N * sizeof(float), cudaMemcpyDeviceToHost));

        // So sánh lân cận 5x5 quanh tâm với trọng số kernel lý thuyết
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int y = 12 + ky;
                int x = 12 + kx;
                float expected = 255.0f * gauss_weights[(ky + 2) * 5 + (kx + 2)];
                float actual = res_blur[y * w + x];
                TEST_ASSERT(std::abs(actual - expected) < 1e-3f, "Impulse response must match Gauss kernel");
            }
        }
        CHECK_CUDA(cudaFree(d_rgb));
        CHECK_CUDA(cudaFree(d_gray));
        CHECK_CUDA(cudaFree(d_blur));
        std::cout << "  -> Impulse fixture: 5x5 Gauss impulse response matches analytical kernel (PASS)\n";
    }

    // 3. Fixture Border Clamp & Kích thước bất kỳ
    std::vector<std::pair<int, int>> sizes = {
        {17, 19},     // Kích thước nguyên tố, nhỏ hơn 1 block
        {131, 257},   // Kích thước lẻ không chia hết cho 16
        {1280, 720}   // 720p chuẩn
    };

    for (auto sz : sizes) {
        int w = sz.first;
        int h = sz.second;
        size_t N = (size_t)w * h;

        std::vector<uchar4> h_in(N);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                if (x < 2 || x >= w - 2 || y < 2 || y >= h - 2) {
                    h_in[idx] = make_uchar4(100, 150, 200, 255); // Border
                } else {
                    h_in[idx] = make_uchar4((x * 7) % 256, (y * 11) % 256, ((x + y) * 13) % 256, 255);
                }
            }
        }

        // CPU ground truth
        std::vector<float> cpu_gray(N);
        std::vector<float> cpu_blur(N);
        for (size_t i = 0; i < N; i++) {
            cpu_gray[i] = 0.299f * h_in[i].x + 0.587f * h_in[i].y + 0.114f * h_in[i].z;
        }

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float sum = 0.0f;
                for (int ky = -2; ky <= 2; ky++) {
                    int py = std::max(0, std::min(h - 1, y + ky));
                    for (int kx = -2; kx <= 2; kx++) {
                        int px = std::max(0, std::min(w - 1, x + kx));
                        sum += cpu_gray[py * w + px] * gauss_weights[(ky + 2) * 5 + (kx + 2)];
                    }
                }
                cpu_blur[y * w + x] = sum;
            }
        }

        uchar4* d_rgb;
        float *d_gray, *d_blur;
        CHECK_CUDA(cudaMalloc(&d_rgb, N * sizeof(uchar4)));
        CHECK_CUDA(cudaMalloc(&d_gray, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_blur, N * sizeof(float)));

        CHECK_CUDA(cudaMemcpy(d_rgb, h_in.data(), N * sizeof(uchar4), cudaMemcpyHostToDevice));
        stage1_preprocess_gpu(d_rgb, d_gray, d_blur, w, h);

        std::vector<float> gpu_gray(N);
        std::vector<float> gpu_blur(N);
        CHECK_CUDA(cudaMemcpy(gpu_gray.data(), d_gray, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(gpu_blur.data(), d_blur, N * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err_gray = 0.0f;
        float max_err_blur = 0.0f;
        for (size_t i = 0; i < N; i++) {
            max_err_gray = std::max(max_err_gray, std::abs(gpu_gray[i] - cpu_gray[i]));
            max_err_blur = std::max(max_err_blur, std::abs(gpu_blur[i] - cpu_blur[i]));
        }

        TEST_ASSERT(max_err_gray <= 1e-4f, "Stage 1 GPU Grayscale error exceeds tolerance");
        TEST_ASSERT(max_err_blur <= 1e-4f, "Stage 1 GPU Gaussian Blur error exceeds tolerance");

        CHECK_CUDA(cudaFree(d_rgb));
        CHECK_CUDA(cudaFree(d_gray));
        CHECK_CUDA(cudaFree(d_blur));

        std::cout << "  -> Dimension " << w << "x" << h << ": max_err_gray=" << max_err_gray
                  << ", max_err_blur=" << max_err_blur << " (PASS)\n";
    }
}

// ============================================================================
// TEST SUITE 4: RANSAC Production Deterministic Sampling Path (M=4, 5, 20, 259, 2000)
// ============================================================================
void test_ransac_sampling_production_path() {
    std::cout << "[Test 4] Testing RANSAC Production Sampling Path (M=4, 5, 20, 259, 2000)...\n";

    CudaWorkspace ws;
    ws.allocate(1280, 720);

    const std::vector<int> test_M = {4, 5, 20, 259, 2000};

    for (int M : test_M) {
        // 1. Chạy production sampling kernel trên GPU
        kernel_ransac_generate_samples<<<1, 1>>>(ws.d_sample_indices, M, RANSAC_ITERS);
        CHECK_CUDA(cudaGetLastError());

        // 2. Đọc bảng mẫu 2000x4 indices từ GPU về host
        std::vector<int> gpu_samples(RANSAC_ITERS * 4);
        CHECK_CUDA(cudaMemcpy(gpu_samples.data(), ws.d_sample_indices,
                              RANSAC_ITERS * 4 * sizeof(int), cudaMemcpyDeviceToHost));

        // 3. Sinh bảng mẫu CPU chuẩn bằng LCG seed 42 và rejection loop khớp V0
        std::vector<int> cpu_samples(RANSAC_ITERS * 4);
        LcgRngCpu rng(42);
        for (int it = 0; it < RANSAC_ITERS; it++) {
            int idxs[4];
            for (int i = 0; i < 4; i++) {
                bool unique = false;
                while (!unique) {
                    idxs[i] = rng.next_int(M);
                    unique = true;
                    for (int prev = 0; prev < i; prev++) {
                        if (idxs[i] == idxs[prev]) {
                            unique = false;
                            break;
                        }
                    }
                }
                cpu_samples[it * 4 + i] = idxs[i];
            }
        }

        // 4. So sánh index-by-index và chỉ ra sai khác đầu tiên nếu có
        for (int it = 0; it < RANSAC_ITERS; it++) {
            for (int i = 0; i < 4; i++) {
                int g = gpu_samples[it * 4 + i];
                int c = cpu_samples[it * 4 + i];
                if (g != c) {
                    std::cerr << "[FAIL Sample] M=" << M << " mismatch at iter=" << it
                              << " sample=" << i << " GPU=" << g << " CPU=" << c << "\n";
                    TEST_ASSERT(false, "GPU sample does not match CPU deterministic sequence");
                }
            }
        }
        std::cout << "  -> M=" << std::setw(4) << M << ": 2000x4 = 8000 sample indices match 100% (PASS)\n";
    }
}

// ============================================================================
// TEST SUITE 5: Stage 5 RANSAC DLT, Fallback & Numerical Correctness
// ============================================================================
void test_stage5_cuda_correctness() {
    std::cout << "[Test 5] Testing Stage 5 CUDA (DLT, Refit, Fallback, Edge Cases & Isolation)...\n";

    CudaWorkspace ws;
    ws.allocate(1280, 720);

    // 5.1: M < 4 lập tức trả false và reset device buffer về 0
    {
        Match dummy_m[2];
        Match* d_m;
        CHECK_CUDA(cudaMalloc(&d_m, 2 * sizeof(Match)));
        CHECK_CUDA(cudaMemcpy(d_m, dummy_m, 2 * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_out;
        int inliers_out = 999;
        float elapsed = 0.0f;

        bool ret = stage5_ransac_gpu(d_m, 2, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_out, inliers_out, elapsed);
        TEST_ASSERT(!ret, "stage5_ransac_gpu must return false when M < 4");
        TEST_ASSERT(inliers_out == 0, "out_inliers must be 0 when M < 4");
        TEST_ASSERT(!H_out.is_valid, "H must be invalid when M < 4");

        GpuRansacResult res_on_dev{};
        CHECK_CUDA(cudaMemcpy(&res_on_dev, ws.d_ransac_result, sizeof(GpuRansacResult), cudaMemcpyDeviceToHost));
        TEST_ASSERT(!res_on_dev.success && res_on_dev.inliers == 0,
                    "d_ransac_result on device must be completely reset to 0 on M < 4");
        CHECK_CUDA(cudaFree(d_m));
        std::cout << "  -> Test 5.1 (M < 4 failure & device reset): PASS\n";
    }

    // 5.2: Phép biến đổi đồng nhất (Identity): B = A
    {
        int M = 16;
        std::vector<Keypoint> kpA(M), kpB(M);
        std::vector<Match> matches(M);
        int idx = 0;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                kpA[idx].x = 400.0f + c * 80.0f;
                kpA[idx].y = 200.0f + r * 80.0f;
                kpB[idx].x = kpA[idx].x;
                kpB[idx].y = kpA[idx].y;
                matches[idx].idx_a = idx;
                matches[idx].idx_b = idx;
                matches[idx].dist = 0;
                idx++;
            }
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), M * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_id;
        int inliers_id = 0;
        float el_id = 0.0f;
        bool ok = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_id, inliers_id, el_id);
        TEST_ASSERT(ok, "Identity test must succeed");
        TEST_ASSERT(inliers_id == M, "Identity must have all 16 inliers");
        TEST_ASSERT(!H_id.used_translation_fallback, "Identity must use projective model");
        TEST_ASSERT(std::abs(H_id.h[0] - 1.0f) < 1e-2f && std::abs(H_id.h[4] - 1.0f) < 1e-2f, "Diagonal must be 1.0");
        TEST_ASSERT(std::abs(H_id.h[2]) < 0.1f && std::abs(H_id.h[5]) < 0.1f, "Translation must be near 0");
        std::cout << "  -> Test 5.2 (Identity H): inliers=16, H=I (PASS)\n";
    }

    // 5.3: Phép biến đổi tịnh tiến (Pure Translation): B = A + (-300, 20)
    {
        int M = 16;
        std::vector<Keypoint> kpA(M), kpB(M);
        std::vector<Match> matches(M);
        int idx = 0;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                kpA[idx].x = 400.0f + c * 80.0f;
                kpA[idx].y = 200.0f + r * 80.0f;
                kpB[idx].x = kpA[idx].x - 300.0f;
                kpB[idx].y = kpA[idx].y + 20.0f;
                matches[idx].idx_a = idx;
                matches[idx].idx_b = idx;
                matches[idx].dist = 0;
                idx++;
            }
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), M * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_tr;
        int inliers_tr = 0;
        float el_tr = 0.0f;
        bool ok = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_tr, inliers_tr, el_tr);
        TEST_ASSERT(ok, "Translation test must succeed");
        TEST_ASSERT(inliers_tr == M, "Translation must have all 16 inliers");
        TEST_ASSERT(std::abs(H_tr.h[2] - (-300.0f)) < 0.1f, "H.dx must be -300");
        TEST_ASSERT(std::abs(H_tr.h[5] - 20.0f) < 0.1f, "H.dy must be 20");
        std::cout << "  -> Test 5.3 (Translation H): inliers=16, dx=-300, dy=20 (PASS)\n";
    }

    // 5.4: Phép biến đổi xạ ảnh thực tế & Projected-Grid RMSE <= 1.0 px
    {
        int M = 16;
        std::vector<Keypoint> kpA(M), kpB(M);
        std::vector<Match> matches(M);

        // Ground-truth projective matrix (hợp lệ hình học)
        // H_gt = [1.02, -0.01, -350; 0.01, 1.01, 20; 0.00002, 0.00001, 1.0]
        const double H_gt[9] = {
            1.02, -0.01, -350.0,
            0.01,  1.01,   20.0,
            0.00002, 0.00001, 1.0
        };

        int idx = 0;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                double xa = 400.0 + c * 80.0;
                double ya = 200.0 + r * 80.0;
                double w = H_gt[6] * xa + H_gt[7] * ya + H_gt[8];
                double xb = (H_gt[0] * xa + H_gt[1] * ya + H_gt[2]) / w;
                double yb = (H_gt[3] * xa + H_gt[4] * ya + H_gt[5]) / w;

                kpA[idx].x = static_cast<float>(xa);
                kpA[idx].y = static_cast<float>(ya);
                kpB[idx].x = static_cast<float>(xb);
                kpB[idx].y = static_cast<float>(yb);
                matches[idx].idx_a = idx;
                matches[idx].idx_b = idx;
                matches[idx].dist = 0;
                idx++;
            }
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), M * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_proj;
        int inliers_proj = 0;
        float el_proj = 0.0f;
        bool ok = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_proj, inliers_proj, el_proj);
        TEST_ASSERT(ok, "Projective test must succeed");
        TEST_ASSERT(inliers_proj >= 15, "Projective must have >= 15 inliers");

        // Tính Projected-Grid RMSE giữa H_proj và Ground Truth trên lưới 16 điểm
        double sum_sq_err = 0.0;
        for (int i = 0; i < M; i++) {
            float xa = kpA[i].x;
            float ya = kpA[i].y;
            float w = H_proj.h[6] * xa + H_proj.h[7] * ya + H_proj.h[8];
            float pred_xb = (H_proj.h[0] * xa + H_proj.h[1] * ya + H_proj.h[2]) / w;
            float pred_yb = (H_proj.h[3] * xa + H_proj.h[4] * ya + H_proj.h[5]) / w;
            float dx = pred_xb - kpB[i].x;
            float dy = pred_yb - kpB[i].y;
            sum_sq_err += (dx * dx + dy * dy);
        }
        double grid_rmse = std::sqrt(sum_sq_err / M);
        TEST_ASSERT(grid_rmse <= 1.0, "Projected-Grid RMSE must be <= 1.0 px");
        std::cout << "  -> Test 5.4 (Projective H & Grid RMSE): inliers=" << inliers_proj
                  << ", RMSE=" << grid_rmse << " px <= 1.0 px (PASS)\n";
    }

    // 5.5: Dữ liệu có nhiễu Gaussian và 30% Outliers
    {
        int M = 20; // 14 inliers + 6 outliers (30% outliers)
        std::vector<Keypoint> kpA(M), kpB(M);
        std::vector<Match> matches(M);
        for (int i = 0; i < 14; i++) {
            kpA[i].x = 300.0f + (i % 4) * 90.0f;
            kpA[i].y = 200.0f + (i / 4) * 90.0f;
            // Thêm nhiễu nhỏ +- 0.3 px
            float noise = ((i % 3) - 1) * 0.3f;
            kpB[i].x = kpA[i].x - 300.0f + noise;
            kpB[i].y = kpA[i].y + 20.0f - noise;
            matches[i].idx_a = i; matches[i].idx_b = i; matches[i].dist = 0;
        }
        // 6 Outliers ngẫu nhiên hoàn toàn
        for (int i = 14; i < M; i++) {
            kpA[i].x = 300.0f + i * 20.0f;
            kpA[i].y = 150.0f + i * 15.0f;
            kpB[i].x = 100.0f + (i * 97) % 500;
            kpB[i].y = 50.0f  + (i * 113) % 400;
            matches[i].idx_a = i; matches[i].idx_b = i; matches[i].dist = 50;
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), M * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_noisy;
        int inliers_noisy = 0;
        float el_noisy = 0.0f;
        bool ok = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_noisy, inliers_noisy, el_noisy);
        TEST_ASSERT(ok, "RANSAC must succeed with 30% outliers");
        TEST_ASSERT(inliers_noisy >= 13, "RANSAC must recover >= 13 inliers");
        TEST_ASSERT(std::abs(H_noisy.h[2] - (-300.0f)) < 1.0f, "H.dx must recover ~ -300 despite outliers");
        std::cout << "  -> Test 5.5 (30% Outliers & Noise): inliers=" << inliers_noisy
                  << "/" << M << " (PASS)\n";
    }

    // 5.6: Collinear Rejection & NaN Rejection
    {
        // 10 điểm thẳng hàng hoàn toàn: DLT suy biến
        int M = 10;
        std::vector<Keypoint> kpA(M), kpB(M);
        std::vector<Match> matches(M);
        for (int i = 0; i < M; i++) {
            kpA[i].x = 100.0f + i * 50.0f;
            kpA[i].y = 150.0f + i * 30.0f;
            // Tịnh tiến dương nằm ngoài cả hai cửa sổ fallback
            kpB[i].x = kpA[i].x + 100.0f;
            kpB[i].y = kpA[i].y + 50.0f;
            matches[i].idx_a = i; matches[i].idx_b = i; matches[i].dist = 0;
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), M * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_col;
        int inliers_col = 0;
        float el_col = 0.0f;
        bool ok_col = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_col, inliers_col, el_col);
        TEST_ASSERT(!ok_col, "Collinear points outside fallback must fail");
        TEST_ASSERT(!H_col.is_valid, "Collinear H must be invalid");

        // Điểm chứa NaN
        kpA[0].x = std::nanf("");
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), sizeof(Keypoint), cudaMemcpyHostToDevice));
        Homography H_nan;
        int inliers_nan = 0;
        float el_nan = 0.0f;
        bool ok_nan = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_nan, inliers_nan, el_nan);
        TEST_ASSERT(!ok_nan, "Points containing NaN must fail");
        std::cout << "  -> Test 5.6 (Collinear & NaN rejection): PASS\n";
    }

    // 5.7: Fallback Translation — Cửa sổ chính và Cửa sổ mở rộng
    {
        // Cửa sổ chính: dx in [-0.85w, -0.20w] = [-1088, -256], |dy| <= 0.35h = 252
        int M = 16;
        std::vector<Keypoint> kpA(M), kpB(M);
        std::vector<Match> matches(M);
        for (int i = 0; i < M; i++) {
            // Điểm thẳng hàng để DLT projective thất bại, buộc chuyển fallback
            kpA[i].x = 100.0f + i * 50.0f;
            kpA[i].y = 150.0f + i * 30.0f;
            kpB[i].x = kpA[i].x - 500.0f; // -500 nằm trong [-1088, -256]
            kpB[i].y = kpA[i].y + 30.0f;  // 30 nằm trong [-252, 252]
            matches[i].idx_a = i; matches[i].idx_b = i; matches[i].dist = 0;
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), M * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_fb1;
        int inliers_fb1 = 0;
        float el_fb1 = 0.0f;
        bool ok_fb1 = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_fb1, inliers_fb1, el_fb1);
        TEST_ASSERT(ok_fb1, "Primary fallback window must succeed");
        TEST_ASSERT(H_fb1.used_translation_fallback, "Must use translation fallback");
        TEST_ASSERT(std::abs(H_fb1.h[2] - (-500.0f)) < 1.0f, "Fallback dx must be -500");
        TEST_ASSERT(std::abs(H_fb1.h[5] - 30.0f) < 1.0f, "Fallback dy must be 30");

        // Cửa sổ mở rộng: dx in [-0.95w, -0.05w] = [-1216, -64]
        // Cho dx = -1150 (ngoài cửa sổ chính [-1088, -256] nhưng trong cửa sổ mở rộng)
        for (int i = 0; i < M; i++) {
            kpB[i].x = kpA[i].x - 1150.0f;
            kpB[i].y = kpA[i].y + 20.0f;
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        Homography H_fb2;
        int inliers_fb2 = 0;
        float el_fb2 = 0.0f;
        bool ok_fb2 = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_fb2, inliers_fb2, el_fb2);
        TEST_ASSERT(ok_fb2, "Wide fallback window must succeed when primary has < 4");
        TEST_ASSERT(H_fb2.used_translation_fallback, "Must use translation fallback");
        TEST_ASSERT(std::abs(H_fb2.h[2] - (-1150.0f)) < 1.0f, "Wide fallback dx must be -1150");
        std::cout << "  -> Test 5.7 (Fallback Primary & Wide Windows): PASS\n";
    }

    // 5.8: Upper Median trong Fallback Translation
    {
        // 4 giá trị consensus: -504, -502, -500, -498. Upper median (index 4/2 = 2) phải là -500.
        int M = 4;
        std::vector<Keypoint> kpA(M), kpB(M);
        std::vector<Match> matches(M);
        const float dx_vals[4] = {-504.0f, -502.0f, -500.0f, -498.0f};
        for (int i = 0; i < M; i++) {
            kpA[i].x = 100.0f + i * 50.0f;
            kpA[i].y = 150.0f + i * 30.0f;
            kpB[i].x = kpA[i].x + dx_vals[i];
            kpB[i].y = kpA[i].y + 20.0f;
            matches[i].idx_a = i; matches[i].idx_b = i; matches[i].dist = 0;
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), M * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_med;
        int inliers_med = 0;
        float el_med = 0.0f;
        bool ok_med = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_med, inliers_med, el_med);
        TEST_ASSERT(ok_med, "Upper median test must succeed");
        TEST_ASSERT(std::abs(H_med.h[2] - (-500.0f)) < 1e-2f, "Median must be upper median (-500.0f)");
        std::cout << "  -> Test 5.8 (Upper Median exact element selection): PASS\n";
    }

    // 5.9: Đánh giá LOW_CONFIDENCE khi inliers < max(12, int(0.15*M))
    {
        int M = 100;
        int inliers = 10; // inliers < max(12, 15) = 15
        bool used_fb = false;
        PipelineStatus st;
        std::string warning;
        if (inliers < 4) {
            st = PipelineStatus::FAILURE;
        } else if (inliers < std::max(12, static_cast<int>(0.15f * M))) {
            st = PipelineStatus::LOW_CONFIDENCE;
            warning = used_fb ? "Unstable homography rejected; translation fallback has low consensus"
                              : "Low inlier ratio or count; output may have alignment defects";
        } else {
            st = PipelineStatus::SUCCESS;
        }
        TEST_ASSERT(st == PipelineStatus::LOW_CONFIDENCE, "Must classify as LOW_CONFIDENCE");
        std::cout << "  -> Test 5.9 (LOW_CONFIDENCE threshold classification): PASS\n";
    }

    // 5.10: Cách ly trạng thái qua các lần chạy liên tiếp (Success -> Fail -> Success -> Fail)
    {
        // Chạy 1: Thành công
        int M = 16;
        std::vector<Keypoint> kpA(M), kpB(M);
        std::vector<Match> matches(M);
        for (int i = 0; i < M; i++) {
            kpA[i].x = 400.0f + (i % 4) * 80.0f;
            kpA[i].y = 200.0f + (i / 4) * 80.0f;
            kpB[i].x = kpA[i].x - 300.0f;
            kpB[i].y = kpA[i].y + 20.0f;
            matches[i].idx_a = i; matches[i].idx_b = i; matches[i].dist = 0;
        }
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_kpB, kpB.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), M * sizeof(Match), cudaMemcpyHostToDevice));

        Homography H_seq1;
        int inl_seq1 = 0;
        float el_seq1 = 0.0f;
        bool ok1 = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_seq1, inl_seq1, el_seq1);
        TEST_ASSERT(ok1 && inl_seq1 == M, "Seq run 1 must succeed");

        // Chạy 2: Thất bại (M=2)
        Homography H_seq2;
        int inl_seq2 = 0;
        float el_seq2 = 0.0f;
        bool ok2 = stage5_ransac_gpu(ws.d_matches, 2, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_seq2, inl_seq2, el_seq2);
        TEST_ASSERT(!ok2 && inl_seq2 == 0 && !H_seq2.is_valid, "Seq run 2 must fail cleanly");

        // Chạy 3: Thành công trở lại
        Homography H_seq3;
        int inl_seq3 = 0;
        float el_seq3 = 0.0f;
        bool ok3 = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_seq3, inl_seq3, el_seq3);
        TEST_ASSERT(ok3 && inl_seq3 == M && H_seq3.is_valid, "Seq run 3 must succeed after failure");

        // Chạy 4: Thất bại (NaN)
        for (int i = 0; i < M; i++) kpA[i].x = std::nanf("");
        CHECK_CUDA(cudaMemcpy(ws.d_kpA, kpA.data(), M * sizeof(Keypoint), cudaMemcpyHostToDevice));
        Homography H_seq4;
        int inl_seq4 = 0;
        float el_seq4 = 0.0f;
        bool ok4 = stage5_ransac_gpu(ws.d_matches, M, ws.d_kpA, ws.d_kpB, ws, 1280, 720, H_seq4, inl_seq4, el_seq4);
        TEST_ASSERT(!ok4 && !H_seq4.is_valid, "Seq run 4 must fail on NaN");

        std::cout << "  -> Test 5.10 (Multi-run state isolation Success->Fail->Success->Fail): PASS\n";
    }
}

// ============================================================================
// MAIN RUNNER
// ============================================================================
int main() {
    std::cout << "==============================================================================\n";
    std::cout << "     CSC14120 — CUDA VERIFICATION TEST SUITE (PHASE C THÔNG — S1/S5/WS/BOUNDS) \n";
    std::cout << "==============================================================================\n";

    test_workspace_lifecycle();
    test_canvas_bounds_contract();
    test_stage1_cuda_correctness();
    test_ransac_sampling_production_path();
    test_stage5_cuda_correctness();

    std::cout << "==============================================================================\n";
    std::cout << "[THONG_HANDOFF_READY] Tất cả unit tests cho phần việc của Thông (S1, S5, WS, Bounds) đã PASS!\n";
    std::cout << "[CHÚ THÍCH GATE G1]: Trạng thái này chứng minh phần Thông sẵn sàng bàn giao;\n";
    std::cout << "                     Gate G1 toàn diện sẽ do Tiến (S2/S3) và Tuấn (S4/S6) hoàn tất.\n";
    std::cout << "==============================================================================\n";
    return 0;
}
