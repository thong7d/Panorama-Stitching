// ============================================================================
// CSC14120 - Lập trình song song | Ghép ảnh Panorama
// Tác giả: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
//          Dao Quoc Tuan (23120392)
// External sources: Tài liệu CUDA Runtime API. Không sao chép mã nguồn bên ngoài.
// ============================================================================

#pragma once

#include <iostream>
#include <chrono>

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "Lỗi CUDA tại " << __FILE__ << ":" << __LINE__        \
                      << " mã=" << err << " \"" << cudaGetErrorString(err)     \
                      << "\"" << std::endl;                                    \
            exit(5);                                                           \
        }                                                                      \
    } while (0)

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void Start(cudaStream_t s = 0) {
        cudaEventRecord(start, s);
    }
    float Stop(cudaStream_t s = 0) {
        cudaEventRecord(stop, s);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

#else

// Lớp bọc GpuTimer dự phòng khi biên dịch chỉ trên CPU
struct GpuTimer {
    std::chrono::steady_clock::time_point t_start;
    void Start() {
        t_start = std::chrono::steady_clock::now();
    }
    float Stop() {
        auto t_end = std::chrono::steady_clock::now();
        std::chrono::duration<float, std::milli> dur = t_end - t_start;
        return dur.count();
    }
};

#define CHECK_CUDA(call) ((void)0)

#endif

// Bộ bấm giờ CPU độ chính xác cao cho các đo đạc trên host
struct CpuTimer {
    std::chrono::steady_clock::time_point t_start;
    void Start() {
        t_start = std::chrono::steady_clock::now();
    }
    float Stop() {
        auto t_end = std::chrono::steady_clock::now();
        std::chrono::duration<float, std::milli> dur = t_end - t_start;
        return dur.count();
    }
};

