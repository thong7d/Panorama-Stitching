#pragma once

#include <iostream>
#include <chrono>

#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << err << " \"" << cudaGetErrorString(err)   \
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

// Fallback GpuTimer wrapper for host-only compilation (CPU)
struct GpuTimer {
    std::chrono::high_resolution_clock::time_point t_start;
    void Start() {
        t_start = std::chrono::high_resolution_clock::now();
    }
    float Stop() {
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> dur = t_end - t_start;
        return dur.count();
    }
};

#define CHECK_CUDA(call) ((void)0)

#endif

// CPU Timer for high-accuracy host measurement
struct CpuTimer {
    std::chrono::high_resolution_clock::time_point t_start;
    void Start() {
        t_start = std::chrono::high_resolution_clock::now();
    }
    float Stop() {
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> dur = t_end - t_start;
        return dur.count();
    }
};
