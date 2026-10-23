// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Common Types and Pipeline Contract (Plan v4)
// Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// ============================================================================

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <utility>
#include <algorithm>

// Vector types tương thích cả NVCC và host C++
#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
    #include <cuda_runtime.h>
#else
    #ifndef UCHAR4_DEFINED
    #define UCHAR4_DEFINED
    struct uchar4 {
        uint8_t x, y, z, w;
    };
    inline uchar4 make_uchar4(uint8_t x, uint8_t y, uint8_t z, uint8_t w) {
        uchar4 u; u.x = x; u.y = y; u.z = z; u.w = w; return u;
    }
    #endif

    #ifndef INT2_DEFINED
    #define INT2_DEFINED
    struct int2 {
        int x, y;
    };
    inline int2 make_int2(int x, int y) {
        int2 p; p.x = x; p.y = y; return p;
    }
    #endif
#endif

// ============================================================================
// HẰNG SỐ BẤT BIẾN TOÀN DỰ ÁN (Plan v4 Section 3)
// ============================================================================
constexpr int DEFAULT_RUNS     = 5;        // Số runs mặc định theo Plan v4
constexpr int DEFAULT_TOP_K    = 1000;     // K tối đa sau Top-K
constexpr float FAST_THRESH    = 20.0f;    // Ngưỡng kiểm tra góc FAST-9
constexpr int BORDER_MARGIN    = 20;       // Biên an toàn loại keypoint sát mép ảnh (bán kính ORB sau xoay)
constexpr float RATIO_THRESH   = 0.75f;    // Lowe's ratio test threshold
constexpr int RANSAC_ITERS     = 2000;     // Số hypotheses RANSAC
constexpr float RANSAC_THRESH  = 3.0f;     // Ngưỡng reprojection error (px)

// ============================================================================
// CORE DATA TYPES (Plan v4 Section 4.2 - RAII Move-Only, Cấm Shallow Copy)
// ============================================================================

// Interleaved RGBA image (uchar4 layout, 4 bytes/pixel, alpha=255)
// Tương thích 100% với cudaTextureObject_t (uchar4)
struct Image {
    int w = 0;
    int h = 0;
    uchar4* data = nullptr;

    Image() = default;

    Image(int width, int height) : w(width), h(height) {
        if (w > 0 && h > 0) {
            data = new uchar4[w * h];
        }
    }

    // CẤM SHALLOW COPY
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    // CHO PHÉP MOVE MOVE-ONLY
    Image(Image&& other) noexcept : w(other.w), h(other.h), data(other.data) {
        other.w = 0;
        other.h = 0;
        other.data = nullptr;
    }

    Image& operator=(Image&& other) noexcept {
        if (this != &other) {
            free();
            w = other.w;
            h = other.h;
            data = other.data;
            other.w = 0;
            other.h = 0;
            other.data = nullptr;
        }
        return *this;
    }

    ~Image() {
        free();
    }

    void allocate(int width, int height) {
        free();
        w = width;
        h = height;
        if (w > 0 && h > 0) {
            data = new uchar4[w * h];
        }
    }

    // Destructor / Free Idempotent
    void free() noexcept {
        if (data) {
            delete[] data;
            data = nullptr;
        }
        w = 0;
        h = 0;
    }

    bool empty() const noexcept {
        return data == nullptr || w <= 0 || h <= 0;
    }
};

// Grayscale Image (float, 32-bit/pixel) - Row-major, RAII Move-Only
struct GrayImage {
    int w = 0;
    int h = 0;
    float* data = nullptr;

    GrayImage() = default;

    GrayImage(int width, int height) : w(width), h(height) {
        if (w > 0 && h > 0) {
            data = new float[w * h];
        }
    }

    GrayImage(const GrayImage&) = delete;
    GrayImage& operator=(const GrayImage&) = delete;

    GrayImage(GrayImage&& other) noexcept : w(other.w), h(other.h), data(other.data) {
        other.w = 0;
        other.h = 0;
        other.data = nullptr;
    }

    GrayImage& operator=(GrayImage&& other) noexcept {
        if (this != &other) {
            free();
            w = other.w;
            h = other.h;
            data = other.data;
            other.w = 0;
            other.h = 0;
            other.data = nullptr;
        }
        return *this;
    }

    ~GrayImage() {
        free();
    }

    void allocate(int width, int height) {
        free();
        w = width;
        h = height;
        if (w > 0 && h > 0) {
            data = new float[w * h];
        }
    }

    void free() noexcept {
        if (data) {
            delete[] data;
            data = nullptr;
        }
        w = 0;
        h = 0;
    }

    bool empty() const noexcept {
        return data == nullptr || w <= 0 || h <= 0;
    }
};

// Keypoint struct: 16 bytes
struct Keypoint {
    float x;
    float y;
    float angle;      // Orientation angle (radians) [-pi, pi]
    float response;   // FAST corner response score
};

// 256-bit Binary ORB Descriptor: 32 bytes (alignas 16, 8 words x 32-bit uint32_t)
struct alignas(16) Descriptor {
    uint32_t words[8];

    Descriptor() {
        for (int i = 0; i < 8; i++) words[i] = 0;
    }
};

// Keypoint match pair
struct Match {
    int idx_a;
    int idx_b;
    int dist;         // Hamming distance [0..256]
};

// Pipeline execution status
enum class PipelineStatus {
    SUCCESS,
    LOW_CONFIDENCE,
    FAILURE
};

inline const char* status_to_string(PipelineStatus s) {
    switch (s) {
        case PipelineStatus::SUCCESS: return "SUCCESS";
        case PipelineStatus::LOW_CONFIDENCE: return "LOW_CONFIDENCE";
        case PipelineStatus::FAILURE: return "FAILURE";
        default: return "UNKNOWN";
    }
}

// 3x3 Homography Matrix H_A_to_B (row-major)
struct Homography {
    float h[9];
    bool is_valid = false;

    Homography() {
        for (int i = 0; i < 9; i++) h[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }

    bool check_finite() const {
        for (int i = 0; i < 9; i++) {
            if (!std::isfinite(h[i])) return false;
        }
        return true;
    }

    void normalize() {
        if (std::abs(h[8]) > 1e-12f) {
            float inv = 1.0f / h[8];
            for (int i = 0; i < 9; i++) h[i] *= inv;
            h[8] = 1.0f;
        }
    }
};

// Pipeline diagnostic metrics
struct PipelineDiagnostics {
    int keypoints_a = 0;
    int keypoints_b = 0;
    int matches = 0;
    int inliers = 0;
    float inlier_ratio = 0.0f;
    Homography H;
    PipelineStatus status = PipelineStatus::FAILURE;
    std::string warning = "None";
};

// Stage execution metrics (Avg + Min) per Plan v4 Section 4.3
struct StageMetric {
    float avg = 0.0f;
    float min = 0.0f;
};

struct TimingSummary {
    StageMetric stage1_preprocess;
    StageMetric stage2_fast_nms_topk;
    StageMetric stage3_orb_descriptor;
    StageMetric stage4_matching;
    StageMetric stage5_ransac;
    StageMetric stage6_warping_blending;
    StageMetric kernel_total;
    StageMetric h2d;
    StageMetric control;
    StageMetric d2h;
    StageMetric end_to_end;
    StageMetric setup;
};

// Comprehensive pipeline execution result
struct PipelineResult {
    Image panorama;
    PipelineDiagnostics diagnostics;
    TimingSummary timings;
    int runs = 1;
    int warmups = 1;
    std::string device_name;
    std::string version_str;
    std::string optimization_str;
    int image_width = 0;
    int image_height = 0;

    bool export_metrics_json(const std::string& json_path) const;
};

// 256 test pairs from OpenCV bit_pattern_31_
extern const int2 ORB_PATTERN_PAIRS[256][2];
