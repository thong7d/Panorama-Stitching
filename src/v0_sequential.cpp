// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: V0 Sequential Baseline Implementation (Plan v4)
// Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// External sources: ORB bit_pattern_31_ adapted from OpenCV modules/features2d/src/orb.cpp
//   (BSD 3-Clause License, Copyright 2000 Intel Corporation, Willow Garage Inc.)
// ============================================================================

#include "v0_sequential.h"
#include "cuda_utils.cuh"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <chrono>

// ============================================================================
// BẢNG MẪU 256 CẶP TỌA ĐỘ CHUẨN OPENCV (bit_pattern_31_)
// Nguồn: OpenCV modules/features2d/src/orb.cpp (BSD License, Willow Garage)
// 256 cặp kiểm tra (512 điểm), tọa độ trong [-13, 13], bán kính tối đa ~18.38 px
// ============================================================================
const int2 ORB_PATTERN_PAIRS[256][2] = {
    { { 8, -3}, { 9,   5} }, { { 4,   2}, { 7, -12} }, { {-11,   9}, {-8,   2} }, { { 7, -12}, {12, -13} },
    { { 2, -13}, { 2,  12} }, { { 1,  -7}, { 1,   6} }, { {-2, -10}, {-2,  -4} }, { {-13, -13}, {-11, -8} },
    { {-13, -3}, {-12, -9} }, { {10,   4}, {11,   9} }, { {-13,  -8}, {-8,  -9} }, { {-11,   7}, {-9,  12} },
    { { 7,   7}, {12,   6} }, { {-4,  -5}, {-3,   0} }, { {-13,   2}, {-12, -3} }, { {-9,   0}, {-7,   5} },
    { {12,  -6}, {12,  -1} }, { {-3,   6}, {-2,  12} }, { {-6, -13}, {-4,  -8} }, { {11, -13}, {12,  -8} },
    { { 4,   7}, { 5,   1} }, { { 5,  -3}, {10,  -3} }, { { 3,  -7}, { 6,  12} }, { {-8,  -7}, {-6,  -2} },
    { {-2,  11}, {-1, -10} }, { {-13, 12}, {-8,  10} }, { {-7,   3}, {-5,  -3} }, { {-4,   2}, {-3,   7} },
    { {-10,-12}, {-6,  11} }, { { 5, -12}, { 6,  -7} }, { { 5,  -6}, { 7,  -1} }, { { 1,   0}, { 4,  -5} },
    { { 9,  11}, {11, -13} }, { { 4,   7}, { 4,  12} }, { { 2,  -1}, { 4,   4} }, { {-4, -12}, {-2,   7} },
    { {-8,  -5}, {-7, -10} }, { { 4,  11}, { 9,  12} }, { { 0,  -8}, { 1, -13} }, { {-13, -2}, {-8,   2} },
    { {-3,  -2}, {-2,   3} }, { {-6,   9}, {-4,  -9} }, { { 8,  12}, {10,   7} }, { { 0,   9}, { 1,   3} },
    { { 7,  -5}, {11, -10} }, { {-13, -6}, {-11,  0} }, { {10,   7}, {12,   1} }, { {-6,  -3}, {-6,  12} },
    { {10,  -9}, {12,  -4} }, { {-13,  8}, {-8, -12} }, { {-13,  0}, {-8,  -4} }, { { 3,   3}, { 7,   8} },
    { { 5,   7}, {10,  -7} }, { {-1,   7}, { 1, -12} }, { { 3, -10}, { 5,   6} }, { { 2,  -4}, { 3, -10} },
    { {-13,  0}, {-13,  5} }, { {-13, -7}, {-12, 12} }, { {-13,  3}, {-11,  8} }, { {-7,  12}, {-4,   7} },
    { { 6, -10}, {12,   8} }, { {-9,  -1}, {-7,  -6} }, { {-2,  -5}, { 0,  12} }, { {-12,  5}, {-7,   5} },
    { { 3, -10}, { 8, -13} }, { {-7,  -7}, {-4,   5} }, { {-3,  -2}, {-1,  -7} }, { { 2,   9}, { 5, -11} },
    { {-11,-13}, {-5, -13} }, { {-1,   6}, { 0,  -1} }, { { 5,  -3}, { 5,   2} }, { {-4, -13}, {-4,  12} },
    { {-9,  -6}, {-9,   6} }, { {-12,-10}, {-8,  -4} }, { {10,   2}, {12,  -3} }, { { 7,  12}, {12,  12} },
    { {-7, -13}, {-6,   5} }, { {-4,   9}, {-3,   4} }, { { 7,  -1}, {12,   2} }, { {-7,   6}, {-5,   1} },
    { {-13, 11}, {-12,  5} }, { {-3,   7}, {-2,  -6} }, { { 7,  -8}, {12,  -7} }, { {-13, -7}, {-11,-12} },
    { { 1,  -3}, {12,  12} }, { { 2,  -6}, { 3,   0} }, { {-4,   3}, {-2, -13} }, { {-1, -13}, { 1,   9} },
    { { 7,   1}, { 8,  -6} }, { { 1,  -1}, { 3,  12} }, { { 9,   1}, {12,   6} }, { {-1,  -9}, {-1,   3} },
    { {-13,-13}, {-10,  5} }, { { 7,   7}, {10,  12} }, { {12,  -5}, {12,   9} }, { { 6,   3}, { 7,  11} },
    { { 5, -13}, { 6,  10} }, { { 2, -12}, { 2,   3} }, { { 3,   8}, { 4,  -6} }, { { 2,   6}, {12,  -1} },
    { { 1,   3}, { 1,  10} }, { { 7, -13}, {12,  -9} }, { {-10,  5}, {-4,   3} }, { {-3,   4}, {-2,   9} },
    { {-12,  3}, {-3,   3} }, { { 1,   2}, { 8,   2} }, { { 2,   0}, {10,  -5} }, { { 3,   8}, { 8,   7} },
    { {-10, 10}, {-8,   1} }, { {-1,  -2}, { 4,   7} }, { {-8,   0}, {-6,  10} }, { {-4, -13}, {-1, -11} },
    { { 8, -11}, {12, -11} }, { { 4,  -3}, { 5,   7} }, { {-6,  -1}, {-1,   1} }, { { 1,   3}, { 7,  -1} },
    { {-4, -11}, {-1, -11} }, { { 8,  -8}, {12,   3} }, { {-11, -4}, {-6,  -4} }, { {-2,  -4}, { 3,   3} },
    { { 4,   1}, { 9,   3} }, { {-6,  -1}, { 6,  12} }, { {12,   3}, {12,   8} }, { { 7,  -5}, {13,   0} },
    { {-1,   2}, { 7,   9} }, { { 3,  -5}, { 7,   6} }, { {-4,  -8}, {-2,  -1} }, { {-10,  4}, {-8,   9} },
    { { 6,   3}, { 7,   5} }, { { 2,  -1}, { 2,   3} }, { {-6,  -5}, {-3,  -6} }, { {-1,   0}, { 4,   2} },
    { {-7,  -3}, {-5,  -1} }, { { 5,   8}, { 8,  11} }, { {-13,  3}, {-9,   2} }, { {-9,  -1}, {-7,   4} },
    { {-1,  -4}, { 5,   9} }, { {-12, -5}, {-8,  -3} }, { { 6,   3}, {11,  -2} }, { {10,  12}, {12,  -9} },
    { { 0,  -8}, { 2,  -2} }, { {-1,  11}, { 1,   4} }, { {-4,   8}, {-2,  13} }, { { 0,   1}, { 8,   0} },
    { {-6,  -4}, {-4,   0} }, { { 8,   2}, { 8,  -4} }, { { 2,   6}, { 7,  -1} }, { { 6,  -6}, { 7,   7} },
    { {-8,   3}, {-7,  -2} }, { {-6,   8}, {-4,   2} }, { { 2,   1}, { 8,  -1} }, { { 2,  11}, { 7,   2} },
    { {-12,  2}, {-6,   1} }, { {-8,  -7}, {-6,   1} }, { { 0,  -3}, { 5,  -5} }, { {-4,   8}, {-3,   1} },
    { {-1,   8}, { 1,  -2} }, { { 3,  -6}, { 6,  -1} }, { {-6,   4}, {-2,  -2} }, { { 1,  -1}, { 8,   8} },
    { { 3,  -2}, { 5,   3} }, { {-4,   1}, {-1,  -4} }, { { 1,   4}, { 6,   0} }, { { 5,  -7}, { 7,   3} },
    { { 6,   4}, { 7,  -8} }, { {-1,  -6}, { 4,  -9} }, { { 4,   3}, { 6,   2} }, { {-7,   5}, {-5,   8} },
    { {-3,  11}, { 4,   3} }, { {-3,  -2}, { 3,  -3} }, { { 0,   6}, { 4,   4} }, { {-4,  -2}, { 0,  -2} },
    { {-1,  -6}, { 4,   7} }, { { 1,   1}, { 8,  -2} }, { { 2,   9}, { 7,  11} }, { { 3,   5}, { 6, -10} },
    { {-5,   2}, {-3,   8} }, { {-2,   5}, { 2,  -1} }, { { 6,  -1}, { 7,   2} }, { { 9,   2}, {10,   6} },
    { {-4,  -1}, { 0,   2} }, { {-1,  10}, { 2,   7} }, { {-4,   1}, {-3,   7} }, { {-9,   2}, {-6,  -2} },
    { {-1,  -2}, { 2,   0} }, { {-7,   4}, {-5,   0} }, { { 0,  10}, { 3,   1} }, { {-6,   7}, {-2,   3} },
    { { 0,   2}, { 4,   4} }, { {-6,   4}, {-3,   2} }, { { 1,   6}, { 4,   1} }, { { 4,   3}, { 9,   1} },
    { { 2,   3}, { 3,   2} }, { {-4,   6}, {-1,   1} }, { {-2,   3}, { 2,   3} }, { {-2,   8}, { 0,   4} },
    { {-1,   3}, { 1,   3} }, { {-1,  -3}, { 1,  -3} }, { {-3,  -2}, { 0,  -4} }, { {-3,   3}, {-1,   0} },
    { { 0,  -2}, { 2,   4} }, { {-1,  -4}, { 0,  -1} }, { {-2,   4}, { 1,   4} }, { {-2,  -1}, { 3,   7} },
    { {-6,   1}, {-1,   1} }, { { 1,  -2}, { 4,   0} }, { { 5,   3}, { 5,   6} }, { { 2,  -4}, { 7,   1} },
    { { 3,   6}, { 6,   5} }, { {-4,   3}, { 0,   7} }, { {-5,  -4}, { 5,   4} }, { {-4,  -6}, {-2,  -3} },
    { {-7,   3}, {-2,   4} }, { {-3,  -8}, { 2,   4} }, { { 0,   4}, { 7,  -3} }, { {-1,   3}, { 1,  -5} },
    { {-1,   3}, { 6,   2} }, { { 3,   7}, { 7,   1} }, { {-4,   2}, { 1,   6} }, { {-2,  -6}, { 5,  -1} },
    { {-4,   4}, { 4,   1} }, { {-2,  -4}, { 0,  -4} }, { { 4,   2}, { 4,   5} }, { { 0,  -7}, { 3,   4} },
    { { 2,  -3}, { 3,   2} }, { { 1,   2}, { 4,   2} }, { {-3,   5}, { 3,   2} }, { {-2,   4}, { 2,  -4} },
    { { 3,   4}, { 4,  -5} }, { {-2,   3}, { 1,  -3} }, { {-4,   2}, { 3,  -2} }, { { 1,   2}, { 6,   2} },
    { {-3,   1}, { 3,   3} }, { { 1,   0}, { 3,   3} }, { {-4,  -1}, { 4,  -1} }, { {-3,   4}, {-2,   2} },
    { { 0,  -2}, { 5,   3} }, { {-4,  -5}, { 1,  -1} }, { { 1,  -2}, { 3,   1} }, { {-4,  -2}, {-2,   1} },
    { { 2,  -2}, { 7,   4} }, { {-3,  -2}, { 1,  -3} }, { {-2,   1}, { 2,   5} }, { { 4,   2}, { 5,   8} },
    { {-4,   0}, { 2,   2} }, { { 1,   7}, { 3,   3} }, { { 4,   4}, { 5,   1} }, { {-3,   6}, { 4,   5} },
    { {-4,  -1}, { 0,   1} }, { { 1,   4}, { 4,   4} }, { { 2,   1}, { 6,   2} }, { {-3,   1}, { 0,   5} },
    { { 1,  -6}, { 3,  -1} }, { {-4,   1}, { 2,   2} }, { { 2,   3}, { 2,   7} }, { {-2,  -1}, { 4,   5} }
};

// ============================================================================
// STAGE 1: PREPROCESSING (Grayscale + Gaussian Blur 5x5)
// Kernel: 2D 5×5 Gaussian, σ≈1.04 (empirically fitted, NOT exact σ=1.0).
// All V1+ versions must use this exact kernel for V0↔V1 correctness verification.
// ============================================================================
static const float GAUSS_KERNEL_5X5[5][5] = {
    { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f },
    { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
    { 0.023792f, 0.094907f, 0.150342f, 0.094907f, 0.023792f },
    { 0.015019f, 0.059912f, 0.094907f, 0.059912f, 0.015019f },
    { 0.003765f, 0.015019f, 0.023792f, 0.015019f, 0.003765f }
};

GrayImage stage1_preprocess_cpu(const Image& img, float& elapsed_ms) {
    CpuTimer timer;
    timer.Start();

    int w = img.w;
    int h = img.h;

    // 1. Grayscale Y = 0.299R + 0.587G + 0.114B
    std::vector<float> gray_temp(w * h);
    for (int i = 0; i < w * h; i++) {
        uchar4 p = img.data[i];
        gray_temp[i] = 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
    }

    // 2. Gaussian Blur 5x5 với boundary clamping
    GrayImage out(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            for (int ky = -2; ky <= 2; ky++) {
                int py = std::clamp(y + ky, 0, h - 1);
                for (int kx = -2; kx <= 2; kx++) {
                    int px = std::clamp(x + kx, 0, w - 1);
                    sum += gray_temp[py * w + px] * GAUSS_KERNEL_5X5[ky + 2][kx + 2];
                }
            }
            out.data[y * w + x] = sum;
        }
    }

    elapsed_ms = timer.Stop();
    return out;
}

// ============================================================================
// STAGE 2: KEYPOINT DETECTION (FAST-9 + Border Pruning + 3x3 NMS + Top-K)
// ============================================================================
static const int CIRCLE_X[16] = { 0,  1,  2,  3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };
static const int CIRCLE_Y[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3,  3,  2,  1,  0, -1, -2, -3 };

static bool is_fast_corner(const float* img, int w, int h, int x, int y, float thresh, float& response) {
    if (x < 3 || x >= w - 3 || y < 3 || y >= h - 3) return false;

    float center_val = img[y * w + x];
    float circle[16];
    for (int i = 0; i < 16; i++) {
        circle[i] = img[(y + CIRCLE_Y[i]) * w + (x + CIRCLE_X[i])];
    }

    // Fast rejection: Kiểm tra 4 điểm 0, 4, 8, 12
    int brighter = (circle[0] > center_val + thresh) +
                   (circle[4] > center_val + thresh) +
                   (circle[8] > center_val + thresh) +
                   (circle[12] > center_val + thresh);
    int darker   = (circle[0] < center_val - thresh) +
                   (circle[4] < center_val - thresh) +
                   (circle[8] < center_val - thresh) +
                   (circle[12] < center_val - thresh);
    if (brighter < 3 && darker < 3) return false;

    // Kiểm tra cung 9 điểm liên tiếp
    int bright_run = 0, max_bright_run = 0;
    int dark_run = 0, max_dark_run = 0;

    for (int i = 0; i < 31; i++) {
        float val = circle[i % 16];
        if (val > center_val + thresh) {
            bright_run++;
            if (bright_run > max_bright_run) max_bright_run = bright_run;
        } else {
            bright_run = 0;
        }

        if (val < center_val - thresh) {
            dark_run++;
            if (dark_run > max_dark_run) max_dark_run = dark_run;
        } else {
            dark_run = 0;
        }
    }

    if (max_bright_run >= 9 || max_dark_run >= 9) {
        float score = 0.0f;
        for (int i = 0; i < 16; i++) {
            float diff = std::abs(circle[i] - center_val);
            if (diff > thresh) score += (diff - thresh);
        }
        response = score;
        return true;
    }

    return false;
}

std::vector<Keypoint> stage2_fast_nms_topk_cpu(const GrayImage& img, int K, float& elapsed_ms) {
    CpuTimer timer;
    timer.Start();

    int w = img.w;
    int h = img.h;

    // 1. FAST Detection trên vùng an toàn (loại bỏ keypoint vi phạm BORDER_MARGIN)
    std::vector<float> response_map(w * h, 0.0f);
    for (int y = BORDER_MARGIN; y < h - BORDER_MARGIN; y++) {
        for (int x = BORDER_MARGIN; x < w - BORDER_MARGIN; x++) {
            float resp = 0.0f;
            if (is_fast_corner(img.data, w, h, x, y, FAST_THRESH, resp)) {
                response_map[y * w + x] = resp;
            }
        }
    }

    // 2. 3x3 Non-Maximum Suppression (NMS)
    std::vector<Keypoint> candidates;
    candidates.reserve(50000);

    for (int y = BORDER_MARGIN; y < h - BORDER_MARGIN; y++) {
        for (int x = BORDER_MARGIN; x < w - BORDER_MARGIN; x++) {
            float r = response_map[y * w + x];
            if (r <= 0.0f) continue;

            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    if (response_map[(y + dy) * w + (x + dx)] >= r) {
                        is_max = false;
                        break;
                    }
                }
            }

            if (is_max) {
                Keypoint kp;
                kp.x = static_cast<float>(x);
                kp.y = static_cast<float>(y);
                kp.angle = 0.0f;
                kp.response = r;
                candidates.push_back(kp);
            }
        }
    }

    // 3. Deterministic Top-K Selection (response DESC, linear_index ASC)
    auto deterministic_comp = [w](const Keypoint& a, const Keypoint& b) {
        if (a.response != b.response) {
            return a.response > b.response;
        }
        int idx_a = static_cast<int>(a.y) * w + static_cast<int>(a.x);
        int idx_b = static_cast<int>(b.y) * w + static_cast<int>(b.x);
        return idx_a < idx_b;
    };

    int select_count = std::min(K, static_cast<int>(candidates.size()));
    if (select_count > 0 && select_count < static_cast<int>(candidates.size())) {
        std::partial_sort(candidates.begin(), 
                          candidates.begin() + select_count, 
                          candidates.end(), 
                          deterministic_comp);
        candidates.resize(select_count);
    } else {
        std::sort(candidates.begin(), candidates.end(), deterministic_comp);
    }

    elapsed_ms = timer.Stop();
    return candidates;
}

// ============================================================================
// STAGE 3: ORB DESCRIPTOR (Oriented BRIEF 256-bit)
// ============================================================================
static float compute_orientation_cpu(const GrayImage& img, float kx, float ky) {
    int cx = static_cast<int>(std::round(kx));
    int cy = static_cast<int>(std::round(ky));
    int w = img.w;
    int h = img.h;

    float m10 = 0.0f, m01 = 0.0f;
    const int R = 15;
    const int R2 = R * R;

    for (int dy = -R; dy <= R; dy++) {
        int py = cy + dy;
        if (py < 0 || py >= h) continue;
        int dy2 = dy * dy;

        for (int dx = -R; dx <= R; dx++) {
            if (dx * dx + dy2 > R2) continue;
            int px = cx + dx;
            if (px < 0 || px >= w) continue;

            float intensity = img.data[py * w + px];
            m10 += dx * intensity;
            m01 += dy * intensity;
        }
    }

    return std::atan2(m01, m10);
}

std::vector<Descriptor> stage3_orb_descriptors_cpu(const GrayImage& img, 
                                                   const std::vector<Keypoint>& kps, 
                                                   float& elapsed_ms) {
    CpuTimer timer;
    timer.Start();

    int n_kps = static_cast<int>(kps.size());
    std::vector<Descriptor> descriptors(n_kps);

    int w = img.w;
    int h = img.h;

    for (int i = 0; i < n_kps; i++) {
        float kx = kps[i].x;
        float ky = kps[i].y;
        float theta = compute_orientation_cpu(img, kx, ky);

        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        Descriptor desc;

        for (int bit = 0; bit < 256; bit++) {
            int2 p = ORB_PATTERN_PAIRS[bit][0];
            int2 q = ORB_PATTERN_PAIRS[bit][1];

            float px_rot = p.x * cos_t - p.y * sin_t;
            float py_rot = p.x * sin_t + p.y * cos_t;
            float qx_rot = q.x * cos_t - q.y * sin_t;
            float qy_rot = q.x * sin_t + q.y * cos_t;

            int px = std::clamp(static_cast<int>(std::round(kx + px_rot)), 0, w - 1);
            int py = std::clamp(static_cast<int>(std::round(ky + py_rot)), 0, h - 1);
            int qx = std::clamp(static_cast<int>(std::round(kx + qx_rot)), 0, w - 1);
            int qy = std::clamp(static_cast<int>(std::round(ky + qy_rot)), 0, h - 1);

            float val_p = img.data[py * w + px];
            float val_q = img.data[qy * w + qx];

            if (val_p < val_q) {
                int word_idx = bit / 32;
                int bit_idx = bit % 32;
                desc.words[word_idx] |= (1u << bit_idx);
            }
        }

        descriptors[i] = desc;
    }

    elapsed_ms = timer.Stop();
    return descriptors;
}

// ============================================================================
// STAGE 4: MATCHING (Brute-force 8x32-bit Hamming + Lowe's Ratio Test)
// ============================================================================
static inline int popcount32(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#elif defined(_MSC_VER)
    return __popcnt(x);
#else
    x = x - ((x >> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
    return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#endif
}

static inline int hamming_dist(const Descriptor& a, const Descriptor& b) {
    int dist = 0;
    for (int i = 0; i < 8; i++) {
        dist += popcount32(a.words[i] ^ b.words[i]);
    }
    return dist;
}

std::vector<Match> stage4_matching_cpu(const std::vector<Descriptor>& descA, 
                                       const std::vector<Descriptor>& descB, 
                                       float& elapsed_ms) {
    CpuTimer timer;
    timer.Start();

    int nA = static_cast<int>(descA.size());
    int nB = static_cast<int>(descB.size());
    std::vector<Match> matches;

    if (nA == 0 || nB == 0) {
        elapsed_ms = timer.Stop();
        return matches;
    }

    for (int i = 0; i < nA; i++) {
        int best1_dist = 9999, best2_dist = 9999;
        int best1_idx = -1;

        for (int j = 0; j < nB; j++) {
            int d = hamming_dist(descA[i], descB[j]);
            if (d < best1_dist) {
                best2_dist = best1_dist;
                best1_dist = d;
                best1_idx = j;
            } else if (d < best2_dist) {
                best2_dist = d;
            }
        }

        // Lowe's Ratio Test
        if (best1_dist < RATIO_THRESH * best2_dist && best1_idx >= 0) {
            Match m;
            m.idx_a = i;
            m.idx_b = best1_idx;
            m.dist = best1_dist;
            matches.push_back(m);
        }
    }

    elapsed_ms = timer.Stop();
    return matches;
}

// ============================================================================
// STAGE 5: HOMOGRAPHY (RANSAC + Hartley Normalized 4-point DLT)
// ============================================================================

// Hartley isotropic normalization for 4 points
static void compute_hartley_norm(const float pts[4][2], float norm_pts[4][2], float T[3][3], float inv_T[3][3]) {
    float cx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25f;
    float cy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25f;

    float mean_dist = 0.0f;
    for (int i = 0; i < 4; i++) {
        float dx = pts[i][0] - cx;
        float dy = pts[i][1] - cy;
        mean_dist += std::sqrt(dx * dx + dy * dy);
    }
    mean_dist *= 0.25f;

    float s = (mean_dist > 1e-6f) ? (std::sqrt(2.0f) / mean_dist) : 1.0f;

    for (int i = 0; i < 4; i++) {
        norm_pts[i][0] = s * (pts[i][0] - cx);
        norm_pts[i][1] = s * (pts[i][1] - cy);
    }

    // T = [s 0 -s*cx; 0 s -s*cy; 0 0 1]
    T[0][0] = s;    T[0][1] = 0.0f; T[0][2] = -s * cx;
    T[1][0] = 0.0f; T[1][1] = s;    T[1][2] = -s * cy;
    T[2][0] = 0.0f; T[2][1] = 0.0f; T[2][2] = 1.0f;

    // inv_T = [1/s 0 cx; 0 1/s cy; 0 0 1]
    float inv_s = 1.0f / s;
    inv_T[0][0] = inv_s; inv_T[0][1] = 0.0f;  inv_T[0][2] = cx;
    inv_T[1][0] = 0.0f;  inv_T[1][1] = inv_s; inv_T[1][2] = cy;
    inv_T[2][0] = 0.0f;  inv_T[2][1] = 0.0f;  inv_T[2][2] = 1.0f;
}

// 8x8 Gaussian Elimination with Relative Partial Pivoting
static bool solve_dlt_8x8_cpu(const float ptsA[4][2], const float ptsB[4][2], float H[9]) {
    float norm_A[4][2], norm_B[4][2];
    float TA[3][3], inv_TA[3][3];
    float TB[3][3], inv_TB[3][3];

    compute_hartley_norm(ptsA, norm_A, TA, inv_TA);
    compute_hartley_norm(ptsB, norm_B, TB, inv_TB);

    float A[8][9]; // Augmented matrix [A | b]
    for (int i = 0; i < 4; i++) {
        float x = norm_A[i][0], y = norm_A[i][1];
        float u = norm_B[i][0], v = norm_B[i][1];

        A[2 * i][0] = x;    A[2 * i][1] = y;    A[2 * i][2] = 1.0f;
        A[2 * i][3] = 0.0f; A[2 * i][4] = 0.0f; A[2 * i][5] = 0.0f;
        A[2 * i][6] = -u * x; A[2 * i][7] = -u * y; A[2 * i][8] = u;

        A[2 * i + 1][0] = 0.0f; A[2 * i + 1][1] = 0.0f; A[2 * i + 1][2] = 0.0f;
        A[2 * i + 1][3] = x;    A[2 * i + 1][4] = y;    A[2 * i + 1][5] = 1.0f;
        A[2 * i + 1][6] = -v * x; A[2 * i + 1][7] = -v * y; A[2 * i + 1][8] = v;
    }

    // Relative Partial Pivoting
    for (int col = 0; col < 8; col++) {
        int max_row = col;
        float max_val = std::abs(A[col][col]);
        for (int r = col + 1; r < 8; r++) {
            float val = std::abs(A[r][col]);
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }

        if (max_val < 1e-6f) return false; // Singular matrix

        if (max_row != col) {
            for (int k = col; k <= 8; k++) {
                std::swap(A[col][k], A[max_row][k]);
            }
        }

        float pivot = A[col][col];
        for (int r = col + 1; r < 8; r++) {
            float factor = A[r][col] / pivot;
            for (int k = col; k <= 8; k++) {
                A[r][k] -= factor * A[col][k];
            }
        }
    }

    // Back-substitution
    float h_tilde[9];
    h_tilde[8] = 1.0f;
    for (int r = 7; r >= 0; r--) {
        float sum = A[r][8];
        for (int c = r + 1; c < 8; c++) {
            sum -= A[r][c] * h_tilde[c];
        }
        h_tilde[r] = sum / A[r][r];
    }

    // Denormalize: H = inv_TB * H_tilde * TA
    float M[3][3], H_final[3][3];
    // M = H_tilde * TA
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            M[r][c] = h_tilde[r * 3 + 0] * TA[0][c] +
                      h_tilde[r * 3 + 1] * TA[1][c] +
                      h_tilde[r * 3 + 2] * TA[2][c];
        }
    }
    // H_final = inv_TB * M
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            H_final[r][c] = inv_TB[r][0] * M[0][c] +
                            inv_TB[r][1] * M[1][c] +
                            inv_TB[r][2] * M[2][c];
        }
    }

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            H[r * 3 + c] = H_final[r][c];
        }
    }

    // Normalize so H[8] = 1.0
    if (std::abs(H[8]) > 1e-12f) {
        float inv = 1.0f / H[8];
        for (int i = 0; i < 9; i++) H[i] *= inv;
        H[8] = 1.0f;
    } else {
        return false;
    }

    // Finite check
    for (int i = 0; i < 9; i++) {
        if (!std::isfinite(H[i])) return false;
    }

    return true;
}

// Deterministic LCG PRNG for repeatable RANSAC sampling across CPU and GPU
struct LcgRng {
    uint32_t state;
    explicit LcgRng(uint32_t seed = 42) : state(seed) {}

    uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }

    int next_int(int max_val) {
        if (max_val <= 0) return 0;
        return static_cast<int>(next() % static_cast<uint32_t>(max_val));
    }
};

bool stage5_ransac_homography_cpu(const std::vector<Match>& matches, 
                                  const std::vector<Keypoint>& kpA, 
                                  const std::vector<Keypoint>& kpB, 
                                  Homography& best_H, 
                                  int& out_inliers, 
                                  float& elapsed_ms) {
    CpuTimer timer;
    timer.Start();

    int n_matches = static_cast<int>(matches.size());
    out_inliers = 0;
    best_H.is_valid = false;

    if (n_matches < 4) {
        elapsed_ms = timer.Stop();
        return false;
    }

    LcgRng rng(42);

    int max_inliers = 0;
    float current_H[9];
    float opt_H[9];
    std::fill(opt_H, opt_H + 9, 0.0f);
    opt_H[0] = opt_H[4] = opt_H[8] = 1.0f;

    for (int iter = 0; iter < RANSAC_ITERS; iter++) {
        // 4 sample indices phân biệt
        int sample_idx[4];
        for (int i = 0; i < 4; i++) {
            bool unique = false;
            while (!unique) {
                sample_idx[i] = rng.next_int(n_matches);
                unique = true;
                for (int prev = 0; prev < i; prev++) {
                    if (sample_idx[i] == sample_idx[prev]) {
                        unique = false;
                        break;
                    }
                }
            }
        }

        float ptsA[4][2], ptsB[4][2];
        for (int i = 0; i < 4; i++) {
            const Match& m = matches[sample_idx[i]];
            ptsA[i][0] = kpA[m.idx_a].x;
            ptsA[i][1] = kpA[m.idx_a].y;
            ptsB[i][0] = kpB[m.idx_b].x;
            ptsB[i][1] = kpB[m.idx_b].y;
        }

        if (!solve_dlt_8x8_cpu(ptsA, ptsB, current_H)) {
            continue;
        }

        // Đếm inliers
        int inliers = 0;
        for (int m = 0; m < n_matches; m++) {
            float xa = kpA[matches[m].idx_a].x;
            float ya = kpA[matches[m].idx_a].y;
            float xb = kpB[matches[m].idx_b].x;
            float yb = kpB[matches[m].idx_b].y;

            float w = current_H[6] * xa + current_H[7] * ya + current_H[8];
            if (std::abs(w) < 1e-6f) continue;

            float pred_xb = (current_H[0] * xa + current_H[1] * ya + current_H[2]) / w;
            float pred_yb = (current_H[3] * xa + current_H[4] * ya + current_H[5]) / w;

            float dx = pred_xb - xb;
            float dy = pred_yb - yb;
            float err = std::sqrt(dx * dx + dy * dy);

            if (err < RANSAC_THRESH) {
                inliers++;
            }
        }

        // Tie-break: Chỉ cập nhật khi inliers strictly lớn hơn (giữ hypothesis ID nhỏ hơn)
        if (inliers > max_inliers) {
            max_inliers = inliers;
            std::memcpy(opt_H, current_H, 9 * sizeof(float));
        }
    }

    out_inliers = max_inliers;
    if (max_inliers >= 4) {
        std::memcpy(best_H.h, opt_H, 9 * sizeof(float));
        best_H.is_valid = true;
    }

    elapsed_ms = timer.Stop();
    return best_H.is_valid;
}

// ============================================================================
// STAGE 6: WARPING & BLENDING (Inverse Warping + Symmetric Feathering)
// ============================================================================
static bool invert_3x3(const float H[9], float invH[9]) {
    float det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
                H[1] * (H[3] * H[8] - H[5] * H[6]) +
                H[2] * (H[3] * H[7] - H[4] * H[6]);

    if (std::abs(det) < 1e-8f) return false;

    float inv_det = 1.0f / det;
    invH[0] = (H[4] * H[8] - H[5] * H[7]) * inv_det;
    invH[1] = (H[2] * H[7] - H[1] * H[8]) * inv_det;
    invH[2] = (H[1] * H[5] - H[2] * H[4]) * inv_det;
    invH[3] = (H[5] * H[6] - H[3] * H[8]) * inv_det;
    invH[4] = (H[0] * H[8] - H[2] * H[6]) * inv_det;
    invH[5] = (H[2] * H[3] - H[0] * H[5]) * inv_det;
    invH[6] = (H[3] * H[7] - H[4] * H[6]) * inv_det;
    invH[7] = (H[1] * H[6] - H[0] * H[7]) * inv_det;
    invH[8] = (H[0] * H[4] - H[1] * H[3]) * inv_det;

    return true;
}

static uchar4 sample_bilinear(const Image& img, float x, float y) {
    if (x < 0.0f || x >= img.w - 1 || y < 0.0f || y >= img.h - 1) {
        return make_uchar4(0, 0, 0, 0);
    }

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float fx = x - x0;
    float fy = y - y0;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    uchar4 p00 = img.data[y0 * img.w + x0];
    uchar4 p10 = img.data[y0 * img.w + x1];
    uchar4 p01 = img.data[y1 * img.w + x0];
    uchar4 p11 = img.data[y1 * img.w + x1];

    uint8_t r = static_cast<uint8_t>(w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x + 0.5f);
    uint8_t g = static_cast<uint8_t>(w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y + 0.5f);
    uint8_t b = static_cast<uint8_t>(w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z + 0.5f);
    uint8_t a = 255;

    return make_uchar4(r, g, b, a);
}

Image stage6_warp_blend_cpu(const Image& imgA, 
                            const Image& imgB, 
                            const Homography& H, 
                            bool& out_success, 
                            float& elapsed_ms) {
    CpuTimer timer;
    timer.Start();
    out_success = false;

    if (!H.is_valid || !H.check_finite()) {
        elapsed_ms = timer.Stop();
        return Image();
    }

    float invH[9];
    if (!invert_3x3(H.h, invH)) {
        elapsed_ms = timer.Stop();
        return Image();
    }

    // Các góc của Image B được chiếu về hệ tọa độ Image A bằng invH
    float corners_B[4][2] = {
        { 0.0f, 0.0f },
        { static_cast<float>(imgB.w), 0.0f },
        { 0.0f, static_cast<float>(imgB.h) },
        { static_cast<float>(imgB.w), static_cast<float>(imgB.h) }
    };

    float min_x = 0.0f, max_x = static_cast<float>(imgA.w);
    float min_y = 0.0f, max_y = static_cast<float>(imgA.h);

    for (int i = 0; i < 4; i++) {
        float bx = corners_B[i][0];
        float by = corners_B[i][1];
        float w = invH[6] * bx + invH[7] * by + invH[8];
        if (std::abs(w) > 1e-6f) {
            float ax = (invH[0] * bx + invH[1] * by + invH[2]) / w;
            float ay = (invH[3] * bx + invH[4] * by + invH[5]) / w;
            if (!std::isfinite(ax) || !std::isfinite(ay)) {
                elapsed_ms = timer.Stop();
                return Image();
            }
            min_x = std::min(min_x, ax);
            max_x = std::max(max_x, ax);
            min_y = std::min(min_y, ay);
            max_y = std::max(max_y, ay);
        } else {
            elapsed_ms = timer.Stop();
            return Image();
        }
    }

    int canvas_x_min = static_cast<int>(std::floor(min_x));
    int canvas_x_max = static_cast<int>(std::ceil(max_x));
    int canvas_y_min = static_cast<int>(std::floor(min_y));
    int canvas_y_max = static_cast<int>(std::ceil(max_y));

    int canvas_w = canvas_x_max - canvas_x_min;
    int canvas_h = canvas_y_max - canvas_y_min;

    // Validate canvas bounds: Không clamp âm thầm, kiểm tra giới hạn an toàn Plan v4
    int max_allowed_w = imgA.w * 4;
    int max_allowed_h = imgA.h * 4;
    size_t max_allowed_pixels = static_cast<size_t>(imgA.w * imgA.h) * 16;

    if (canvas_w <= 0 || canvas_h <= 0 ||
        canvas_w > max_allowed_w || canvas_h > max_allowed_h ||
        static_cast<size_t>(canvas_w) * canvas_h > max_allowed_pixels) {
        std::cerr << "[Error] Canvas bounds validation failed: " << canvas_w << "x" << canvas_h 
                  << " exceeds safety threshold!\n";
        elapsed_ms = timer.Stop();
        return Image();
    }

    Image panorama(canvas_w, canvas_h);
    std::memset(panorama.data, 0, (size_t)canvas_w * canvas_h * sizeof(uchar4));

    // Inverse Warping với Symmetric Feathering
    for (int cy = 0; cy < canvas_h; cy++) {
        float ax = static_cast<float>(cy + canvas_y_min);
        for (int cx = 0; cx < canvas_w; cx++) {
            float cur_ax = static_cast<float>(cx + canvas_x_min);

            bool in_A = (cur_ax >= 0.0f && cur_ax < imgA.w && ax >= 0.0f && ax < imgA.h);
            uchar4 col_A = make_uchar4(0, 0, 0, 0);
            if (in_A) {
                int iax = static_cast<int>(cur_ax);
                int iay = static_cast<int>(ax);
                col_A = imgA.data[iay * imgA.w + iax];
            }

            // Chiếu tọa độ frame A sang frame B bằng H: p_B = H * p_A
            float w = H.h[6] * cur_ax + H.h[7] * ax + H.h[8];
            bool in_B = false;
            uchar4 col_B = make_uchar4(0, 0, 0, 0);
            float bx = 0.0f, by = 0.0f;

            if (std::abs(w) > 1e-6f) {
                bx = (H.h[0] * cur_ax + H.h[1] * ax + H.h[2]) / w;
                by = (H.h[3] * cur_ax + H.h[4] * ax + H.h[5]) / w;
                if (bx >= 0.0f && bx < imgB.w - 1 && by >= 0.0f && by < imgB.h - 1) {
                    col_B = sample_bilinear(imgB, bx, by);
                    in_B = (col_B.w > 0);
                }
            }

            uchar4 out_col = make_uchar4(0, 0, 0, 0);
            if (in_A && in_B) {
                // Symmetric feathering dựa trên khoảng cách tới cả 4 biên của A và B
                float dist_A = std::min({cur_ax, static_cast<float>(imgA.w - 1 - cur_ax),
                                         ax,     static_cast<float>(imgA.h - 1 - ax)});
                float dist_B = std::min({bx,     static_cast<float>(imgB.w - 1 - bx),
                                         by,     static_cast<float>(imgB.h - 1 - by)});

                float alpha = (dist_A + 1e-6f) / (dist_A + dist_B + 2e-6f);
                alpha = std::clamp(alpha, 0.0f, 1.0f);

                out_col.x = static_cast<uint8_t>(alpha * col_A.x + (1.0f - alpha) * col_B.x);
                out_col.y = static_cast<uint8_t>(alpha * col_A.y + (1.0f - alpha) * col_B.y);
                out_col.z = static_cast<uint8_t>(alpha * col_A.z + (1.0f - alpha) * col_B.z);
                out_col.w = 255;
            } else if (in_A) {
                out_col = col_A;
                out_col.w = 255;
            } else if (in_B) {
                out_col = col_B;
                out_col.w = 255;
            }

            panorama.data[cy * canvas_w + cx] = out_col;
        }
    }

    out_success = true;
    elapsed_ms = timer.Stop();
    return panorama;
}

// ============================================================================
// EXPORT METRICS JSON (Plan v4 Section 4.3 Schema)
// ============================================================================
bool PipelineResult::export_metrics_json(const std::string& json_path) const {
    std::ofstream ofs(json_path);
    if (!ofs.is_open()) return false;

    auto now = std::chrono::system_clock::now();
    std::time_t t_now = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t_now));

    ofs << "{\n";
    ofs << "  \"schema_version\": \"4.0\",\n";
    ofs << "  \"timestamp\": \"" << buf << "\",\n";
    ofs << "  \"version\": \"" << version_str << "\",\n";
    ofs << "  \"parent_version\": \"None\",\n";
    ofs << "  \"optimization\": \"" << optimization_str << "\",\n";
    ofs << "  \"device_name\": \"" << device_name << "\",\n";
    ofs << "  \"image_width\": " << image_width << ",\n";
    ofs << "  \"image_height\": " << image_height << ",\n";
    ofs << "  \"runs\": " << runs << ",\n";
    ofs << "  \"warmups\": " << warmups << ",\n";
    ofs << "  \"status\": \"" << status_to_string(diagnostics.status) << "\",\n";
    ofs << "  \"warning\": \"" << diagnostics.warning << "\",\n";
    ofs << "  \"keypoints_a\": " << diagnostics.keypoints_a << ",\n";
    ofs << "  \"keypoints_b\": " << diagnostics.keypoints_b << ",\n";
    ofs << "  \"matches\": " << diagnostics.matches << ",\n";
    ofs << "  \"inliers\": " << diagnostics.inliers << ",\n";
    ofs << "  \"inlier_ratio\": " << std::fixed << std::setprecision(4) << diagnostics.inlier_ratio << ",\n";

    ofs << "  \"homography_a_to_b\": [";
    for (int i = 0; i < 9; i++) {
        ofs << std::setprecision(6) << diagnostics.H.h[i] << (i < 8 ? ", " : "");
    }
    ofs << "],\n";

    ofs << "  \"timing_ms\": {\n";
    auto write_metric = [&ofs](const char* name, const StageMetric& m, bool comma = true) {
        ofs << "    \"" << name << "\": {\"avg\": " << std::fixed << std::setprecision(3) 
            << m.avg << ", \"min\": " << m.min << "}" << (comma ? ",\n" : "\n");
    };

    write_metric("stage1_preprocess", timings.stage1_preprocess);
    write_metric("stage2_fast_nms_topk", timings.stage2_fast_nms_topk);
    write_metric("stage3_orb_descriptor", timings.stage3_orb_descriptor);
    write_metric("stage4_matching", timings.stage4_matching);
    write_metric("stage5_ransac", timings.stage5_ransac);
    write_metric("stage6_warping_blending", timings.stage6_warping_blending);
    write_metric("kernel_total", timings.kernel_total);
    write_metric("h2d", timings.h2d);
    write_metric("control", timings.control);
    write_metric("d2h", timings.d2h);
    write_metric("end_to_end", timings.end_to_end);
    write_metric("setup", timings.setup, false);
    ofs << "  }\n";
    ofs << "}\n";

    return true;
}

// ============================================================================
// PIPELINE RUNNER V0
// ============================================================================
PipelineResult run_v0_pipeline(const Image& imgA, 
                               const Image& imgB, 
                               int runs) {
    if (runs <= 0) runs = DEFAULT_RUNS;

    PipelineResult result;
    result.runs = runs;
    result.warmups = 1;
    result.version_str = "V0";
    result.optimization_str = "Sequential Baseline (Single-thread C++)";
    result.image_width = imgA.w;
    result.image_height = imgA.h;
    result.device_name = "Host CPU";

    std::vector<float> t_s1(runs, 0.0f), t_s2(runs, 0.0f), t_s3(runs, 0.0f);
    std::vector<float> t_s4(runs, 0.0f), t_s5(runs, 0.0f), t_s6(runs, 0.0f);
    std::vector<float> t_e2e(runs, 0.0f);

    // Warm-up Run
    {
        float t1A, t1B, t2A, t2B, t3A, t3B, t4, t5, t6;
        GrayImage gA = stage1_preprocess_cpu(imgA, t1A);
        GrayImage gB = stage1_preprocess_cpu(imgB, t1B);
        auto kpA = stage2_fast_nms_topk_cpu(gA, DEFAULT_TOP_K, t2A);
        auto kpB = stage2_fast_nms_topk_cpu(gB, DEFAULT_TOP_K, t2B);
        auto descA = stage3_orb_descriptors_cpu(gA, kpA, t3A);
        auto descB = stage3_orb_descriptors_cpu(gB, kpB, t3B);
        auto matches = stage4_matching_cpu(descA, descB, t4);
        Homography H;
        int inliers = 0;
        bool ok_ransac = stage5_ransac_homography_cpu(matches, kpA, kpB, H, inliers, t5);
        if (ok_ransac) {
            bool ok_warp = false;
            Image pano = stage6_warp_blend_cpu(imgA, imgB, H, ok_warp, t6);
        }
    }

    // Benchmark Loop
    for (int r = 0; r < runs; r++) {
        CpuTimer timer_e2e;
        timer_e2e.Start();

        float t1A, t1B, t2A, t2B, t3A, t3B, t4, t5, t6;

        GrayImage gA = stage1_preprocess_cpu(imgA, t1A);
        GrayImage gB = stage1_preprocess_cpu(imgB, t1B);
        t_s1[r] = t1A + t1B;

        auto kpA = stage2_fast_nms_topk_cpu(gA, DEFAULT_TOP_K, t2A);
        auto kpB = stage2_fast_nms_topk_cpu(gB, DEFAULT_TOP_K, t2B);
        t_s2[r] = t2A + t2B;

        auto descA = stage3_orb_descriptors_cpu(gA, kpA, t3A);
        auto descB = stage3_orb_descriptors_cpu(gB, kpB, t3B);
        t_s3[r] = t3A + t3B;

        auto matches = stage4_matching_cpu(descA, descB, t4);
        t_s4[r] = t4;

        Homography H;
        int inliers = 0;
        bool ok_ransac = stage5_ransac_homography_cpu(matches, kpA, kpB, H, inliers, t5);
        t_s5[r] = t5;

        // Lưu thông số diagnostics ở lần chạy đầu tiên
        if (r == 0) {
            result.diagnostics.keypoints_a = static_cast<int>(kpA.size());
            result.diagnostics.keypoints_b = static_cast<int>(kpB.size());
            result.diagnostics.matches = static_cast<int>(matches.size());
            result.diagnostics.inliers = inliers;
            result.diagnostics.inlier_ratio = (matches.empty()) ? 0.0f : (static_cast<float>(inliers) / matches.size());
            result.diagnostics.H = H;

            if (!ok_ransac || inliers < 4) {
                result.diagnostics.status = PipelineStatus::FAILURE;
                result.diagnostics.warning = "RANSAC failed: less than 4 inliers or singular system";
            } else if (inliers < std::max(12, static_cast<int>(0.15f * matches.size()))) {
                result.diagnostics.status = PipelineStatus::LOW_CONFIDENCE;
                result.diagnostics.warning = "Low inlier ratio or count; output may have alignment defects";
            } else {
                result.diagnostics.status = PipelineStatus::SUCCESS;
                result.diagnostics.warning = "None";
            }
        }

        bool ok_warp = false;
        if (ok_ransac) {
            Image pano = stage6_warp_blend_cpu(imgA, imgB, H, ok_warp, t6);
            if (r == runs - 1 && ok_warp) {
                result.panorama = std::move(pano);
            }
        } else {
            t6 = 0.0f;
        }
        t_s6[r] = t6;

        t_e2e[r] = timer_e2e.Stop();
    }

    // Tính thống kê Avg và Min
    auto compute_stat = [runs](const std::vector<float>& vec) -> StageMetric {
        float sum = 0.0f;
        float min_val = std::numeric_limits<float>::max();
        for (float v : vec) {
            sum += v;
            min_val = std::min(min_val, v);
        }
        return {sum / runs, min_val};
    };

    result.timings.stage1_preprocess       = compute_stat(t_s1);
    result.timings.stage2_fast_nms_topk    = compute_stat(t_s2);
    result.timings.stage3_orb_descriptor   = compute_stat(t_s3);
    result.timings.stage4_matching         = compute_stat(t_s4);
    result.timings.stage5_ransac           = compute_stat(t_s5);
    result.timings.stage6_warping_blending = compute_stat(t_s6);
    result.timings.end_to_end              = compute_stat(t_e2e);

    // Kernel total cho V0 tương ứng tổng các stage tính toán
    std::vector<float> t_total(runs, 0.0f);
    for (int r = 0; r < runs; r++) {
        t_total[r] = t_s1[r] + t_s2[r] + t_s3[r] + t_s4[r] + t_s5[r] + t_s6[r];
    }
    result.timings.kernel_total = compute_stat(t_total);

    // H2D, control, d2h, setup cho V0 CPU đều là 0
    result.timings.h2d     = {0.0f, 0.0f};
    result.timings.control = {0.0f, 0.0f};
    result.timings.d2h     = {0.0f, 0.0f};
    result.timings.setup   = {0.0f, 0.0f};

    return result;
}
