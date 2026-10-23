// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: V0 Sequential Baseline Header (Plan v4)
// Authors: Group 19 - Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// External code: none (ORB pattern table is declared here, defined in v0_sequential.cpp)
// ============================================================================

#pragma once

#include "common.h"
#include <vector>

// Stage 1: Preprocessing (Grayscale + Gaussian Blur 5x5)
GrayImage stage1_preprocess_cpu(const Image& img, float& elapsed_ms);

// Stage 2: Keypoint Detection (FAST-9 + Border Filter + 3x3 NMS + Deterministic Top-K)
std::vector<Keypoint> stage2_fast_nms_topk_cpu(const GrayImage& img, int K, float& elapsed_ms);

// Stage 3: Description (Oriented BRIEF / ORB 256-bit with OpenCV bit_pattern_31_)
std::vector<Descriptor> stage3_orb_descriptors_cpu(const GrayImage& img, 
                                                   const std::vector<Keypoint>& kps, 
                                                   float& elapsed_ms);

// Stage 4: Matching (Brute-force 8x32-bit Hamming + Lowe's Ratio Test)
std::vector<Match> stage4_matching_cpu(const std::vector<Descriptor>& descA, 
                                       const std::vector<Descriptor>& descB, 
                                       float& elapsed_ms);

// Stage 5: Homography Estimation (RANSAC 2000 + Normalized 4-point DLT + Partial Pivoting
//          + least-squares refit on all inliers, 3 rounds)
bool stage5_ransac_homography_cpu(const std::vector<Match>& matches, 
                                  const std::vector<Keypoint>& kpA, 
                                  const std::vector<Keypoint>& kpB, 
                                  Homography& best_H, 
                                  int& out_inliers, 
                                  float& elapsed_ms);

// Stage 6: Warping & Blending (Inverse Warping + Bilinear Sampling + Symmetric Feathering)
Image stage6_warp_blend_cpu(const Image& imgA, 
                            const Image& imgB, 
                            const Homography& H, 
                            bool& out_success, 
                            float& elapsed_ms);

// Pipeline Runner V0: Handles warmups, repeated runs, Avg/Min telemetry, and PipelineResult creation
PipelineResult run_v0_pipeline(const Image& imgA, 
                               const Image& imgB, 
                               int runs = DEFAULT_RUNS);
