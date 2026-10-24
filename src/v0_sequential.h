// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama (Panorama Image Stitching)
// Module: Khai báo Baseline tuần tự V0 (Plan v4)
// Tác giả: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
//          Dao Quoc Tuan (23120392)
// External sources: Mẫu ORB OpenCV và các bài báo tham khảo thuật toán; xem
// THIRD_PARTY_NOTICES.md. Giao diện hàm là mã gốc của đồ án.
// ============================================================================

#pragma once

#include "common.h"
#include <vector>

// Giai đoạn 1: Tiền xử lý (Mức xám + Lọc nhiễu Gaussian 5x5)
GrayImage stage1_preprocess_cpu(const Image& img, float& elapsed_ms);

// Giai đoạn 2: Phát hiện đặc trưng (FAST-9 + Lọc biên + 3x3 NMS + Top-K tất định)
std::vector<Keypoint> stage2_fast_nms_topk_cpu(const GrayImage& img, int K, float& elapsed_ms);

// Giai đoạn 3: Trích xuất mô tả (Oriented BRIEF / ORB 256-bit với mẫu OpenCV bit_pattern_31_)
std::vector<Descriptor> stage3_orb_descriptors_cpu(const GrayImage& img, 
                                                   const std::vector<Keypoint>& kps, 
                                                   float& elapsed_ms);

// Giai đoạn 4: Ghép cặp đặc trưng (Vét cạn Hamming 8x32-bit + Kiểm định tỷ số Lowe's)
std::vector<Match> stage4_matching_cpu(const std::vector<Descriptor>& descA, 
                                       const std::vector<Descriptor>& descB, 
                                       float& elapsed_ms);

// Giai đoạn 5: Ước lượng Homography (RANSAC 2000 + Normalized 4-point DLT + Khử Gauss)
bool stage5_ransac_homography_cpu(const std::vector<Match>& matches, 
                                  const std::vector<Keypoint>& kpA, 
                                  const std::vector<Keypoint>& kpB, 
                                  Homography& best_H, 
                                  int& out_inliers, 
                                  float& elapsed_ms);

// Giai đoạn 6: Chiếu hình & Trộn ảnh (Inverse Warping + Nội suy Bilinear + Trộn mềm Symmetric Feathering)
Image stage6_warp_blend_cpu(const Image& imgA, 
                            const Image& imgB, 
                            const Homography& H, 
                            bool& out_success, 
                            float& elapsed_ms);

// Điều phối viên Pipeline V0: Xử lý warm-up, đo đạc lặp lại Avg/Median/Min, và đóng gói PipelineResult
PipelineResult run_v0_pipeline(const Image& imgA, 
                               const Image& imgB, 
                               int runs = DEFAULT_RUNS);

