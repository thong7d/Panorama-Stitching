// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Phase A CPU Unit Tests (Plan v4 Section 6 & Quality Gate A)
// Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
//          Dao Quoc Tuan (23120392)
// External sources: OpenCV ORB pattern properties; see THIRD_PARTY_NOTICES.md.
// ============================================================================

#include <iostream>
#include <cassert>
#include <cmath>
#include <set>
#include <vector>
#include <limits>

#include "common.h"
#include "v0_sequential.h"

void test_orb_pattern() {
    std::cout << "[Test 1] Testing ORB bit pattern properties (OpenCV bit_pattern_31_)...\n";
    
    // 1. Verify exactly 256 test pairs
    int count = 0;
    std::set<std::pair<std::pair<int, int>, std::pair<int, int>>> unique_pairs;
    
    int max_coord = 0;
    float max_radius_sq = 0.0f;

    for (int i = 0; i < 256; i++) {
        int2 p1 = ORB_PATTERN_PAIRS[i][0];
        int2 p2 = ORB_PATTERN_PAIRS[i][1];
        count++;

        auto pair_key = std::make_pair(std::make_pair(p1.x, p1.y), std::make_pair(p2.x, p2.y));
        unique_pairs.insert(pair_key);

        max_coord = std::max(max_coord, std::abs(p1.x));
        max_coord = std::max(max_coord, std::abs(p1.y));
        max_coord = std::max(max_coord, std::abs(p2.x));
        max_coord = std::max(max_coord, std::abs(p2.y));

        float r1_sq = static_cast<float>(p1.x * p1.x + p1.y * p1.y);
        float r2_sq = static_cast<float>(p2.x * p2.x + p2.y * p2.y);
        max_radius_sq = std::max(max_radius_sq, std::max(r1_sq, r2_sq));
    }

    assert(count == 256 && "ORB pattern must have exactly 256 test pairs");
    assert(unique_pairs.size() == 256 && "All 256 ORB pattern pairs must be strictly unique!");
    assert(max_coord <= 15 && "ORB coordinate must be within patch radius (-15..+15)");
    
    float max_radius = std::sqrt(max_radius_sq);
    assert(max_radius < BORDER_MARGIN && "Max sampling radius must be strictly within BORDER_MARGIN (20 px)");

    std::cout << "  -> Count: " << count << " pairs | Unique: " << unique_pairs.size() 
              << " | Max coord: " << max_coord << " | Max radius: " << max_radius << " px (PASS)\n";
}

void test_raii_contracts() {
    std::cout << "[Test 2] Testing RAII move-only contracts for Image & GrayImage...\n";

    // 1. Image test
    {
        Image img;
        assert(img.empty());
        img.allocate(64, 48);
        assert(!img.empty());
        assert(img.w == 64 && img.h == 48);
        img.data[0] = make_uchar4(12, 34, 56, 255);

        // Move constructor
        Image moved(std::move(img));
        assert(img.empty());
        assert(img.data == nullptr);
        assert(!moved.empty());
        assert(moved.w == 64 && moved.h == 48);
        assert(moved.data[0].x == 12 && moved.data[0].y == 34 && moved.data[0].z == 56 && moved.data[0].w == 255);

        // Move assignment
        Image assigned;
        assigned = std::move(moved);
        assert(moved.empty());
        assert(assigned.w == 64);
        assert(assigned.data[0].x == 12 && assigned.data[0].y == 34 && assigned.data[0].z == 56 && assigned.data[0].w == 255);

        // Idempotent free
        assigned.free();
        assert(assigned.empty());
        assigned.free(); // Safe double-free check
    }

    // 2. GrayImage test
    {
        GrayImage gimg;
        assert(gimg.empty());
        gimg.allocate(32, 24);
        assert(!gimg.empty());
        gimg.data[0] = 45.67f;

        // Move constructor
        GrayImage moved(std::move(gimg));
        assert(gimg.empty());
        assert(!moved.empty());
        assert(moved.w == 32 && std::abs(moved.data[0] - 45.67f) < 1e-5f);

        // Move assignment
        GrayImage assigned;
        assigned = std::move(moved);
        assert(moved.empty());
        assert(assigned.w == 32);

        // Idempotent free
        assigned.free();
        assert(assigned.empty());
        assigned.free(); // Safe double-free check
    }

    std::cout << "  -> Image and GrayImage RAII move-only semantics verified (PASS)\n";
}

void test_descriptor_hamming() {
    std::cout << "[Test 3] Testing 256-bit Hamming Distance (8x32-bit popcount)...\n";

    Descriptor d1, d2;
    // Identical descriptors -> dist = 0
    int dist0 = 0;
    for (int i = 0; i < 8; i++) {
        uint32_t x = d1.words[i] ^ d2.words[i];
#if defined(_MSC_VER)
        dist0 += static_cast<int>(__popcnt(x));
#else
        dist0 += __builtin_popcount(x);
#endif
    }
    assert(dist0 == 0 && "Identical descriptors must have distance 0");

    // Single bit flip in word 0 -> dist = 1
    d2.words[0] = 1;
    int dist1 = 0;
    for (int i = 0; i < 8; i++) {
        uint32_t x = d1.words[i] ^ d2.words[i];
#if defined(_MSC_VER)
        dist1 += static_cast<int>(__popcnt(x));
#else
        dist1 += __builtin_popcount(x);
#endif
    }
    assert(dist1 == 1 && "1 bit flip must give distance 1");

    // All 256 bits inverted -> dist = 256
    for (int i = 0; i < 8; i++) {
        d2.words[i] = 0xFFFFFFFF;
    }
    int dist256 = 0;
    for (int i = 0; i < 8; i++) {
        uint32_t x = d1.words[i] ^ d2.words[i];
#if defined(_MSC_VER)
        dist256 += static_cast<int>(__popcnt(x));
#else
        dist256 += __builtin_popcount(x);
#endif
    }
    assert(dist256 == 256 && "All bits inverted must give distance 256");

    std::cout << "  -> Hamming distances: 0, 1, 256 exact (PASS)\n";
}

void test_homography_properties() {
    std::cout << "[Test 4] Testing Homography validation & normalization...\n";

    Homography H;
    assert(H.check_finite() && "Identity homography must be finite");
    assert(std::abs(H.h[0] - 1.0f) < 1e-6f);
    assert(std::abs(H.h[4] - 1.0f) < 1e-6f);
    assert(std::abs(H.h[8] - 1.0f) < 1e-6f);

    // Test scaling normalization
    for (int i = 0; i < 9; i++) H.h[i] *= 5.0f;
    assert(std::abs(H.h[8] - 5.0f) < 1e-6f);
    H.normalize();
    assert(std::abs(H.h[8] - 1.0f) < 1e-6f);
    assert(std::abs(H.h[0] - 1.0f) < 1e-6f);

    // Test non-finite check
    H.h[2] = std::numeric_limits<float>::quiet_NaN();
    assert(!H.check_finite() && "NaN in Homography must fail check_finite()");

    std::cout << "  -> Homography finite and normalization checks passed (PASS)\n";
}

void test_fast9_two_cardinal_shortcut() {
    std::cout << "[Test 5] Testing FAST-9 shortcut does not reject valid 9-pixel arcs...\n";
    GrayImage image(64, 64);
    std::fill(image.data, image.data + image.w * image.h, 100.0f);
    const int cx = 32, cy = 32;
    const int circle_x[16] = {0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1};
    const int circle_y[16] = {-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3};
    // Indices 1..9 form a valid FAST-9 arc but include only cardinal samples 4 and 8.
    for (int i = 1; i <= 9; i++) {
        image.data[(cy + circle_y[i]) * image.w + cx + circle_x[i]] = 160.0f;
    }
    float elapsed = 0.0f;
    const auto keypoints = stage2_fast_nms_topk_cpu(image, 100, elapsed);
    const bool found_center = std::any_of(
        keypoints.begin(), keypoints.end(), [cx, cy](const Keypoint& kp) {
            return static_cast<int>(kp.x) == cx && static_cast<int>(kp.y) == cy;
        });
    assert(found_center && "A valid FAST-9 arc with two cardinal samples was rejected");
    std::cout << "  -> Valid two-cardinal FAST-9 arc retained (PASS)\n";
}

void test_ransac_refit_accuracy() {
    std::cout << "[Test 6] Testing RANSAC consensus refit on noisy correspondences...\n";
    const float gt[9] = {
        1.01f, -0.02f, 40.0f,
        0.01f, 0.99f, 15.0f,
        0.00001f, -0.00002f, 1.0f
    };
    std::vector<Keypoint> points_a, points_b;
    std::vector<Match> matches;
    for (int i = 0; i < 60; i++) {
        const float x = 40.0f + static_cast<float>((i % 10) * 70);
        const float y = 35.0f + static_cast<float>((i / 10) * 65);
        const float w = gt[6] * x + gt[7] * y + gt[8];
        float u = (gt[0] * x + gt[1] * y + gt[2]) / w;
        float v = (gt[3] * x + gt[4] * y + gt[5]) / w;
        if (i < 50) {
            u += 0.15f * std::sin(static_cast<float>(i));
            v += 0.15f * std::cos(static_cast<float>(i));
        } else {
            u += 80.0f + i;
            v -= 60.0f + i;
        }
        points_a.push_back({x, y, 0.0f, 1.0f});
        points_b.push_back({u, v, 0.0f, 1.0f});
        matches.push_back({i, i, 0});
    }

    Homography estimated;
    int inliers = 0;
    float elapsed = 0.0f;
    const bool ok = stage5_ransac_homography_cpu(
        matches, points_a, points_b, estimated, inliers, elapsed);
    assert(ok && estimated.is_valid);
    assert(inliers >= 48 && "RANSAC should retain nearly all 50 synthetic inliers");

    double squared_error = 0.0;
    int count = 0;
    for (int i = 0; i < 50; i++) {
        const float x = points_a[i].x, y = points_a[i].y;
        const float wg = gt[6] * x + gt[7] * y + gt[8];
        const float we = estimated.h[6] * x + estimated.h[7] * y + estimated.h[8];
        const float ug = (gt[0] * x + gt[1] * y + gt[2]) / wg;
        const float vg = (gt[3] * x + gt[4] * y + gt[5]) / wg;
        const float ue = (estimated.h[0] * x + estimated.h[1] * y + estimated.h[2]) / we;
        const float ve = (estimated.h[3] * x + estimated.h[4] * y + estimated.h[5]) / we;
        squared_error += (ue - ug) * (ue - ug) + (ve - vg) * (ve - vg);
        count++;
    }
    const double rmse = std::sqrt(squared_error / count);
    assert(rmse < 0.5 && "Consensus-refitted homography must be sub-pixel accurate");
    std::cout << "  -> Inliers: " << inliers << " | grid RMSE: " << rmse << " px (PASS)\n";
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "Running Phase A CPU Unit Tests (Plan v4 Quality Gate A)\n";
    std::cout << "=========================================================\n";

    test_orb_pattern();
    test_raii_contracts();
    test_descriptor_hamming();
    test_homography_properties();
    test_fast9_two_cardinal_shortcut();
    test_ransac_refit_accuracy();

    std::cout << "=========================================================\n";
    std::cout << "[ALL UNIT TESTS PASSED SUCCESSFULLY]\n";
    std::cout << "=========================================================\n";
    return 0;
}
