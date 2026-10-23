// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Phase A CPU Unit Tests (Plan v4 Section 6 & Quality Gate A)
// Authors: Group 19 - Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// External code: none
// ============================================================================

#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include <set>
#include <vector>

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

// Stage 5 on synthetic correspondences with a known homography: Gaussian noise on the inliers and
// 30% gross outliers. Checks that RANSAC + least-squares refit returns a model that is accurate
// everywhere in the overlap, not only on the 4 points of the winning minimal sample.
void test_stage5_ransac_refit() {
    std::cout << "[Test 5] Testing Stage 5 RANSAC + inlier refit on noisy correspondences with outliers...\n";

    const float GT[9] = { 1.05f, 0.02f, 300.0f,  -0.01f, 1.03f, 10.0f,  2e-5f, 1e-5f, 1.0f };
    auto project = [](const float* H, float x, float y, float& u, float& v) {
        float w = H[6] * x + H[7] * y + H[8];
        u = (H[0] * x + H[1] * y + H[2]) / w;
        v = (H[3] * x + H[4] * y + H[5]) / w;
    };

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> ux(50.0f, 1230.0f), uy(50.0f, 670.0f), ub(0.0f, 1280.0f), ubv(0.0f, 720.0f);
    std::normal_distribution<float> noise(0.0f, 0.6f);

    std::vector<Keypoint> kpA, kpB;
    std::vector<Match> matches;
    int true_inliers = 0;
    while (static_cast<int>(matches.size()) < 300) {
        float x = ux(gen), y = uy(gen), u, v;
        project(GT, x, y, u, v);
        if (u < 0.0f || u >= 1280.0f || v < 0.0f || v >= 720.0f) continue;
        bool outlier = (matches.size() % 10) < 3;           // 30% gross outliers
        if (outlier) { u = ub(gen); v = ubv(gen); }
        else         { u += noise(gen); v += noise(gen); true_inliers++; }
        Keypoint a{x, y, 0.0f, 1.0f}, b{u, v, 0.0f, 1.0f};
        Match m;
        m.idx_a = static_cast<int>(kpA.size());
        m.idx_b = static_cast<int>(kpB.size());
        m.dist = 0;
        kpA.push_back(a);
        kpB.push_back(b);
        matches.push_back(m);
    }

    Homography H;
    int inliers = 0;
    float ms = 0.0f;
    bool ok = stage5_ransac_homography_cpu(matches, kpA, kpB, H, inliers, ms);
    assert(ok && H.is_valid && H.check_finite() && "Stage 5 must return a valid finite homography");
    assert(inliers >= static_cast<int>(0.95f * true_inliers) && "RANSAC must recover almost all true inliers");

    // RMS transfer error over a grid of the overlap region (pixels in image B)
    double se = 0.0;
    int n = 0;
    for (int gy = 0; gy < 10; gy++) {
        for (int gx = 0; gx < 10; gx++) {
            float x = 100.0f + gx * 100.0f, y = 80.0f + gy * 55.0f, u0, v0, u1, v1;
            project(GT, x, y, u0, v0);
            project(H.h, x, y, u1, v1);
            se += (u0 - u1) * (u0 - u1) + (v0 - v1) * (v0 - v1);
            n++;
        }
    }
    double rmse = std::sqrt(se / n);
    assert(rmse < 0.5 && "Stage 5 grid RMSE against the true homography must stay below 0.5 px");

    std::cout << "  -> true inliers: " << true_inliers << " | found: " << inliers
              << " | grid RMSE: " << rmse << " px (PASS)\n";
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "Running Phase A CPU Unit Tests (Plan v4 Quality Gate A)\n";
    std::cout << "=========================================================\n";

    test_orb_pattern();
    test_raii_contracts();
    test_descriptor_hamming();
    test_homography_properties();
    test_stage5_ransac_refit();

    std::cout << "=========================================================\n";
    std::cout << "[ALL UNIT TESTS PASSED SUCCESSFULLY]\n";
    std::cout << "=========================================================\n";
    return 0;
}
