// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Phase A CPU Unit Tests (Plan v4 Section 6 & Quality Gate A)
// Authors: GroupXX (Members A, B, C)
// ============================================================================

#include <iostream>
#include <cassert>
#include <cmath>
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
        img.allocate(64, 48, 3);
        assert(!img.empty());
        assert(img.w == 64 && img.h == 48 && img.c == 3);
        img.data[0] = 123;

        // Move constructor
        Image moved(std::move(img));
        assert(img.empty());
        assert(img.data == nullptr);
        assert(!moved.empty());
        assert(moved.w == 64 && moved.h == 48 && moved.data[0] == 123);

        // Move assignment
        Image assigned;
        assigned = std::move(moved);
        assert(moved.empty());
        assert(assigned.w == 64 && assigned.data[0] == 123);

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

int main() {
    std::cout << "=========================================================\n";
    std::cout << "Running Phase A CPU Unit Tests (Plan v4 Quality Gate A)\n";
    std::cout << "=========================================================\n";

    test_orb_pattern();
    test_raii_contracts();
    test_descriptor_hamming();
    test_homography_properties();

    std::cout << "=========================================================\n";
    std::cout << "[ALL UNIT TESTS PASSED SUCCESSFULLY]\n";
    std::cout << "=========================================================\n";
    return 0;
}
