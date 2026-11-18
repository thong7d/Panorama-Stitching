// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: Stage 4 CUDA Kernels — Brute-force Hamming Matching + Lowe's Ratio Test
// Phụ trách & Tác giả chính: Dao Quoc Tuan (23120392)
// Nhóm tác giả: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// External sources: David Lowe (2004) "Distinctive Image Features from Scale-Invariant Keypoints";
//   NVIDIA CUDA C++ Programming Guide (developer.nvidia.com) cho lệnh nội tại __popc().
//   Bất biến thuật toán khớp 100% V0 CPU (v0_sequential.cpp Stage 4).
// ============================================================================

#include "cuda_pipeline.cuh"

// ============================================================================
// Kernel 1: Brute-force Hamming Distance + Lowe's Ratio Test
// Mỗi thread xử lý 1 descriptor A, duyệt toàn bộ mảng descriptor B để tìm:
//   - best1_dist: khoảng cách Hamming nhỏ nhất (lân cận gần nhất)
//   - best2_dist: khoảng cách Hamming nhỏ nhì (lân cận gần nhì)
//   - best1_idx: chỉ số của lân cận gần nhất trong B
//
// Lệnh nội tại __popc(): tính số bit 1 trong 32-bit integer trong 1 cycle GPU.
// Descriptor 256-bit gồm 8 words 32-bit: tổng popcount trên 8 words.
//
// Kết quả ghi tạm thời vào d_matches_out[i]:
//   - Nếu thỏa mãn ratio test: Match{idx_a = i, idx_b = best1_idx, dist = best1_dist}
//   - Nếu không thỏa mãn: Match{idx_a = -1, idx_b = -1, dist = 9999}
// ============================================================================
__global__ void kernel_stage4_matching(const Descriptor* __restrict__ d_descA,
                                       int nA,
                                       const Descriptor* __restrict__ d_descB,
                                       int nB,
                                       float ratio_threshold,
                                       Match* __restrict__ d_matches_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nA) return;

    // Tải descriptor A vào thanh ghi của thread
    Descriptor descA_i = d_descA[i];

    int best1_dist = 9999;
    int best2_dist = 9999;
    int best1_idx = -1;

    // Duyệt toàn bộ descriptor trong ảnh B
    // Các thread trong warp cùng đọc descB[j] tại cùng chu kỳ -> Broadcast load hiệu quả
    for (int j = 0; j < nB; j++) {
        Descriptor descB_j = d_descB[j];
        int d = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            d += __popc(descA_i.words[k] ^ descB_j.words[k]);
        }

        if (d < best1_dist) {
            best2_dist = best1_dist;
            best1_dist = d;
            best1_idx = j;
        } else if (d < best2_dist) {
            best2_dist = d;
        }
    }

    // Kiểm tra Lowe's Ratio Test nghiêm ngặt (best1 < ratio * best2)
    // Nếu best1 == best2 (tie-break), điều kiện sai -> điểm bị loại (khớp 100% V0 CPU)
    Match m;
    m.idx_a = -1;
    m.idx_b = -1;
    m.dist = 9999;

    if (best1_dist < ratio_threshold * static_cast<float>(best2_dist) && best1_idx >= 0) {
        m.idx_a = i;
        m.idx_b = best1_idx;
        m.dist = best1_dist;
    }

    d_matches_out[i] = m;
}

// ============================================================================
// Kernel 2: Deterministic In-Place Stream Compaction
// Thu gọn mảng d_matches_out tại chỗ để loại bỏ các phần tử có idx_a == -1.
// Đảm bảo thứ tự xuất mảng d_matches luôn TĂNG DẦN THEO idx_a (Plan v4 Mục 18.4).
// Vì nA <= 2000, 1 thread duy nhất trên GPU thực thi trong < 3 micro-giây từ L1/L2 cache,
// đảm bảo tính tất định 100%, không cần bộ nhớ đệm phụ và không bị race condition.
// ============================================================================
__global__ void kernel_stage4_compact(Match* __restrict__ d_matches,
                                       int nA,
                                       int* __restrict__ d_nmatches) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int count = 0;
    for (int i = 0; i < nA; i++) {
        Match m = d_matches[i];
        if (m.idx_a >= 0) {
            d_matches[count] = m;
            count++;
        }
    }
    *d_nmatches = count;
}

// ============================================================================
// stage4_matching_gpu — API công khai (khai báo trong cuda_pipeline.cuh)
// Đo thời gian bằng cudaEvent trên stream được chỉ định.
// ============================================================================
float stage4_matching_gpu(const Descriptor* d_descA,
                          int               nA,
                          const Descriptor* d_descB,
                          int               nB,
                          Match*            d_matches,
                          int*              d_nmatches,
                          float             ratio_threshold,
                          cudaStream_t      stream) {
    if (d_matches == nullptr || d_nmatches == nullptr) {
        std::cerr << "[Lỗi Stage 4 GPU] Con trỏ d_matches hoặc d_nmatches là nullptr!\n";
        return 0.0f;
    }

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start, stream));

    // Trường hợp biên: nA <= 0 hoặc nB < 2
    // Không thể thực hiện Lowe's ratio test nếu B có ít hơn 2 điểm -> trả về 0 matches
    if (nA <= 0 || nB < 2 || d_descA == nullptr || d_descB == nullptr) {
        CHECK_CUDA(cudaMemsetAsync(d_nmatches, 0, sizeof(int), stream));
        CHECK_CUDA(cudaEventRecord(ev_stop, stream));
        CHECK_CUDA(cudaEventSynchronize(ev_stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
        CHECK_CUDA(cudaEventDestroy(ev_start));
        CHECK_CUDA(cudaEventDestroy(ev_stop));
        return elapsed_ms;
    }

    // Giới hạn an toàn nA, nB không vượt quá capacity
    int clamped_nA = (nA > MAX_MATCHES) ? MAX_MATCHES : nA;
    int clamped_nB = (nB > MAX_DESCRIPTORS) ? MAX_DESCRIPTORS : nB;

    // Pass 1: Hamming distance + Ratio test
    dim3 block(128);
    dim3 grid((clamped_nA + block.x - 1) / block.x);
    kernel_stage4_matching<<<grid, block, 0, stream>>>(
        d_descA, clamped_nA, d_descB, clamped_nB, ratio_threshold, d_matches);
    CHECK_CUDA(cudaGetLastError());

    // Pass 2: Stream compaction bảo toàn thứ tự idx_a tăng dần
    kernel_stage4_compact<<<1, 1, 0, stream>>>(
        d_matches, clamped_nA, d_nmatches);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));

    return elapsed_ms;
}
