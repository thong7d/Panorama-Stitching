// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: CudaWorkspace Implementation + V1 Pipeline Orchestrator
// Tác giả: Pham Minh Thong (23120364)
// Phụ trách: CudaWorkspace RAII, điều phối viên run_v1_basic_cuda,
//            H2D/D2H, stubs CPU cho Stage 3-6 chờ Tiến & Tuấn triển khai GPU
// Phụ trách Stage 3-4 (sẽ thay stub): Nguyen Le Tan Tien (23120369)
// Phụ trách Stage 5-6 (sẽ thay stub): Dao Quoc Tuan (23120392)
// External sources: NVIDIA CUDA Best Practices Guide (developer.nvidia.com);
//   Plan v4 Mục C.3 (CudaWorkspace RAII, timing policy, benchmark structure).
// ============================================================================

#include "cuda_pipeline.cuh"
#include "v0_sequential.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstdio>

// ============================================================================
// CudaWorkspace::allocate
// Cấp phát toàn bộ GPU buffer theo kích thước ảnh đầu vào.
// Phải gọi DUY NHẤT MỘT LẦN trước warm-up run.
// Plan v4 C.3: "CudaWorkspace RAII, allocation trước warm-up, reuse."
// canvas_w = 2*w, canvas_h = h (ước tính conservative cho panorama A+B)
// ============================================================================
void CudaWorkspace::allocate(int w, int h) {
    img_w    = w;
    img_h    = h;
    canvas_w = 2 * w;   // Ước lượng safe cho Stage 6 (plan v4: "tối đa 4× mỗi chiều")
    canvas_h = h;

    const size_t N   = static_cast<size_t>(w) * h;        // MAX_PIXELS cho ảnh thực tế
    const size_t MAX_C = MAX_CANDIDATES;                   // Capacity candidates

    // ---- Input images ----
    CHECK_CUDA(cudaMalloc(&d_imgA,  N * sizeof(uchar4)));
    CHECK_CUDA(cudaMalloc(&d_imgB,  N * sizeof(uchar4)));

    // ---- Stage 1: Grayscale + Blur ----
    CHECK_CUDA(cudaMalloc(&d_grayA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_grayB, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_blurA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_blurB, N * sizeof(float)));

    // ---- Stage 2: Response maps ----
    CHECK_CUDA(cudaMalloc(&d_respA, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_respB, N * sizeof(float)));

    // ---- Stage 2: Keypoints sau Top-K ----
    CHECK_CUDA(cudaMalloc(&d_kpA,   MAX_KEYPOINTS * sizeof(Keypoint)));
    CHECK_CUDA(cudaMalloc(&d_kpB,   MAX_KEYPOINTS * sizeof(Keypoint)));
    CHECK_CUDA(cudaMalloc(&d_nkpA,  sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_nkpB,  sizeof(int)));

    // ---- Stage 2: Candidate buffers (trước sort) ----
    CHECK_CUDA(cudaMalloc(&d_candA, MAX_C * sizeof(Keypoint)));
    CHECK_CUDA(cudaMalloc(&d_candB, MAX_C * sizeof(Keypoint)));
    CHECK_CUDA(cudaMalloc(&d_ncandA, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ncandB, sizeof(int)));

    // ---- Sort keys/vals: nhân đôi để có in/out buffer cho CUB ----
    // Bố cục: [0 .. MAX_C-1] = input, [MAX_C .. 2*MAX_C-1] = output
    CHECK_CUDA(cudaMalloc(&d_sort_keysA, 2 * MAX_C * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_sort_keysB, 2 * MAX_C * sizeof(uint64_t)));
    CHECK_CUDA(cudaMalloc(&d_sort_valsA, 2 * MAX_C * sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc(&d_sort_valsB, 2 * MAX_C * sizeof(uint32_t)));

    // ---- CUB sort temp storage: ước lượng 32 MB, đủ cho MAX_CANDIDATES=~8M ----
    sort_tmp_bytes = 32ULL * 1024 * 1024;  // 32 MB
    CHECK_CUDA(cudaMalloc(&d_sort_tmp, sort_tmp_bytes));

    // ---- Stage 3: ORB Descriptors (Tiến phụ trách) ----
    CHECK_CUDA(cudaMalloc(&d_descA, MAX_DESCRIPTORS * sizeof(Descriptor)));
    CHECK_CUDA(cudaMalloc(&d_descB, MAX_DESCRIPTORS * sizeof(Descriptor)));

    // ---- Stage 4: Matches (Tiến phụ trách) ----
    CHECK_CUDA(cudaMalloc(&d_matches,  MAX_MATCHES * sizeof(Match)));
    CHECK_CUDA(cudaMalloc(&d_nmatches, sizeof(int)));

    // ---- Stage 5: RANSAC (Thông phụ trách - Plan v4 Mục 18.2) ----
    CHECK_CUDA(cudaMalloc(&d_sample_indices, MAX_RANSAC_HYPOTHESES * 4 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_inlier_counts,  MAX_RANSAC_HYPOTHESES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_all_H,          MAX_RANSAC_HYPOTHESES * 9 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_best_hyp,       sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_ransac_result,  sizeof(GpuRansacResult)));

    // ---- Stage 6: Panorama output (Tuấn phụ trách) ----
    const size_t CANVAS = static_cast<size_t>(canvas_w) * canvas_h;
    CHECK_CUDA(cudaMalloc(&d_pano,         CANVAS * sizeof(uchar4)));
    CHECK_CUDA(cudaMalloc(&d_canvas_bounds, 4 * sizeof(int)));

    std::cout << "[CudaWorkspace] Đã cấp phát GPU memory: "
              << img_w << "x" << img_h << " px, "
              << "canvas " << canvas_w << "x" << canvas_h << "\n";
}

// ============================================================================
// CudaWorkspace::reset_counters
// Reset tất cả atomic counter về 0 trước mỗi run (kể cả warm-up).
// Plan v4 C.3: "Atomic counts reset trước mỗi run."
// ============================================================================
void CudaWorkspace::reset_counters() {
    CHECK_CUDA(cudaMemset(d_nkpA,    0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_nkpB,    0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_ncandA,  0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_ncandB,  0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_nmatches, 0, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_best_hyp, 0, sizeof(int)));
}

// ============================================================================
// CudaWorkspace::free_all
// Giải phóng toàn bộ GPU memory — gọi từ destructor.
// Mọi pointer được null sau khi free để tránh double-free.
// ============================================================================
void CudaWorkspace::free_all() {
    auto safe_free = [](void*& ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    };
    safe_free(reinterpret_cast<void*&>(d_imgA));
    safe_free(reinterpret_cast<void*&>(d_imgB));
    safe_free(reinterpret_cast<void*&>(d_grayA));
    safe_free(reinterpret_cast<void*&>(d_grayB));
    safe_free(reinterpret_cast<void*&>(d_blurA));
    safe_free(reinterpret_cast<void*&>(d_blurB));
    safe_free(reinterpret_cast<void*&>(d_respA));
    safe_free(reinterpret_cast<void*&>(d_respB));
    safe_free(reinterpret_cast<void*&>(d_kpA));
    safe_free(reinterpret_cast<void*&>(d_kpB));
    safe_free(reinterpret_cast<void*&>(d_nkpA));
    safe_free(reinterpret_cast<void*&>(d_nkpB));
    safe_free(reinterpret_cast<void*&>(d_candA));
    safe_free(reinterpret_cast<void*&>(d_candB));
    safe_free(reinterpret_cast<void*&>(d_ncandA));
    safe_free(reinterpret_cast<void*&>(d_ncandB));
    safe_free(reinterpret_cast<void*&>(d_sort_keysA));
    safe_free(reinterpret_cast<void*&>(d_sort_keysB));
    safe_free(reinterpret_cast<void*&>(d_sort_valsA));
    safe_free(reinterpret_cast<void*&>(d_sort_valsB));
    safe_free(d_sort_tmp);
    safe_free(reinterpret_cast<void*&>(d_descA));
    safe_free(reinterpret_cast<void*&>(d_descB));
    safe_free(reinterpret_cast<void*&>(d_matches));
    safe_free(reinterpret_cast<void*&>(d_nmatches));
    safe_free(reinterpret_cast<void*&>(d_sample_indices));
    safe_free(reinterpret_cast<void*&>(d_inlier_counts));
    safe_free(reinterpret_cast<void*&>(d_all_H));
    safe_free(reinterpret_cast<void*&>(d_best_hyp));
    safe_free(reinterpret_cast<void*&>(d_ransac_result));
    safe_free(reinterpret_cast<void*&>(d_pano));
    safe_free(reinterpret_cast<void*&>(d_canvas_bounds));
    sort_tmp_bytes = 0;
    img_w = img_h = canvas_w = canvas_h = 0;
}

// ============================================================================
// Helper: tính median của vector float
// ============================================================================
static float compute_median(std::vector<float> v) {
    if (v.empty()) return 0.0f;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) * 0.5f : v[n/2];
}

// ============================================================================
// Helper: điền TimingEntry từ vector raw runs
// ============================================================================
static TimingEntry make_timing_entry(const std::vector<float>& runs) {
    TimingEntry e{};
    if (runs.empty()) return e;
    float sum = 0.0f;
    float mn  = runs[0];
    for (float v : runs) { sum += v; if (v < mn) mn = v; }
    e.avg    = sum / static_cast<float>(runs.size());
    e.median = compute_median(runs);
    e.min    = mn;
    return e;
}

// ============================================================================
// run_v1_basic_cuda — Điều phối viên V1 Pipeline
//
// Cấu trúc (Plan v4 C.3 + Mục 18.2):
//   1. CudaWorkspace::allocate(w, h) — cấp phát GPU memory
//   2. 1 lần warm-up đầy đủ (không tính vào benchmark)
//   3. `runs` lần đo đạc: reset_counters → H2D → S1 → S2 → S3(stub) →
//      S4(stub) → S5(stub) → S6(stub) → D2H
//   4. Tổng hợp Avg/Median/Min
//   5. Trả về PipelineResult
//
// STAGE 3-6: CPU STUB — TODO: thay bằng CUDA kernel khi Tiến/Tuấn triển khai.
// Stub sẽ: D2H copy kết quả Stage 2 → gọi hàm V0 CPU → H2D copy kết quả.
// ============================================================================
PipelineResult run_v1_basic_cuda(const Image& imgA, const Image& imgB, int runs) {
    PipelineResult result;
    result.version_str = "V1";

    const int w = imgA.w;
    const int h = imgA.h;
    const size_t N = static_cast<size_t>(w) * h;

    // ---- Cấp phát GPU workspace ----
    CudaWorkspace ws;
    ws.allocate(w, h);

    // ============================================================
    // Lambda: thực thi một lần đầy đủ pipeline và điền timing vào
    // các vector tương ứng. Dùng cho cả warm-up và benchmark runs.
    // ============================================================
    auto run_once = [&](
        std::vector<float>& t_h2d_runs,
        std::vector<float>& t_s1_runs,
        std::vector<float>& t_s2_runs,
        std::vector<float>& t_s3_runs,
        std::vector<float>& t_s4_runs,
        std::vector<float>& t_s5_runs,
        std::vector<float>& t_s6_runs,
        std::vector<float>& t_d2h_runs,
        std::vector<float>& t_e2e_runs,
        Image& out_pano,
        PipelineDiagnostics& diag,
        bool record
    ) -> bool {
        ws.reset_counters();

        CpuTimer e2e_timer;
        e2e_timer.Start();

        // ---- H2D: Copy ảnh lên GPU ----
        GpuTimer h2d_timer;
        h2d_timer.Start();
        CHECK_CUDA(cudaMemcpy(ws.d_imgA, imgA.data.data(), N * sizeof(uchar4), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ws.d_imgB, imgB.data.data(), N * sizeof(uchar4), cudaMemcpyHostToDevice));
        float t_h2d = h2d_timer.Stop();

        // ---- Stage 1: Grayscale + Gaussian Blur (GPU) ----
        float t_s1A = stage1_preprocess_gpu(ws.d_imgA, ws.d_grayA, ws.d_blurA, w, h);
        float t_s1B = stage1_preprocess_gpu(ws.d_imgB, ws.d_grayB, ws.d_blurB, w, h);
        float t_s1  = t_s1A + t_s1B;

        // ---- Stage 2: FAST-9 + NMS + Top-K (GPU) ----
        float t_s2A = stage2_fast_nms_topk_gpu(
            ws.d_blurA, ws.d_respA,
            ws.d_candA, ws.d_ncandA,
            ws.d_kpA,   ws.d_nkpA,
            ws.d_sort_keysA, ws.d_sort_valsA,
            ws.d_sort_tmp, ws.sort_tmp_bytes,
            w, h, DEFAULT_TOP_K);
        float t_s2B = stage2_fast_nms_topk_gpu(
            ws.d_blurB, ws.d_respB,
            ws.d_candB, ws.d_ncandB,
            ws.d_kpB,   ws.d_nkpB,
            ws.d_sort_keysB, ws.d_sort_valsB,
            ws.d_sort_tmp, ws.sort_tmp_bytes,
            w, h, DEFAULT_TOP_K);
        float t_s2 = t_s2A + t_s2B;

        // ---- D2H: Lấy kết quả Stage 2 về host (cho CPU stubs S3-S6) ----
        int nkpA = 0, nkpB = 0;
        CHECK_CUDA(cudaMemcpy(&nkpA, ws.d_nkpA, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&nkpB, ws.d_nkpB, sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<Keypoint> h_kpA(nkpA), h_kpB(nkpB);
        if (nkpA > 0) CHECK_CUDA(cudaMemcpy(h_kpA.data(), ws.d_kpA, nkpA * sizeof(Keypoint), cudaMemcpyDeviceToHost));
        if (nkpB > 0) CHECK_CUDA(cudaMemcpy(h_kpB.data(), ws.d_kpB, nkpB * sizeof(Keypoint), cudaMemcpyDeviceToHost));

        // Lấy blur images về host (cho Stage 3 ORB descriptor CPU stub)
        GrayImage blurA_host(w, h);
        GrayImage blurB_host(w, h);
        CHECK_CUDA(cudaMemcpy(blurA_host.data, ws.d_blurA, N * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(blurB_host.data, ws.d_blurB, N * sizeof(float), cudaMemcpyDeviceToHost));

        // ================================================================
        // Stage 3: ORB Descriptor — CPU STUB
        // TODO: Thay bằng stage3_orb_descriptors_gpu() khi Tiến triển khai
        // ================================================================
        float t_s3 = 0.0f;
        std::vector<Descriptor> descA, descB;
        {
            CpuTimer s3_timer;
            s3_timer.Start();
            // [STUB] Gọi V0 CPU với blur image từ GPU
            descA = stage3_orb_descriptors_cpu(blurA_host, h_kpA, t_s3);
            float dummy3b = 0.0f;
            descB = stage3_orb_descriptors_cpu(blurB_host, h_kpB, dummy3b);
            t_s3  = s3_timer.Stop();
            // TODO(Tiến): Thay bằng GPU kernel, đo bằng cudaEvent
            //   stage3_orb_descriptors_gpu(ws.d_blurA, ws.d_kpA, nkpA, ws.d_descA, ...);
            //   stage3_orb_descriptors_gpu(ws.d_blurB, ws.d_kpB, nkpB, ws.d_descB, ...);
        }

        // ================================================================
        // Stage 4: Hamming Matching — CPU STUB
        // TODO: Thay bằng stage4_matching_gpu() khi Tiến triển khai
        // ================================================================
        float t_s4 = 0.0f;
        std::vector<Match> matches;
        {
            // [STUB] Gọi V0 CPU matching
            matches = stage4_matching_cpu(descA, descB, t_s4);
            // TODO(Tiến): Thay bằng GPU kernel, đo bằng cudaEvent
            //   stage4_matching_gpu(ws.d_descA, nkpA, ws.d_descB, nkpB,
            //                       ws.d_matches, ws.d_nmatches, ...);
        }

        // ================================================================
        // Stage 5: RANSAC Homography — GPU (Thông phụ trách - Plan v4 Mục 18.2)
        // ================================================================
        float t_s5 = 0.0f;
        Homography best_H{};
        int inliers = 0;
        bool homo_ok = false;
        {
            int n_matches = static_cast<int>(matches.size());
            if (n_matches > 0) {
                CHECK_CUDA(cudaMemcpy(ws.d_matches, matches.data(), n_matches * sizeof(Match), cudaMemcpyHostToDevice));
            }
            CHECK_CUDA(cudaMemcpy(ws.d_nmatches, &n_matches, sizeof(int), cudaMemcpyHostToDevice));

            // Gọi GPU kernel Stage 5
            homo_ok = stage5_ransac_gpu(
                ws.d_matches, n_matches, ws.d_kpA, ws.d_kpB,
                ws, w, h, best_H, inliers, t_s5);
        }

        // ================================================================
        // Stage 6: Warp + Blend — CPU STUB
        // TODO: Thay bằng stage6_warp_blend_gpu() khi Tuấn triển khai
        // ================================================================
        float t_s6 = 0.0f;
        bool warp_ok = false;
        {
            // [STUB] Gọi V0 CPU warp+blend
            out_pano = stage6_warp_blend_cpu(imgA, imgB, best_H, warp_ok, t_s6);
            // TODO(Tuấn): Thay bằng GPU kernel, đo bằng cudaEvent
            //   stage6_warp_blend_gpu(ws.d_imgA, ws.d_imgB, best_H,
            //                         ws.d_pano, ws.d_canvas_bounds, ...);
        }

        // ---- D2H: Copy panorama về host (stage 6 stub chạy trên CPU nên không cần D2H) ----
        // Khi Stage 6 GPU hoạt động: CHECK_CUDA(cudaMemcpy(out_pano.data.data(), ws.d_pano, ...));
        GpuTimer d2h_timer;
        d2h_timer.Start();
        // [Hiện tại không có D2H thực sự vì S6 là CPU stub]
        // Khi GPU: cudaMemcpy panorama D2H ở đây
        float t_d2h = d2h_timer.Stop();

        float t_e2e = e2e_timer.Stop();

        // Điền diagnostics
        diag.keypoints_a  = nkpA;
        diag.keypoints_b  = nkpB;
        diag.matches      = static_cast<int>(matches.size());
        diag.inliers      = inliers;
        diag.inlier_ratio = (matches.size() > 0)
                            ? static_cast<float>(inliers) / matches.size()
                            : 0.0f;
        diag.H = best_H;
        if (!homo_ok || !warp_ok) {
            diag.status  = PipelineStatus::FAILURE;
            diag.warning = "RANSAC hoặc warp thất bại";
        } else if (inliers < 4) {
            diag.status  = PipelineStatus::FAILURE;
            diag.warning = "Inliers < 4: H suy biến";
        } else {
            diag.status  = PipelineStatus::SUCCESS;
            diag.warning = "";
        }

        if (record) {
            t_h2d_runs.push_back(t_h2d);
            t_s1_runs .push_back(t_s1);
            t_s2_runs .push_back(t_s2);
            t_s3_runs .push_back(t_s3);
            t_s4_runs .push_back(t_s4);
            t_s5_runs .push_back(t_s5);
            t_s6_runs .push_back(t_s6);
            t_d2h_runs.push_back(t_d2h);
            t_e2e_runs.push_back(t_e2e);
        }
        return (diag.status == PipelineStatus::SUCCESS);
    };

    // ---- Vectơ lưu raw timing ----
    std::vector<float> t_h2d, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6, t_d2h, t_e2e;

    // ---- Warm-up: 1 lần đầy đủ, không ghi timing ----
    PipelineDiagnostics warmup_diag{};
    Image warmup_pano;
    std::cout << "[V1] Warm-up run...\n";
    run_once(t_h2d, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6, t_d2h, t_e2e,
             warmup_pano, warmup_diag, /*record=*/false);

    // ---- Benchmark runs ----
    std::cout << "[V1] Bắt đầu " << runs << " benchmark run(s)...\n";
    PipelineDiagnostics last_diag{};
    Image last_pano;
    for (int r = 0; r < runs; r++) {
        bool ok = run_once(t_h2d, t_s1, t_s2, t_s3, t_s4, t_s5, t_s6, t_d2h, t_e2e,
                           last_pano, last_diag, /*record=*/true);
        std::printf("[V1] Run %d/%d — S1: %.2f ms, S2: %.2f ms, E2E: %.2f ms%s\n",
                    r + 1, runs,
                    t_s1.back(), t_s2.back(), t_e2e.back(),
                    ok ? "" : " [CẢNH BÁO: pipeline thất bại]");
    }

    // ---- Tổng hợp timing ----
    result.diagnostics = last_diag;
    result.panorama    = std::move(last_pano);

    // H2D + D2H: lưu vào kernel_total thay vì các stage
    // Plan v4: "H2D/D2H measured separately and reported in JSON"
    TimingEntry h2d_entry  = make_timing_entry(t_h2d);
    TimingEntry d2h_entry  = make_timing_entry(t_d2h);

    result.timings.stage1_preprocess      = make_timing_entry(t_s1);
    result.timings.stage2_fast_nms_topk   = make_timing_entry(t_s2);
    result.timings.stage3_orb_descriptor  = make_timing_entry(t_s3);
    result.timings.stage4_matching        = make_timing_entry(t_s4);
    result.timings.stage5_ransac          = make_timing_entry(t_s5);
    result.timings.stage6_warping_blending = make_timing_entry(t_s6);

    // kernel_total = tổng S1-S6 (không bao H2D/D2H)
    std::vector<float> t_kernel(runs);
    for (int r = 0; r < runs; r++) {
        t_kernel[r] = t_s1[r] + t_s2[r] + t_s3[r] + t_s4[r] + t_s5[r] + t_s6[r];
    }
    result.timings.kernel_total = make_timing_entry(t_kernel);
    result.timings.end_to_end   = make_timing_entry(t_e2e);

    // Lưu H2D/D2H vào diagnostics warning string nếu cần (hoặc JSON extension)
    std::printf("[V1] H2D avg: %.3f ms | D2H avg: %.3f ms\n",
                h2d_entry.avg, d2h_entry.avg);

    // Amdahl analysis (Plan v4 Mục 4.4): S1+S2+S6 là phần song song hóa
    if (result.timings.end_to_end.avg > 0.0f) {
        float par = result.timings.stage1_preprocess.avg
                  + result.timings.stage2_fast_nms_topk.avg
                  + result.timings.stage6_warping_blending.avg;
        result.amdahl_parallel_fraction = par / result.timings.end_to_end.avg;
        float seq_frac = 1.0f - result.amdahl_parallel_fraction;
        result.amdahl_max_speedup = (seq_frac > 1e-6f) ? (1.0f / seq_frac) : 999.0f;
    }

    std::cout << "[V1] Hoàn thành. Keypoints A=" << last_diag.keypoints_a
              << " B=" << last_diag.keypoints_b
              << " Matches=" << last_diag.matches
              << " Inliers=" << last_diag.inliers << "\n";

    return result;
}

