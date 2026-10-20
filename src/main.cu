// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Main CLI Entrypoint (Plan v4 Section 4.1)
// Authors: GroupXX (Members A, B, C)
//
// Exit Codes (Plan v4 Section 4.1):
//   0: Success
//   2: CLI / Input parameter error
//   3: Pipeline hard failure (inliers < 4, singular H, invalid canvas)
//   4: Output image / metrics write error
//   5: CUDA runtime error
// ============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <cmath>

#include "common.h"
#include "cuda_utils.cuh"
#include "stb_impl.h"
#include "versions.h"

// Exit code constants
constexpr int EXIT_CLI_ERROR = 2;
constexpr int EXIT_PIPELINE_ERROR = 3;
constexpr int EXIT_IO_ERROR = 4;
constexpr int EXIT_CUDA_ERR = 5;

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <left_img> <right_img> <output_img> --version <0|1|2|3|4|5> [--runs N] [--metrics-json <path>]\n";
    std::cout << "Options:\n";
    std::cout << "  --version <0..5>        Pipeline version (0 = Sequential C++, 1 = Basic CUDA, ...)\n";
    std::cout << "  --runs <N>              Number of benchmark iterations (default: 5)\n";
    std::cout << "  --metrics-json <path>   Optional path to export JSON metrics (Plan v4 Section 4.3)\n";
}

int main(int argc, char** argv) {
    if (argc < 6) {
        print_usage(argv[0]);
        return EXIT_CLI_ERROR;
    }

    std::string left_path  = argv[1];
    std::string right_path = argv[2];
    std::string out_path   = argv[3];

    int version = -1;
    int runs = 5; // Default is 5 per Plan v4 Section 4.1
    std::string metrics_json_path = "";

    for (int i = 4; i < argc; i++) {
        if (std::strcmp(argv[i], "--version") == 0 && i + 1 < argc) {
            version = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--metrics-json") == 0 && i + 1 < argc) {
            metrics_json_path = argv[++i];
        } else {
            std::cerr << "[Error] Unknown or malformed CLI argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return EXIT_CLI_ERROR;
        }
    }

    if (version < 0) {
        std::cerr << "[Error] Missing or invalid --version argument.\n";
        print_usage(argv[0]);
        return EXIT_CLI_ERROR;
    }

    if (runs <= 0) {
        std::cerr << "[Error] --runs must be a positive integer, got: " << runs << "\n";
        return EXIT_CLI_ERROR;
    }

    // 1. Load input images (using RAII move-only Image)
    Image imgA, imgB;
    if (!load_image(left_path, imgA)) {
        std::cerr << "[Error] Failed to load left image: " << left_path << "\n";
        return EXIT_CLI_ERROR;
    }
    if (!load_image(right_path, imgB)) {
        std::cerr << "[Error] Failed to load right image: " << right_path << "\n";
        return EXIT_CLI_ERROR;
    }

    // 2. Query execution device info
    std::string device_name = "Host CPU (No CUDA runtime detected)";
#if defined(__CUDACC__) || defined(__CUDA_RUNTIME_H__)
    int device_count = 0;
    cudaError_t cu_err = cudaGetDeviceCount(&device_count);
    if (cu_err == cudaSuccess && device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        device_name = prop.name;
    }
#endif

    std::cout << "=========================================================\n";
    std::cout << "CSC14120 Panorama Stitching | Plan v4 Pipeline\n";
    std::cout << "Device:  " << device_name << "\n";
    std::cout << "Version: V" << version << " | Runs: " << runs << "\n";
    std::cout << "Left:    " << left_path  << " (" << imgA.w << "x" << imgA.h << ")\n";
    std::cout << "Right:   " << right_path << " (" << imgB.w << "x" << imgB.h << ")\n";
    std::cout << "Output:  " << out_path   << "\n";
    if (!metrics_json_path.empty()) {
        std::cout << "Metrics: " << metrics_json_path << "\n";
    }
    std::cout << "=========================================================\n";

    // 3. Execute Pipeline by Version
    PipelineResult result;

    if (version == 0) {
        result = run_v0_pipeline(imgA, imgB, runs);
        result.device_name = device_name;
    } else {
        std::cerr << "[Error] Version V" << version 
                  << " is not yet implemented. Only V0 (Sequential C++) is implemented in Phase A/B.\n";
        return EXIT_CLI_ERROR;
    }

    // 4. Output detailed execution timing table
    std::cout << "\n---------------------------------------------------------\n";
    std::cout << "Stage Timing Breakdown (ms) [Runs: " << runs << "]:\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Stage 1 (Preprocess):        Avg: " << std::setw(8) << result.timings.stage1_preprocess.avg 
              << " ms | Min: " << std::setw(8) << result.timings.stage1_preprocess.min << " ms\n";
    std::cout << "  Stage 2 (FAST+NMS+TopK):     Avg: " << std::setw(8) << result.timings.stage2_fast_nms_topk.avg 
              << " ms | Min: " << std::setw(8) << result.timings.stage2_fast_nms_topk.min << " ms\n";
    std::cout << "  Stage 3 (ORB Descriptor):    Avg: " << std::setw(8) << result.timings.stage3_orb_descriptor.avg 
              << " ms | Min: " << std::setw(8) << result.timings.stage3_orb_descriptor.min << " ms\n";
    std::cout << "  Stage 4 (Hamming Match):     Avg: " << std::setw(8) << result.timings.stage4_matching.avg 
              << " ms | Min: " << std::setw(8) << result.timings.stage4_matching.min << " ms\n";
    std::cout << "  Stage 5 (RANSAC Homography): Avg: " << std::setw(8) << result.timings.stage5_ransac.avg 
              << " ms | Min: " << std::setw(8) << result.timings.stage5_ransac.min << " ms\n";
    std::cout << "  Stage 6 (Warp & Blend):      Avg: " << std::setw(8) << result.timings.stage6_warping_blending.avg 
              << " ms | Min: " << std::setw(8) << result.timings.stage6_warping_blending.min << " ms\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "  Kernel/Stages Total:         Avg: " << std::setw(8) << result.timings.kernel_total.avg 
              << " ms | Min: " << std::setw(8) << result.timings.kernel_total.min << " ms\n";
    std::cout << "  End-to-End Pipeline:         Avg: " << std::setw(8) << result.timings.end_to_end.avg 
              << " ms | Min: " << std::setw(8) << result.timings.end_to_end.min << " ms\n";
    std::cout << "---------------------------------------------------------\n";

    std::cout << "\nDiagnostics:\n";
    std::cout << "  Keypoints A: " << result.diagnostics.keypoints_a 
              << " | Keypoints B: " << result.diagnostics.keypoints_b << "\n";
    std::cout << "  Matches:     " << result.diagnostics.matches 
              << " | Inliers: " << result.diagnostics.inliers 
              << " (" << std::setprecision(2) << result.diagnostics.inlier_ratio * 100.0f << "%)\n";
    std::cout << "  Status:      " << status_to_string(result.diagnostics.status) << "\n";
    std::cout << "  Warning:     " << result.diagnostics.warning << "\n";
    std::cout << "  Homography H_A_to_B:\n";
    for (int r = 0; r < 3; r++) {
        std::cout << "    [";
        for (int c = 0; c < 3; c++) {
            std::cout << std::setw(11) << std::setprecision(5) << result.diagnostics.H.h[r * 3 + c] << (c < 2 ? ", " : "");
        }
        std::cout << "]\n";
    }
    std::cout << "=========================================================\n";

    // 5. Amdahl's Law Speedup Analysis for V0 (Plan v4 Section 4.4 & A4)
    if (version == 0) {
        float parallel_time = result.timings.stage1_preprocess.avg + 
                              result.timings.stage2_fast_nms_topk.avg + 
                              result.timings.stage3_orb_descriptor.avg + 
                              result.timings.stage4_matching.avg + 
                              result.timings.stage5_ransac.avg + 
                              result.timings.stage6_warping_blending.avg;
        float e2e_time = result.timings.end_to_end.avg;
        float p = (e2e_time > 0.0f) ? (parallel_time / e2e_time) : 0.0f;
        if (p > 1.0f) p = 1.0f;

        std::cout << "\n[Amdahl's Law Analysis for V0]\n";
        std::cout << "Parallel fraction p = " << std::fixed << std::setprecision(4) << p * 100.0f << "%\n";
        if (p >= 0.999f) {
            std::cout << "Theoretical Max Speedup (S_max): idealized unbounded / below measurement resolution\n";
            std::cout << "Note: In real execution, CUDA memory transfers (H2D/D2H) and kernel launch overheads will limit achievable speedup.\n";
        } else {
            float s_max = 1.0f / (1.0f - p);
            std::cout << "Theoretical Max Speedup (S_max): " << std::setprecision(2) << s_max << "x\n";
        }
        std::cout << "=========================================================\n";
    }

    // 6. Export telemetry metrics JSON if requested
    if (!metrics_json_path.empty()) {
        if (!result.export_metrics_json(metrics_json_path)) {
            std::cerr << "[Error] Failed to write metrics JSON to: " << metrics_json_path << "\n";
            return EXIT_IO_ERROR;
        }
        std::cout << "[OK] Telemetry metrics JSON exported to: " << metrics_json_path << "\n";
    }

    // 7. Check for pipeline hard failure
    if (result.diagnostics.status == PipelineStatus::FAILURE || result.panorama.empty()) {
        std::cerr << "[Error] Pipeline execution failed: " << result.diagnostics.warning << "\n";
        return EXIT_PIPELINE_ERROR;
    }

    // 8. Save panorama output image
    if (!save_image(out_path, result.panorama)) {
        std::cerr << "[Error] Failed to save panorama output image to: " << out_path << "\n";
        return EXIT_IO_ERROR;
    }
    std::cout << "[OK] Panorama successfully saved to: " << out_path 
              << " (" << result.panorama.w << "x" << result.panorama.h << ")\n";

    return 0;
}
