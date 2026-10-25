// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama (Panorama Image Stitching)
// Module: Điểm vào CLI chính (Kế hoạch v4 Mục 4.1)
// Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
//          Dao Quoc Tuan (23120392)
// External sources: Tài liệu CUDA Runtime API; ghi nhận bản quyền
// stb/OpenCV tại THIRD_PARTY_NOTICES.md. Mã CLI và telemetry là tự phát triển.
//
// Mã thoát (Plan v4 Mục 4.1):
//   0: Thành công
//   2: Lỗi CLI / Tham số đầu vào
//   3: Lỗi nghiêm trọng pipeline (inliers < 4, H suy biến, canvas không hợp lệ)
//   4: Lỗi ghi ảnh kết quả / số liệu JSON
//   5: Lỗi CUDA runtime
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

// Các hằng số mã thoát
constexpr int EXIT_CLI_ERROR = 2;
constexpr int EXIT_PIPELINE_ERROR = 3;
constexpr int EXIT_IO_ERROR = 4;
constexpr int EXIT_CUDA_ERR = 5;

void print_usage(const char* prog) {
    std::cout << "Hướng dẫn sử dụng: " << prog << " <anh_trai> <anh_phai> <anh_dau_ra> --version <0|1|2|3|4|5> [--runs N] [--metrics-json <duong_dan>]\n";
    std::cout << "Tùy chọn:\n";
    std::cout << "  --version <0..5>        Phiên bản thuật toán (0 = Tuần tự C++, 1 = CUDA cơ bản, ...)\n";
    std::cout << "  --runs <N>              Số lần lặp đo đạc benchmark (mặc định: 5)\n";
    std::cout << "  --metrics-json <path>   Đường dẫn xuất số liệu JSON (Plan v4 Mục 4.3)\n";
}

int main(int argc, char** argv) {
    if (argc >= 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    if (argc < 6) {
        print_usage(argv[0]);
        return EXIT_CLI_ERROR;
    }

    std::string left_path  = argv[1];
    std::string right_path = argv[2];
    std::string out_path   = argv[3];

    int version = -1;
    int runs = 5; // Mặc định là 5 theo Plan v4 Mục 4.1
    std::string metrics_json_path = "";

    for (int i = 4; i < argc; i++) {
        if (std::strcmp(argv[i], "--version") == 0 && i + 1 < argc) {
            version = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--metrics-json") == 0 && i + 1 < argc) {
            metrics_json_path = argv[++i];
        } else {
            std::cerr << "[Lỗi] Tham số dòng lệnh không nhận diện được: " << argv[i] << "\n";
            print_usage(argv[0]);
            return EXIT_CLI_ERROR;
        }
    }

    if (version < 0) {
        std::cerr << "[Lỗi] Thiếu hoặc không hợp lệ tham số --version.\n";
        print_usage(argv[0]);
        return EXIT_CLI_ERROR;
    }

    if (runs <= 0) {
        std::cerr << "[Lỗi] --runs phải là số nguyên dương, nhận được: " << runs << "\n";
        return EXIT_CLI_ERROR;
    }

    // 1. Tải ảnh đầu vào (dùng RAII move-only Image)
    Image imgA, imgB;
    if (!load_image(left_path, imgA)) {
        std::cerr << "[Lỗi] Không thể tải ảnh trái: " << left_path << "\n";
        return EXIT_CLI_ERROR;
    }
    if (!load_image(right_path, imgB)) {
        std::cerr << "[Lỗi] Không thể tải ảnh phải: " << right_path << "\n";
        return EXIT_CLI_ERROR;
    }

    // 2. Truy vấn thông tin thiết bị thực thi
    std::string device_name = "Host CPU (Không phát hiện CUDA runtime)";
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
    std::cout << "CSC14120 Ghép ảnh Panorama | Quy trình Plan v4\n";
    std::cout << "Thiết bị:    " << device_name << "\n";
    std::cout << "Phiên bản:   V" << version << " | Số lần chạy: " << runs << "\n";
    std::cout << "Ảnh trái:    " << left_path  << " (" << imgA.w << "x" << imgA.h << ")\n";
    std::cout << "Ảnh phải:    " << right_path << " (" << imgB.w << "x" << imgB.h << ")\n";
    std::cout << "Đầu ra:      " << out_path   << "\n";
    if (!metrics_json_path.empty()) {
        std::cout << "Số liệu JSON: " << metrics_json_path << "\n";
    }
    std::cout << "=========================================================\n";

    // 3. Thực thi Pipeline theo phiên bản
    PipelineResult result;

    if (version == 0) {
        result = run_v0_pipeline(imgA, imgB, runs);
        result.device_name = device_name;
    } else {
        std::cerr << "[Lỗi] Phiên bản V" << version 
                  << " chưa được triển khai. Chỉ V0 (Tuần tự C++) được hoàn thiện ở Phase A/B.\n";
        return EXIT_CLI_ERROR;
    }

    // 4. In bảng thời gian thực thi chi tiết
    std::cout << "\n---------------------------------------------------------\n";
    std::cout << "Phân rã thời gian từng giai đoạn (ms) [Số lần: " << runs << ", sau 1 lần warm-up]:\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Giai đoạn 1 (Tiền xử lý):      TB: " << std::setw(8) << result.timings.stage1_preprocess.avg 
              << " ms | Trung vị: " << std::setw(8) << result.timings.stage1_preprocess.median
              << " ms | Tối thiểu: " << std::setw(8) << result.timings.stage1_preprocess.min << " ms\n";
    std::cout << "  Giai đoạn 2 (FAST+NMS+TopK):   TB: " << std::setw(8) << result.timings.stage2_fast_nms_topk.avg 
              << " ms | Trung vị: " << std::setw(8) << result.timings.stage2_fast_nms_topk.median
              << " ms | Tối thiểu: " << std::setw(8) << result.timings.stage2_fast_nms_topk.min << " ms\n";
    std::cout << "  Giai đoạn 3 (Mô tả ORB):       TB: " << std::setw(8) << result.timings.stage3_orb_descriptor.avg 
              << " ms | Trung vị: " << std::setw(8) << result.timings.stage3_orb_descriptor.median
              << " ms | Tối thiểu: " << std::setw(8) << result.timings.stage3_orb_descriptor.min << " ms\n";
    std::cout << "  Giai đoạn 4 (Khớp Hamming):    TB: " << std::setw(8) << result.timings.stage4_matching.avg 
              << " ms | Trung vị: " << std::setw(8) << result.timings.stage4_matching.median
              << " ms | Tối thiểu: " << std::setw(8) << result.timings.stage4_matching.min << " ms\n";
    std::cout << "  Giai đoạn 5 (RANSAC DLT):      TB: " << std::setw(8) << result.timings.stage5_ransac.avg 
              << " ms | Trung vị: " << std::setw(8) << result.timings.stage5_ransac.median
              << " ms | Tối thiểu: " << std::setw(8) << result.timings.stage5_ransac.min << " ms\n";
    std::cout << "  Giai đoạn 6 (Chiếu & Trộn):    TB: " << std::setw(8) << result.timings.stage6_warping_blending.avg 
              << " ms | Trung vị: " << std::setw(8) << result.timings.stage6_warping_blending.median
              << " ms | Tối thiểu: " << std::setw(8) << result.timings.stage6_warping_blending.min << " ms\n";
    std::cout << "---------------------------------------------------------\n";
    std::cout << "  Tổng thời gian các Stage:      TB: " << std::setw(8) << result.timings.kernel_total.avg 
              << " ms | Trung vị: " << std::setw(8) << result.timings.kernel_total.median
              << " ms | Tối thiểu: " << std::setw(8) << result.timings.kernel_total.min << " ms\n";
    std::cout << "  Toàn quy trình End-to-End:     TB: " << std::setw(8) << result.timings.end_to_end.avg 
              << " ms | Trung vị: " << std::setw(8) << result.timings.end_to_end.median
              << " ms | Tối thiểu: " << std::setw(8) << result.timings.end_to_end.min << " ms\n";
    std::cout << "---------------------------------------------------------\n";

    std::cout << "\nThông số chẩn đoán (Diagnostics):\n";
    std::cout << "  Điểm đặc trưng A: " << result.diagnostics.keypoints_a 
              << " | Điểm đặc trưng B: " << result.diagnostics.keypoints_b << "\n";
    std::cout << "  Số cặp khớp:      " << result.diagnostics.matches 
              << " | Số inliers: " << result.diagnostics.inliers 
              << " (" << std::setprecision(2) << result.diagnostics.inlier_ratio * 100.0f << "%)\n";
    std::cout << "  Trạng thái:       " << status_to_string(result.diagnostics.status) << "\n";
    std::cout << "  Cảnh báo:         " << result.diagnostics.warning << "\n";
    std::cout << "  Ma trận Homography H_A_to_B:\n";
    for (int r = 0; r < 3; r++) {
        std::cout << "    [";
        for (int c = 0; c < 3; c++) {
            std::cout << std::setw(11) << std::setprecision(5) << result.diagnostics.H.h[r * 3 + c] << (c < 2 ? ", " : "");
        }
        std::cout << "]\n";
    }
    std::cout << "=========================================================\n";

    // 5. Phân tích Tăng tốc Định luật Amdahl cho V0 (Plan v4 Mục 4.4 & A4)
    if (version == 0) {
        std::cout << "\n[Phân tích Định luật Amdahl cho V0]\n";
        std::cout << "Mô hình: các giai đoạn hot được chọn S1 + S2 + S6 được tăng tốc; "
                     "S3-S5 và chi phí host giữ tuần tự.\n";
        std::cout << "Tỷ lệ song song hóa p = " << std::fixed << std::setprecision(4)
                  << result.amdahl_parallel_fraction * 100.0f << "%\n";
        std::cout << "Tăng tốc tối đa theo lý thuyết (S_max): " << std::setprecision(2)
                  << result.amdahl_max_speedup << "x\n";
        std::cout << "Ghi chú: Đây là giới hạn trên; chi phí truyền dữ liệu, "
                     "khởi chạy kernel và đồng bộ sẽ làm giảm tốc độ thực tế.\n";
        std::cout << "=========================================================\n";
    }

    // 6. Xuất số liệu telemetry JSON nếu được yêu cầu
    if (!metrics_json_path.empty()) {
        if (!result.export_metrics_json(metrics_json_path)) {
            std::cerr << "[Lỗi] Không thể ghi số liệu JSON ra: " << metrics_json_path << "\n";
            return EXIT_IO_ERROR;
        }
        std::cout << "[OK] Đã xuất số liệu JSON tới: " << metrics_json_path << "\n";
    }

    // 7. Kiểm tra lỗi nghiêm trọng của pipeline
    if (result.diagnostics.status == PipelineStatus::FAILURE || result.panorama.empty()) {
        std::cerr << "[Lỗi] Thực thi pipeline thất bại: " << result.diagnostics.warning << "\n";
        return EXIT_PIPELINE_ERROR;
    }

    // 8. Lưu ảnh panorama kết quả
    if (!save_image(out_path, result.panorama)) {
        std::cerr << "[Lỗi] Không thể lưu ảnh panorama ra: " << out_path << "\n";
        return EXIT_IO_ERROR;
    }
    std::cout << "[OK] Đã lưu ảnh panorama thành công tại: " << out_path 
              << " (" << result.panorama.w << "x" << result.panorama.h << ")\n";

    return 0;
}

