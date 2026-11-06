# Hướng dẫn cộng tác cho repository Group 19

## Nguồn quyết định

- Đọc `project_plan_v4.md` trước khi nhận việc; đây là kế hoạch duy nhất.
- Đề `CSC14120_2026_Final Project.pdf` và thông báo giảng viên là căn cứ requirements.
- Mục 16–24 của plan là hướng dẫn Final 20 ngày, 05/11–24/11/2026 (UTC+7).
- Midterm đã nộp; bảo toàn bản nộp. Viết Final bằng artifact mới.
- Đây là quy tắc cho nhiệm vụ được giao, không tự cấp quyền triển khai mọi task,
  đổi thuật toán, push/publish/nộp bài hoặc liên lạc bên ngoài.

## Khi bắt đầu task

1. Đọc Mục 3–4, phần task ở Mục 18, tiêu chí Mục 20–22 và prerequisite.
2. Xem `git status`/diff; không ghi đè thay đổi của người khác.
3. Xác định task ID, owner/reviewer, input SHA/data/config, file scope và output.
4. Kiểm công cụ/môi trường có thực. Có compiler CPU không có nghĩa có GPU/NVCC.
5. Nếu chỉ được yêu cầu viết/review hướng dẫn, chỉ sửa tài liệu; không tự triển khai CUDA.

## Ranh giới trách nhiệm

- Thông: S1/S5, contract, workspace, dispatcher, Makefile/build và tích hợp.
- Tiến: S2/S3, manifest/preprocessing, dataset audit và gallery.
- Tuấn: S4/S6, validator/schema, benchmark/profile, Final notebook/package/video.
- Đổi API/file chung phải phối hợp owner và cập nhật caller/callee/test.
- Giữ ownership/layout/count/capacity/stream/failure contract của plan.

## Bất biến kỹ thuật

- V0 single-thread; mọi version cùng methods/parameters/dataset.
- K=2000, FAST T=20, ratio=0.75, RANSAC=2000, seed LCG=42, border=20;
  ORB pattern, rounding/order/refit/geometry/translation fallback theo V0 đã khóa.
- H_A_to_B ánh xạ A→B; output panorama trong frame A.
- Cùng seed không đồng nghĩa cùng sample sequence: làm theo Mục 18.2.
- V1 đủ sáu stage trên GPU, kể cả refit/fallback; không dùng CPU solver hoặc
  OpenCV thay thuật toán trong đường chạy CUDA.
- V2 dựa V1, V3 dựa V2; mỗi bản một tối ưu chính có parent profiling evidence.
- Không đổi threshold/tolerance hoặc bỏ test để khiến một phiên bản vượt gate.
- Không dựng timing/ảnh/model status. UNKNOWN/NOT_RUN phải ghi đúng.

## Kiểm tra và bàn giao

- Dùng checklist Mục 18, 20 và gate tương ứng Mục 21 cho cả người lẫn agent.
- CPU fixture/test không thay cho CUDA execution/sanitizer.
- Khi thiếu GPU/tool/dataset, tiếp tục phần độc lập và ghi rõ phần NOT_RUN/BLOCKED.
- Bàn giao files/API đổi, command/exit code, actual vs expected, log/artifact paths,
  hashes, limitation và task cần reviewer làm; không tự đánh dấu reviewer đã duyệt.
- Không chạy benchmark chính dưới profiler/sanitizer, không trộn session hoặc
  timing policy; Average bắt buộc, lưu raw runs.
- Nguồn/code tham khảo phải cite tại nơi dùng và References; header tác giả/nguồn
  ở file viết/sửa. AI usage declaration phải đúng thực tế.
- Comment/report tiếng Việt có dấu; identifiers và git commit tiếng Anh.

## Lệnh hiện có và lệnh dự kiến

- Hiện có: `make cpu`, `make test` (Linux shell), `python tests/test_scripts.py`,
  `scripts/validate_metrics.py`, preprocessing/synthetic/Midterm scripts.
- `make cuda` hiện chưa chứng minh V1 đã được triển khai; kiểm dispatcher/source.
- `benchmark_final.py`, `validate_cpu_gpu.py`, `build_final_notebook.py`,
  `package_final.py` và các kernel CUDA là task dự kiến cho tới khi có source/test.
- Không báo chạy thành công một file/lệnh chỉ mới được mô tả trong plan.
