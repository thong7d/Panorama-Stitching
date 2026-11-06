# PROJECT PLAN v4 — CUDA-Accelerated Panorama Image Stitching

> **Trạng thái:** Nguồn kế hoạch duy nhất; hướng dẫn Final cập nhật ngày 05/11/2026
> **Môn học:** CSC14120 — Parallel Programming  
> **Chủ đề:** Panorama Image Stitching với FAST-9, Oriented BRIEF/ORB, Hamming Matching, RANSAC Homography và CUDA  
> **Phạm vi:** Midterm, V0 Sequential, V1 Basic CUDA, ít nhất hai phiên bản tối ưu, Final Report và Oral Defense

File này hợp nhất các nội dung còn giá trị của plan v3 với kết quả audit kỹ thuật trong v4. Khi có mâu thuẫn với tài liệu cũ, **quyết định trong file này được ưu tiên**. Không triển khai theo các version/tối ưu cũ nếu chưa vượt qua quality gate tương ứng.

**Kế hoạch 20 ngày:** D01 = 05/11/2026, D20 = 24/11/2026, múi giờ Asia/Bangkok (UTC+7). Mục tiêu hoàn tất kỹ thuật D18; D19–D20 là dự phòng và bàn giao. Đây là deadline nội bộ do yêu cầu nhóm, không phải khẳng định deadline môn học. Nếu deadline Moodle sớm hơn, lấy mốc sớm hơn và rút bớt phạm vi tùy chọn.

**Cách đọc:** cả nhóm đọc Mục 1–4 và 12; sau đó làm theo hướng dẫn Mục 16–24. Mục 6–7 lưu quy trình/tiêu chí Midterm đã hoàn thành, không phải danh sách phải làm lại. Mục 8–11 là hợp đồng kỹ thuật Final; Mục 16–24 cụ thể hóa bằng task ID, artifact và checklist. `AGENTS.md` chỉ định quy trình dùng tài liệu này, không tạo kế hoạch thứ hai.

**Thứ tự căn cứ:** đề `CSC14120_2026_Final Project.pdf` và thông báo giảng viên → hợp đồng đã khóa trong tài liệu này → code/test đã nghiệm thu → tài liệu lịch sử. Nếu phát hiện lệch nhau, ghi issue và đồng bộ trước khi đo; không chọn tùy tiện quy định dễ đạt hơn.

Điểm bắt đầu đã kiểm tra ngày 05/11/2026: commit `240e694730dd0c51b67febc8f79c88ac5d087831` (`feat: midterm v1`), worktree sạch trước lần cập nhật hướng dẫn này. Commit này là mốc tham chiếu lịch sử, không thay thế commit SHA được chốt khi bắt đầu Final.

### Tra cứu nhanh hướng dẫn Final

| Cần xem | Đi tới |
|---|---|
| Requirements và bằng chứng | [Mục 16](#final-requirements) |
| Môi trường, công cụ, API tích hợp | [Mục 17](#final-environment) |
| I/O các stage và việc cụ thể từng người | [Mục 18](#final-members) |
| Lịch đủ 20 ngày và xử lý khi trễ | [Mục 19](#final-timeline) |
| Correctness, timing, benchmark, profiler | [Mục 20](#final-testing) |
| Checklist nghiệm thu G0–G6 | [G0 tại Mục 16](#final-requirements), [G1–G6 tại Mục 21](#final-gates) |
| Mẫu giao việc, review và dùng agent | [Mục 22](#final-workflow) |
| Notebook, PDF, slides, video, zip | [Mục 23](#final-deliverables) |
| Việc bắt đầu ngay và checklist nhóm trưởng | [Mục 24](#final-start) |

---

## 1. Mục tiêu và quy tắc bất biến

### 1.1 Mục tiêu sản phẩm

Nhận hai ảnh RGB có vùng chồng lấn, ước lượng homography và tạo panorama trong hệ tọa độ của ảnh A. Chương trình phải:

- Có một binary xử lý mọi version qua `--version`.
- Có V0 single-thread C/C++ làm chuẩn correctness và speedup.
- Có V1 đưa toàn bộ sáu stage lên GPU bằng kernel cơ bản.
- Có ít nhất hai optimized versions, mỗi version chỉ thêm **một ý tưởng tối ưu chính**.
- Chạy được trên Google Colab T4 bằng notebook từ đầu đến cuối.
- Báo cáo correctness, thời gian từng stage, transfer, end-to-end và speedup trên 720p, 1080p, 4K.
- Có kết quả từ ba site của nhóm, một challenging pair và dataset giảng viên khi dataset này được cung cấp.

### 1.2 Quy tắc triển khai

1. **Correctness trước performance:** V0 phải vượt quality gate trước V1; mỗi CUDA version phải nhất quán với V0.
2. **Một nguồn code duy nhất:** notebook chỉ build các file trong `src/`, không chứa bản sao source độc lập.
3. **APOD:** profile → chọn bottleneck → áp dụng một tối ưu → verify → benchmark → phân tích.
4. **Không chọn tối ưu bằng phỏng đoán:** tên và thứ tự V2/V3 được khóa sau profiling V1.
5. **Công bằng:** cùng thuật toán, tham số, input, GPU và Colab session cho bảng so sánh.
6. **Số liệu thật:** không điền số ước lượng vào report; mọi con số phải sinh từ output cell hoặc JSON metrics.
7. **Tái lập:** seed, tie-break, dataset manifest, lệnh build và benchmark đều cố định.
8. **Học thuật:** mọi file sửa/viết có header tác giả và nguồn; notebook có references và AI usage declaration.
9. **Không gộp tối ưu:** CUDA streams và kernel fusion, nếu làm, phải nằm ở hai version khác nhau.
10. **Quy tắc ngôn ngữ:**
    - Tiếng Việt có dấu: giao tiếp trong hội thoại, plan chi tiết, comment trong code, những phần in ra màn hình, các cell giải thích notebook.
    - Tiếng Anh: code (tên biến, tên hàm, tên struct/class, hằng số, từ khóa lập trình...), git commit.
    - Tuyệt đối không thay đổi mã code hay các mạch logic thuật toán khi dịch chuyển ngôn ngữ.

### 1.3 Workflow bắt buộc

```text
Phase A  Dataset + contract + sửa V0
   ↓ quality gate V0
Phase B  Midterm notebook/package
   ↓ quality gate Midterm
Phase C  V1 Basic CUDA
   ↓ profile V1
Phase D  V2 optimization #1
   ↓ profile V2
Phase D  V3 optimization #2
   ↓ V4+ ngoài phạm vi lịch 20 ngày này
Phase E  Full benchmark + Final report + video + oral defense
```

---

## 2. Hiện trạng codebase và các lỗi phải xử lý

### 2.1 Thành phần hiện có

- `src/`: V0 C++ single-thread hoàn chỉnh, CLI, metrics Avg/Median/Min,
  Amdahl hữu hạn, RANSAC consensus refit, stb image I/O và provenance headers.
- `scripts/`: preprocess, synthetic GT, metrics validation, OpenCV homography
  cross-check, notebook build/execute và package quality gate.
- `tests/`: C++ unit/regression tests và Python integration/reproducibility tests.
- `notebooks/Group19_Midterm.ipynb`: snapshot ngày 05/11 có 25 cells, 15 code
  cells và không có error output. Số cell không phải tiêu chí đánh giá; D01 kiểm
  tra riêng execution count và nội dung trước khi lưu bản Midterm bất biến.
- `dataset/`: manifest có synthetic/ba standard pairs/challenging, cấu hình ba
  resolution; `dataset/provided/` chưa tồn tại ở snapshot kiểm tra. Trạng thái
  chưa có dữ liệu trong repo không chứng minh giảng viên chưa phát hành; Tiến
  kiểm tra LMS ngày D01 và D16.
- `output/`: metrics/panorama tái sinh khi Run All và bị Git ignore.
- V0 build warning-clean bằng g++ 16.2; V1/V2/V3 CUDA chưa triển khai.

### 2.2 Lỗi chặn correctness/reproducibility

Các blocker Phase A/B đã được đóng và có regression test:

- Notebook hợp lệ, đúng Group 19, có outputs thật và source/package hash.
- Synthetic GT có projected-grid RMSE ở cả ba độ phân giải.
- ORB pattern chuẩn có 256 cặp duy nhất, attribution BSD-3-Clause.
- RAII move-only, failure propagation, canvas/finite checks và deterministic Top-K.
- FAST-9 shortcut đúng cho cung 9 điểm; RANSAC refit từ toàn bộ consensus inliers.
- Mặc định 1 warm-up + 5 timed runs; report Avg/Median/Min.
- Amdahl dùng S1+S2+S6 đo bằng median trên end-to-end nên luôn hữu hạn.
- Ba standard sites và challenging pair đều có timing đủ ba độ phân giải.
- Mô hình biến đổi ảnh thật được đối chiếu độc lập với OpenCV và kiểm tra nhất
  quán trong hệ tọa độ chuẩn hóa; metrics ghi rõ homography hay translation fallback.

Các kết quả Midterm là bằng chứng lịch sử, không thay cho Gate Final. Phần chưa
có: implementation V1–V3, validator CPU/GPU, raw timing runs, Final notebook và
package. D01 phải đánh giá lại chất lượng ảnh, overlap thực tế và provided data.

---

## 3. Phương pháp và tham số bất biến giữa các version

| Stage | Phương pháp | Tham số/contract cố định |
|---|---|---|
| 1. Preprocess | RGB→grayscale, Gaussian blur 5×5 | `Y=0.299R+0.587G+0.114B`, σ=1.0, boundary clamp |
| 2. Detect | FAST-9, NMS 3×3, Top-K | threshold=20, K=2000 tối đa, sort response giảm dần rồi linear index tăng dần |
| 3. Describe | Oriented BRIEF/ORB 256-bit | intensity centroid, pattern 256 cặp đã kiểm chứng, loại keypoint không đủ border |
| 4. Match | Brute-force Hamming + Lowe ratio | `best1 < 0.75 × best2`, một chiều A→B |
| 5. Transform | RANSAC + normalized 4-point DLT; translation fallback khi H suy biến | 2000 hypotheses, reprojection threshold=3 px, seed=42, partial pivoting; fallback dùng displacement consensus + median |
| 6. Warp | B được warp vào frame A | inverse mapping, bilinear sampling, symmetric feathering |

### 3.1 Quy ước hình học

- `H_A_to_B` ánh xạ điểm ảnh A sang tọa độ ảnh B.
- Canvas nằm trong frame A; để tìm biên panorama, các góc B được ánh xạ về A bằng `inverse(H_A_to_B)`.
- Khi render một pixel canvas/A, dùng `H_A_to_B` để lấy mẫu từ B.
- Homography metrics dùng projected-grid error, không so trực tiếp chín hệ số chưa chuẩn hóa.
- H projective phải qua kiểm tra pole, hướng, diện tích và span. Nếu không đạt,
  Stage 5 dùng mô hình tịnh tiến robust cho panorama quay ngang và ghi
  transform_model=translation_fallback.

### 3.2 Quy ước quality

- Hard failure nếu cả RANSAC Homography lẫn translation fallback có dưới 4
  inliers, ma trận không hữu hạn/không khả nghịch hoặc canvas không hợp lệ.
- `low_confidence` nếu inliers `< max(12, 15% số matches)`; có thể xuất ảnh để phân tích nhưng report phải đánh dấu.
- Không gọi Stage 6 nếu Stage 5 hard-fail.
- Canvas không được “clamp cho chạy tiếp”. Giới hạn an toàn: tối đa 4× mỗi chiều và 16× số pixel lớn nhất của input; vượt giới hạn phải trả lỗi rõ ràng.

---

## 4. Giao diện, kiểu dữ liệu và telemetry

### 4.1 CLI

```bash
./stitch <left_image> <right_image> <output_image> \
  --version <0|1|2|3|...> [--runs N] [--metrics-json path]
```

- `--runs` mặc định 5; giá trị phải là số nguyên dương.
- `--metrics-json` là tùy chọn nhưng bắt buộc dùng trong notebook/benchmark automation.
- In mỗi lần chạy: GPU/CPU name, version, optimization name, kích thước input và số runs.
- Mã thoát: `0` thành công; `2` CLI/input; `3` pipeline; `4` ghi output/metrics; `5` CUDA.

### 4.2 Core types

- `Image` và `GrayImage`: RAII move-only, cấm shallow copy, destructor/free idempotent.
- `Descriptor`: `alignas(16)` và chứa `uint32_t words[8]`; CPU dùng popcount tương đương, CUDA dùng `__popc`.
- `Homography`: lưu `H_A_to_B[9]`, model flag fallback và có hàm normalize/finite validation.
- `PipelineDiagnostics`: keypoint A/B, matches, inliers, inlier ratio, H, status và warning.
- `TimingSummary`: Avg/Median/Min cho sáu stage, kernel total, H2D, control copy, D2H, end-to-end và one-time setup.
- `PipelineResult`: panorama + diagnostics + timings.
- Dispatcher mục tiêu: `run_pipeline_version(version, inputs, config)` hoặc một
  switch chung tương đương; hiện `main.cu` mới dispatch V0, `versions.h` chỉ khai
  báo V1. Thông tích hợp dispatcher; các version dùng chung orchestration.

### 4.3 JSON metrics schema

Mỗi benchmark tạo một JSON gồm:

```text
schema_version, timestamp, version, parent_version, optimization,
device_name, image_width, image_height, runs, warmups,
status, warning, keypoints_a, keypoints_b, matches, inliers, transform_model,
homography_a_to_b[9],
timing_ms.{stage1..stage6,kernel_total,h2d,control,d2h,end_to_end,setup}.{avg,median,min}
```

Notebook chỉ đọc JSON để lập bảng/biểu đồ; không copy số liệu bằng tay.

### 4.4 Timing contract

- V0: đề yêu cầu `std::chrono::high_resolution_clock`; code hiện dùng
  `steady_clock`. D02 thống nhất timer với đề, ghi thay đổi instrumentation,
  xác nhận tính đơn điệu trên môi trường và đo lại V0 Final; không sửa thuật toán.
- CUDA: `cudaEventRecord/cudaEventElapsedTime` cho kernel và transfer; CPU wall timer có synchronization cho end-to-end.
- Một warm-up không ghi vào thống kê; tối thiểu năm timed runs.
- Exclude disk load/save; report setup allocation riêng.
- Chính sách Final: preallocate/reuse cho cả CPU và GPU sau refactor chung D02;
  kiểm tra output V0 tương đương snapshot trước khi khóa. Hiện V0 còn allocation
  trong timed loop, nên chưa được gọi là steady-state đã hoàn tất. Nếu không
  refactor kịp, phải tính allocation cần thiết trong end-to-end cho mọi version,
  ghi rõ `timing_policy` và dùng cùng chính sách trong một bảng so sánh.
- GPU benchmark gồm H2D input, các control copies thực sự cần thiết và D2H panorama.

---

## 5. Cấu trúc repository mục tiêu

```text
Prj/
├── src/
│   ├── common.h
│   ├── cuda_utils.cuh
│   ├── pipeline_contract.h
│   ├── versions.h
│   ├── main.cu
│   ├── stb_impl.cpp/.h
│   ├── v0_sequential.cpp/.h
│   ├── cuda_pipeline.cu/.cuh
│   └── kernels/
├── scripts/
│   ├── preprocess_dataset.py
│   ├── generate_synthetic_test.py
│   ├── validate_metrics.py
│   └── benchmark.py
├── tests/
│   ├── test_cpu.cpp
│   └── test_scripts.py
├── dataset/
│   ├── manifest.json
│   ├── synthetic/
│   ├── site1_indoor/
│   ├── site2_outdoor/
│   ├── site3_custom/
│   ├── challenging/
│   └── provided/
├── notebooks/
│   ├── Group19_Midterm.ipynb
│   └── Group19_Final.ipynb
├── output/
├── Makefile
├── requirements.txt
├── README.md
└── project_plan_v4.md
```

Không bắt buộc tạo từng file đúng tên nếu một module nhỏ hơn rõ ràng hơn, nhưng contract, testability và tránh duplicate orchestration là bắt buộc.

## 6. Phase A — Dataset, reproducibility và V0 Golden Baseline

### A0. Môi trường, dependency và provenance

- Khóa Python dependencies tối thiểu: NumPy, Pillow, pillow-heif, OpenCV headless, matplotlib, nbformat và thư viện metric được dùng.
- Makefile có `cpu`, `cuda`, `test`, `clean`; CUDA target dùng C++17 và `sm_75` cho T4.
- Mọi file tự viết/sửa có comment đầu file: thành viên, nhiệm vụ, nguồn adaptation và license liên quan.
- `stb_image`/`stb_image_write`, OpenCV ORB pattern, sách/paper và AI tool được trích dẫn trong notebook.

### A1. Dataset manifest và preprocessing

Manifest mô tả rõ từng pair, không suy luận theo chuỗi `left/right` trong tên file.
Ví dụ dưới đây phản ánh tên JPG hiện tại:

```json
{
  pairs: [
    {id:site1_indoor,kind:standard,left_raw:in1.jpg,right_raw:in2.jpg},
    {id:site2_outdoor,kind:standard,left_raw:out1.jpg,right_raw:out2.jpg},
    {id:site3_custom,kind:standard,left_raw:cus1.jpg,right_raw:cus2.jpg,overlap_crop_fraction:0.02}
  ]
}
```

Preprocessor phải:

- Đăng ký HEIF opener trước khi mở HEIC.
- Áp dụng EXIF orientation.
- Convert RGB, crop tâm 16:9 và resize:
  - Small: 1280×720.
  - Medium: 1920×1080.
  - Large: 3840×2160.
- Ghi `left_<size>.jpg`, `right_<size>.jpg`, kiểm tra lại dimension và trả non-zero khi lỗi.
- Không chọn “file đầu tiên” bằng glob không ổn định.
- Bổ sung tối thiểu một challenging pair: low texture, repetitive pattern, moving objects, night hoặc parallax.
- Dataset giảng viên hiện thiếu: đánh dấu external blocker cho Final Gallery; không thay bằng ảnh synthetic.

Quy tắc chụp: đứng một vị trí và xoay camera, overlap 30–50%, không bước ngang để hạn chế parallax.

### A2. Synthetic ground truth

- Sinh cùng một cảnh giàu corner ở ba resolution.
- Dùng transform biết trước và lưu metadata chứa `H_A_to_B`, overlap mask và validation grid.
- Không tuyên bố ground truth nếu generator không xuất matrix thực tế.
- Metric chuẩn: symmetric/projected-grid RMSE tính theo pixel và theo tỷ lệ đường chéo.

### A3. Sửa V0

1. Thay raw ownership bằng RAII move-only.
2. Thay ORB table lặp bằng pattern 256 cặp chuẩn từ OpenCV `bit_pattern_31_`, kèm BSD attribution và test số lượng/độ duy nhất/phạm vi.
3. Tính border từ bán kính cực đại của pattern sau xoay; keypoint không đủ patch bị loại trước Top-K.
4. Top-K deterministic theo `(response desc, linear_index asc)` và trả `min(K, candidates)`.
5. Hamming dùng tám word 32-bit.
6. DLT áp dụng Hartley normalization, partial pivoting tương đối và kiểm tra finite.
7. RANSAC hiện dùng LCG tuần tự seed 42, có rejection để bốn sample index phân
   biệt, tie-break hypothesis ID nhỏ nhất. CUDA phải tái tạo đúng chuỗi đó;
   Mục 18 quy định cách sinh bảng samples trên GPU. Không tự thay bằng
   `seed=42+hypothesis_id` vì sẽ đổi các giả thuyết so với V0.
8. Nếu RANSAC fail, trả status; tuyệt đối không truyền stack `H` chưa khởi tạo.
9. Stage 6 validate canvas, bilinear bounds và dùng feather weights dựa trên khoảng cách tới biên của cả A và B.
10. Tách diagnostics/timings khỏi stdout để notebook sử dụng JSON.

### A4. Benchmark V0 và Amdahl

Chạy synthetic và ba site ở Small/Medium/Large. Với bảng Midterm chính, report sáu stage:

| Stage | 720p Median [Avg/Min] | 1080p Median [Avg/Min] | 4K Median [Avg/Min] |
|---|---:|---:|---:|
| Preprocess | | | |
| FAST + NMS + Top-K | | | |
| ORB descriptor | | | |
| Matching | | | |
| RANSAC | | | |
| Warp + blend | | | |
| End-to-end | | | |

Amdahl:

```text
p = mean_r(S1_r + S2_r + S6_r) / mean_r(V0_end_to_end_r)
S_max = 1 / (1 - p)
```

Final dùng raw runs và tỷ lệ trung bình theo cùng tập runs để tránh nhầm
`sum(median(stage))` với `median(sum(stage))`. Telemetry Midterm đang dùng tổng
median từng stage / median end-to-end: lưu như kết quả lịch sử, không tự gán
nhãn cho công thức Final. Average là thống kê chính theo đề; Median/Min bổ sung.

S1/S2/S6 là tập hot-stage bảo thủ được chọn từ profile Medium/Large. S3-S5
và host overhead thuộc serial fraction, vì vậy `S_max` phải hữu hạn. Transfer,
launch và synchronization chưa có trong V0 vẫn phải được giải thích là chi phí
làm speedup thực tế thấp hơn giới hạn này.

### Quality gate A

- CPU build thành công trên Colab và không warning nghiêm trọng.
- Synthetic grid-RMSE mọi size ≤1 px; normalized RMSE mọi size ≤0,1% đường chéo.
- Ba standard pairs tạo panorama hoặc được ghi rõ low-confidence/failure với lý do.
- Kết quả giữa năm runs deterministic về counts/H trong tolerance.
- Unit/integration tests Phase A pass.
- Chỉ khi gate này đạt mới đóng gói Midterm.

---

## 7. Phase B — Midterm Submission Quality Gate

### B1. Nội dung notebook

| Mục | Nội dung bắt buộc |
|---|---|
| Team information | Tên, MSSV, contribution theo stage và version |
| Problem understanding | Diễn giải sáu-stage pipeline bằng lời nhóm và data-flow diagram |
| Chosen methods | Phương pháp, tham số, quality rationale và GPU suitability |
| V0 design | Interface, đoạn code ngắn cần thiết, không copy toàn source |
| Correctness | Synthetic ground-truth homography error |
| Panoramas | Một output cho mỗi site của nhóm |
| Profiling | Per-stage Avg/Median/Min của synthetic và mọi ảnh thật ở ba resolution, hot-stage scaling |
| Amdahl | p, giới hạn lý thuyết và hạn chế của ước lượng |
| Dataset | Bối cảnh chụp, overlap, loại thử thách, prepared pairs, challenging pair và Drive link |
| Real-image H | Đối chiếu OpenCV `findHomography` và cross-resolution consistency; không gọi reference là ground truth |
| GPU plan | Stage mapping V1; profiling evidence; 2–3 candidate tối ưu kèm rủi ro/mitigation |

### B2. Notebook execution contract

- Tạo notebook bằng `nbformat`, không viết JSON thủ công.
- Setup cell dùng đúng `src/` trong package; nếu thiếu thì prompt upload zip một lần.
- Cài dependency, tạo output, generate synthetic, build, run benchmark, parse JSON, vẽ bảng/biểu đồ tự động.
- `Runtime → Run all` trên fresh Colab T4 thành công sau khi source/dataset được cung cấp.
- Mọi output được lưu trong file nộp; mọi số trong markdown xuất phát từ output cell.
- Group, tên/MSSV và link dataset phải là thông tin thật; không để placeholder.

### B3. Package

```text
Group19_Midterm.zip
├── Group19_Midterm.ipynb
└── src/
    ├── common.h
    ├── cuda_utils.cuh
    ├── versions.h
    ├── main.cu
    ├── stb_impl.cpp/.h
    └── v0_sequential.cpp/.h
```

Đính kèm link dataset trong notebook; source files phải có author/source header.

### Quality gate B

- [x] Midterm đã được nhóm xác nhận nộp xong; notebook hợp lệ theo `nbformat`, code cells
  đã execute, không có error output.
- [x] V0 correctness, panorama, timing ba resolution, hot-stage scaling,
  OpenCV cross-check và Amdahl hữu hạn đầy đủ.
- [x] Zip đúng tên/cấu trúc, integrity pass và build từ source duy nhất.
- [x] Attribution ở header, notebook và `THIRD_PARTY_NOTICES.md`.

Phase C chỉ bắt đầu từ commit kế tiếp sau khi archive Midterm được lưu an toàn.

---

## 8. Phase C — V1 Basic CUDA

### C1. Data flow

```text
Host RGB A/B
  → H2D
Device RGB
  → Stage 1 gray/blur
  → Stage 2 FAST/NMS/Top-K
  → Stage 3 descriptors
  → Stage 4 matches
  → Stage 5 H_A_to_B
  → small control copy for validated canvas bounds
  → Stage 6 panorama
  → D2H panorama + diagnostics
```

Descriptor/keypoint/match arrays không round-trip về Host. Các control copy nhỏ không được che giấu; phải đo và report.

### C2. V1 thread mapping

| Stage | Mapping V1 | Block |
|---|---|---|
| RGB→gray | 1 thread/output pixel | 16×16 |
| Gaussian 5×5 | 1 thread/output pixel, global reads | 16×16 |
| FAST response | 1 thread/pixel | 256 1D |
| NMS + compaction | 1 thread/pixel + bounded atomic | 256 1D |
| Deterministic Top-K | composite radix key + GPU sort | library-managed |
| ORB descriptor | 1 thread/keypoint | 128 1D |
| Matching | 1 thread/keypoint A, loop all B | 256 1D |
| RANSAC | 1 block/hypothesis | 256 1D |
| Best-H argmax | parallel deterministic reduction | 256 1D |
| Warp/blend | 1 thread/output pixel | 16×16 |

### C3. Memory and safety

- `CudaWorkspace` RAII, allocation trước warm-up, reuse cho mọi run.
- Candidate capacity là `w*h` hoặc compaction có capacity được chứng minh; không dùng 500.000 như bảo đảm tuyệt đối.
- Atomic counts reset trước mỗi run.
- Top-K composite key đảm bảo atomic arrival order không ảnh hưởng kết quả.
- DLT matrix 8×9 nằm trong shared memory; không tạo mảng dynamic-indexed lớn trên register/local memory.
- Mọi kernel có launch-error check; synchronization errors được bắt tại timing boundary.
- `cudaGetLastError`/`compute-sanitizer` phải sạch trên synthetic Small.

### C4. V1 evidence

- Timing Avg/Median/Min: từng stage, kernel total, H2D, control, D2H, end-to-end.
- Speedup V1/V0 ở ba resolution.
- Projected-grid RMSE với V0/GT và panorama similarity.
- Nsight Systems timeline xác định stage chi phối; Nsight Compute lấy metrics cho các candidate bottleneck.

### Quality gate C

- V1 correctness đạt ngưỡng trong Section 11.
- Không OOB/leak/CUDA error ở Small và 4K smoke test.
- Full timing/profile evidence được lưu.
- Chỉ sau gate này mới khóa optimization mapping cho V2.

---

## 9. Phase D — Profiling-driven V2, V3 và version tùy chọn

### D1. Quy tắc chọn version

Profile Medium và Large. Candidate đủ điều kiện khi:

1. Stage chiếm ít nhất 10% kernel time ở ít nhất một resolution; và
2. Profiling hoặc per-stage CUDA-event timing xác nhận stage giới hạn. Chỉ
   kết luận bottleneck phần cứng cụ thể khi có metrics/timeline tương ứng.

Xếp hạng theo tỷ trọng stage trung bình Medium/Large; tie-break: Matching → Blur → RANSAC. Candidate cao nhất là V2. Sau khi V2 hoàn thành, profile lại và áp dụng cùng quy tắc để chọn V3. Nếu chưa đủ hai candidate vượt 10%, chọn stage còn lại có tỷ trọng đo được lớn nhất như một thử nghiệm và phân tích trung thực nếu chậm hơn.

`version_manifest` phải ghi mapping thực tế:

```text
V0: sequential reference
V1: basic CUDA
V2: <optimization selected from V1 profile>, parent V1
V3: <optimization selected from V2 profile>, parent V2
V4+: optional, parent previous
```

### D2. Candidate A — Descriptor matching tiling

Điều kiện: Stage 4 đáng kể và memory analysis cho thấy repeated/inefficient global reads.

- `__popc` và descriptor words đã có từ V1; không tính là tối ưu mới.
- Nạp B descriptors hợp tác theo các `uint32_t` liên tiếp vào shared memory.
- Threads đọc cùng descriptor word theo broadcast, tránh bank conflict.
- Chỉ thay data access của matching; không đồng thời đổi algorithm/cross-check/ratio.
- Evidence: Stage 4 time, DRAM throughput, L1/shared behavior, long-scoreboard stalls.

### D3. Candidate B — Gaussian shared-memory tiling

Điều kiện: Stage 1 đáng kể và blur memory-bound.

- Tile nội tâm 16×16, halo 2, shared tile 20×20 float.
- Cooperative load có boundary clamp, một synchronization trước convolution.
- Filter weights/pattern constants là baseline data contract, không thêm optimization thứ hai.
- Không đồng thời chuyển sang separable convolution hoặc fusion.
- Evidence: blur time, global-load reduction, memory throughput, occupancy.

### D4. Candidate C — Warp-per-hypothesis RANSAC

Điều kiện: Stage 5 đáng kể và profiler cho thấy block-level synchronization/occupancy là giới hạn.

- Đổi 1 block/hypothesis thành 1 warp/hypothesis.
- Lane 0 giải normalized DLT; matrix workspace nằm trong per-warp shared memory.
- Các lanes chia matches và reduce inliers bằng `__shfl_down_sync` với active mask hợp lệ.
- H của mọi hypothesis vẫn được lưu; argmax vẫn deterministic.
- Evidence: Stage 5 time, achieved occupancy, synchronization stalls, warp execution efficiency và local-memory traffic.

### D5. Optional versions — ngoài phạm vi 20 ngày

Danh sách ý tưởng cho sau đợt Final này; lịch Mục 19 chỉ thực hiện V0–V3:

- CUDA streams cho hai nhánh A/B là một version riêng.
- Grayscale+blur fusion là một version riêng khác.
- Texture-object bilinear warp là một version riêng nếu Stage 6 được profile là bottleneck.
- Mỗi version vẫn phải verify V0 và report speedup-vs-previous, kể cả khi chậm hơn.

### Five-part report template cho mỗi optimized version

1. Bottleneck: số liệu của version trước.
2. Idea: một tối ưu và lý do phần cứng.
3. Implementation: thay đổi kernel/data movement.
4. Result: correctness, Avg/Median/Min, speedup-vs-V0 và speedup-vs-previous.
5. Analysis: kỳ vọng so với thực tế, profiler explanation và trade-off.

---

## 10. Phase E — Final integration, report và submission

### E1. Benchmark matrix

Trong cùng một Colab T4 session:

```text
versions: V0, V1, V2, V3, [optional versions]
sizes:    Small, Medium, Large
inputs:   synthetic + standard sites + challenging; thêm provided/failure khi áp dụng
runs:     1 warm-up + 5 timed runs tối thiểu
```

Bảng chính phải có Avg, Min, speedup-vs-V0 và speedup-vs-previous. In GPU name/driver/session metadata cùng bảng.

Biểu đồ bắt buộc:

1. Speedup-vs-V0 theo version, ba đường cho ba resolution.
2. Per-stage stacked bar V0 so với best version.
3. H2D + kernels + control + D2H + end-to-end.
4. Speedup-vs-previous cho từng optimization.

### E2. Final notebook structure

1. Team information và contribution table.
2. Introduction và pipeline overview.
3. Chosen methods, parameters và quality/GPU rationale.
4. V0 Sequential: design, correctness, panorama, profiling, Amdahl.
5. V1 Basic CUDA: mapping, data flow, verification, timing.
6. Mỗi optimized version theo five-part template.
7. Final comparison và per-stage breakdown.
8. Results gallery: synthetic, ba site, provided dataset và challenging failure case.
9. Conclusion: lessons, limitations, future work.
10. References, source attribution và AI usage declaration.

### E3. Gallery và failure analysis

- Tối thiểu một panorama cho mỗi standard site.
- Có provided dataset ngay khi giảng viên cung cấp.
- Có ít nhất một challenging case; mô tả nguyên nhân như low texture, repetitive descriptors, motion, exposure, parallax hoặc thiếu overlap.
- Không chỉnh sửa ảnh kết quả để che lỗi thuật toán.

### E4. Video 20–30 phút

| Phần | Thời gian |
|---|---:|
| Problem overview + approach | 4–6 phút |
| Live demo build/run | 6–9 phút |
| Performance results + profiling | 6–9 phút |
| Optimizations + lessons learned | 4–6 phút |

Upload YouTube Unlisted; link xuất hiện trong notebook và README, không upload video trực tiếp lên hệ thống môn học.

### E5. Final package

- `Group19_Final.ipynb` với outputs saved và Run All pass.
- PDF export của notebook.
- Toàn bộ `.cpp/.cu/.h/.cuh` cần thiết.
- README: dependencies, CUDA, build, usage, hardware, expected outputs và video link.
- Slides PDF.
- YouTube Unlisted link.
- Dataset link/manifest.
- Author/source headers và AI declaration.

---

## 11. Test plan và acceptance criteria

### 11.1 Unit tests

- Image ownership: move/free nhiều lần không double-free.
- Grayscale/Gaussian: ảnh constant giữ nguyên; impulse response khớp kernel.
- FAST: known corner, flat image và border behavior.
- ORB: đúng 256 pairs, không có block lặp ngoài source chuẩn, mọi rotated sample trong border contract.
- Hamming: zero, all-bit-different và known vectors.
- DLT: identity, translation, projective case và degenerate collinear points.
- Matrix inverse: identity/random invertible/singular.
- Warp: identity, translation, canvas bounds và symmetric feathering.
- JSON schema và CLI parsing.

### 11.2 Integration tests

- Generate synthetic Small → V0 → metrics → GT validation.
- Chạy cùng input nhiều lần, counts và H ổn định trong tolerance.
- Với mỗi CUDA version:
  - Projected-grid RMSE với V0 ≤1 px.
  - Normalized RMSE với GT ≤0,1% đường chéo.
  - Panorama SSIM trên common valid region ≥0,98.
  - Không CUDA error.
- Invalid CLI/input/output path trả đúng exit code.
- Dưới 4 matches, singular H, NaN/Inf và canvas quá lớn fail có kiểm soát.
- 4K smoke test không OOB và không vượt capacity.
- `compute-sanitizer` pass trên synthetic Small.

### 11.3 Benchmark validation

- `--runs 5` thực hiện đúng năm timed runs sau warm-up.
- Atomic counters không tích lũy giữa runs.
- Avg/Median/Min hữu hạn, Min ≤ Median; giữ raw runs để audit nhiễu.
- Kernel total nhất quán với tổng stage trong tolerance đo.
- End-to-end bao gồm transfer/control đã khai báo và loại disk I/O.
- Không đặt speedup tối thiểu bắt buộc: version chậm hơn vẫn hợp lệ nếu correctness giữ nguyên và analysis trung thực.

---

## 12. Phân công nhóm và timeline theo gate

### 12.1 Vai trò cố định, kế thừa bảng phân công Midterm

| Vai trò | Trách nhiệm chính |
|---|---|
| Phạm Minh Thông — 23120364 (Thông) | Trưởng nhóm; S1/S5 CUDA; workspace, dispatcher, build; refit/fallback, tích hợp và chốt release |
| Nguyễn Lê Tấn Tiến — 23120369 (Tiến) | Dataset/manifest; S2/S3 CUDA; Top-K/ORB, kiểm tra trung gian, gallery và phần phương pháp |
| Đào Quốc Tuấn — 23120392 (Tuấn) | S4/S6 CUDA; test harness, timing/profiling, benchmark/biểu đồ, Final notebook/package/video |

Mọi thành viên phải review và giải thích được toàn bộ pipeline; contribution table ghi theo file/stage/version thực tế, không chỉ theo vai trò danh nghĩa.

### 12.2 Milestones

| Milestone | Hoàn thành khi |
|---|---|
| M0 — Environment/Data | Manifest, HEIC preprocessing, synthetic GT và dependencies hoạt động |
| M1 — V0 Gate | Correctness/tests/benchmark V0 pass |
| M2 — Midterm Gate | Notebook Run All, outputs saved, package đúng format |
| M3 — V1 Gate | Full CUDA correctness, timing và profile evidence pass |
| M4 — V2 Gate | Một optimization có evidence và five-part analysis |
| M5 — V3 Gate | Optimization thứ hai có evidence và full comparison |
| M6 — Final Gate | Notebook/PDF/slides/video/gallery/package hoàn chỉnh |

Lịch nội bộ bắt buộc D01–D20 ở Mục 19, được đặt theo yêu cầu 20 ngày của nhóm.
Khối lượng dự kiến 4–6 giờ tập trung/người/ngày; D01 xác nhận khả năng thực tế.
Thiếu thời gian thì bỏ V4+ và giảm số thử nghiệm tối ưu, không bỏ V1 đầy đủ,
V2/V3, correctness, bằng chứng đo đạc hoặc sản phẩm bắt buộc.

---

## 13. Rủi ro và biện pháp

| Rủi ro | Mức | Biện pháp bắt buộc |
|---|---|---|
| Dataset provided chưa có trong repo | Cao | Tiến kiểm tra LMS D01/D16; thêm dữ liệu thật khi có; không tự miễn yêu cầu gallery |
| Chưa có challenging pair | Cao | Chụp/bổ sung trước Midterm dataset gate |
| HEIC không đọc được | Cao | `pillow-heif`, EXIF orientation và validation output |
| Notebook invalid/không Run All | Cao | Sinh bằng `nbformat`, fresh-runtime smoke test |
| FAST candidate overflow 4K | Cao | Capacity `w*h` hoặc compaction được chứng minh; bounds check |
| RANSAC degenerate/NaN | Cao | Normalized DLT, relative pivot, finite checks, failure propagation |
| DLT register/local spilling | Cao | Matrix trong shared memory; kiểm tra compiler/Nsight metrics |
| Optimization không đúng bottleneck | Cao | Chỉ chọn sau profile; threshold và ranking rule cố định |
| Atomic state tích lũy | Trung bình | Reset trước mỗi run và deterministic repeat test |
| Colab timeout/session drift | Trung bình | Lưu JSON/output sau mỗi benchmark; chạy comparison trong cùng session |
| CUDA streams bị implicit sync | Trung bình | Preallocate, async API đúng stream; chỉ làm ở version riêng |
| Speedup thấp hoặc âm | Trung bình | Giữ kết quả thật, phân tích bottleneck migration/overhead |
| Canvas allocation bất thường | Trung bình | Finite/bounds/pixel-count validation, fail thay vì clamp |
| Thiếu attribution/AI declaration | Trung bình | Header checklist và References gate |
| Video sai thời lượng/link private | Thấp | Rehearse 22–25 phút, kiểm tra Unlisted từ tài khoản khác |

---

## 14. Oral defense checklist

Mỗi thành viên phải giải thích được:

- Vì sao chọn FAST/ORB/Hamming và giới hạn quality so với SIFT-like methods.
- Top-K deterministic và lý do atomic order không được ảnh hưởng kết quả.
- Hướng `H_A_to_B` và tại sao warping vẫn là inverse sampling.
- Hartley normalization, partial pivoting và cách loại hypothesis suy biến.
- Khác nhau giữa V1 mapping và optimization thực sự của V2/V3.
- `__popc`, shared-memory tiling, halo, bank conflict, coalescing và warp shuffle.
- Tại sao control copy/synchronization được report thay vì che giấu.
- Cách CUDA events khác CPU timer quanh asynchronous launch.
- Amdahl estimate, transfer overhead và bottleneck migration.
- Failure case thực tế và giới hạn homography khi có parallax/motion.
- Vì sao một optimized version có thể chậm hơn nhưng vẫn là kết quả khoa học hợp lệ.

Không trả lời “đây là phần của thành viên khác”.

---

## 15. Trạng thái và checklist thực thi

### Inputs bên ngoài còn thiếu

- [ ] Kiểm tra LMS và nhập dataset giảng viên vào `dataset/provided/` khi có.
- [x] Challenging pair của nhóm.
- [x] Tên/MSSV/Group 19 thật và dataset Drive link.
- [x] Mốc nộp Midterm: cuối tuần 7 theo đề; nhóm tự khóa archive trước khi nộp.

### Trạng thái thực thi

- [x] A0 — dependency, Makefile, provenance contract.
- [x] A1 — manifest + preprocessing 4 real pairs × 3 resolutions.
- [x] A2 — synthetic ground truth.
- [x] A3 — core types và sáu stage V0.
- [x] A4 — tests + V0 benchmark/metrics + OpenCV reference audit.
- [x] Gate A.
- [x] Phase B Midterm notebook/package.
- [x] Gate B.
- [ ] Phase C V1 CUDA.
- [ ] Gate C + profiling.
- [ ] Phase D V2 rồi V3 theo profiling.
- [ ] Phase E Final integration/submission.

### Definition of Done toàn dự án

- V0, V1 và ít nhất hai optimized versions build từ một source tree.
- Mọi version đạt correctness contract và không có memory/CUDA error.
- Benchmark đủ ba resolution, cùng GPU/session, có Avg/Median/Min và hai loại speedup.
- Mỗi optimized version có profiling evidence và đúng một thay đổi chính.
- Notebook Midterm/Final Run All, output saved và không có số liệu điền tay.
- Gallery, failure analysis, provided dataset, README, PDF, slides, video và attribution đầy đủ.

---

<a id="final-requirements"></a>

## 16. Bản đồ requirements Final và bằng chứng phải bàn giao

Đọc cùng đề gốc [CSC14120_2026_Final Project.pdf](<CSC14120_2026_Final Project.pdf>).
Trang bên dưới là số trang PDF, không phải số trang sách. Những ngưỡng số học
trong Mục 11/20 là quality gate nội bộ của nhóm, không phải trích nguyên đề.

| ID | Yêu cầu và căn cứ | Artifact để kiểm tra | Người chịu trách nhiệm; người review |
|---|---|---|---|
| R01 | Ba site khác nhau đã đăng ký, mỗi site một standard pair, một challenging pair; 16:9 và ba size — tr.1–2 | manifest + raw/prepared hashes + contact sheet + xác nhận site đăng ký | Tiến; Thông |
| R02 | Sáu stage, phương pháp hợp lý, chất lượng ảnh và khả năng giải thích — tr.2–4 | bảng I/O Mục 18, source, ảnh ghép có đánh giá trực quan | Mỗi người theo stage; reviewer vòng chéo |
| R03 | V0 single-thread C/C++, cùng methods/parameters mọi version — tr.4 | snapshot Midterm, V0 Final đã khóa, config hash + tests | Thông; Tuấn |
| R04 | V1 mọi stage chạy GPU — tr.4 | CUDA source S1–S6, trace, không gọi CPU FAST/ORB/RANSAC/fallback/warp | Ba người; Thông tích hợp |
| R05 | Ít nhất V2 và V3; mỗi bản một tối ưu chính, kế thừa bản trước — tr.4–5 | version_manifest, diff, optimization note + profile của parent | Owner tối ưu; hai người còn lại review |
| R06 | Một chương trình/CLI cho mọi version — tr.5 | binary ./stitch, switch/dispatcher, CLI tests | Thông; Tuấn |
| R07 | CUDA events, chrono CPU, một warm-up, ít nhất năm runs, không disk I/O — tr.6 | raw_runs.jsonl + aggregated metrics + timer boundaries | Tuấn; Thông |
| R08 | Sáu stage, total kernels, transfers, E2E; GPU name; cùng GPU/session — tr.6 | run_manifest + 60 kết quả lõi V0–V3, thêm provided/failure khi áp dụng | Tuấn; Tiến |
| R09 | Speedup-vs-V0 và speedup-vs-parent đủ ba size — tr.6 | CSV + charts tạo từ metrics; không lấy timing Midterm Windows | Tuấn; Thông |
| R10 | Bằng chứng profiling cho từng optimized version — tr.6 | trace hoặc bảng events đã lưu + Nsight nếu khả dụng; giải thích bottleneck | Owner tối ưu; Tuấn |
| R11 | Colab fresh Run All, chỉ dataset upload nếu cần; outputs saved — tr.7 | Final notebook tự lấy đúng source bất biến; fresh-run log + SHA | Tuấn; Tiến chạy độc lập |
| R12 | Notebook đủ 10 phần, mỗi tối ưu đủ 5 ý — tr.7–8 | checklist nội dung + outputs sinh mọi số liệu | Tuấn tổng hợp; cả nhóm |
| R13 | Gallery ảnh nhóm, provided, một failure case có giải thích — tr.8 | ảnh thật từ binary + status/model + case note | Tiến; Thông |
| R14 | Notebook + PDF, source + README, slides, video 20–30 phút Unlisted — tr.6–7 | final zip đã giải nén thử, PDF, slide PDF/PPTX, link video | Tuấn; Thông chốt |
| R15 | Attribution tại nơi sử dụng + References + AI declaration; mọi người giải thích được — tr.8–9 | headers, citations, AI usage log, biên bản mock defense | Cả nhóm review chéo |

Không đánh dấu R13 xong nếu chưa kiểm tra nguồn giảng viên. Nếu LMS chưa có,
lưu bằng chứng kiểm tra ngày giờ, người phụ trách và trao đổi với giảng viên.
Chỉ ngoại lệ được giảng viên xác nhận mới đóng được mục tương ứng; nhóm không
tự thay provided bằng synthetic hoặc tự coi là đã hoàn thành.

### 16.1 Đóng Midterm và khóa baseline Final — task G0

Đầu vào: commit hiện tại, zip/notebook Midterm đã nộp, manifest và ảnh gốc.
Đầu ra: `evidence/g0/baseline_manifest.json`, test log, data audit và gate note.
Thông điều phối; Tiến phụ trách dữ liệu, Tuấn phụ trách logs.

1. Lưu SHA-256 của zip/notebook/source Midterm; dùng bản đã nộp làm lịch sử.
   Không sinh lại hoặc ghi đè bản nộp đó khi viết Final.
2. Ghi git SHA thật, compiler flags, tham số trong `src/common.h`, hashes của
   raw/prepared ảnh, synthetic GT và chính sách crop Site 3.
3. Build và chạy tests hiện có trên Linux/Colab. Chạy V0 đủ 15 cấu hình để lấy
   trạng thái nền. Không sao chép con số từ hội thoại vào manifest bằng chứng.
4. Kiểm tra ảnh: mở panorama ít nhất Small/Medium từng site, soi đường thẳng,
   vùng nối, vật thể bị đôi, ảnh bị kéo cực đoan, mất nội dung. Ghi nhận 4K
   bằng thumbnail + crop vùng nối, không cần mở ảnh khổng lồ toàn bộ.
5. Kiểm tra overlap bằng ảnh/vùng tương ứng thực tế. Các số 38/33/49/39 trong
   manifest là ước lượng hiện có, không phải ground truth. Cắt 2% không tự chứng
   minh Site 3 đạt 49%; lưu crop box và phép đo/bằng chứng dùng để kết luận.
6. Xác nhận ba site khác nhau đã đăng ký và khác dataset nhóm khác theo quy trình
   lớp. Cảnh bếp và challenging có thể cùng nơi; không dùng challenging thay
   standard site thứ ba.
7. Kiểm tra provided trên LMS. Thêm dưới manifest riêng khi có, giữ nguyên raw.
8. D02 đồng bộ timer/allocation/schema và test hồi quy V0 trước khi khóa
   `baseline_final_sha`. Ghi mọi khác biệt với Midterm; đo lại V0 cùng các bản GPU.

Nếu chất lượng standard pair kém: ưu tiên chỉnh dataset/crop có ghi nhận hoặc
sửa chung CPU/GPU trước khi khóa. Sau G0, mọi thay đổi thuật toán/threshold phải
có issue, review cả ba người, chạy lại V0 và tất cả version hậu duệ. Không bỏ
fallback đã có chỉ ở GPU để vượt test, cũng không đổi ảnh giữa các phiên bản.
Đề vẫn trừ điểm khi panorama ghép sai nặng dù code chạy nhanh.

### 16.2 Gate G0

- [ ] Zip Midterm đã bảo toàn; commit/hash baseline ghi được.
- [ ] Test CPU/Python hiện có chạy thành công; log có command/exit code.
- [ ] Synthetic GT có ma trận và lưới thật; không dùng fallback thay Homography để qua correctness.
- [ ] Bốn cặp thật đủ ba size, raw/crop không mơ hồ; đã review panorama.
- [ ] Provided đã kiểm tra và có trạng thái/bằng chứng.
- [ ] Timer/allocation và hợp đồng Stage 5 được chốt bằng văn bản.
- [ ] Tuấn và Tiến review G0; Thông ký chốt baseline Final.

---

<a id="final-environment"></a>

## 17. Môi trường thống nhất, thư mục và cách bắt đầu

### 17.1 Vai trò của từng môi trường

| Môi trường | Dùng làm gì | Điều kiện nghiệm thu |
|---|---|---|
| Máy cá nhân Windows/VS Code | Đọc code, sửa file, Git, test CPU nếu toolchain phù hợp | Không dùng timing này trong bảng so GPU Colab |
| WSL2/Linux cục bộ | Build/test tự động, có thể debug CUDA nếu đã có GPU hỗ trợ | Ghi đúng hardware/toolchain; không mặc định mọi máy có CUDA |
| Google Colab Linux GPU | Môi trường chấp nhận cuối: Run All, V0–V3, benchmark chung | GPU NVIDIA thật, nvcc build được, ghi session ID/model/VRAM/driver/toolkit |
| Linux GPU có quyền profiler | Thu Nsight/sanitizer khi Colab hạn chế | Log nguyên môi trường; không trộn số timing chính giữa hai máy |

T4/sm_75 là mục tiêu sẵn có trong Makefile. D01 kiểm tra GPU thực tế; không
giả định Colab luôn cấp T4 hoặc bộ profiler luôn được cài. Nếu dùng GPU khác,
Thông sửa cấu hình kiến trúc build phù hợp thiết bị và khóa lại cho cả phiên.
Không tự nâng/hạ toolkit nếu chưa có lỗi build cần giải quyết.

Đầu ra kiểm tra môi trường: `evidence/environment/<session_id>/environment.txt`
và `run_manifest.json`, gồm UTC timestamp, git/source SHA, Python, OS, CPU,
GPU name/VRAM/compute capability, driver, nvcc, g++, flags, package versions,
data/config hashes và `timing_policy`. Không ghi token hoặc mật khẩu.

### 17.2 Kiểm tra ban đầu trên Colab/Linux

Các dòng bash chạy trong terminal Linux hoặc cell `%%bash` sau khi đã ở root
repo. Đây là kiểm tra hiện có; setup notebook Final sẽ tự làm phần tương đương.

```bash
pwd
git rev-parse HEAD
python --version
g++ --version
nvidia-smi
nvcc --version
command -v nsys
command -v ncu
command -v compute-sanitizer
python -m pip install -r requirements.txt
python -m pip freeze
make cpu
make test
python tests/test_scripts.py
```

Nếu một profiler không có thì ghi `UNAVAILABLE` và nguyên nhân, không ghi test
pass. Ngày D01 phải thử một kernel smoke-test CUDA có CUDA events và
`CHECK_CUDA` để phân biệt “nhìn thấy GPU” với “toolchain thực sự chạy được”.
`make cuda` hiện chỉ build source V0; build được chưa chứng minh V1 tồn tại.
`make test` hiện dùng cú pháp shell Unix, không chạy nguyên xi trong PowerShell.

Ví dụ V0 có sẵn, cần tạo output folder trước:

```bash
mkdir -p output/dev
./stitch dataset/synthetic/left_small.jpg dataset/synthetic/right_small.jpg output/dev/pano_v0.jpg --version 0 --runs 5 --metrics-json output/dev/metrics_v0.json
python scripts/validate_metrics.py output/dev/metrics_v0.json
```

`requirements.txt` hiện là lower-bound dependencies. Tuấn tạo lock đã kiểm
chứng trong D02, thêm thư viện SSIM/export nếu dùng, và ghi đúng phiên bản;
không dùng pip freeze của máy Windows làm lock mặc định cho Colab Linux.

### 17.3 Cây thư mục Final mục tiêu

```text
src/
  common.h, versions.h, main.cu, cuda_utils.cuh
  v0_sequential.cpp/.h, stb_impl.*, stb_image*.h
  pipeline_contract.h             # chung, được tạo ở D02
  cuda_pipeline.cu/.cuh           # orchestration chung V1–V3
  cuda_workspace.cu/.cuh          # ownership/capacity/events
  kernels/stage1_preprocess.cu/.cuh
  kernels/stage2_detect.cu/.cuh
  kernels/stage3_describe.cu/.cuh
  kernels/stage4_match.cu/.cuh
  kernels/stage5_transform.cu/.cuh
  kernels/stage6_warp.cu/.cuh
scripts/
  benchmark_final.py              # tạo ở D02–D03
  validate_cpu_gpu.py             # tạo ở D02–D03
  build_final_notebook.py         # tạo ở D03, hoàn thiện D16–D18
  package_final.py                # tạo ở D03, hoàn thiện D18
  ...                            # tái sử dụng scripts đã có
tests/test_cuda_stages.cu          # bộ đối chiếu stage, tạo khi triển khai V1
notebooks/Group19_Final.ipynb
evidence/
  g0/, environment/, reviews/, correctness/, optimizations/
  profiles/<session>/<version>/<pair>/<resolution>/
output/final/<session>/<version>/<pair>/<resolution>/
  raw_runs.jsonl, metrics.json, pano.png, validity_mask.png
version_manifest.json
requirements-colab.lock.txt
```

Các file đánh dấu “tạo” là **sản phẩm phải viết**, chưa phải công cụ chạy được
trong repo hiện tại. Đường dẫn kernel có thể gộp/tách nếu cả nhóm thống nhất
trước; không âm thầm đổi chữ ký khi thành viên khác đang dùng.
`output/` và zip đang bị Git ignore: commit notes/config/checksums cần thiết,
lưu artifact lớn ở Drive hoặc release archive có hash, không chỉ để trên
runtime tạm của Colab. Không cần push dữ liệu lớn vào Git.

### 17.4 Hợp đồng tích hợp chung — task I0

Thông tạo header chung D02, có reviewer Tiến/Tuấn:

- Device buffers có pointer + count + capacity + stream; giữ layout RGBA
  `uchar4`, grayscale float, descriptor 8×uint32_t như V0.
- Các wrapper stage nhận buffer/input đã có; trả count/status thông qua buffer
  chuẩn. Nêu rõ ai allocate/reset/free, stream nào và khi nào output sẵn sàng.
- S2/S4 trả count 0 hợp lệ; S5 xử lý <4 match bằng failure, không fake H identity.
- Host chỉ load/save, cấp phát/launch, điều phối/timing và kiểm tra bounds từ
  metadata nhỏ. Các thuật toán S1–S6, kể cả refit/fallback, chạy trên GPU.
- Cho phép control copy counts/H/bounds/status nhỏ, ghi byte count và thời gian.
  Không copy toàn bộ keypoints/descriptors/matches về CPU để tính tiếp.
- V1 một stream mặc định, chưa fusion/texture interpolation/streams A-B. V2/V3
  bật từng lựa chọn qua config chung; không copy cả pipeline thành ba bản.
- Dữ liệu trung gian xuất về CPU chỉ ở chế độ debug, ngoài bảng timing chính.
- Buffer candidates tối đa w×h hoặc có chứng minh capacity và overflow flag;
  dùng size_t cho byte-size. Ghi peak VRAM ở 4K trước khi chọn capacity canvas.

V1 có thể dùng Thrust/CUB cho sort/reduce cơ bản, phải cite và đo thời gian.
Không dùng OpenCV CUDA hoặc thư viện stitching để thay implementation sáu stage.

---

<a id="final-members"></a>

## 18. Hướng dẫn công việc cho từng thành viên

### 18.1 Chuỗi I/O sáu stage dùng chung

Ký hiệu N = w×h, nA/nB ≤ K=2000, M ≤ nA. Hai nhánh A/B đều phải xong S3
trước khi gọi S4. Các stage sau nhận output thật của stage trước, không tự
phát hiện/matching lại bằng thuật toán khác.

| Task/stage | Owner | Input | Output | Điều kiện chuyển bước |
|---|---|---|---|---|
| K1 / S1 | Thông | Device RGBA A/B, w,h | gray float đã blur A/B | So sánh CPU/GPU flat/impulse/ảnh thật |
| K2 / S2 | Tiến | gray A/B | keypoints có thứ tự + nA/nB | Border, NMS, Top-K/count/tie-break khớp |
| K3 / S3 | Tiến | gray + keypoints từ K2 | descriptors có cùng index với keypoints | So sánh đủ 256 bits, không reorder |
| K4 / S4 | Tuấn | descriptors A/B, nA/nB | matches A→B theo idx_a + M | Best/second-best/ratio/tie khớp |
| K5 / S5 | Thông | matches, keypoints, M,w,h | H_A_to_B + model/status/inliers | DLT/refit/fallback và ngưỡng khớp V0 |
| K6 / S6 | Tuấn | ảnh RGBA, H + bounds hợp lệ | panorama + validity mask + dimensions | Canvas an toàn, sampling/blend khớp |
| B0 / harness | Tuấn | binary/versions/data/config đã khóa | raw runs + JSON/CSV/plots | Schema/correctness/timing gate pass |
| F0 / Final report | Tuấn tổng hợp | artifacts G0–G3 và B0 | notebook/PDF/slides/package/video | Run All độc lập + review cả nhóm |

K2–K6 có thể phát triển bằng fixture nhỏ trước khi upstream merge. Chỉ gọi đó là
unit-level progress; gate V1 chỉ qua khi fixture được thay bằng output liên
stage thật và chạy end-to-end. Không đợi S1 hoàn tất rồi mới bắt đầu viết S6.

### 18.2 Phạm Minh Thông — workspace, S1, S5 và integration

**Thứ tự:** G0 → I0 → K1/K5 → tích hợp G1 → điều phối G2/G3 → release.
**Tool:** C++17/NVCC, CUDA Runtime, CUDA events, Compute Sanitizer, Git;
Nsight khi cần kiểm tra register/local-memory của solver.

K1:

1. Đọc đúng hai loop CPU gray/blur và Gaussian coefficients trước khi port.
2. Kernel grayscale 1 thread/pixel; kernel blur 1 thread/output pixel, đọc
   global memory 25 taps; clamp biên giống CPU. Đầu ra float cùng miền 0–255.
3. Dùng phép toán/thứ tự gần CPU; chưa bật fast math hoặc separable blur.
4. Wrapper dùng I0; bàn giao cho Tiến fixture gray + log sai số và kernel build.

K5 — phải làm đủ sáu phần:

1. **Samples:** một kernel 1 thread sinh bảng 2000×4 index bằng đúng LCG hiện
   tại và vòng loại trùng, giữ state xuyên hypothesis. Nó đọc M trên device.
   Bảng này ở GPU; chi phí tính vào S5. V0 test fixture sinh bảng chuẩn để
   so index-by-index cho M=4,5,20 và một M thực. Không đổi sang seed+ID.
2. **Hypotheses:** 1 block/hypothesis. Thread 0 giải 4-point normalized DLT;
   workspace matrix shared, broadcast model. Các thread đếm inliers song song.
   Các thread phải cùng đi qua barrier; mẫu suy biến dùng valid flag.
3. **Chọn model:** reduce theo inliers giảm dần rồi hypothesis ID tăng dần.
   Không để thứ tự atomic kết thúc quyết định model.
4. **Refit:** giữ normalized least squares double precision và quy tắc nhận
   refit khi số inlier không giảm. V1 cho phép một thread GPU giải hệ nhỏ;
   không copy consensus về CPU để dùng solver CPU.
5. **Geometry:** kiểm tra cùng bốn góc, dấu denominator/pole, area_scale
   [0.35,3.0], span ≤3× mỗi chiều và denominator_ratio ≤6; confidence floor
   max(12, int(0.15×M)). Đây là contract hiện tại, không phải tối ưu V2.
6. **Fallback:** lọc displacement như CPU; cửa sổ chính dx [-0.85w,-0.20w],
   |dy|≤0.35h; chỉ mở rộng nếu candidate<4 sang dx [-0.95w,-0.05w],
   |dy|≤0.60h. Threshold = max(12,0.03×sqrt(w×h)), chấm mọi candidate, tie
   chọn index sớm nhất; median là phần tử n/2 (upper median), không lấy trung
   bình hai phần tử. Đếm lại với 1.5×threshold; giữ model/status/flag như V0.
   Preserve thứ tự candidates bằng stable compaction hoặc sort về idx_a.

Kernel một thread cho hệ nhỏ/median ở V1 được chấp nhận trong kế hoạch cơ bản
nếu thuật toán thật chạy GPU và chi phí được đo. Tối ưu phần này chỉ sau profile.

Checklist bàn giao của Thông:

- [ ] I0 ghi đầy đủ ownership, counts, stream, failure policy; hai người review.
- [ ] Build CPU còn chạy; NVCC link đủ object mới; dispatcher 0–3 không trỏ nhầm.
- [ ] K1 có test gray/blur biên, flat/impulse và ảnh thật.
- [ ] K5 có identity/translation/projective/noisy/collinear/<4/NaN/fallback.
- [ ] Samples và tie-break khớp V0; refit/fallback đều chạy device.
- [ ] Mọi CUDA API/launch/timer errors được kiểm tra; buffer RAII free đúng.
- [ ] Có VRAM peak/capacity 4K; buffer overflow trả lỗi có kiểm soát.
- [ ] Gate integration có bằng chứng, không chỉ báo “build thành công”.

### 18.3 Nguyễn Lê Tấn Tiến — dataset, S2/S3, correctness và gallery

**Thứ tự:** data audit G0 → K2 → K3 → verifier G1 → review G2/G3 → gallery.
**Tool:** Pillow/NumPy/OpenCV cho preprocessing/reference, C++17/NVCC,
Thrust/CUB cho sort, test harness của Tuấn, công cụ xem ảnh.

K2:

1. Port FAST-9 response 1 thread/pixel, T=20; giữ shortcut hai cardinal
   samples hiện có. Đừng lấy bản FAST-12 phổ biến thay thế.
2. NMS 3×3 loại khi neighbor response ≥ response hiện tại, border 20 pixel.
3. Compaction có bound/count/overflow; sau đó sort deterministic
   (response DESC, linear_index ASC), lấy min(2000, candidates).
   Nếu encode composite key, test cả ties; không sort response đơn thuần.
4. Reset response/counter mỗi run; không chọn Top-K theo thứ tự atomic.

K3:

1. 1 thread/keypoint, đúng intensity-centroid patch radius 15 và vòng tròn.
2. Đúng OpenCV pattern 256 pairs có attribution; giữ rounding/clamp/index.
   Trên GPU roundf và round-to-nearest-even không mặc định tương đương;
   fixture phải bao phủ giá trị .5 và góc xoay.
3. Descriptor trả cùng thứ tự keypoints. Pattern đặt constant memory từ V1
   là baseline choice; sau đó không gọi lại đây là “optimization mới” của V2.
4. Compare đủ tám words, không chỉ Hamming trung bình. Khi sai, truy ra gray,
   orientation và tọa độ từng sample; không nới ratio threshold để giấu lỗi.

Dataset/gallery:

1. Giữ raw và crop config; tạo data lock/hashes, contact sheet và descriptions.
2. Bổ sung provided khi có; xác minh GPU fallback hạn chế hướng dx âm trên
   provided. Nếu cần chuẩn hóa thứ tự A/B, ghi vào manifest và dùng đồng nhất
   mọi version; nếu thuật toán cần đổi phải qua quy tắc thay baseline.
3. Thu một failure case thật, tái lập bằng code. LOW_CONFIDENCE tự nó không
   chứng minh failure thị giác; phải chỉ ra lỗi vùng nối/matching hoặc status
   FAILURE thật. Nếu challenging hiện ghép chấp nhận được, thêm một cặp khó
   riêng (`failure_01`), không ghi đè ba standard pairs để tạo failure.
4. Không sửa panorama hậu kỳ. Chỉ thumbnail/crop có ghi nhãn để phân tích.

Checklist bàn giao của Tiến:

- [ ] Raw/prepared hashes; crop Site 3 có box và overlap evidence.
- [ ] Ba site đăng ký/khác nhau đã xác nhận; provided có owner/status.
- [ ] K2 flat/corner/border/tie/candidate-overflow/repeated-runs pass.
- [ ] K3 descriptor ordering, 256 bit pattern, rounding/orientation pass.
- [ ] Gray/keypoint/descriptor diff có artifact; không tự sửa test threshold.
- [ ] Gallery đủ standard sites/provided/failure và giải thích giới hạn fallback.
- [ ] Phần phương pháp/reference ghi đúng code, thông số K=2000.

### 18.4 Đào Quốc Tuấn — S4/S6, evidence, notebook và package

**Thứ tự:** harness B0 song song K4/K6 → profile → G2/G3 → benchmark → F0.
**Tool:** C++17/NVCC, CUDA events/NVTX, Python/NumPy/matplotlib/nbformat,
Compute Sanitizer, Nsight Systems/Compute, trình export PDF/ghi màn hình.

K4:

1. V1 1 thread/keypoint A, loop mọi descriptor B, tính XOR/popcount 8 words.
2. Giữ đúng best1/best2, cả case tie và nB=0/1; ratio 0.75 như CPU. Nếu cần
   sửa hành vi nB=1 thì phải sửa chung V0 trước G0, không chỉ CUDA.
3. Matches xuất theo idx_a tăng để K5 nhìn thấy cùng thứ tự như CPU.
   Compaction atomic phải có bước phục hồi thứ tự.
4. Shared-memory tiling là candidate sau profile, không đưa vào V1 rồi tính
   nó lần nữa là V2.

K6:

1. Khóa hướng H_A_to_B; dùng inverse H để tính bounds của B trong A-frame.
2. Copy metadata nhỏ nếu cần cấp phát; invalid H/canvas trả FAILURE thật và
   warning rõ, không để JSON SUCCESS trong khi panorama trống.
3. 1 thread/pixel canvas, bilinear và symmetric feather giống CPU. Giữ alpha/
   validity mask riêng; vùng đen vẫn có thể là pixel ảnh hợp lệ, không lấy
   RGB=0 để suy luận mask.
4. Preserve negative canvas origins, floor/ceil, clamp và đường biên bilinear.
   Bound 4× mỗi chiều/16× input pixels như hiện tại; không clamp canvas sai.
5. V1 software bilinear; texture interpolation chỉ là candidate sau profile.

Harness/Final:

1. D02 tạo schema/raw-run collector và validator theo Mục 20, không chờ V1 xong.
2. D03 tạo notebook Final skeleton, setup source bất biến và package skeleton.
3. Mỗi merge có quick check; D07/D11/D14 có full gate. D16 đo chính thức.
4. Mọi bảng, biểu đồ và câu có số xuất từ outputs; markdown không lưu timing
   điền tay. GPU profiling chạy riêng, không lấy thời gian instrumented làm
   số speedup.
5. D18 chuyển source/archive Final đã khóa vào setup; D19 Tiến chạy lại trên
   runtime sạch. Không phụ thuộc src đã tình cờ nằm trong session.

Checklist bàn giao của Tuấn:

- [ ] K4 zero/one/tied descriptor, exact matches/order tests.
- [ ] K6 identity/translation/projective/negative-origin/bounds/mask tests.
- [ ] B0 lưu raw runs và schema metadata; wrong status/NaN bị bắt.
- [ ] Có speedup Average-vs-V0/vs-parent, đủ ba size và session nhất quán.
- [ ] Có evidence parent trước mỗi tối ưu và note đủ năm phần.
- [ ] Final notebook tự lấy đúng source, Run All và saved outputs.
- [ ] PDF/slides/video/package đã kiểm độc lập; không dùng package_midterm để
  tạo Final vì nó chưa biết source CUDA mới.

### 18.5 Review và trách nhiệm khi có lỗi

| Sản phẩm | Author chính | Reviewer bắt buộc |
|---|---|---|
| S1/S5/workspace/contract | Thông | Tiến kiểm correctness; Tuấn kiểm timing/integration |
| S2/S3/dataset | Tiến | Tuấn kiểm interface/tests; Thông kiểm memory/algorithm |
| S4/S6/harness/report | Tuấn | Thông kiểm integration/timing; Tiến kiểm data/visual |
| V2/V3 | Owner của stage được chọn | Hai người còn lại |
| Final release | Thông chốt, Tuấn đóng gói | Tiến fresh-run độc lập |

Không tự nghiệm thu phần của mình. Người dùng agent vẫn là author chịu trách
nhiệm review từng diff, chạy test thực và giải thích được thay đổi.

---

<a id="final-timeline"></a>

## 19. Timeline 20 ngày — 05/11 đến hết 24/11/2026

Mỗi ngày có một đầu ra nhìn thấy được. Các ô dưới đây là kế hoạch, không phải
xác nhận đã xong. Cuối ngày 21:00 cả nhóm review 15 phút; 21:30 lưu task status,
SHA, link evidence, blocker, người xử lý và bước kế tiếp. Mỗi ngày dành khoảng
30–45 phút cho review và 20 phút học chéo/vấn đáp, nằm trong 4–6 giờ dự kiến.

| Ngày | Thông | Tiến | Tuấn | Đầu ra/ngưỡng kết thúc ngày |
|---|---|---|---|---|
| D01 — 05/11 | G0: bảo toàn Midterm, đọc S5, thử GPU/toolchain | Audit raw/crop/site registration; kiểm LMS provided | Chạy CPU/tests, thu env, chốt nơi lưu evidence | Baseline manifest + data issues + lịch GPU; chưa viết tối ưu |
| D02 — 06/11 | I0: header/device workspace, timer/allocation, khóa V0 Final | CPU fixtures K2/K3; data lock; review contract | B0 schema/raw-run harness/validator CLI bản đầu | G0 pass, API được cả ba review; V0 tương đương Midterm |
| D03 — 07/11 | K1 gray/blur CUDA; bảng samples K5 | K2 FAST/NMS CUDA | K4 matching CUDA; skeleton Final notebook/package | Build từng module, fixture-level tests; harness dùng được |
| D04 — 08/11 | K5 normalized DLT + inlier reduction | K2 compaction/sort Top-K; bắt đầu K3 | K6 inverse warp/bilinear/feather; B0 diff tool | S1→S2→S3 nhánh đầu; S4/S6 fixture pass |
| D05 — 09/11 | K5 argmax/refit/geometry/fallback GPU | Hoàn thiện K3; full intermediate comparison | K6 bounds/mask; timing H2D/control/D2H | Sáu stage có implementation; không CPU algorithm fallback |
| D06 — 10/11 | Tích hợp dispatcher/workspace/build V1; sửa kernel errors | Test flat/tie/border/repeated state, compare K2/K3 | E2E V0/V1 Small/Medium; sanitizer đầu | V1 chạy E2E synthetic + ít nhất một site |
| D07 — 11/11 | Sửa S5/4K; review G1 | Review visual/model/data; provided smoke nếu có | Full V0/V1 15×2; sanitizer + raw metadata | G1 V1 pass; tag/commit nghiệm thu nội bộ |
| D08 — 12/11 | Đọc profile S1/S5 và memory | Phân tích S2/S3; audit lựa chọn tối ưu | Profile V1 Medium/Large + timing breakdown | evidence parent V1, optimization note đề xuất V2 |
| D09 — 13/11 | Chốt mapping V2 theo bằng chứng; hỗ trợ owner | Owner nếu chọn S2/S3; nếu khác thì verifier | Owner nếu chọn S4/S6; benchmark harness | V2 chỉ bật một ý tưởng, parent V1, API không đổi |
| D10 — 14/11 | Owner nếu S1/S5; tích hợp V2 | Correctness/visual V2 và regression V1 | Đo V2-vs-V1; Nsight/events evidence | Correctness V2 pass, diff và note năm phần |
| D11 — 15/11 | Chốt G2 và chọn V3 từ profile V2 | Kiểm tra không mất thay đổi V2 | Profile lại V2; lưu rank candidates V3 | G2 pass + mapping V3/parent V2 có reviewer |
| D12 — 16/11 | Tích hợp/owner V3 nếu S1/S5 | Owner V3 nếu S2/S3; nếu khác thì tests | Owner V3 nếu S4/S6; nếu khác thì profiling | V3 implementation một ý tưởng mới trên V2 |
| D13 — 17/11 | Sửa V3 integration/numerics | Full model/descriptor/data verification | Benchmark V0–V3, memory/sync checks | V3 correctness pass; kết quả nhanh/chậm đều giữ thật |
| D14 — 18/11 | Chốt G3; khóa source thuật toán | Gallery/failure/provided audit | Note V3 + charts bốn version | G3 pass; không thêm V4+, feature freeze |
| D15 — 19/11 | Rà fair baseline/CLI/errors/README | Failure analysis, phương pháp/citations | Hoàn thiện benchmark suite + notebook biểu đồ | G4 dry-run; mọi yêu cầu có đường dẫn evidence |
| D16 — 20/11 | Review bản build đo cuối | Kiểm LMS provided lần cuối; chạy gallery chuẩn | Cùng session GPU đo toàn bộ V0–V3, lưu artifacts | G4 benchmark đủ ma trận, session/data/config hashes |
| D17 — 21/11 | Kết luận kỹ thuật/limitations, review report | Soi panorama và coverage gallery; slide phương pháp | Final notebook/CSV/charts/PDF draft; slide performance | Báo cáo dự thảo có output thật; không số điền tay |
| D18 — 22/11 | Chốt release candidate source/archive | Diễn tập phần mình; kiểm citations/video outline | Fresh Run All RC; export PDF, slides; quay demo/video | G5 kỹ thuật; notebook/PDF/source/slides/video RC đủ |
| D19 — 23/11 | Mock defense cả ba; sửa blocker có kiểm soát | Fresh runtime độc lập; xác nhận data/video links | Export lại tài liệu nếu cần; zip unpack/build audit | Mọi checkbox release được reviewer xác nhận |
| D20 — 24/11 | Chốt G6, ký checksum và bàn giao | Kiểm quyền xem dataset/provided/video | Lưu bản nộp cuối, README/index và bản sao evidence | Hoàn tất trước 23:59 UTC+7; nhóm quyết định thao tác nộp LMS |

Không tự dùng tool upload/nộp bài hoặc nhắn giảng viên chỉ vì có task trong
timeline; người được phân công thực hiện theo quyền nhóm đã thống nhất.

### 19.1 Khi trễ hoặc thiếu tài nguyên

- Kẹt API/memory trên 2 giờ: ghi issue nhỏ có reproducer, Thông xử lý trước
  khi tiếp tục upstream-dependent task; thành viên còn lại làm fixtures/report.
- Cuối D05 chưa đủ sáu stage: Thông và Tuấn pair-debug S5/integration, Tiến
  giữ tests/data; bỏ mọi tối ưu ngoài V2/V3. Ưu tiên implementation đúng, đơn giản.
- D07 G1 chưa đạt: dùng D08–D09 khắc phục; rút số candidate thử, giữ tối thiểu
  hai tối ưu riêng và thời gian kiểm tra cuối. Không coi bản hybrid CPU là V1.
- Nsight counter bị chặn: dùng CUDA-event per-stage timing làm evidence theo
  đề; lưu lỗi quyền, nêu rõ không kết luận occupancy/bank conflict khi chưa đo.
  Tìm runtime Linux GPU khác nếu cần sanitizer; số liệu performance chính vẫn
  đo lại đủ version trong một session.
- Mất Colab session/GPU đổi: artifacts phiên cũ vẫn lưu; bảng comparison mới
  phải đo lại V0–V3 trên cùng session mới, không ghép timing hai phiên.
- D14 code chưa ổn: không mở rộng scope; giữ tối ưu chậm nhưng đúng và phân
  tích trung thực. Một version sai correctness không được tính là version đạt.
- Lỗi sau D18: sửa trên RC, chạy lại tests liên quan và mọi output/benchmark
  bị ảnh hưởng; cập nhật SHA/source archive/notebook/PDF/video nếu nội dung đổi.
  Không vá code xong giữ nguyên số liệu cũ.
- Provided đến muộn: thêm gallery/correctness case; không thay benchmark scene
  chuẩn đã khóa. Nếu thiếu dữ liệu bắt buộc, ghi blocker với evidence và xin
  hướng dẫn giảng viên; không đánh dấu G6 hoàn thành toàn bộ requirements.

### 19.2 Quy trình V2 và V3 — task O2, O3

Đầu vào O2 = source V1 đã qua G1 + profile V1; đầu vào O3 = source V2 đã qua
G2 + profile V2. Không phát triển V3 trên V1 rồi gọi đó là kế thừa.

1. Tuấn tính tỷ trọng stage từ raw events Medium/Large. Owner xem evidence
   nguồn; theo ranking Mục 9. Stage ≥10% là tiêu chí nội bộ ưu tiên.
2. Chọn một ý tưởng, chỉ rõ kernel/data access thay đổi và phần giữ nguyên.
   Nếu cùng một stage chọn cho hai version thì hai ý tưởng phải tách biệt và
   có profile parent riêng; không đổi tên cùng một patch thành hai tối ưu.
3. Thông chốt owner theo stage: S1/S5 Thông, S2/S3 Tiến, S4/S6 Tuấn; hai người
   kia là reviewer. Nếu cần phân tải, ghi thay đổi trách nhiệm trong task note.
4. Bảo toàn V1/V2 bằng flags/config; V3 giữ mọi tối ưu V2 trừ khi note giải
   thích rõ lý do bỏ. Giữ executable có thể chạy lại các version cũ.
5. Chạy correctness trước benchmark. Failure thì sửa implementation; không
   giảm K/iters/precision chỉ cho version mới để lấy speedup.
6. Benchmark không profiler ≥5 runs, rồi thu profile chẩn đoán riêng. Giữ cả
   speedup âm; mỗi bản phải có note đủ bottleneck/idea/implementation/result/
   analysis và liên kết artifacts tương ứng.

Mẫu `version_manifest.json` phải có các trường:

```text
version, parent_version, main_optimization, changed_kernels,
inherited_optimizations, removed_optimizations_with_reason,
source_sha, config_hash, dataset_hash, parent_evidence,
correctness_evidence, performance_evidence, reviewers
```

Đây là schema cần tạo, không phải kết quả V2/V3 đã tồn tại.

---

<a id="final-testing"></a>

## 20. Hợp đồng kiểm thử, timing và benchmark — task B0

### 20.1 Tách ba câu hỏi phải chứng minh

| Câu hỏi | Cách kiểm | Cách kết luận |
|---|---|---|
| Thuật toán đúng trên dữ liệu biết H? | Synthetic projected-grid RMSE vs GT | Dùng GT thật, không dùng OpenCV làm nhãn |
| CUDA thực hiện cùng thuật toán V0? | Stage fixtures + end-to-end vs V0 | Giữ model/status/ordering và các ngưỡng dưới đây |
| Ảnh thực tế đủ tốt để nộp? | Visual review + failure analysis | Ghi cả lỗi còn lại; counts/SSIM cao không thay thế review |

CPU/GPU consistent không tự chứng minh panorama thật đẹp. Một homography
OpenCV bị geometry check loại vẫn phải lưu matrix/inlier count/lý do; không
xóa reference hoặc nới tiêu chí để che nhược điểm phương pháp đã chọn.

### 20.2 Bộ kiểm tra tối thiểu phải viết

Tuấn tạo runner, mỗi owner thêm fixture theo stage mình. Các fixture phải kiểm
vấn đề thật: flat/corner/border/tie/degenerate, ảnh chuẩn và trường hợp failure.
Chạy isolated stage bằng **cùng buffer input**, sau đó mới kiểm end-to-end.

| Đối tượng | Gate nội bộ ban đầu | Evidence khi lỗi |
|---|---|---|
| S1 gray/blur | float max_abs_error ≤1e-3 trên miền 0–255; cùng boundary policy | max-error coordinate + CPU/GPU values |
| S2 | cùng input: candidate count, Top-K indices và thứ tự bằng nhau | response/NMS/tie tại pixel lệch |
| S3 | cùng gray/keypoints: đủ 8 words descriptor bằng nhau | angle, sample coordinates và bit đầu tiên khác |
| S4 | cùng descriptors: best/second/ratio, matches và thứ tự bằng nhau | idx_a, distance1/2, chosen idx_b |
| S5 samples | bảng 2000×4 mẫu bằng nhau khi cùng M | hypothesis ID + expected/actual indices |
| S5 model | cùng matches: model flag và status bằng nhau; H finite/khả nghịch | raw H, selected hypothesis, consensus/refit/fallback |
| Synthetic V0 | RMSE vs GT ≤1 px và ≤0.1% đường chéo mọi size | JSON points/projections/RMSE, không chỉ PASS text |
| CUDA V1–V3 | grid RMSE vs V0 ≤1 px; synthetic vs GT ≤0.1% đường chéo | H/grid/error/tolerance trong report |
| Panorama | SSIM ≥0.98 trên common valid ROI; coverage IoU ≥0.99 khi so CPU/GPU | mask/ROI/origin/dimensions + SSIM + visual crop |
| Repeated runs | counts, model/status và H ổn định; không state/counter tích lũy | diagnostics của từng run |
| CLI/failure | invalid args/input/output/singular/too-large canvas trả lỗi đúng | command, exit code, stderr và metrics status |

SSIM/coverage tính từ ảnh lossless hoặc buffer, đặt hai output lên cùng canvas
theo origin A-frame; không resize/register riêng để làm SSIM tăng giả. Điều
kiện ROI phải ghi rõ, không để common-region rất nhỏ vẫn “pass”.

Sai số floating-point tại ngưỡng FAST/ratio/RANSAC có thể làm count khác trong
end-to-end. Không tự nới threshold hay bỏ test: ghi reproducer và diff stage,
giải thích phép toán/rounding, reviewer xác nhận cách sửa hoặc ngoại lệ có
bằng chứng trước gate. Nếu đổi acceptance threshold, ghi quyết định vào plan,
rerun mọi version liên quan; không sửa chỉ file test cho version đang fail.

### 20.3 Schema mới cần thêm, tương thích metrics hiện có

Không rename/xóa các khóa hiện có mà không migration reader. Tuấn bổ sung:

- Run metadata: `session_id`, `source_sha`, `source_hash`, `data_hash`,
  `config_hash`, compiler/flags, GPU/CPU/toolkit/driver, version/parent.
- Mỗi raw run: warmup hay timed, index, S1–S6, kernel_total, H2D, control,
  D2H, E2E, residual host overhead, diagnostics/status/model và CUDA errors.
- Aggregate: runs/warmups, Average/Median/Min/std hoặc CV, transform_model,
  dimensions/origin, schema_version và timing_policy.
- Correctness: tolerance, actual error/SSIM/coverage, input hashes, status
  PASS/FAIL/NOT_RUN, đường dẫn log. Không dùng 0 thay cho số đo thiếu.
- Model thất bại không ghi H identity như thể estimated H hợp lệ; có flag/
  status rõ, không dùng metrics failure trong speedup table thành công.

Current exporter đang hard-code parent None; khi thêm V2/V3 phải lấy từ version
manifest. Current metrics chỉ lưu diagnostics run đầu; Final phải lưu per-run
ở raw evidence để kiểm tất định. Không ghi nhanh hơn từ early failure S5/S6.

### 20.4 Biên đo thống nhất

1. Load RGB/preprocess dataset từ đĩa trước timing, save panorama sau timing.
2. Tạo context, chuẩn bị workspace theo timing_policy, chạy một warm-up đầy đủ.
3. Với mỗi timed run GPU: CPU wall start → H2D input → S1–S5 → control copies/
   host bounds nếu có → S6 → D2H panorama/diagnostics → synchronization hoàn
   tất → CPU wall stop. V0 đo tương đương từ input đã load đến output sẵn sàng.
4. Stage CUDA events ghi trên stream chạy kernel, kiểm lỗi event/sync. Gộp
   thời gian của hai nhánh A/B và library calls vào stage tương ứng.
5. `kernel_total` = tổng compute của sáu stage với V1 một stream; `control`
   chứa metadata transfer/sync thực cần, không giấu thời gian vào số 0.
6. E2E khác tổng các event do host dispatch/sync/allocation policy; lưu residual
   và giải thích. Không ép các số bằng nhau bằng cách bịa overhead.
7. Average là số chính theo đề; Median/Min hỗ trợ. So tổng trên raw run trước
   khi aggregate; median của tổng không cần bằng tổng các median.
8. Benchmark performance chạy không profiler/sanitizer/debug-dumps. Profile/
   sanitizer dùng binary/source cùng SHA, log riêng và không nhập timing đó
   vào bảng speedup. Nếu đổi flags profiling, ghi flags riêng.

### 20.5 Ma trận benchmark và công thức

Lõi bắt buộc theo kế hoạch nhóm: 4 version × 5 pairs (synthetic, site1, site2,
site3, challenging) × 3 sizes = **60 cấu hình**, mỗi cấu hình 1 warm-up +
5 timed runs = 300 timed pipeline runs. Thêm provided và failure case riêng;
không thay thế một trong năm pairs lõi.

Full comparison của bốn versions, kể cả V0, đo cùng một Colab GPU session,
cùng scene/hash/flags/timing policy. Chọn trước `site2_outdoor` làm scene của
bảng speedup chính; vẫn lưu bảng tất cả pairs và không đổi sang scene có
speedup cao nhất sau khi đã xem số. Low-confidence được đánh dấu rõ.

```text
speedup_vs_v0(v, pair, size) = avg_E2E(V0, pair, size) / avg_E2E(v, pair, size)
speedup_vs_parent(v, pair, size) = avg_E2E(parent(v), pair, size) / avg_E2E(v, pair, size)
```

V0-vs-parent = N/A; V1 parent V0; V2 parent V1; V3 parent V2. Kernel-only
speedup có thể thêm riêng và phải dán nhãn; không thay nó cho E2E speedup.
Nếu CV >10% (ngưỡng nội bộ), giữ log 5 runs gốc, kiểm tải nền/nhiệt rồi đo
10 runs cho toàn nhóm comparison liên quan cùng chính sách; không chọn từng
run nhanh nhất hay tự bỏ outlier. Version chậm hơn vẫn có thể đạt.

### 20.6 Lệnh giao diện mục tiêu cho harness

Các file/lệnh này do Tuấn **phải tạo ở D02–D03**; đây là giao diện đề xuất
để các thành viên thống nhất, không phải lệnh đang có sẵn.

```bash
python scripts/benchmark_final.py --help
python scripts/benchmark_final.py --versions 0 1 2 3 --cases synthetic site1_indoor site2_outdoor site3_custom challenging --runs 5 --session-id final-20261120-a --output-root output/final
python scripts/validate_cpu_gpu.py --baseline output/final/final-20261120-a/V0 --candidate output/final/final-20261120-a/V1 --json-out evidence/correctness/v1.json
```

Harness đọc sizes từ manifest, tạo directories, gọi CLI hiện có, lưu stdout/
stderr/exit code và raw runs, kiểm correctness trước khi in speedup. Nó phải
từ chối so sánh session/data/config/timing_policy khác nhau.

### 20.7 Profiler và sanitizer: lệnh tham khảo đã kiểm tài liệu

Sau I0/Makefile tích hợp source CUDA mới, build debug-line giữ -O3 và không
bật fast math. Tạo thư mục evidence trước. Ví dụ Linux:

```bash
make cuda NVCC_FLAGS="-O3 -arch=sm_75 -std=c++17 -lineinfo"
mkdir -p evidence/profiles evidence/sanitizer output/dev
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o evidence/profiles/v1_medium ./stitch dataset/site2_outdoor/left_medium.jpg dataset/site2_outdoor/right_medium.jpg output/dev/profile.jpg --version 1 --runs 1
ncu --set basic --launch-count 8 -o evidence/profiles/v1_kernels ./stitch dataset/site2_outdoor/left_medium.jpg dataset/site2_outdoor/right_medium.jpg output/dev/profile_ncu.jpg --version 1 --runs 1
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 ./stitch dataset/synthetic/left_small.jpg dataset/synthetic/right_small.jpg output/dev/sanitizer.jpg --version 1 --runs 1
compute-sanitizer --tool racecheck --error-exitcode 99 ./stitch dataset/synthetic/left_small.jpg dataset/synthetic/right_small.jpg output/dev/racecheck.jpg --version 1 --runs 1
compute-sanitizer --tool synccheck --error-exitcode 99 ./stitch dataset/synthetic/left_small.jpg dataset/synthetic/right_small.jpg output/dev/synccheck.jpg --version 1 --runs 1
```

Kiến trúc sm_75 chỉ dùng khi GPU đúng T4/compute capability tương ứng. Ncu ví dụ
thu tối đa tám launches đầu, có thể thuộc warm-up; sau khi xem trace phải lọc
đúng kernel bottleneck theo tên, không coi ví dụ này đã profile đủ mọi stage.
Chạy memcheck trước racecheck/synccheck; lỗi sanitizer không được coi pass chỉ
vì ứng dụng trả 0. Log stderr và report lỗi thật. Chạy thêm initcheck khi nghi
buffer chưa reset. 4K cần smoke/memcheck theo ngân sách runtime.

Căn cứ công cụ: [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/),
[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
và [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).
Các ngưỡng correctness phía trên do nhóm đề xuất, không phải do NVIDIA đặt.

---

<a id="final-gates"></a>

## 21. Checklist nghiệm thu từng gate

Mỗi gate có `evidence/reviews/Gx.md`: input SHA/data/config, lệnh đã chạy,
artifact paths/hashes, output mong đợi/thực tế, PASS/FAIL/BLOCKED/NOT_RUN,
blocker và chữ ký reviewer. Checklist trống là chưa kiểm; không mặc định PASS.

### G1 — V1 hoàn chỉnh, hạn D07

Đầu vào G0/I0 pass + K1–K6; đầu ra source V1 đã khóa + correctness/profile
ban đầu. Người chốt Thông; Tiến/Tuấn review.

- [ ] S1–S6, samples/refit/geometry/fallback chạy device thật.
- [ ] Mọi intermediate counts/buffers đi qua pipeline thật; debug CPU path tắt.
- [ ] Cùng K/FAST/ratio/RANSAC/pattern/rounding/fallback với V0.
- [ ] Sort/matching/samples/argmax deterministic; reset state mỗi run.
- [ ] Synthetic cả ba size qua GT gate; V1 vs V0 và visual review qua gate.
- [ ] Standard/challenging có output/status thật; không bỏ case khó khỏi log.
- [ ] Memcheck sạch, race/sync được kiểm; 4K không OOB/capacity overflow.
- [ ] CUDA events/CPU synchronization và schema/raw runs đã kiểm.
- [ ] ./stitch --version 0 và 1 build từ cùng source tree trên Colab.
- [ ] Evidence lưu ngoài runtime trước khi chốt G1.

### G2 — V2 tối ưu thứ nhất, hạn D11

Đầu vào G1 + profile V1; đầu ra source V2, parent evidence, note đủ năm ý.

- [ ] Bottleneck lấy từ V1 đúng cùng size/scene; có log/timing thật.
- [ ] Một ý tưởng chính, thay đổi code có giới hạn; owner/reviewer rõ.
- [ ] Parent V1, methods/config/data không đổi; V1 vẫn chạy lại được.
- [ ] Correctness, determinism, memory/sync checks liên quan đều pass.
- [ ] Có Avg/Median/Min, speedup-vs-V0 và vs-V1 đủ ba size.
- [ ] Kết quả âm vẫn giữ, giải thích dựa trên đo đạc, không sửa ảnh kết quả.
- [ ] Version manifest + note + artifact paths/hashes hoàn chỉnh.

### G3 — V3 tối ưu thứ hai, hạn D14

Đầu vào G2 + profile V2; đầu ra source V3 và comparison bốn versions.

- [ ] Profiling motivation là V2, không copy lại số liệu motivation của V2 từ V1.
- [ ] V3 giữ tối ưu V2 hoặc giải thích cụ thể vì sao bỏ.
- [ ] Thêm đúng một ý tưởng mới; không tách cùng một patch thành hai versions.
- [ ] V0–V3 đều CLI-selectable; correctness theo Mục 20 qua gate.
- [ ] Speedup-vs-parent lấy V2; đủ ba sizes, không hard-code số.
- [ ] Note năm ý, profile before/after và review chéo đầy đủ.
- [ ] Feature freeze; không làm V4+ trong lịch 20 ngày này.

### G4 — Benchmark chính thức, hạn D16

Đầu vào G3 + fixed source/data/config; đầu ra raw/aggregate/CSV/charts.

- [ ] Có 60 cấu hình lõi, thêm provided/failure theo inventory thực tế.
- [ ] Mỗi cấu hình một warm-up, ≥5 timed runs; raw diagnostics được giữ.
- [ ] Cùng GPU/session/compiler flags/timing_policy; V0 đo lại trong session đó.
- [ ] Average bắt buộc, Min/Median đầy đủ; không lấy debug/profile timing.
- [ ] H2D/control/D2H/host residual được khai báo; disk I/O ngoài E2E.
- [ ] Từng kết quả in GPU name và có source/data/config hashes.
- [ ] Mọi failure/low-confidence ghi thật; không tính speedup cho early failure.
- [ ] Hai loại speedup + stage breakdown + scalability charts đọc từ JSON.
- [ ] Artifact copy khỏi Colab, không chỉ còn link runtime đã mất.

### G5 — Báo cáo/Run All, hạn D18

Đầu vào G4; đầu ra release candidate notebook/PDF/source/slides/video.

- [ ] Final notebook đủ 10 phần của đề, V2/V3 mỗi bản đủ năm ý.
- [ ] Mọi số trong text truy được về output cell hoặc artifact được cell in.
- [ ] Source setup tự lấy đúng bản bất biến, verify SHA, build đủ files CUDA.
- [ ] Fresh Colab Run All không cần sửa path/cell hoặc upload source thủ công;
  chỉ dataset upload nếu cần. Không lệ thuộc Drive mount có tương tác bắt buộc.
- [ ] Code cells có execution count, outputs saved, không exception bị giấu.
- [ ] Gallery có standard/provided/failure và giải thích; nếu thiếu provided
  phải có ngoại lệ thật từ giảng viên, không tự đóng requirement.
- [ ] PDF đầy đủ bảng/charts/panorama, tiếng Việt không mất dấu/cắt cột.
- [ ] README có setup/CUDA/build/usage/hardware/expected outputs/video/data.
- [ ] Attribution tại nơi dùng và References; AI usage khai đúng công cụ/vai trò.
- [ ] Tiến chạy độc lập được; feedback được xử lý trước G6.

### G6 — Gói bàn giao và hiểu bài, hạn D20

- [ ] Notebook/PDF/source/README/slides/video/data links đủ và trùng release SHA.
- [ ] Giải nén zip sang thư mục mới, build/chạy smoke-test thành công.
- [ ] Không thiếu .cu/.cuh/kernel/header/script/lock file cần cho Run All.
- [ ] Không private token, binary cũ, .venv, object, cache hoặc output bị sửa tay.
- [ ] Video Unlisted 20–30 phút, phát được ngoài tài khoản chủ; link trong cả
  notebook và README. Không upload video trực tiếp Moodle.
- [ ] Dataset/source links tải được với quyền người chấm, đúng hash/version.
- [ ] Cả ba thành viên trả lời và sửa được code các stage ngoài phần mình.
- [ ] Reviewer xác nhận từng R01–R15; còn BLOCKED/NOT_RUN thì ghi rõ chưa đủ.
- [ ] SHA-256 zip cuối được lưu cùng biên bản release; có bản sao độc lập.

---

<a id="final-workflow"></a>

## 22. Cách làm thống nhất cho người và agent

### 22.1 Một task luôn dùng cùng một phiếu giao việc

Thông/Tiến/Tuấn dùng mẫu này trong issue hoặc `evidence/reviews/<task_id>.md`.
Không cần tạo cả issue và file nếu chúng chỉ lặp cùng nội dung.

```text
Task ID / owner / reviewer:
Gate và ngày phải xong:
Input: git SHA, data/config hash, prerequisite task đã PASS:
Mục plan/requirement liên quan:
File được phép sửa:
API nhận vào / trả ra / ownership / stream / failure status:
Kết quả mong đợi và tolerance:
Các thay đổi không nằm trong task:
Command/test phải chạy:
Artifacts phải nộp:
Actual result / exit code / evidence path:
Known limitations hoặc BLOCKED/NOT_RUN:
Reviewer decision:
```

**Task DONE** = output đúng + test thật + artifact đọc được + reviewer duyệt.
“Đã code xong”, “agent báo pass”, “biên dịch được” hoặc ảnh chụp terminal
không gắn input/SHA đều chưa phải DONE. Nếu không có GPU, task CUDA chỉ được
ghi IMPLEMENTED/NOT_RUN; người có GPU phải nghiệm thu trước merge gate.

### 22.2 Git và ranh giới file

1. Mỗi task một branch/worktree riêng từ commit integration đã review, ví dụ
   `feat/tien-s2-s3`, `feat/thong-s1-s5`, `feat/tuan-s4-s6`.
2. Trước sửa, đọc git status và diff; giữ nguyên thay đổi của người khác.
   Không dùng reset/checkout để làm sạch shared worktree.
3. Thông sở hữu các file chung common/contract/dispatcher/workspace/Makefile.
   Khi cần đổi API, đề xuất signature trước, cập nhật cả caller/callee/test.
4. Tuấn sở hữu schema/scripts/notebook/package. Không đổi tên JSON key hoặc
   output path từ kernel task mà không cập nhật reader/validator.
5. Merge nhỏ hằng ngày vào nhánh integration; reviewer chạy kiểm tra liên quan.
   Thông tích hợp sau review, không đợi ba nhánh lớn tới D07 mới ghép.
6. V2 bắt đầu từ G1; V3 bắt đầu từ G2. Task kernel độc lập dùng fixtures được
   phép song song; hai tối ưu tích lũy bắt buộc nối tiếp.
7. Commit message tiếng Anh, ghi task ID. Báo cáo/comment tiếng Việt có dấu,
   identifiers code tiếng Anh. Không dịch identifiers khi chỉ sửa văn phong.

### 22.3 Prompt chung khi giao agent

Sao chép mẫu, điền task ID/owner/files thực; acceptance lấy nguyên từ plan:

```text
Đọc AGENTS.md và project_plan_v4.md, đặc biệt Mục 3–4, 18 và 20–22.
Thực hiện task <ID> cho <owner>, prerequisite <IDs> đã PASS tại commit <SHA>.
Chỉ sửa <file paths được giao>; giữ các thay đổi của người khác.
Input/output và tiêu chí nghiệm thu theo task card đính kèm.
Methods, K=2000, T=20, ratio=0.75, 2000 RANSAC hypotheses, pattern,
RNG/order/refit/fallback phải nhất quán với baseline Final đã khóa.
Không tự đổi thuật toán hoặc tolerance để vượt test.
Triển khai và chạy test phù hợp nếu môi trường có khả năng.
Nếu thiếu GPU/tool/dataset, ghi NOT_RUN/BLOCKED với bằng chứng, không bịa PASS.
Bàn giao: diff, API/files đổi, lệnh+exit code, metrics/log paths+hash,
requirement đã đáp ứng, giới hạn còn lại và bước reviewer cần làm.
Không tự push/publish/nộp bài hoặc nhắn người khác khi chưa có ủy quyền.
```

Các task sau là cách điền phạm vi, không phải lệnh tự triển khai tất cả:

| Người giao agent | Task đầu tiên phù hợp | Đầu vào cần đưa | Sản phẩm cần đòi |
|---|---|---|---|
| Thông | I0 rồi K1/K5 | V0 source + contract đã review + Stage 5 fixtures | Wrapper/device kernels, sample/solver/fallback tests |
| Tiến | Data G0 rồi K2/K3 | data lock + gray fixtures + sorted keypoints chuẩn | Audit/crop evidence, deterministic Top-K, descriptors diff |
| Tuấn | B0 rồi K4/K6 | schema + descriptor fixtures + H fixture | Collector/validator, match/warp tests, timing evidence |

Không giao một prompt “làm hết CUDA” rồi chấp nhận source mà không stage
evidence. Khi agent muốn đổi assumption, ghi task note và reviewer xử lý theo
quyền sẵn có; routine edit/test trong scope không cần xin duyệt lại từng lệnh.

### 22.4 Phiên làm việc và bàn giao hằng ngày

- Đầu phiên: xem gate hiện tại, prerequisite, git SHA và task card; chạy smoke
  test phần liên quan trước nếu vừa đổi integration commit.
- Trong phiên: làm một đầu ra nhỏ, test ngay; gặp API/data/precision mismatch
  thì lưu reproducer trước khi sửa. Không tiếp tục phụ thuộc vào output sai.
- Cuối phiên: điền actual vs expected, evidence paths, commands/exit codes,
  phần chưa chạy và đề xuất task tiếp. Reviewer có thể kiểm lại không cần hỏi.
- Cuối ngày: cập nhật Mục 15/gate note theo thực tế; không sửa tiêu chí để biến
  việc chưa xong thành hoàn thành.

---

<a id="final-deliverables"></a>

## 23. Hoàn thiện Final notebook, PDF, slides, video và archive

### 23.1 Nối tiếp Midterm nhưng không làm thay đổi bản đã nộp

Tuấn tạo `scripts/build_final_notebook.py` tái sử dụng nội dung/logic setup,
dataset và plot còn đúng từ Midterm. Không gọi builder Midterm để ghi đè bản
đã nộp; notebook Final là artifact mới build từ source thực tế V0–V3.
Final phải đo lại V0, không lấy timings Windows trong Midterm làm baseline.

Đầu vào: source G3, benchmark G4, version manifest, dataset lock, references.
Đầu ra: `notebooks/Group19_Final.ipynb` và PDF cùng nội dung.

Cell sequence:

1. Team/IDs/contribution thực theo stage và version; mục tiêu/sơ đồ sáu stage.
2. Setup tự lấy source từ archive URL/commit bất biến, verify SHA-256, giải
   nén và build; tự cài dependencies đã kiểm chứng. Không lấy nhánh main mới
   nhất. Nếu source đã giải nén cùng notebook thì xác minh hash rồi dùng.
3. Dataset cell kiểm manifest/hashes; tự lấy dataset công khai nếu có, nếu
   không mới cho phép upload dataset. Tránh thêm bước tương tác mount Drive
   bắt buộc vì đề chỉ cho manual dataset upload.
4. Environment/timing-policy cell in metadata và cấu hình thật.
5. Build/test V0–V3 từ cùng src; tạo output root; chạy benchmark V0 và CUDA.
6. Correctness GT + CPU/GPU + model flags; in số liệu vào output.
7. V0/V1 timing, mapping/dataflow; V2/V3 đủ năm ý, evidence parent.
8. Final speedup table/plots/stage-transfer breakdown, có Average/Median/Min.
9. Gallery đủ site/provided/challenging/failure, nêu parallax/repetitive
   pattern/fallback restrictions và quan sát lỗi thực.
10. Kết luận/giới hạn/hướng tiếp + references/AI usage/video/dataset links.
11. Preflight kiểm source/data hashes, saved outputs và artifact inventory.

Cell phân tích expected failure phải kiểm exit code/status dự kiến và hiển thị
lỗi để giải thích; không dùng runner tự dừng notebook ở case đó. Một lỗi ngoài
danh sách expected vẫn phải làm preflight fail, không catch rồi im lặng bỏ qua.

Source archive dùng cho setup phải loại notebook/output để tránh vòng lặp
“notebook chứa hash của archive chứa chính nó”. Artifact source bất biến có thể
là release URL hoặc snapshot repo theo SHA, cần quyền tải người chấm. Nhóm
chuẩn bị và xác minh URL ở D18; việc công bố thực tế tuân theo quyền repo.
Thiếu cơ chế tự lấy source thì chưa qua fresh-runtime gate.

Nếu dữ liệu large/canvas khiến runtime hết RAM/VRAM, giảm debug dumps/preview
và xử lý case tuần tự, không âm thầm hạ resolution hay K. Theo dõi memory của
NumPy/Matplotlib và close figures sau display.

### 23.2 PDF và slides

1. Lưu notebook đã execute trước khi export.
2. Có thể dùng nbconvert HTML rồi trình duyệt Print→Save as PDF; hoặc PDF/
   WebPDF exporter được kiểm trong môi trường nhóm. Chốt một cách làm ở D17,
   ghi dependency nếu exporter cần; không chờ ngày cuối mới cài LaTeX.
3. Mở PDF kiểm cả đầu/cuối, bảng rộng, font tiếng Việt, mọi panorama/charts/
   source citation/video link và không có cell bị cắt mất.
4. Slides cập nhật từ output Final: problem, phương pháp, CUDA mapping,
   hai tối ưu có evidence, scaling, failure/limitations, contributions.
   Không dùng các con số Midterm/slide cũ khi dataset/implementation đổi.
5. Có file slides gốc và PDF để người xem không cần đúng phần mềm.

### 23.3 Video 25 phút, cả ba người đều tham gia

Khung thời lượng 25 phút dưới đây nằm trong 20–30 phút của đề và các khoảng
gợi ý từng phần. Tuấn ghép/quay; mỗi người chịu trách nhiệm nội dung mình:

| Phần | Thời lượng | Người dẫn/nội dung |
|---|---:|---|
| Problem/approach | 5 phút | Tiến: dataset, sáu stage, overlap, chosen methods/giới hạn |
| Live demo | 7 phút | Thông + Tuấn: build từ source, chạy CLI chọn versions, correctness, panorama |
| Performance/profiling | 8 phút | Tuấn + Tiến: bảng ba sizes, stage/transfer, hai loại speedup, profile parent |
| Optimizations/lessons | 5 phút | Thông + cả nhóm: V2/V3, kết quả âm nếu có, failure, bài học CUDA |

Quay command thực và output từ bản release; text đọc được, âm thanh rõ. Video
không cần chờ chạy lại toàn bộ benchmark dài nếu giải thích phần output lưu,
nhưng phần live demo phải là chạy code thật. Kiểm link Unlisted từ cửa sổ/
tài khoản khác; đặt link vào README và notebook trước export/package cuối.

### 23.4 Cấu trúc gói Final mục tiêu — task F0

Đây là quy cách nội bộ vì đề không ấn định toàn bộ tên zip Final. Nếu Moodle
có naming riêng, dùng naming của Moodle và ghi mapping trong README.

```text
Group19_Final.zip
  Group19_Final.ipynb
  Group19_Final.pdf
  README.md
  Makefile
  requirements.txt
  requirements-colab.lock.txt
  THIRD_PARTY_NOTICES.md
  project_plan_v4.md
  version_manifest.json
  src/                         # tất cả source/kernels cần build
  scripts/                     # các script notebook gọi và dependencies nội bộ
  tests/                       # fixtures/runner cần cho reproducibility
  dataset/manifest.json
  dataset/synthetic/ground_truth.json
  evidence/                    # note/bảng/hash; artifact lớn qua link có quyền
  slides/Group19_Slides.pdf
  slides/Group19_Slides.pptx
  artifact_manifest.json
```

Raw/prepared ảnh lớn qua dataset link đúng version theo đề; source setup
archive riêng được notebook tải và hash kiểm chứng. README liệt kê rõ cái gì
được đóng kèm, cái gì tải ở đâu, hash, dung lượng và quyền truy cập. Video qua
link, không đóng nguyên video vào zip.

Tuấn viết package script mới, kiểm inventory bắt buộc, notebook errors/
execution counts, citation/link placeholders, path traversal khi giải nén và
source hashes. Không chỉ sửa tên Group19_Midterm thành Final vì packager cũ
hard-code source list V0 và sẽ thiếu .cu/.cuh mới.

Sau khi notebook được lưu đủ outputs, chạy package **bên ngoài** notebook lần
cuối; nếu package cell chạy trước lúc notebook save thì zip có thể thiếu output
cuối. Artifact manifest ghi hash từng file khác; hash của zip lưu ở biên bản
release bên ngoài zip để tránh self-reference.

### 23.5 Handover thực tế cho nhóm

Thông chỉ chốt release khi Tiến giải nén sang folder mới và xác nhận:

- [ ] Inventory đủ; source hash khớp bản notebook đã build.
- [ ] Build từ src mới, CLI 0–3, synthetic Small smoke-test pass.
- [ ] Fresh Colab Run All trên dataset chuẩn không sửa tay cell.
- [ ] Notebook/PDF/slide/video kể cùng methods/config/results.
- [ ] Links source/dataset/video đọc được với quyền người chấm.
- [ ] Tất cả R01–R15 có evidence hoặc blocker/ngoại lệ được xác nhận rõ ràng.

---

<a id="final-start"></a>

## 24. Bắt đầu từ đâu và hồ sơ review cuối

### 24.1 Ba bước ngay D01

1. **Thông:** mở Mục 16.1 + 17.4 + 18.2; lập G0/I0 cards, bảo toàn Midterm,
   kiểm GPU toolchain và chốt owners file chung.
2. **Tiến:** mở Mục 16.1 + 18.3; kiểm dataset/site/overlap/provided, tạo contact
   sheet và data audit, chưa đổi ảnh chuẩn sau khi lock nếu không có review.
3. **Tuấn:** mở Mục 17.2 + 18.4 + 20; lưu env/test logs, thiết kế raw-run schema
   và validator, chuẩn bị notebook/package Final skeleton.

Cả ba đọc timeline Mục 19 và checklist Mục 21 trước khi nhận task. Làm tay hay
dùng agent đều dùng cùng task card Mục 22; reviewer nghiệm thu theo artifact.

### 24.2 Chín câu hỏi của nhóm trưởng trước khi chốt mỗi bản

- [ ] Bản này nhận input từ bản/gate nào, SHA và config đã khóa chưa?
- [ ] Có thực sự chạy sáu stage cần thiết hay đang gọi fixture/CPU thay thế?
- [ ] Methods/thresholds/RNG/order/fallback có giống V0 và parent?
- [ ] V2/V3 có đúng một ý tưởng chính và evidence bottleneck của parent?
- [ ] Test đã chạy ở đâu, actual value bao nhiêu, log ở đâu?
- [ ] E2E có tính transfer/control/allocation đúng policy và loại disk I/O?
- [ ] Có case fail/low-confidence/ảnh nối xấu bị bỏ khỏi report không?
- [ ] Thành viên khác có chạy lại và giải thích code được không?
- [ ] Tài liệu/source/notebook/artifacts có cùng phiên bản không?

### 24.3 Nguồn dùng để soạn hướng dẫn

- Đề gốc `CSC14120_2026_Final Project.pdf`, toàn bộ 9 trang: dataset, version
  rules, measurement, deliverables, report structure và academic integrity.
- Code/test/manifest/Makefile thực tại commit mở đầu đã ghi phía trên; D01
  luôn ghi lại SHA hiện hành, không tự coi nhận xét lịch sử là test mới.
- [CUDA Best Practices Guide — APOD](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html):
  kiểm bottleneck, port CUDA, đo và tối ưu theo chu kỳ.
- Tài liệu NVIDIA về Nsight/Compute Sanitizer được dẫn tại Mục 20.7; các câu
  lệnh phải kiểm `--help`/version trên môi trường cài thật trước khi dùng.

Hướng dẫn này là sản phẩm lập kế hoạch. V1/V2/V3, harness và file Final mới
được mô tả như công việc phải làm; chưa được đánh dấu đã triển khai chỉ vì
tài liệu hoặc prompt đã tồn tại.

---
