# PROJECT PLAN v4 — CUDA-Accelerated Panorama Image Stitching

> **Trạng thái:** Nguồn kế hoạch duy nhất của dự án từ ngày 19/10/2026  
> **Môn học:** CSC14120 — Parallel Programming  
> **Chủ đề:** Panorama Image Stitching với FAST-9, Oriented BRIEF/ORB, Hamming Matching, RANSAC Homography và CUDA  
> **Phạm vi:** Midterm, V0 Sequential, V1 Basic CUDA, ít nhất hai phiên bản tối ưu, Final Report và Oral Defense

File này hợp nhất các nội dung còn giá trị của plan v3 với kết quả audit kỹ thuật trong v4. Khi có mâu thuẫn với tài liệu cũ, **quyết định trong file này được ưu tiên**. Không triển khai theo các version/tối ưu cũ nếu chưa vượt qua quality gate tương ứng.

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
   ↓ optional V4 nếu còn thời gian
Phase E  Full benchmark + Final report + video + oral defense
```

---

## 2. Hiện trạng codebase và các lỗi phải xử lý

### 2.1 Thành phần hiện có

- `src/`: V0 sequential, CLI, common types, CUDA timer wrapper và stb image I/O.
- `scripts/`: preprocess ảnh và tạo synthetic pair.
- `notebooks/GroupXX_Midterm.ipynb`: notebook khung nhưng chưa chạy.
- `dataset/`: ba cặp HEIC tại ba site; `challenging/` và `provided/` trống.
- `output/`: trống.
- Chưa có V1/V2/V3 CUDA implementation và chưa có test suite.
- Workspace hiện không phải Git repository và máy local không có C++/CUDA compiler; benchmark chính thức phải chạy trên Colab T4.

### 2.2 Lỗi chặn correctness/reproducibility

- Notebook hiện sai JSON do escape `\sigma`, dùng thông tin nhóm mẫu và toàn bộ output rỗng.
- Script preprocess không nhận tên `In1/In2`, `Out1/Out2`, `Free1/Free2`; môi trường hiện chưa có HEIC plugin.
- Synthetic generator chưa lưu ground-truth homography.
- ORB table hiện có 256 entries nhưng chỉ 133 cặp duy nhất.
- V0 có thể truyền `H` chưa khởi tạo sang warping khi RANSAC thất bại.
- `Image`/`GrayImage` sở hữu raw pointer nhưng chưa có copy/move contract an toàn.
- Canvas hiện bị cắt kích thước âm thầm và thiếu kiểm tra finite/overflow.
- Top-K chưa có tie-break deterministic giữa CPU và GPU.
- `--runs` mặc định là 1, trái yêu cầu mặc định 5.
- Amdahl hiện được tính từ tổng stage chia chính tổng stage, dẫn tới kết quả giả `999×`.
- `MAX_CANDIDATES=500000` không phải bảo đảm toán học và vẫn có thể overflow.
- Plan cũ chọn sẵn tối ưu trước profiling và gộp nhiều thay đổi trong cùng version.

Tất cả mục trên thuộc Phase A và phải được giải quyết trước khi gọi V0 là “golden baseline”.

---

## 3. Phương pháp và tham số bất biến giữa các version

| Stage | Phương pháp | Tham số/contract cố định |
|---|---|---|
| 1. Preprocess | RGB→grayscale, Gaussian blur 5×5 | `Y=0.299R+0.587G+0.114B`, σ=1.0, boundary clamp |
| 2. Detect | FAST-9, NMS 3×3, Top-K | threshold=20, K=1000 tối đa, sort response giảm dần rồi linear index tăng dần |
| 3. Describe | Oriented BRIEF/ORB 256-bit | intensity centroid, pattern 256 cặp đã kiểm chứng, loại keypoint không đủ border |
| 4. Match | Brute-force Hamming + Lowe ratio | `best1 < 0.75 × best2`, một chiều A→B |
| 5. Homography | RANSAC + normalized 4-point DLT | 2000 hypotheses, reprojection threshold=3 px, seed=42, partial pivoting |
| 6. Warp | B được warp vào frame A | inverse mapping, bilinear sampling, symmetric feathering |

### 3.1 Quy ước hình học

- `H_A_to_B` ánh xạ điểm ảnh A sang tọa độ ảnh B.
- Canvas nằm trong frame A; để tìm biên panorama, các góc B được ánh xạ về A bằng `inverse(H_A_to_B)`.
- Khi render một pixel canvas/A, dùng `H_A_to_B` để lấy mẫu từ B.
- Homography metrics dùng projected-grid error, không so trực tiếp chín hệ số chưa chuẩn hóa.

### 3.2 Quy ước quality

- Hard failure nếu có dưới 4 matches/inliers, H không hữu hạn, H không khả nghịch hoặc canvas không hợp lệ.
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
- `Homography`: lưu `H_A_to_B[9]`, có hàm normalize/finite/invertibility validation.
- `PipelineDiagnostics`: keypoint A/B, matches, inliers, inlier ratio, H, status và warning.
- `TimingSummary`: Avg/Min cho sáu stage, kernel total, H2D, control copy, D2H, end-to-end và one-time setup.
- `PipelineResult`: panorama + diagnostics + timings.
- Dispatcher công khai: `run_pipeline_version(version, inputs, config)`; các version dùng chung orchestration thay vì copy toàn bộ pipeline.

### 4.3 JSON metrics schema

Mỗi benchmark tạo một JSON gồm:

```text
schema_version, timestamp, version, parent_version, optimization,
device_name, image_width, image_height, runs, warmups,
status, warning, keypoints_a, keypoints_b, matches, inliers,
homography_a_to_b[9],
timing_ms.{stage1..stage6,kernel_total,h2d,control,d2h,end_to_end,setup}.{avg,min}
```

Notebook chỉ đọc JSON để lập bảng/biểu đồ; không copy số liệu bằng tay.

### 4.4 Timing contract

- V0: `std::chrono::high_resolution_clock`.
- CUDA: `cudaEventRecord/cudaEventElapsedTime` cho kernel và transfer; CPU wall timer có synchronization cho end-to-end.
- Một warm-up không ghi vào thống kê; tối thiểu năm timed runs.
- Exclude disk load/save; report setup allocation riêng.
- Preallocate và reuse workspace cho cả CPU lẫn GPU để so steady-state công bằng.
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
│   ├── GroupXX_Midterm.ipynb
│   └── GroupXX_Final.ipynb
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

Manifest mô tả rõ từng pair, không suy luận theo chuỗi `left/right` trong tên file:

```json
{
  pairs: [
    {id:site1_indoor,kind:standard,left_raw:In1.heic,right_raw:In2.heic},
    {id:site2_outdoor,kind:standard,left_raw:Out1.heic,right_raw:Out2.heic},
    {id:site3_custom,kind:standard,left_raw:Free1.heic,right_raw:Free2.heic}
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
7. RANSAC dùng stateless PRNG giống nhau trên CPU/GPU, seed 42, bốn sample index phân biệt và tie-break hypothesis ID nhỏ nhất.
8. Nếu RANSAC fail, trả status; tuyệt đối không truyền stack `H` chưa khởi tạo.
9. Stage 6 validate canvas, bilinear bounds và dùng feather weights dựa trên khoảng cách tới biên của cả A và B.
10. Tách diagnostics/timings khỏi stdout để notebook sử dụng JSON.

### A4. Benchmark V0 và Amdahl

Chạy synthetic và ba site ở Small/Medium/Large. Với bảng Midterm chính, report sáu stage:

| Stage | 720p Avg/Min | 1080p Avg/Min | 4K Avg/Min |
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
p = tổng thời gian các stage dự kiến chuyển GPU / V0 end-to-end
S_max = 1 / (1 - p)
```

Nếu serial fraction nhỏ hơn độ phân giải đo, report “idealized unbounded / below measurement resolution” và giải thích transfer/setup thực tế sẽ giới hạn speedup; không in `999×`.

### Quality gate A

- CPU build thành công trên Colab và không warning nghiêm trọng.
- Synthetic Small grid-RMSE so với GT ≤1 px; normalized RMSE mọi size ≤0,1% đường chéo.
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
| Profiling | Per-stage Avg/Min ở ba resolution, hot-stage analysis |
| Amdahl | p, giới hạn lý thuyết và hạn chế của ước lượng |
| Dataset | Registered sites, prepared pairs, challenging pair và Drive link |
| GPU plan | Stage mapping V1; candidate tối ưu được trình bày là giả thuyết chờ profiling |

### B2. Notebook execution contract

- Tạo notebook bằng `nbformat`, không viết JSON thủ công.
- Setup cell dùng đúng `src/` trong package; nếu thiếu thì prompt upload zip một lần.
- Cài dependency, tạo output, generate synthetic, build, run benchmark, parse JSON, vẽ bảng/biểu đồ tự động.
- `Runtime → Run all` trên fresh Colab T4 thành công sau khi source/dataset được cung cấp.
- Mọi output được lưu trong file nộp; mọi số trong markdown xuất phát từ output cell.
- Thay toàn bộ `GroupXX`, tên/MSSV mẫu và link placeholder trước khi nộp.

### B3. Package

```text
GroupXX_Midterm.zip
├── GroupXX_Midterm.ipynb
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

- Notebook hợp lệ theo `nbformat`, Run All pass và output saved.
- V0 correctness, panorama, timing, hot stages và Amdahl đầy đủ.
- Zip đúng tên/cấu trúc và build từ source duy nhất.
- Không bắt đầu V1 trước khi gate này đạt hoặc Midterm đã được nộp an toàn.

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

- Timing Avg/Min: từng stage, kernel total, H2D, control, D2H, end-to-end.
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
2. Profiler xác nhận đúng loại bottleneck dự kiến.

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

### D5. Optional versions

Chỉ làm khi V0–V3 hoàn chỉnh và còn đủ thời gian:

- CUDA streams cho hai nhánh A/B là một version riêng.
- Grayscale+blur fusion là một version riêng khác.
- Texture-object bilinear warp là một version riêng nếu Stage 6 được profile là bottleneck.
- Mỗi version vẫn phải verify V0 và report speedup-vs-previous, kể cả khi chậm hơn.

### Five-part report template cho mỗi optimized version

1. Bottleneck: số liệu của version trước.
2. Idea: một tối ưu và lý do phần cứng.
3. Implementation: thay đổi kernel/data movement.
4. Result: correctness, Avg/Min, speedup-vs-V0 và speedup-vs-previous.
5. Analysis: kỳ vọng so với thực tế, profiler explanation và trade-off.

---

## 10. Phase E — Final integration, report và submission

### E1. Benchmark matrix

Trong cùng một Colab T4 session:

```text
versions: V0, V1, V2, V3, [optional versions]
sizes:    Small, Medium, Large
inputs:   synthetic + standard sites; main speedup table dùng scene cố định
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

- `GroupXX_Final.ipynb` với outputs saved và Run All pass.
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
- Avg/Min hữu hạn, Min ≤ Avg.
- Kernel total nhất quán với tổng stage trong tolerance đo.
- End-to-end bao gồm transfer/control đã khai báo và loại disk I/O.
- Không đặt speedup tối thiểu bắt buộc: version chậm hơn vẫn hợp lệ nếu correctness giữ nguyên và analysis trung thực.

---

## 12. Phân công nhóm và timeline theo gate

### 12.1 Vai trò đề xuất cho ba thành viên

| Vai trò | Trách nhiệm chính |
|---|---|
| Member A — Pipeline lead | Stages 1–3, dataset preprocessing, ORB/FAST correctness, candidate blur |
| Member B — CUDA lead | Stages 4–6, RANSAC/warp/matching kernels, profiler và memory safety |
| Member C — Integration/report lead | Notebook automation, benchmark JSON, charts, README, slides, video và package |

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

Không gắn ngày giả khi chưa có lịch môn học chính thức. Khi biết deadline, đặt ngày nội bộ cho mỗi milestone trước deadline thật tối thiểu 2–3 ngày.

---

## 13. Rủi ro và biện pháp

| Rủi ro | Mức | Biện pháp bắt buộc |
|---|---|---|
| Dataset provided chưa có | Cao | Theo dõi nguồn môn học; blocker Final Gallery, không giả lập thay thế |
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

- [ ] Dataset giảng viên trong `dataset/provided/`.
- [ ] Challenging pair của nhóm.
- [ ] Tên/MSSV/GroupXX thật và dataset Drive link.
- [ ] Deadline chính thức để khóa lịch nội bộ.

### Thứ tự công việc sau khi plan được duyệt

- [ ] A0 — dependency, Makefile, provenance contract.
- [ ] A1 — manifest + HEIC preprocessing.
- [ ] A2 — synthetic ground truth.
- [ ] A3 — sửa core types và sáu stage V0.
- [ ] A4 — tests + V0 benchmark/metrics.
- [ ] Gate A.
- [ ] Phase B Midterm notebook/package.
- [ ] Gate B.
- [ ] Phase C V1 CUDA.
- [ ] Gate C + profiling.
- [ ] Phase D V2 rồi V3 theo profiling.
- [ ] Phase E Final integration/submission.

### Definition of Done toàn dự án

- V0, V1 và ít nhất hai optimized versions build từ một source tree.
- Mọi version đạt correctness contract và không có memory/CUDA error.
- Benchmark đủ ba resolution, cùng GPU/session, có Avg/Min và hai loại speedup.
- Mỗi optimized version có profiling evidence và đúng một thay đổi chính.
- Notebook Midterm/Final Run All, output saved và không có số liệu điền tay.
- Gallery, failure analysis, provided dataset, README, PDF, slides, video và attribution đầy đủ.

---


