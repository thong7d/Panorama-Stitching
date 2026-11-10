# TÀI LIỆU BÀN GIAO KỸ THUẬT CHO TIẾN (PHASE C)
**Dự án:** Ghép ảnh Panorama trên GPU bằng CUDA (CSC14120 — Lập trình song song)  
**Người bàn giao:** Phạm Minh Thông (23120364) — Pipeline Lead (Phụ trách S1/S5, CudaWorkspace, Bounds, Dispatcher)  
**Người nhận bàn giao:** Nguyễn Lê Tấn Tiến (23120369) — Phụ trách Stage 2 (tiếp quản) & Stage 3 (triển khai mới)  
**Baseline SHA:** `cba0e603f16243376d28b2f52c6335fe467b09f7`  
**Ngày cập nhật:** 10/11/2026  
**Trạng thái Review:** PENDING (Đang chờ Tiến xác nhận và triển khai Stage 3)

---

## 1. Mục tiêu và Phạm vi công việc của Tiến

Theo phân công tại [project_plan_v4.md](../project_plan_v4.md#L656) Mục 18:
1. **Stage 2 (FAST-9 + 3x3 NMS + Deterministic Top-K):**
   - Đã được Thông viết nền tảng hoàn chỉnh trong [`src/cuda_stage2_fast_nms_topk.cu`](../src/cuda_stage2_fast_nms_topk.cu).
   - Tiến tiếp quản file này để bảo trì, kiểm tra trung gian và làm căn cứ cho Stage 3.
   - Không có cấp phát động `cudaMalloc` trong vòng lặp; CUB scratch `d_sort_tmp` được cấp phát tĩnh từ `CudaWorkspace`.
2. **Stage 3 (Trích xuất ORB Descriptors 256-bit trên GPU):**
   - Tiến tạo mới tập tin `src/cuda_stage3_orb.cu`.
   - Triển khai kernel trích xuất vector đặc trưng nhị phân 256-bit bằng CUDA thay thế CPU stub hiện tại trong [`src/cuda_pipeline.cu`](../src/cuda_pipeline.cu).
   - Thêm `src/cuda_stage3_orb.cu` vào biến `CUDA_LIB_SRCS` trong [`Makefile`](../Makefile). Thay đổi file chung do Thông điều phối.

---

## 2. Hợp đồng Kỹ thuật và API cho Stage 3 (CUDA Contract)

### 2.1. Signature hàm bắt buộc (đã khai báo trong [`src/cuda_pipeline.cuh`](../src/cuda_pipeline.cuh))
```cpp
float stage3_orb_descriptors_gpu(const float*    d_blur,
                                 const Keypoint* d_kp,
                                 int             n_kp,
                                 Descriptor*     d_desc,
                                 int w, int h,
                                 cudaStream_t stream = 0);
```

### 2.2. Dữ liệu đầu vào và đầu ra
- **Đầu vào:**
  - `d_blur`: Ảnh xám sau khi làm mịn Gaussian 5x5 trên device (`float*`, kích thước $w \times h$). Nằm tại `ws.d_blurA` hoặc `ws.d_blurB`.
  - `d_kp`: Mảng điểm đặc trưng sau Top-K trên device (`Keypoint*`, tối đa `MAX_KEYPOINTS = 2000`). Nằm tại `ws.d_kpA` hoặc `ws.d_kpB`.
  - `n_kp`: Số lượng điểm đặc trưng thực tế cần tính toán ($\le K$). Lấy từ scalar `ws.d_nkpA` hoặc `ws.d_nkpB`.
  - `w, h`: Kích thước ảnh gốc (ví dụ $1280 \times 720$).
- **Đầu ra:**
  - `d_desc`: Mảng lưu kết quả descriptor trên device (`Descriptor*`, tối đa `MAX_DESCRIPTORS = 2000`). Nằm tại `ws.d_descA` hoặc `ws.d_descB`.
  - Hàm trả về thời gian thực thi kernel GPU tính bằng mili-giây (đo bằng `cudaEvent`).

### 2.3. Cấu trúc dữ liệu và Bất biến thuật toán
- **Struct `Descriptor`:** Định nghĩa tại [`src/common.h`](../src/common.h):
  - Kích thước 32 bytes, `alignas(16)`.
  - Gồm 8 từ 32-bit: `uint32_t words[8]`.
  - Bit thứ $i$ ($0 \le i < 256$) tương ứng với cặp kiểm tra thứ $i$ trong bảng `ORB_PATTERN_PAIRS[256][2]`.
- **Bất biến thuật toán (Bắt buộc khớp 100% V0):**
  - **Góc định hướng:** Tính tâm cường độ (Intensity Centroid) trên patch tròn bán kính $R=15$:
    $$m_{10} = \sum x \cdot I(x, y), \quad m_{01} = \sum y \cdot I(x, y), \quad \theta = \text{atan2}(m_{01}, m_{10})$$
  - **Lấy mẫu xoay — QUY TẮC LÀM TRÒN BẮT BUỘC:**
    Tọa độ thực của điểm lấy mẫu sau khi xoay góc $\theta$:
    $$u = kp.x + (x \cos\theta - y \sin\theta) = kp.x + \text{rot\_offset\_x}$$
    $$v = kp.y + (x \sin\theta + y \cos\theta) = kp.y + \text{rot\_offset\_y}$$
    Lấy mẫu pixel: **BẮT BUỘC** làm tròn tổng rồi mới clamp:
    $$\text{sample\_x} = \text{std::clamp}(\text{static\_cast<int>}(\text{std::round}(u)), 0, w - 1)$$
    $$\text{sample\_y} = \text{std::clamp}(\text{static\_cast<int>}(\text{std::round}(v)), 0, h - 1)$$
    > [!IMPORTANT]
    > **Tuyệt đối KHÔNG dùng** $kp.x + \text{round}(\text{rot\_offset\_x})$. Vì $kp.x$ có thể là số thực (sub-pixel), việc cộng rồi làm tròn $\text{round}(kp + \text{offset})$ khác với $kp + \text{round}(\text{offset})$ và sẽ làm sai lệch bit descriptor!
  - **Lưu ý số học trên GPU:** Hàm `roundf()` trên CUDA có thể làm tròn half-integer khác với `std::round` nếu không kiểm soát rounding mode. Cần kiểm tra kỹ các trường hợp half-integer âm (ví dụ $-0.5 \to -1.0$, $-1.5 \to -2.0$) và các góc xoay ($0, \pi/4, \pi/2, \pi, -\pi/3$).
  - **So sánh bit:**
    $$\text{bit} = (I(\text{sample\_1}) < I(\text{sample\_2})) \ ? \ 1 : 0$$
    Bit được ghi vào từ `words[i / 32] |= (bit << (i % 32))`.

---

## 3. Chỉ dẫn Tích hợp Từng bước (Khi S3 có GPU nhưng S4 còn CPU Stub)

Trong giai đoạn chuyển tiếp khi Tiến hoàn thành S3 trên GPU nhưng Tuấn chưa hoàn thành S4 trên GPU:
1. **Debug Bridge:** Trong [`src/cuda_pipeline.cu`](../src/cuda_pipeline.cu), sau khi gọi `stage3_orb_descriptors_gpu`, **phải sao chép `ws.d_descA` và `ws.d_descB` về host** để CPU stub của Stage 4 có dữ liệu thực hiện khớp:
   ```cpp
   // [DEBUG BRIDGE: S3 GPU -> S4 CPU Stub (Giai đoạn chuyển tiếp)]
   // Cần sao chép descriptor từ GPU về host để CPU stub Stage 4 hoạt động
   std::vector<Descriptor> descA(nkpA), descB(nkpB);
   if (nkpA > 0) CHECK_CUDA(cudaMemcpy(descA.data(), ws.d_descA, nkpA * sizeof(Descriptor), cudaMemcpyDeviceToHost));
   if (nkpB > 0) CHECK_CUDA(cudaMemcpy(descB.data(), ws.d_descB, nkpB * sizeof(Descriptor), cudaMemcpyDeviceToHost));
   ```
   > [!WARNING]
   > Không được để `descA` và `descB` rỗng trên host; nếu không, Stage 4 CPU stub sẽ nhận 0 descriptor và toàn bộ pipeline sẽ trả về 0 matches!
2. **Cập nhật Makefile:**
   Thêm `src/cuda_stage3_orb.cu` vào `CUDA_LIB_SRCS` trong [`Makefile`](../Makefile):
   ```makefile
   CUDA_LIB_SRCS = src/cuda_pipeline.cu \
                   src/cuda_stage1_preprocess.cu \
                   src/cuda_stage2_fast_nms_topk.cu \
                   src/cuda_stage3_orb.cu \
                   src/cuda_stage5_ransac.cu
   ```

---

## 4. Hướng dẫn Biên dịch và Kiểm thử trên Google Colab (Tesla T4)

Sử dụng môi trường Colab tài khoản `thongphamfff@gmail.com`:

```bash
# 1. Kiểm tra trạng thái git local và biên dịch
make clean && make test_cuda

# 2. Chạy bộ unit tests xác minh
./test_cuda

# 3. Biên dịch binary chính
make cuda

# 4. Kiểm tra an toàn bộ nhớ GPU (Gate G1 yêu cầu 0 errors, 0 leak)
compute-sanitizer --tool memcheck ./stitch \
    dataset/synthetic/left_small.jpg dataset/synthetic/right_small.jpg \
    /tmp/test.jpg --version 1 --runs 1
```

---

## 5. Checklist Bàn giao và Tự kiểm

- [x] Nền tảng Stage 2 (FAST-9 + NMS + Top-K) hoàn chỉnh, keypoint A=852, B=917 khớp tuyệt đối V0 trên `synthetic/small`.
- [x] Không có dynamic allocation trong vòng lặp (CUB scratch dùng buffer tĩnh `ws.d_sort_tmp`).
- [x] CudaWorkspace có move constructor/assignment `noexcept` và idempotent `free_all()`.
- [x] Pano capacity mở rộng an toàn $8N$ ($7,372,800\text{ px}$ cho 720p), hỗ trợ mọi fixture hình học hợp lệ.
- [ ] **Việc Tiến cần làm:** Tạo `src/cuda_stage3_orb.cu`, triển khai `stage3_orb_descriptors_gpu`.
- [ ] **Việc Tiến cần làm:** Đảm bảo công thức làm tròn $\text{clamp}(\text{round}(kp + \text{rot}))$, kiểm thử fixture half-integer âm và góc xoay.
- [ ] **Việc Tiến cần làm:** So sánh descriptor bit-by-bit (256 bits) với V0 CPU stub trên cùng tập keypoints.
- [ ] **Reviewer:** PENDING (Thông & Tuấn sẽ review khi Tiến nộp PR).
