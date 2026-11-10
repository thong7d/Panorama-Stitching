# TÀI LIỆU BÀN GIAO KỸ THUẬT CHO TUẤN (PHASE C)
**Dự án:** Ghép ảnh Panorama trên GPU bằng CUDA (CSC14120 — Lập trình song song)  
**Người bàn giao:** Phạm Minh Thông (23120364) — Pipeline Lead (Phụ trách S1/S5, CudaWorkspace, Bounds, Dispatcher)  
**Người nhận bàn giao:** Đào Quốc Tuấn (23120392) — Phụ trách Stage 4, Stage 6 & Benchmark/Packaging  
**Baseline SHA:** `cba0e603f16243376d28b2f52c6335fe467b09f7`  
**Ngày cập nhật:** 10/11/2026  
**Trạng thái Review:** PENDING (Đang chờ Tuấn xác nhận và triển khai Stage 4 & 6)

---

## 1. Mục tiêu và Phạm vi công việc của Tuấn

Theo phân công tại [project_plan_v4.md](../project_plan_v4.md#L657) Mục 18:
1. **Stage 4 (Khớp Descriptor bằng khoảng cách Hamming + Lowe's Ratio trên GPU):**
   - Tạo mới tập tin `src/cuda_stage4_matching.cu`.
   - Triển khai kernel Brute-force Hamming matching dùng lệnh nội tại `__popc()` trên 8 words (256-bit).
   - Áp dụng kiểm tra Lowe's ratio nghiêm ngặt $\text{best1} < \text{RATIO\_THRESH} \times \text{best2}$ với $\text{RATIO\_THRESH} = 0.75f$.
2. **Stage 6 (Chiếu tọa độ Inverse Warping + Bilinear Sampling + Feathering Blend trên GPU):**
   - Tạo mới tập tin `src/cuda_stage6_warp.cu`.
   - Nhận ma trận $H_{A\to B}$ (đã được Stage 5 GPU của Thông tính toán) và `CanvasBounds` đã kiểm tra an toàn.
   - Inverse Warping từ tọa độ canvas frame A sang ảnh B; lấy mẫu song tuyến tính (Bilinear Interpolation); hòa trộn bằng trọng số khoảng cách đối xứng (Symmetric Feathering).
3. **Cập nhật Makefile:**
   - Thêm `src/cuda_stage4_matching.cu` và `src/cuda_stage6_warp.cu` vào `CUDA_LIB_SRCS` trong [`Makefile`](../Makefile). Thay đổi file chung do Thông điều phối.
4. **Thay thế CPU Stub trong [`src/cuda_pipeline.cu`](../src/cuda_pipeline.cu):**
   - Cắm các kernel GPU trực tiếp vào pipeline để loại bỏ hoàn toàn các cầu nối copy trung gian giữa CPU và GPU, đưa V1 lên trạng thái **Full V1 Basic CUDA**.

---

## 2. Hợp đồng Kỹ thuật và API cho Stage 4 (Hamming Matching)

### 2.1. Signature hàm bắt buộc (đã khai báo trong [`src/cuda_pipeline.cuh`](../src/cuda_pipeline.cuh))
```cpp
float stage4_matching_gpu(const Descriptor* d_descA,
                          int               nA,
                          const Descriptor* d_descB,
                          int               nB,
                          Match*            d_matches,
                          int*              d_nmatches,
                          float             ratio_threshold = RATIO_THRESH,
                          cudaStream_t      stream = 0);
```

### 2.2. Dữ liệu đầu vào và đầu ra
- **Đầu vào:**
  - `d_descA`, `d_descB`: Con trỏ mảng descriptor trên GPU (`Descriptor*`). Lấy từ `ws.d_descA` và `ws.d_descB`.
  - `nA, nB`: Số lượng descriptors ảnh A và B ($\le 2000$).
  - `ratio_threshold`: Ngưỡng tỉ lệ khoảng cách của Lowe (mặc định $\text{RATIO\_THRESH} = 0.75f$ định nghĩa tại [`src/common.h`](../src/common.h)).
- **Đầu ra:**
  - `d_matches`: Mảng lưu các cặp khớp tìm được (`Match*`, capacity `MAX_MATCHES = 2000`). Nằm tại `ws.d_matches`.
  - `d_nmatches`: Con trỏ int scalar trên device lưu số lượng cặp khớp thực tế. Nằm tại `ws.d_nmatches`.
  - Hàm trả về thời gian GPU đo bằng `cudaEvent` (ms).

### 2.3. Bất biến thuật toán Stage 4 (Khớp 100% V0 CPU)
- **Khoảng cách Hamming:** Giữa hai descriptor $D_1, D_2$ tính bằng tổng số bit khác biệt trên 8 words:
  $$\text{dist} = \sum_{k=0}^{7} \text{\_\_popc}(D_1.\text{words}[k] \oplus D_2.\text{words}[k])$$
- **Tìm kiếm lân cận gần nhất:** Tìm `best1_dist` (khoảng cách nhỏ nhất) và `best2_dist` (khoảng cách nhỏ nhì) từ mỗi keypoint $A$ sang toàn bộ keypoints $B$.
- **Lowe's Ratio Test nghiêm ngặt:**
  $$\text{best1\_dist} < \text{RATIO\_THRESH} \times \text{best2\_dist}$$
- **Quy tắc xử lý Tie và Trường hợp biên (Edge Cases):**
  - **Trường hợp bằng nhau (Tie):** Nếu $\text{best1\_dist} == \text{best2\_dist}$, bất đẳng thức nghiêm ngặt $\text{best1} < 0.75 \times \text{best2}$ là **SAI** (vì $0.75 \times \text{best2} < \text{best2}$). Do đó điểm này **bị loại** (không tạo match), khớp 100% với V0 CPU.
  - **Trường hợp $n_B = 0$:** Không có điểm nào trong $B \implies$ trả về $M = 0$.
  - **Trường hợp $n_B = 1$:** Chỉ có 1 điểm trong $B$, không thể xác định `best2_dist` $\implies$ theo V0 CPU không thể thỏa mãn ratio test $\implies$ trả về $M = 0$.
- **Thứ tự xuất mảng `d_matches`:** Phải được sắp xếp theo thứ tự chỉ số $idx\_a$ tăng dần để Stage 5 GPU nhìn thấy cùng thứ tự như V0 CPU.

---

## 3. Hợp đồng Kỹ thuật và API cho Stage 6 (Warping & Blending)

### 3.1. Signature hàm bắt buộc (đã khai báo trong [`src/cuda_pipeline.cuh`](../src/cuda_pipeline.cuh))
```cpp
float stage6_warp_blend_gpu(const uchar4*       d_imgA,
                            const uchar4*       d_imgB,
                            int                 img_w,
                            int                 img_h,
                            const Homography&   H_A_to_B,
                            const CanvasBounds& bounds,
                            uchar4*             d_pano,
                            uint8_t*            d_pano_mask = nullptr,
                            cudaStream_t        stream = 0);
```

### 3.2. Quy ước hệ tọa độ Canvas và Giới hạn an toàn (Canvas Contract)
- **Hệ quy chiếu:** Canvas nằm trọn vẹn trong hệ tọa độ **frame A**.
- **Cấu trúc `CanvasBounds`:**
  - `x_min, y_min`: Tọa độ góc trên bên trái của canvas trong frame A (có thể âm nếu B dịch sang trái/lên trên).
  - `x_max, y_max`: Tọa độ góc dưới bên phải.
  - `width = x_max - x_min`, `height = y_max - y_min`.
  - `total_pixels = width * height`.
- **Đảm bảo bộ nhớ:** `bounds.total_pixels <= ws.pano_capacity` đã được kiểm tra nghiêm ngặt trước khi gọi hàm. Dung lượng `pano_capacity = N * 8` ($7,372,800\text{ px}$ cho 720p) đảm bảo đủ cho mọi ma trận $H$ hợp lệ, kể cả fixture $3200 \times 1200 = 3,840,000\text{ px}$.
- **Buffer đích:** `d_pano` là mảng `uchar4` có kích thước ít nhất `bounds.total_pixels`. Pixel tại vị trí $(cx, cy)$ của canvas tương ứng với chỉ số `cy * bounds.width + cx`.
- **Mask hợp lệ riêng biệt (`d_pano_mask`):** `d_pano_mask` kiểu `uint8_t` lưu giá trị 1 nếu pixel được chiếu từ A hoặc B, 0 nếu là nền ngoài canvas. Không dùng màu RGB=(0,0,0) để suy luận mask vì pixel ảnh gốc hoàn toàn có thể có màu đen hợp lệ.

### 3.3. Thuật toán Inverse Warping & Symmetric Feathering (Khớp 100% V0)
- Với mỗi pixel $(cx, cy)$ trên canvas:
  - Tọa độ frame A: $ax = cx + bounds.x\_min, \quad ay = cy + bounds.y\_min$.
  - Kiểm tra thuộc ảnh A: $in\_A = (0 \le ax < img\_w \text{ and } 0 \le ay < img\_h)$.
  - Chiếu sang ảnh B bằng $H_{A\to B}$:
    $$w = H[6] \cdot ax + H[7] \cdot ay + H[8]$$
    $$bx = (H[0] \cdot ax + H[1] \cdot ay + H[2]) / w, \quad by = (H[3] \cdot ax + H[4] \cdot ay + H[5]) / w$$
  - Kiểm tra thuộc ảnh B: $in\_B = (0 \le bx < img\_w - 1 \text{ and } 0 \le by < img\_h - 1)$. Nếu thuộc, lấy mẫu song tuyến tính (Bilinear Interpolation).
  - **Trộn màu (Feathering):**
    - Nếu chỉ thuộc A: lấy màu A.
    - Nếu chỉ thuộc B: lấy màu B.
    - Nếu thuộc cả A và B: tính khoảng cách tới 4 biên:
      $$d_A = \min(ax, img\_w - 1 - ax, ay, img\_h - 1 - ay)$$
      $$d_B = \min(bx, img\_w - 1 - bx, by, img\_h - 1 - by)$$
      $$\alpha = \frac{d_A + 10^{-6}}{d_A + d_B + 2 \times 10^{-6}}$$
      $$\text{color} = \alpha \cdot \text{col}_A + (1 - \alpha) \cdot \text{col}_B$$

---

## 4. Hướng dẫn Biên dịch và Kiểm thử trên Google Colab (Tesla T4)

Sử dụng môi trường Colab tài khoản `thongphamfff@gmail.com`:

```bash
# 1. Kiểm tra trạng thái git local và biên dịch test suite
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

- [x] Hạ tầng `CudaWorkspace`, Stage 1, Stage 2 và Stage 5 (RANSAC GPU) đã hoàn thành và đạt chuẩn THONG_HANDOFF_READY.
- [x] Canvas Bounds Contract và Fixtures kiểm tra an toàn bộ nhớ ($2,156,544\text{ px}$ và $3,840,000\text{ px} \le \text{pano\_capacity}$) đã được thiết kế và kiểm thử thành công.
- [x] Các API declaration của Stage 4 và Stage 6 đã sẵn sàng trong `src/cuda_pipeline.cuh`.
- [ ] **Việc Tuấn cần làm:** Tạo `src/cuda_stage4_matching.cu` (triển khai `stage4_matching_gpu` với strict ratio test và tie-break).
- [ ] **Việc Tuấn cần làm:** Tạo `src/cuda_stage6_warp.cu` (triển khai `stage6_warp_blend_gpu` với inverse warping và feathering).
- [ ] **Việc Tuấn cần làm:** Cập nhật `CUDA_LIB_SRCS` trong `Makefile`.
- [ ] **Việc Tuấn cần làm:** Thay thế các CPU stubs trong `src/cuda_pipeline.cu`, chạy benchmark Full V1 và thu thập Nsight profiling metrics cho Phase D.
- [ ] **Reviewer:** PENDING (Thông & Tiến sẽ review khi Tuấn nộp PR).
