// ============================================================================
// CSC14120 - Lập trình song song
// Đồ án: Ghép ảnh Panorama
// Module: Stage 5 CUDA Kernels — RANSAC Homography + Fallback Translation
// Tác giả: Pham Minh Thong (23120364)
// Phụ trách: Stage 5 GPU kernels (ước lượng ma trận Homography và Fallback)
// External sources: Thuật toán Normalized DLT (Hartley & Zisserman, 2004);
//   RANSAC (Fischler & Bolles, 1981); mã nguồn gốc đồ án tại v0_sequential.cpp.
// ============================================================================

#include "cuda_pipeline.cuh"
#include <cmath>

// ============================================================================
// PRNG LCG trên GPU cho RANSAC tất định (Seed 42)
// Khớp 100% với chuỗi số sinh từ LcgRng của V0 (v0_sequential.cpp dòng 692-705)
// ============================================================================
struct LcgRngDev {
    uint32_t state;
    __device__ explicit LcgRngDev(uint32_t seed = 42) : state(seed) {}
    __device__ uint32_t next() {
        state = state * 1664525u + 1013904223u;
        return state;
    }
    __device__ int next_int(int max_val) {
        if (max_val <= 0) return 0;
        return (int)(next() % (uint32_t)max_val);
    }
};

// ============================================================================
// Hartley Isotropic Normalization cho 4 điểm trên GPU
// Khớp chính xác V0 (v0_sequential.cpp dòng 434-463)
// ============================================================================
__device__ void compute_hartley_norm_dev(
    const float pts[4][2],
    float norm_pts[4][2],
    float T[3][3],
    float inv_T[3][3])
{
    float cx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) * 0.25f;
    float cy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) * 0.25f;

    float mean_dist = 0.0f;
    for (int i = 0; i < 4; i++) {
        float dx = pts[i][0] - cx;
        float dy = pts[i][1] - cy;
        mean_dist += sqrtf(dx * dx + dy * dy);
    }
    mean_dist *= 0.25f;

    float s = (mean_dist > 1e-6f) ? (sqrtf(2.0f) / mean_dist) : 1.0f;

    for (int i = 0; i < 4; i++) {
        norm_pts[i][0] = s * (pts[i][0] - cx);
        norm_pts[i][1] = s * (pts[i][1] - cy);
    }

    // T = [s 0 -s*cx; 0 s -s*cy; 0 0 1]
    T[0][0] = s;    T[0][1] = 0.0f; T[0][2] = -s * cx;
    T[1][0] = 0.0f; T[1][1] = s;    T[1][2] = -s * cy;
    T[2][0] = 0.0f; T[2][1] = 0.0f; T[2][2] = 1.0f;

    // inv_T = [1/s 0 cx; 0 1/s cy; 0 0 1]
    float inv_s = 1.0f / s;
    inv_T[0][0] = inv_s; inv_T[0][1] = 0.0f;  inv_T[0][2] = cx;
    inv_T[1][0] = 0.0f;  inv_T[1][1] = inv_s; inv_T[1][2] = cy;
    inv_T[2][0] = 0.0f;  inv_T[2][1] = 0.0f;  inv_T[2][2] = 1.0f;
}

// ============================================================================
// Giải hệ DLT 8×8 bằng khử Gauss với Relative Partial Pivoting trên GPU
// Khớp chính xác V0 solve_dlt_8x8_cpu (v0_sequential.cpp dòng 466-568)
// ============================================================================
__device__ bool solve_dlt_8x8_dev(
    const float ptsA[4][2],
    const float ptsB[4][2],
    float H[9])
{
    float norm_A[4][2], norm_B[4][2];
    float TA[3][3], inv_TA[3][3];
    float TB[3][3], inv_TB[3][3];

    compute_hartley_norm_dev(ptsA, norm_A, TA, inv_TA);
    compute_hartley_norm_dev(ptsB, norm_B, TB, inv_TB);

    float A[8][9]; // Ma trận mở rộng [A | b]
    for (int i = 0; i < 4; i++) {
        float x = norm_A[i][0], y = norm_A[i][1];
        float u = norm_B[i][0], v = norm_B[i][1];

        A[2 * i][0] = x;    A[2 * i][1] = y;    A[2 * i][2] = 1.0f;
        A[2 * i][3] = 0.0f; A[2 * i][4] = 0.0f; A[2 * i][5] = 0.0f;
        A[2 * i][6] = -u * x; A[2 * i][7] = -u * y; A[2 * i][8] = u;

        A[2 * i + 1][0] = 0.0f; A[2 * i + 1][1] = 0.0f; A[2 * i + 1][2] = 0.0f;
        A[2 * i + 1][3] = x;    A[2 * i + 1][4] = y;    A[2 * i + 1][5] = 1.0f;
        A[2 * i + 1][6] = -v * x; A[2 * i + 1][7] = -v * y; A[2 * i + 1][8] = v;
    }

    // Relative Partial Pivoting
    for (int col = 0; col < 8; col++) {
        int max_row = col;
        float max_val = fabsf(A[col][col]);
        for (int r = col + 1; r < 8; r++) {
            float val = fabsf(A[r][col]);
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }

        if (max_val < 1e-6f) return false; // Ma trận suy biến

        if (max_row != col) {
            for (int k = col; k <= 8; k++) {
                float tmp = A[col][k];
                A[col][k] = A[max_row][k];
                A[max_row][k] = tmp;
            }
        }

        float pivot = A[col][col];
        for (int r = col + 1; r < 8; r++) {
            float factor = A[r][col] / pivot;
            for (int k = col; k <= 8; k++) {
                A[r][k] -= factor * A[col][k];
            }
        }
    }

    // Thế ngược (Back-substitution)
    float h_tilde[9];
    h_tilde[8] = 1.0f;
    for (int r = 7; r >= 0; r--) {
        float sum = A[r][8];
        for (int c = r + 1; c < 8; c++) {
            sum -= A[r][c] * h_tilde[c];
        }
        h_tilde[r] = sum / A[r][r];
    }

    // Khử chuẩn hóa: H = inv_TB * H_tilde * TA
    float M[3][3], H_final[3][3];
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            M[r][c] = h_tilde[r * 3 + 0] * TA[0][c] +
                      h_tilde[r * 3 + 1] * TA[1][c] +
                      h_tilde[r * 3 + 2] * TA[2][c];
        }
    }
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            H_final[r][c] = inv_TB[r][0] * M[0][c] +
                            inv_TB[r][1] * M[1][c] +
                            inv_TB[r][2] * M[2][c];
        }
    }

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            H[r * 3 + c] = H_final[r][c];
        }
    }

    // Chuẩn hóa H[8] = 1.0
    if (fabsf(H[8]) > 1e-12f) {
        float inv = 1.0f / H[8];
        for (int i = 0; i < 9; i++) H[i] *= inv;
        H[8] = 1.0f;
    } else {
        return false;
    }

    // Kiểm tra tính hữu hạn
    for (int i = 0; i < 9; i++) {
        if (!isfinite(H[i])) return false;
    }

    return true;
}

// ============================================================================
// Kiểm tra hình học hợp lệ của ma trận Homography (Plan v4 Mục 18.2)
// Khớp chính xác V0 homography_geometry_plausible (v0_sequential.cpp dòng 707-766):
//   - Chiếu 4 góc của ảnh A sang ảnh B
//   - Kiểm tra pole projective (không đi qua khung ảnh)
//   - Tỉ lệ diện tích: area_scale in [0.35, 3.0]
//   - Span: span_x <= 3.0 * w, span_y <= 3.0 * h
//   - Tỉ lệ mẫu số: denominator_ratio <= 6.0
// ============================================================================
__device__ bool homography_geometry_plausible_dev(
    const float H[9],
    int image_w,
    int image_h)
{
    if (image_w <= 0 || image_h <= 0) return true;

    const float corners[4][2] = {
        {0.0f, 0.0f},
        {(float)image_w, 0.0f},
        {(float)image_w, (float)image_h},
        {0.0f, (float)image_h}
    };
    float projected[4][2];
    float min_x = 1e30f;
    float min_y = 1e30f;
    float max_x = -1e30f;
    float max_y = -1e30f;
    float min_abs_den = 1e30f;
    float max_abs_den = 0.0f;
    float first_den = 0.0f;

    for (int i = 0; i < 4; i++) {
        const float x = corners[i][0];
        const float y = corners[i][1];
        const float den = H[6] * x + H[7] * y + H[8];
        if (!isfinite(den) || fabsf(den) < 1e-4f) return false;
        if (i == 0) first_den = den;
        if (den * first_den <= 0.0f) return false;  // Pole cắt ngang ảnh

        const float px = (H[0] * x + H[1] * y + H[2]) / den;
        const float py = (H[3] * x + H[4] * y + H[5]) / den;
        if (!isfinite(px) || !isfinite(py)) return false;
        projected[i][0] = px;
        projected[i][1] = py;
        if (px < min_x) min_x = px;
        if (py < min_y) min_y = py;
        if (px > max_x) max_x = px;
        if (py > max_y) max_y = py;
        float abs_den = fabsf(den);
        if (abs_den < min_abs_den) min_abs_den = abs_den;
        if (abs_den > max_abs_den) max_abs_den = abs_den;
    }

    double twice_area = 0.0;
    for (int i = 0; i < 4; i++) {
        const int j = (i + 1) % 4;
        twice_area += (double)projected[i][0] * (double)projected[j][1]
                    - (double)projected[j][0] * (double)projected[i][1];
    }
    const double area = 0.5 * twice_area;
    const double source_area = (double)image_w * (double)image_h;
    const double area_scale = area / source_area;
    const float span_x = max_x - min_x;
    const float span_y = max_y - min_y;
    const float denominator_ratio = max_abs_den / min_abs_den;

    return area_scale >= 0.35 && area_scale <= 3.0
        && span_x <= 3.0f * (float)image_w
        && span_y <= 3.0f * (float)image_h
        && denominator_ratio <= 6.0f;
}

// ============================================================================
// Shell Sort trên device cho mảng float nhỏ (phục vụ tìm upper median)
// Ổn định, không đệ quy, độ phức tạp O(n log^2 n), n <= 2000
// ============================================================================
__device__ void shell_sort_dev(float* arr, int n) {
    for (int gap = n / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < n; i++) {
            float temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}

// GpuRansacResult đã được khai báo trong cuda_pipeline.cuh

// ============================================================================
// Kernel 1: kernel_ransac_generate_samples
// Sinh bảng 2000×4 indices ngẫu nhiên bằng LCG seed 42 với rejection loop.
// Chạy trên GPU với 1 thread (Thread 0) duy nhất để đảm bảo tính tất định
// và khớp 100% với chuỗi số sinh từ V0 CPU (Plan v4 Mục 18.2).
// ============================================================================
__global__ void kernel_ransac_generate_samples(
    int* __restrict__ d_samples,
    int n_matches,
    int iters)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    LcgRngDev rng(42);
    for (int iter = 0; iter < iters; iter++) {
        int sample_idx[4];
        for (int i = 0; i < 4; i++) {
            bool unique = false;
            while (!unique) {
                sample_idx[i] = rng.next_int(n_matches);
                unique = true;
                for (int prev = 0; prev < i; prev++) {
                    if (sample_idx[i] == sample_idx[prev]) {
                        unique = false;
                        break;
                    }
                }
            }
            d_samples[iter * 4 + i] = sample_idx[i];
        }
    }
}

// ============================================================================
// Kernel 2: kernel_ransac_hypotheses_and_inliers
// Grid: iters blocks (2000 blocks), Block: 128 threads.
// Mỗi block giải 1 hypothesis 4 điểm bằng Normalized DLT trong shared memory,
// sau đó 128 threads song song chiếu và đếm inliers trên toàn bộ matches.
// ============================================================================
__global__ void kernel_ransac_hypotheses_and_inliers(
    const Match* __restrict__ d_matches,
    int n_matches,
    const Keypoint* __restrict__ d_kpA,
    const Keypoint* __restrict__ d_kpB,
    const int* __restrict__ d_samples,
    float* __restrict__ d_all_H,
    int* __restrict__ d_inlier_counts)
{
    __shared__ float s_H[9];
    __shared__ bool s_valid;
    __shared__ int s_inliers;

    if (threadIdx.x == 0) {
        s_inliers = 0;
        int s_idx[4];
        for (int i = 0; i < 4; i++) {
            s_idx[i] = d_samples[blockIdx.x * 4 + i];
        }

        float ptsA[4][2], ptsB[4][2];
        for (int i = 0; i < 4; i++) {
            Match m = d_matches[s_idx[i]];
            ptsA[i][0] = d_kpA[m.idx_a].x;
            ptsA[i][1] = d_kpA[m.idx_a].y;
            ptsB[i][0] = d_kpB[m.idx_b].x;
            ptsB[i][1] = d_kpB[m.idx_b].y;
        }

        s_valid = solve_dlt_8x8_dev(ptsA, ptsB, s_H);
    }
    __syncthreads();

    if (!s_valid) {
        if (threadIdx.x == 0) {
            d_inlier_counts[blockIdx.x] = 0;
            for (int i = 0; i < 9; i++) {
                d_all_H[blockIdx.x * 9 + i] = 0.0f;
            }
        }
        return;
    }

    int thread_inliers = 0;
    const float thresh_sq = RANSAC_THRESH * RANSAC_THRESH;
    for (int m = threadIdx.x; m < n_matches; m += blockDim.x) {
        Match match = d_matches[m];
        float xa = d_kpA[match.idx_a].x;
        float ya = d_kpA[match.idx_a].y;
        float xb = d_kpB[match.idx_b].x;
        float yb = d_kpB[match.idx_b].y;

        float w = s_H[6] * xa + s_H[7] * ya + s_H[8];
        if (fabsf(w) < 1e-6f) continue;

        float pred_xb = (s_H[0] * xa + s_H[1] * ya + s_H[2]) / w;
        float pred_yb = (s_H[3] * xa + s_H[4] * ya + s_H[5]) / w;

        float dx = pred_xb - xb;
        float dy = pred_yb - yb;
        if (dx * dx + dy * dy < thresh_sq) {
            thread_inliers++;
        }
    }

    atomicAdd(&s_inliers, thread_inliers);
    __syncthreads();

    if (threadIdx.x == 0) {
        d_inlier_counts[blockIdx.x] = s_inliers;
        for (int i = 0; i < 9; i++) {
            d_all_H[blockIdx.x * 9 + i] = s_H[i];
        }
    }
}

// ============================================================================
// Kernel 3: kernel_ransac_select_best (kernel_select_best_hypothesis)
// 1 block gồm 256 threads, reduction tìm hypothesis có số inliers tối đa.
// Tie-break: inliers bằng nhau thì hypothesis ID nhỏ hơn thắng (khớp V0: inliers > max_inliers).
// ============================================================================
__global__ void kernel_ransac_select_best(
    const int* __restrict__ d_inlier_counts,
    int* __restrict__ d_best_hyp,
    int iters)
{
    __shared__ int s_inliers[256];
    __shared__ int s_indices[256];

    int best_inliers = -1;
    int best_idx = 0;

    for (int i = threadIdx.x; i < iters; i += blockDim.x) {
        int cnt = d_inlier_counts[i];
        if (cnt > best_inliers) {
            best_inliers = cnt;
            best_idx = i;
        }
    }

    s_inliers[threadIdx.x] = best_inliers;
    s_indices[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            int other_inliers = s_inliers[threadIdx.x + stride];
            int other_idx = s_indices[threadIdx.x + stride];
            if (other_inliers > s_inliers[threadIdx.x] ||
                (other_inliers == s_inliers[threadIdx.x] && other_idx < s_indices[threadIdx.x]))
            {
                s_inliers[threadIdx.x] = other_inliers;
                s_indices[threadIdx.x] = other_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_best_hyp[0] = s_indices[0];
    }
}

#define kernel_select_best_hypothesis kernel_ransac_select_best

// ============================================================================
// Kernel 4: kernel_ransac_refit_and_finalize
// 1 block, 1 thread (Thread 0):
// - Đọc best_hyp và nạp opt_H
// - Refit Least Squares (Double Precision) trên toàn bộ inliers nếu max_inliers >= 4
// - Kiểm tra hình học projective: pole, area scale [0.35, 3.0], span <= 3x,
//   denominator ratio <= 6.0, confidence floor >= max(12, int(0.15 * n_matches))
// - Nếu hình học hợp lệ: xuất Homography hoàn chỉnh (used_translation_fallback = false)
// - Nếu hình học KHÔNG hợp lệ: thực hiện Fallback Translation trên GPU (khớp V0)
// ============================================================================
__global__ void kernel_ransac_refit_and_finalize(
    const Match* __restrict__ d_matches,
    int n_matches,
    const Keypoint* __restrict__ d_kpA,
    const Keypoint* __restrict__ d_kpB,
    const float* __restrict__ d_all_H,
    const int* __restrict__ d_inlier_counts,
    const int* __restrict__ d_best_hyp,
    int image_w,
    int image_h,
    GpuRansacResult* __restrict__ d_result)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    __shared__ float s_cand_dx[DEFAULT_TOP_K];
    __shared__ float s_cand_dy[DEFAULT_TOP_K];
    __shared__ float s_val_dx[DEFAULT_TOP_K];
    __shared__ float s_val_dy[DEFAULT_TOP_K];

    int best_idx = d_best_hyp[0];
    int max_inliers = d_inlier_counts[best_idx];

    float opt_H[9];
    for (int i = 0; i < 9; i++) {
        opt_H[i] = d_all_H[best_idx * 9 + i];
    }

    const float threshold_sq = RANSAC_THRESH * RANSAC_THRESH;

    // ---- 1. Refit least squares in double precision ----
    if (max_inliers >= 4) {
        double cax = 0.0, cay = 0.0, cbx = 0.0, cby = 0.0;
        int n_inl = 0;

        for (int m = 0; m < n_matches; m++) {
            Match match = d_matches[m];
            float xa = d_kpA[match.idx_a].x;
            float ya = d_kpA[match.idx_a].y;
            float xb = d_kpB[match.idx_b].x;
            float yb = d_kpB[match.idx_b].y;

            float w = opt_H[6] * xa + opt_H[7] * ya + opt_H[8];
            if (fabsf(w) < 1e-6f) continue;
            float px = (opt_H[0] * xa + opt_H[1] * ya + opt_H[2]) / w;
            float py = (opt_H[3] * xa + opt_H[4] * ya + opt_H[5]) / w;
            float dx = px - xb;
            float dy = py - yb;
            if (dx * dx + dy * dy < threshold_sq) {
                cax += (double)xa;
                cay += (double)ya;
                cbx += (double)xb;
                cby += (double)yb;
                n_inl++;
            }
        }

        if (n_inl >= 4) {
            cax /= n_inl; cay /= n_inl;
            cbx /= n_inl; cby /= n_inl;

            double mean_a = 0.0, mean_b = 0.0;
            for (int m = 0; m < n_matches; m++) {
                Match match = d_matches[m];
                float xa = d_kpA[match.idx_a].x;
                float ya = d_kpA[match.idx_a].y;
                float xb = d_kpB[match.idx_b].x;
                float yb = d_kpB[match.idx_b].y;

                float w = opt_H[6] * xa + opt_H[7] * ya + opt_H[8];
                if (fabsf(w) < 1e-6f) continue;
                float px = (opt_H[0] * xa + opt_H[1] * ya + opt_H[2]) / w;
                float py = (opt_H[3] * xa + opt_H[4] * ya + opt_H[5]) / w;
                float dx = px - xb;
                float dy = py - yb;
                if (dx * dx + dy * dy < threshold_sq) {
                    double dxa = (double)xa - cax;
                    double dya = (double)ya - cay;
                    double dxb = (double)xb - cbx;
                    double dyb = (double)yb - cby;
                    mean_a += sqrt(dxa * dxa + dya * dya);
                    mean_b += sqrt(dxb * dxb + dyb * dyb);
                }
            }
            mean_a /= n_inl;
            mean_b /= n_inl;

            if (mean_a >= 1e-9 && mean_b >= 1e-9) {
                double sa = sqrt(2.0) / mean_a;
                double sb = sqrt(2.0) / mean_b;
                double normal[8][9];
                for (int r = 0; r < 8; r++) {
                    for (int c = 0; c < 9; c++) normal[r][c] = 0.0;
                }

                for (int m = 0; m < n_matches; m++) {
                    Match match = d_matches[m];
                    float xa = d_kpA[match.idx_a].x;
                    float ya = d_kpA[match.idx_a].y;
                    float xb = d_kpB[match.idx_b].x;
                    float yb = d_kpB[match.idx_b].y;

                    float w = opt_H[6] * xa + opt_H[7] * ya + opt_H[8];
                    if (fabsf(w) < 1e-6f) continue;
                    float px = (opt_H[0] * xa + opt_H[1] * ya + opt_H[2]) / w;
                    float py = (opt_H[3] * xa + opt_H[4] * ya + opt_H[5]) / w;
                    float dx = px - xb;
                    float dy = py - yb;
                    if (dx * dx + dy * dy < threshold_sq) {
                        double x = sa * ((double)xa - cax);
                        double y = sa * ((double)ya - cay);
                        double u = sb * ((double)xb - cbx);
                        double v = sb * ((double)yb - cby);
                        double rows[2][8] = {
                            {x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y},
                            {0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y}
                        };
                        double rhs[2] = {u, v};
                        for (int eq = 0; eq < 2; eq++) {
                            for (int r = 0; r < 8; r++) {
                                for (int c = 0; c < 8; c++) {
                                    normal[r][c] += rows[eq][r] * rows[eq][c];
                                }
                                normal[r][8] += rows[eq][r] * rhs[eq];
                            }
                        }
                    }
                }

                // Giải 8x8 bằng relative partial pivoting (khớp chính xác V0 lines 620-649)
                bool solve_ok = true;
                for (int col = 0; col < 8; col++) {
                    int pivot_row = -1;
                    double best_ratio = 0.0;
                    for (int r = col; r < 8; r++) {
                        double row_scale = 0.0;
                        for (int c = col; c < 8; c++) {
                            double val = fabs(normal[r][c]);
                            if (val > row_scale) row_scale = val;
                        }
                        if (row_scale <= 1e-15) continue;
                        double ratio = fabs(normal[r][col]) / row_scale;
                        if (ratio > best_ratio) {
                            best_ratio = ratio;
                            pivot_row = r;
                        }
                    }
                    if (pivot_row < 0 || fabs(normal[pivot_row][col]) < 1e-12) {
                        solve_ok = false;
                        break;
                    }
                    if (pivot_row != col) {
                        for (int c = col; c <= 8; c++) {
                            double tmp = normal[col][c];
                            normal[col][c] = normal[pivot_row][c];
                            normal[pivot_row][c] = tmp;
                        }
                    }
                    for (int r = col + 1; r < 8; r++) {
                        double factor = normal[r][col] / normal[col][col];
                        for (int c = col; c <= 8; c++) {
                            normal[r][c] -= factor * normal[col][c];
                        }
                    }
                }

                if (solve_ok) {
                    double ht[9];
                    for (int i = 0; i < 9; i++) ht[i] = 0.0;
                    ht[8] = 1.0;
                    for (int r = 7; r >= 0; r--) {
                        double val = normal[r][8];
                        for (int c = r + 1; c < 8; c++) {
                            val -= normal[r][c] * ht[c];
                        }
                        ht[r] = val / normal[r][r];
                    }

                    const double TA[9] = {sa, 0.0, -sa * cax,
                                          0.0, sa, -sa * cay,
                                          0.0, 0.0, 1.0};
                    const double invTB[9] = {1.0 / sb, 0.0, cbx,
                                             0.0, 1.0 / sb, cby,
                                             0.0, 0.0, 1.0};
                    double middle[9] = {0.0}, denorm[9] = {0.0};
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 3; c++) {
                            for (int k = 0; k < 3; k++) {
                                middle[r * 3 + c] += ht[r * 3 + k] * TA[k * 3 + c];
                            }
                        }
                    }
                    for (int r = 0; r < 3; r++) {
                        for (int c = 0; c < 3; c++) {
                            for (int k = 0; k < 3; k++) {
                                denorm[r * 3 + c] += invTB[r * 3 + k] * middle[k * 3 + c];
                            }
                        }
                    }

                    if (isfinite(denorm[8]) && fabs(denorm[8]) >= 1e-12) {
                        double inv_h33 = 1.0 / denorm[8];
                        float refined_H[9];
                        bool refined_finite = true;
                        for (int i = 0; i < 9; i++) {
                            double val = denorm[i] * inv_h33;
                            if (!isfinite(val)) {
                                refined_finite = false;
                                break;
                            }
                            refined_H[i] = (float)val;
                        }
                        refined_H[8] = 1.0f;

                        if (refined_finite) {
                            int refined_inliers = 0;
                            for (int m = 0; m < n_matches; m++) {
                                Match match = d_matches[m];
                                float xa = d_kpA[match.idx_a].x;
                                float ya = d_kpA[match.idx_a].y;
                                float xb = d_kpB[match.idx_b].x;
                                float yb = d_kpB[match.idx_b].y;

                                float w = refined_H[6] * xa + refined_H[7] * ya + refined_H[8];
                                if (fabsf(w) < 1e-6f) continue;
                                float px = (refined_H[0] * xa + refined_H[1] * ya + refined_H[2]) / w;
                                float py = (refined_H[3] * xa + refined_H[4] * ya + refined_H[5]) / w;
                                float dx = px - xb;
                                float dy = py - yb;
                                if (dx * dx + dy * dy < threshold_sq) {
                                    refined_inliers++;
                                }
                            }

                            if (refined_inliers >= max_inliers) {
                                for (int i = 0; i < 9; i++) opt_H[i] = refined_H[i];
                                max_inliers = refined_inliers;
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- 2. Geometry check & Confidence floor ----
    const int confidence_floor = (12 > (int)(0.15f * n_matches)) ? 12 : (int)(0.15f * n_matches);
    const bool geometry_ok = homography_geometry_plausible_dev(opt_H, image_w, image_h);
    const bool projective_model_ok = (max_inliers >= confidence_floor) && geometry_ok;

    if (max_inliers >= 4 && (image_w <= 0 || image_h <= 0 || projective_model_ok)) {
        for (int i = 0; i < 9; i++) d_result->H.h[i] = opt_H[i];
        d_result->H.is_valid = true;
        d_result->H.used_translation_fallback = false;
        d_result->inliers = max_inliers;
        d_result->success = true;
        return;
    }

    // ---- 3. Fallback Translation (khi hình học thất bại) ----
    if (image_w <= 0 || image_h <= 0) {
        d_result->H.is_valid = false;
        d_result->inliers = 0;
        d_result->success = false;
        return;
    }

    // Bước 3.1: Lọc candidates trong cửa sổ chính dx [-0.85w, -0.20w], |dy| <= 0.35h
    int n_cand = 0;
    const float min_dx = -0.85f * (float)image_w;
    const float max_dx = -0.20f * (float)image_w;
    const float max_abs_dy = 0.35f * (float)image_h;

    for (int m = 0; m < n_matches; m++) {
        Match match = d_matches[m];
        if (match.idx_a < 0 || match.idx_b < 0) continue;
        float dx = d_kpB[match.idx_b].x - d_kpA[match.idx_a].x;
        float dy = d_kpB[match.idx_b].y - d_kpA[match.idx_a].y;
        if (dx >= min_dx && dx <= max_dx && fabsf(dy) <= max_abs_dy) {
            if (n_cand < DEFAULT_TOP_K) {
                s_cand_dx[n_cand] = dx;
                s_cand_dy[n_cand] = dy;
                n_cand++;
            }
        }
    }

    // Bước 3.2: Nếu < 4 điểm, mở rộng sang dx [-0.95w, -0.05w], |dy| <= 0.60h
    if (n_cand < 4) {
        n_cand = 0;
        const float min_dx_wide = -0.95f * (float)image_w;
        const float max_dx_wide = -0.05f * (float)image_w;
        const float max_abs_dy_wide = 0.60f * (float)image_h;

        for (int m = 0; m < n_matches; m++) {
            Match match = d_matches[m];
            if (match.idx_a < 0 || match.idx_b < 0) continue;
            float dx = d_kpB[match.idx_b].x - d_kpA[match.idx_a].x;
            float dy = d_kpB[match.idx_b].y - d_kpA[match.idx_a].y;
            if (dx >= min_dx_wide && dx <= max_dx_wide && fabsf(dy) <= max_abs_dy_wide) {
                if (n_cand < DEFAULT_TOP_K) {
                    s_cand_dx[n_cand] = dx;
                    s_cand_dy[n_cand] = dy;
                    n_cand++;
                }
            }
        }
    }

    if (n_cand < 4) {
        d_result->H.is_valid = false;
        d_result->inliers = 0;
        d_result->success = false;
        return;
    }

    // Ngưỡng đồng thuận: T = max(12.0f, 0.03f * sqrt(w * h))
    float T = 0.03f * sqrtf((float)image_w * (float)image_h);
    if (T < 12.0f) T = 12.0f;
    const float T_sq = T * T;

    // Tìm candidate có số phiếu đồng thuận cao nhất (tie-break index nhỏ hơn)
    int best_count = 0;
    int best_index = 0;
    for (int i = 0; i < n_cand; i++) {
        int count = 0;
        float c_dx = s_cand_dx[i];
        float c_dy = s_cand_dy[i];
        for (int j = 0; j < n_cand; j++) {
            float ex = s_cand_dx[j] - c_dx;
            float ey = s_cand_dy[j] - c_dy;
            if (ex * ex + ey * ey <= T_sq) {
                count++;
            }
        }
        if (count > best_count) {
            best_count = count;
            best_index = i;
        }
    }

    if (best_count < 4) {
        d_result->H.is_valid = false;
        d_result->inliers = 0;
        d_result->success = false;
        return;
    }

    // Thu thập tập đồng thuận để tính upper median
    int n_val = 0;
    float best_dx = s_cand_dx[best_index];
    float best_dy = s_cand_dy[best_index];
    for (int j = 0; j < n_cand; j++) {
        float ex = s_cand_dx[j] - best_dx;
        float ey = s_cand_dy[j] - best_dy;
        if (ex * ex + ey * ey <= T_sq) {
            s_val_dx[n_val] = s_cand_dx[j];
            s_val_dy[n_val] = s_cand_dy[j];
            n_val++;
        }
    }

    // Sắp xếp để lấy upper median (vị trí n/2)
    shell_sort_dev(s_val_dx, n_val);
    shell_sort_dev(s_val_dy, n_val);
    const int middle = n_val / 2;
    const float median_dx = s_val_dx[middle];
    const float median_dy = s_val_dy[middle];

    // Đếm số inliers cuối cùng với ngưỡng nới rộng 1.5 * T
    const float final_T = 1.5f * T;
    const float final_T_sq = final_T * final_T;
    int fallback_inliers = 0;
    for (int j = 0; j < n_cand; j++) {
        float ex = s_cand_dx[j] - median_dx;
        float ey = s_cand_dy[j] - median_dy;
        if (ex * ex + ey * ey <= final_T_sq) {
            fallback_inliers++;
        }
    }

    if (fallback_inliers >= 4) {
        for (int i = 0; i < 9; i++) {
            d_result->H.h[i] = (i % 4 == 0) ? 1.0f : 0.0f;
        }
        d_result->H.h[2] = median_dx;
        d_result->H.h[5] = median_dy;
        d_result->H.is_valid = true;
        d_result->H.used_translation_fallback = true;
        d_result->inliers = fallback_inliers;
        d_result->success = true;
    } else {
        d_result->H.is_valid = false;
        d_result->inliers = 0;
        d_result->success = false;
    }
}

// ============================================================================
// stage5_ransac_gpu — Public API (khai báo trong cuda_pipeline.cuh)
// Thực hiện toàn bộ Stage 5 trên GPU theo đúng Plan v4 Mục 18.2:
//   1. kernel_ransac_generate_samples<<<1, 1>>>
//   2. kernel_ransac_hypotheses_and_inliers<<<RANSAC_ITERS, 128>>>
//   3. kernel_ransac_select_best<<<1, 256>>>
//   4. kernel_ransac_refit_and_finalize<<<1, 1>>>
// ============================================================================
bool stage5_ransac_gpu(
    const Match*    d_matches,
    int             n_matches,
    const Keypoint* d_kpA,
    const Keypoint* d_kpB,
    CudaWorkspace&  ws,
    int image_w,
    int image_h,
    Homography&     best_H,
    int&            out_inliers,
    float&          elapsed_ms)
{
    out_inliers = 0;
    best_H = Homography();
    best_H.is_valid = false;
    elapsed_ms = 0.0f;

    if (n_matches < 4) {
        return false;
    }

    // Dùng buffer kết quả đã được cấp phát tĩnh trong CudaWorkspace (không cudaMalloc giữa các run)
    GpuRansacResult* d_result = ws.d_ransac_result;
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(GpuRansacResult)));

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start));

    // 1. Sinh bảng samples ngẫu nhiên trên GPU
    kernel_ransac_generate_samples<<<1, 1>>>(ws.d_sample_indices, n_matches, RANSAC_ITERS);
    CHECK_CUDA(cudaGetLastError());

    // 2. Tính 2000 hypotheses và đếm inliers song song
    kernel_ransac_hypotheses_and_inliers<<<RANSAC_ITERS, 128>>>(
        d_matches, n_matches, d_kpA, d_kpB,
        ws.d_sample_indices, ws.d_all_H, ws.d_inlier_counts);
    CHECK_CUDA(cudaGetLastError());

    // 3. Chọn hypothesis có số inlier lớn nhất (tie-break ID nhỏ hơn)
    kernel_ransac_select_best<<<1, 256>>>(
        ws.d_inlier_counts, ws.d_best_hyp, RANSAC_ITERS);
    CHECK_CUDA(cudaGetLastError());

    // 4. Refit least squares, kiểm tra hình học và fallback nếu cần
    kernel_ransac_refit_and_finalize<<<1, 1>>>(
        d_matches, n_matches, d_kpA, d_kpB,
        ws.d_all_H, ws.d_inlier_counts, ws.d_best_hyp,
        image_w, image_h, d_result);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));

    // D2H copy kết quả RANSAC
    GpuRansacResult h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(GpuRansacResult), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));

    best_H = h_result.H;
    out_inliers = h_result.inliers;
    return h_result.success;
}
