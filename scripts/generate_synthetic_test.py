#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Generate Synthetic Image Pair with Ground-Truth Homography
Authors: Group 19 - Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
External code: none
Description:
  - Generates a corner-rich synthetic scene at 3 resolutions (720p, 1080p, 4K)
  - Uses a rigorous projective transform (Homography H_A_to_B)
  - Exports left_<size>.jpg, right_<size>.jpg to dataset/synthetic/
  - Exports dataset/synthetic/ground_truth.json containing:
      - H_A_to_B (3x3 row-major)
      - inv_H (3x3 row-major)
      - validation_grid points
      - diagonal length for normalized RMSE computation
  - Provides compute_grid_rmse(estimated_H, gt_H, w, h) evaluation function
============================================================================
"""

import os
import math
import json
import numpy as np
from PIL import Image, ImageDraw

RESOLUTIONS = {
    'small':  (1280, 720),
    'medium': (1920, 1080),
    'large':  (3840, 2160)
}

def create_rich_canvas(w: int, h: int) -> Image.Image:
    """Tạo canvas nền giàu góc cạnh (corners) và kết cấu đa dạng cho FAST detector"""
    img = Image.new('RGB', (w, h), color=(245, 245, 248))
    draw = ImageDraw.Draw(img)

    # 1. Bàn cờ ô vuông nền (checkerboard grid)
    grid_size = max(16, h // 30)
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            if ((x // grid_size) + (y // grid_size)) % 2 == 0:
                draw.rectangle([x, y, x + grid_size, y + grid_size], fill=(225, 228, 232))

    # 2. Sinh các khối hình học với màu sắc tương phản cao và nhiều góc nhọn
    rng = np.random.default_rng(12345)
    num_shapes = max(80, (w * h) // 15000)
    colors = [
        (220, 50, 47), (38, 139, 210), (133, 153, 0), (181, 137, 0),
        (211, 54, 130), (42, 161, 152), (108, 113, 196), (20, 20, 20),
        (255, 127, 0), (0, 128, 128), (75, 0, 130), (238, 130, 238)
    ]

    for _ in range(num_shapes):
        cx = int(rng.integers(50, w - 50))
        cy = int(rng.integers(50, h - 50))
        size = int(rng.integers(max(15, h // 35), max(40, h // 12)))
        color = colors[int(rng.integers(0, len(colors)))]
        shape_type = int(rng.integers(0, 4))

        if shape_type == 0:
            # Hình chữ nhật có viền tương phản
            draw.rectangle([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2],
                           fill=color, outline=(0, 0, 0), width=2)
        elif shape_type == 1:
            # Tam giác
            draw.polygon([(cx, cy - size // 2), 
                          (cx - size // 2, cy + size // 2), 
                          (cx + size // 2, cy + size // 2)],
                         fill=color, outline=(0, 0, 0))
        elif shape_type == 2:
            # Ngôi sao 5 cánh (tạo nhiều góc FAST nhọn)
            pts = []
            for k in range(10):
                r = size // 2 if k % 2 == 0 else size // 4
                ang = k * math.pi / 5 - math.pi / 2
                pts.append((cx + int(r * math.cos(ang)), cy + int(r * math.sin(ang))))
            draw.polygon(pts, fill=color, outline=(0, 0, 0))
        else:
            # Chữ thập / Cross
            arm = size // 6
            half = size // 2
            draw.rectangle([cx - arm, cy - half, cx + arm, cy + half], fill=color, outline=(0, 0, 0))
            draw.rectangle([cx - half, cy - arm, cx + half, cy + arm], fill=color, outline=(0, 0, 0))

    return img

def get_ground_truth_homography(w: int, h: int):
    """
    Tạo ma trận Homography ground-truth chuẩn H_A_to_B:
    Bao gồm tịnh tiến ngang ~35% chiều rộng (overlap ~45%),
    xoay nhẹ ~1.2 độ, và biến dạng phối cảnh nhẹ (projective tilt).
    """
    theta = math.radians(1.2)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Tịnh tiến: t_x tương ứng ~35% w
    tx = 0.35 * w
    ty = 0.015 * h

    # Thành phần phối cảnh (perspective / projective terms)
    p_x = 0.00002 * (1280.0 / w)
    p_y = 0.00001 * (720.0 / h)

    H_A_to_B = np.array([
        [cos_t, -sin_t, tx],
        [sin_t,  cos_t, ty],
        [p_x,    p_y,   1.0]
    ], dtype=np.float64)

    # Chuẩn hóa sao cho H[2, 2] = 1.0
    H_A_to_B /= H_A_to_B[2, 2]
    inv_H = np.linalg.inv(H_A_to_B)
    inv_H /= inv_H[2, 2]

    return H_A_to_B, inv_H

def generate_synthetic_pair_for_res(res_name: str, w: int, h: int, out_dir: str):
    """Sinh cặp ảnh và metadata cho 1 độ phân giải"""
    # 1. Tạo ma trận ground-truth
    H_A_to_B, inv_H = get_ground_truth_homography(w, h)

    # 2. Tạo canvas tổng thể đủ rộng chứa cả Image A và không gian chiếu của Image B
    # Kích thước canvas lớn hơn ảnh chuẩn:
    canvas_w = int(w * 1.8)
    canvas_h = int(h * 1.3)
    canvas = create_rich_canvas(canvas_w, canvas_h)
    canvas_arr = np.array(canvas, dtype=np.uint8)

    # 3. Trích xuất Image A: vùng [0..w, 0..h] trên canvas
    img_A = canvas.crop((0, 0, w, h))

    # 4. Trích xuất Image B: áp dụng phép chiếu ngược
    # Đối với mỗi pixel (u, v) trong Image B, tìm tọa độ tương ứng (x, y) trên canvas:
    # Điểm trong B liên hệ với điểm trong A qua: [u, v, 1]^T ~ H_A_to_B * [x, y, 1]^T
    # Do đó [x, y, 1]^T ~ inv_H * [u, v, 1]^T
    u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones_like(u_coords)

    # Tọa độ trong B
    pts_B = np.stack([u_coords, v_coords, ones], axis=-1).reshape(-1, 3).T # (3, w*h)

    # Chiếu về canvas A
    pts_A_proj = inv_H @ pts_B
    x_canvas = pts_A_proj[0] / pts_A_proj[2]
    y_canvas = pts_A_proj[1] / pts_A_proj[2]

    x_canvas = x_canvas.reshape(h, w)
    y_canvas = y_canvas.reshape(h, w)

    # Nội suy bilinear
    x0 = np.clip(np.floor(x_canvas).astype(int), 0, canvas_w - 2)
    y0 = np.clip(np.floor(y_canvas).astype(int), 0, canvas_h - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (x_canvas - x0)[..., None]
    wy = (y_canvas - y0)[..., None]

    ia = canvas_arr[y0, x0].astype(np.float32)
    ib = canvas_arr[y0, x1].astype(np.float32)
    ic = canvas_arr[y1, x0].astype(np.float32)
    id_val = canvas_arr[y1, x1].astype(np.float32)

    top = ia * (1.0 - wx) + ib * wx
    bottom = ic * (1.0 - wx) + id_val * wx
    sampled_B = top * (1.0 - wy) + bottom * wy
    img_B = Image.fromarray(np.clip(sampled_B, 0, 255).astype(np.uint8))

    # 5. Lưu ảnh Left (A) và Right (B)
    left_path = os.path.join(out_dir, f"left_{res_name}.jpg")
    right_path = os.path.join(out_dir, f"right_{res_name}.jpg")
    img_A.save(left_path, format='JPEG', quality=95)
    img_B.save(right_path, format='JPEG', quality=95)
    print(f"[OK] Generated synthetic {res_name}:")
    print(f"     Left:  {left_path} ({w}x{h})")
    print(f"     Right: {right_path} ({w}x{h})")

    # 6. Tạo validation grid points trên vùng overlap
    # Vùng overlap x thuộc [0.2*w, 0.9*w], y thuộc [0.1*h, 0.9*h]
    grid_pts_A = []
    grid_pts_B = []
    for gy in np.linspace(0.15 * h, 0.85 * h, 10):
        for gx in np.linspace(0.25 * w, 0.85 * w, 10):
            pA = np.array([gx, gy, 1.0])
            pB = H_A_to_B @ pA
            bx = pB[0] / pB[2]
            by = pB[1] / pB[2]
            if 0 <= bx < w and 0 <= by < h:
                grid_pts_A.append([float(gx), float(gy)])
                grid_pts_B.append([float(bx), float(by)])

    diagonal = math.sqrt(w * w + h * h)

    metadata = {
        "resolution": res_name,
        "width": w,
        "height": h,
        "diagonal": diagonal,
        "H_A_to_B": H_A_to_B.flatten().tolist(),
        "inv_H": inv_H.flatten().tolist(),
        "num_validation_points": len(grid_pts_A),
        "validation_grid_A": grid_pts_A,
        "validation_grid_B": grid_pts_B
    }

    return metadata

def compute_grid_rmse(estimated_H: np.ndarray, gt_metadata: dict) -> tuple:
    """Tính toán projected-grid RMSE giữa ma trận ước lượng và Ground Truth"""
    grid_A = np.array(gt_metadata["validation_grid_A"]) # (N, 2)
    grid_B_gt = np.array(gt_metadata["validation_grid_B"]) # (N, 2)
    diagonal = gt_metadata["diagonal"]

    ones = np.ones((len(grid_A), 1))
    pts_A_homo = np.hstack([grid_A, ones]).T # (3, N)

    pts_B_pred_homo = estimated_H @ pts_A_homo
    pred_x = pts_B_pred_homo[0] / pts_B_pred_homo[2]
    pred_y = pts_B_pred_homo[1] / pts_B_pred_homo[2]
    pred_B = np.stack([pred_x, pred_y], axis=1)

    errors = np.linalg.norm(pred_B - grid_B_gt, axis=1)
    rmse_px = float(np.sqrt(np.mean(errors ** 2)))
    normalized_rmse_pct = float((rmse_px / diagonal) * 100.0)

    return rmse_px, normalized_rmse_pct

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(repo_root, "dataset", "synthetic")
    os.makedirs(out_dir, exist_ok=True)

    all_metadata = {}
    for res_name, (w, h) in RESOLUTIONS.items():
        meta = generate_synthetic_pair_for_res(res_name, w, h, out_dir)
        all_metadata[res_name] = meta

    gt_json_path = os.path.join(out_dir, "ground_truth.json")
    with open(gt_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n[Success] Ground-truth metadata written to: {gt_json_path}")

if __name__ == '__main__':
    main()
