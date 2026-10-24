#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Validate JSON Metrics against Schema (Plan v4 Section 4.3)
Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
         Dao Quoc Tuan (23120392)
External sources: Python standard-library documentation; no copied code.
Description:
  - Validates output JSON metrics from benchmark runs.
  - Ensures required keys, datatypes, and value constraints are met.
  - Exits with 0 if valid, non-zero on failure.
============================================================================
"""

import sys
import json
import os
import math

REQUIRED_TOP_KEYS = [
    "schema_version", "timestamp", "version", "parent_version", "optimization",
    "device_name", "image_width", "image_height", "runs", "warmups",
    "status", "warning", "keypoints_a", "keypoints_b", "matches", "inliers",
    "inlier_ratio", "homography_a_to_b", "timing_ms", "amdahl_model",
    "amdahl_parallel_fraction", "amdahl_smax"
]

REQUIRED_TIMING_STAGES = [
    "stage1_preprocess", "stage2_fast_nms_topk", "stage3_orb_descriptor",
    "stage4_matching", "stage5_ransac", "stage6_warping_blending",
    "kernel_total", "h2d", "control", "d2h", "end_to_end", "setup"
]

def validate_metrics_json(json_path: str) -> bool:
    if not os.path.isfile(json_path):
        print(f"[Lỗi] Không tìm thấy tệp: {json_path}")
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Lỗi] Phân tích JSON thất bại trong {json_path}: {e}")
        return False

    # Kiểm tra các khóa cấp cao nhất
    for k in REQUIRED_TOP_KEYS:
        if k not in data:
            print(f"[Lỗi] Thiếu khóa bắt buộc cấp cao nhất: '{k}'")
            return False

    # Kiểm tra mảng homography
    H = data["homography_a_to_b"]
    if (not isinstance(H, list) or len(H) != 9 or
            not all(isinstance(v, (int, float)) and math.isfinite(v) for v in H)):
        print(f"[Lỗi] 'homography_a_to_b' phải là danh sách 9 số hữu hạn, nhận được: {H}")
        return False

    for key in ["image_width", "image_height", "runs", "warmups",
                "keypoints_a", "keypoints_b", "matches", "inliers"]:
        if not isinstance(data[key], int) or data[key] < 0:
            print(f"[Lỗi] '{key}' phải là số nguyên không âm")
            return False
    if data["image_width"] <= 0 or data["image_height"] <= 0 or data["runs"] <= 0:
        print("[Lỗi] Kích thước ảnh và số lần chạy runs phải là số dương")
        return False

    p = data["amdahl_parallel_fraction"]
    smax = data["amdahl_smax"]
    if (not isinstance(p, (int, float)) or not math.isfinite(p) or
            p < 0.0 or p >= 1.0):
        print(f"[Lỗi] Tỷ lệ song song hóa Amdahl p không hợp lệ: {p}")
        return False
    if (not isinstance(smax, (int, float)) or not math.isfinite(smax) or
            abs(smax - 1.0 / (1.0 - p)) > max(0.01, 0.002 * smax)):
        print(f"[Lỗi] Giá trị Amdahl S_max không nhất quán với p: p={p}, S_max={smax}")
        return False

    # Kiểm tra cấu trúc timing
    timing = data["timing_ms"]
    for stage in REQUIRED_TIMING_STAGES:
        if stage not in timing:
            print(f"[Lỗi] Thiếu giai đoạn đo thời gian: '{stage}'")
            return False
        stage_timing = timing[stage]
        if (not isinstance(stage_timing, dict) or
                not all(k in stage_timing for k in ["avg", "median", "min"])):
            print(f"[Lỗi] Giai đoạn '{stage}' phải chứa đủ các trường 'avg', 'median', và 'min'")
            return False
        values = [stage_timing[k] for k in ["avg", "median", "min"]]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
            print(f"[Lỗi] Thời gian của giai đoạn '{stage}' không thể là số âm")
            return False
        if stage_timing["min"] > stage_timing["median"] + 1e-6:
            print(f"[Lỗi] Giai đoạn '{stage}' có min > median")
            return False

    # Kiểm tra các giá trị status
    if data["status"] not in ["SUCCESS", "LOW_CONFIDENCE", "FAILURE"]:
        print(f"[Lỗi] Trạng thái '{data['status']}' không hợp lệ, kỳ vọng SUCCESS, LOW_CONFIDENCE, hoặc FAILURE")
        return False

    if data["inliers"] > data["matches"]:
        print("[Lỗi] Số lượng inlier không thể vượt quá số lượng matches")
        return False
    expected_ratio = data["inliers"] / data["matches"] if data["matches"] else 0.0
    if abs(data["inlier_ratio"] - expected_ratio) > 5e-4:
        print("[Lỗi] inlier_ratio không nhất quán với inliers/matches")
        return False

    print(f"[OK] Số liệu JSON hợp lệ: {json_path}")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Hướng dẫn: python validate_metrics.py <duong_dan_metrics.json>")
        sys.exit(1)

    all_valid = True
    for p in sys.argv[1:]:
        if not validate_metrics_json(p):
            all_valid = False

    sys.exit(0 if all_valid else 1)

