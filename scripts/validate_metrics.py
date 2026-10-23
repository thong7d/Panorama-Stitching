#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Validate JSON Metrics against Schema (Plan v4 Section 4.3)
Authors: Group 19 - Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
External code: none
Description:
  - Validates output JSON metrics from benchmark runs.
  - Ensures required keys, datatypes, and value constraints are met.
  - Exits with 0 if valid, non-zero on failure.
============================================================================
"""

import sys
import json
import os

REQUIRED_TOP_KEYS = [
    "schema_version", "timestamp", "version", "parent_version", "optimization",
    "device_name", "image_width", "image_height", "runs", "warmups",
    "status", "warning", "keypoints_a", "keypoints_b", "matches", "inliers",
    "homography_a_to_b", "timing_ms"
]

REQUIRED_TIMING_STAGES = [
    "stage1_preprocess", "stage2_fast_nms_topk", "stage3_orb_descriptor",
    "stage4_matching", "stage5_ransac", "stage6_warping_blending",
    "kernel_total", "h2d", "control", "d2h", "end_to_end", "setup"
]

def validate_metrics_json(json_path: str) -> bool:
    if not os.path.isfile(json_path):
        print(f"[Error] File not found: {json_path}")
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Error] Failed to parse JSON in {json_path}: {e}")
        return False

    # Check top-level keys
    for k in REQUIRED_TOP_KEYS:
        if k not in data:
            print(f"[Error] Missing required top-level key: '{k}'")
            return False

    # Check homography array
    H = data["homography_a_to_b"]
    if not isinstance(H, list) or len(H) != 9:
        print(f"[Error] 'homography_a_to_b' must be a list of 9 numbers, got: {H}")
        return False

    # Check timing structure
    timing = data["timing_ms"]
    for stage in REQUIRED_TIMING_STAGES:
        if stage not in timing:
            print(f"[Error] Missing timing stage: '{stage}'")
            return False
        stage_timing = timing[stage]
        if not isinstance(stage_timing, dict) or "avg" not in stage_timing or "min" not in stage_timing:
            print(f"[Error] Stage '{stage}' must contain 'avg' and 'min' fields")
            return False
        if stage_timing["avg"] < 0 or stage_timing["min"] < 0:
            print(f"[Error] Stage '{stage}' timing cannot be negative")
            return False

    # Check status values
    if data["status"] not in ["SUCCESS", "LOW_CONFIDENCE", "FAILURE"]:
        print(f"[Error] Invalid status '{data['status']}', expected SUCCESS, LOW_CONFIDENCE, or FAILURE")
        return False

    print(f"[OK] Valid metrics JSON: {json_path}")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_metrics.py <path_to_metrics.json>")
        sys.exit(1)

    all_valid = True
    for p in sys.argv[1:]:
        if not validate_metrics_json(p):
            all_valid = False

    sys.exit(0 if all_valid else 1)
