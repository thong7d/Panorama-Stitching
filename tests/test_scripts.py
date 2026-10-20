#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Project: Panorama Image Stitching
Module: Phase A Python Integration Tests (Plan v4 Section 6)
Authors: GroupXX (Members A, B, C)
============================================================================
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

def test_manifest_structure():
    print("[Python Test 1] Validating dataset/manifest.json...")
    manifest_path = WORKSPACE_ROOT / "dataset" / "manifest.json"
    assert manifest_path.is_file(), f"Manifest file not found: {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "pairs" in data, "Manifest missing 'pairs' list"
    assert any(p.get("status") == "external_blocker" for p in data["pairs"]), "Manifest missing external_blocker entry"
    
    pair_ids = [p["id"] for p in data["pairs"]]
    required_ids = ["site1_indoor", "site2_outdoor", "site3_custom", "challenging"]
    for rid in required_ids:
        assert rid in pair_ids, f"Required pair ID '{rid}' missing from manifest"
    
    print(f"  -> Found {len(data['pairs'])} pairs in manifest: {pair_ids} (PASS)")

def test_synthetic_ground_truth():
    print("[Python Test 2] Validating dataset/synthetic/ground_truth.json...")
    gt_path = WORKSPACE_ROOT / "dataset" / "synthetic" / "ground_truth.json"
    assert gt_path.is_file(), f"Ground truth file not found: {gt_path}"
    
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
        
    for res_name in ["small", "medium", "large"]:
        assert res_name in gt, f"Missing resolution '{res_name}' in ground_truth.json"
        res_info = gt[res_name]
        assert "H_A_to_B" in res_info, f"Missing H_A_to_B in {res_name}"
        assert len(res_info["H_A_to_B"]) == 9, f"H_A_to_B must be 9 values in {res_name}"
        assert abs(res_info["H_A_to_B"][8] - 1.0) < 1e-4, f"H_A_to_B[8] must be 1.0"
        assert "validation_grid_A" in res_info and "validation_grid_B" in res_info, "Missing validation grid"
        
        left_file = WORKSPACE_ROOT / "dataset" / "synthetic" / f"left_{res_name}.jpg"
        right_file = WORKSPACE_ROOT / "dataset" / "synthetic" / f"right_{res_name}.jpg"
        assert left_file.is_file(), f"Synthetic left image missing: {left_file}"
        assert right_file.is_file(), f"Synthetic right image missing: {right_file}"
        
    print(f"  -> Synthetic ground truth H and multi-resolution files verified (PASS)")

def test_validate_metrics_script():
    print("[Python Test 3] Validating scripts/validate_metrics.py...")
    from scripts.validate_metrics import validate_metrics_json
    
    # Valid sample JSON
    sample_valid = {
        "schema_version": "4.0",
        "timestamp": "2026-10-20T00:00:00Z",
        "version": "V0",
        "parent_version": "None",
        "optimization": "Sequential Baseline",
        "device_name": "Host CPU",
        "image_width": 1280,
        "image_height": 720,
        "runs": 5,
        "warmups": 1,
        "status": "SUCCESS",
        "warning": "None",
        "keypoints_a": 850,
        "keypoints_b": 870,
        "matches": 420,
        "inliers": 290,
        "homography_a_to_b": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "timing_ms": {
            "stage1_preprocess": {"avg": 5.0, "min": 4.8},
            "stage2_fast_nms_topk": {"avg": 12.0, "min": 11.5},
            "stage3_orb_descriptor": {"avg": 8.0, "min": 7.8},
            "stage4_matching": {"avg": 15.0, "min": 14.7},
            "stage5_ransac": {"avg": 25.0, "min": 24.1},
            "stage6_warping_blending": {"avg": 30.0, "min": 29.5},
            "kernel_total": {"avg": 95.0, "min": 92.4},
            "h2d": {"avg": 0.0, "min": 0.0},
            "control": {"avg": 0.0, "min": 0.0},
            "d2h": {"avg": 0.0, "min": 0.0},
            "end_to_end": {"avg": 98.0, "min": 95.2},
            "setup": {"avg": 0.0, "min": 0.0}
        }
    }
    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sample_valid, f)
        temp_path = f.name
        
    try:
        assert validate_metrics_json(temp_path) == True, "Failed to validate legitimate metrics JSON"
    finally:
        os.remove(temp_path)
        
    # Invalid sample (missing keys)
    sample_invalid = {"schema_version": "4.0"}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sample_invalid, f)
        temp_bad_path = f.name
        
    try:
        assert validate_metrics_json(temp_bad_path) == False, "Validator accepted invalid JSON"
    finally:
        os.remove(temp_bad_path)
        
    print("  -> Schema validator logic tested (PASS)")

def test_preprocessed_images_exist():
    print("[Python Test 4] Checking preprocessed image files...")
    sites = ["site1_indoor", "site2_outdoor", "site3_custom", "challenging"]
    resolutions = ["small", "medium", "large"]
    
    for s in sites:
        for r in resolutions:
            left_img = WORKSPACE_ROOT / "dataset" / s / f"left_{r}.jpg"
            right_img = WORKSPACE_ROOT / "dataset" / s / f"right_{r}.jpg"
            assert left_img.is_file(), f"Missing {left_img}"
            assert right_img.is_file(), f"Missing {right_img}"
            
def test_midterm_notebook_validity():
    print("[Python Test 5] Checking Group19_Midterm.ipynb validity with nbformat...")
    import nbformat
    nb_path = WORKSPACE_ROOT / "notebooks" / "Group19_Midterm.ipynb"
    assert nb_path.is_file(), f"Notebook file not found at: {nb_path}"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    assert len(nb.cells) >= 20, f"Expected at least 20 cells in Group19_Midterm.ipynb, got {len(nb.cells)}"
    print(f"  -> Group19_Midterm.ipynb parsed successfully with {len(nb.cells)} cells (PASS)")

if __name__ == "__main__":
    print("=========================================================")
    print("Running Python Integration Tests (Plan v4 Quality Gates)")
    print("=========================================================")
    test_manifest_structure()
    test_synthetic_ground_truth()
    test_validate_metrics_script()
    test_preprocessed_images_exist()
    test_midterm_notebook_validity()
    print("=========================================================")
    print("[ALL PYTHON INTEGRATION TESTS PASSED]")
    print("=========================================================")
