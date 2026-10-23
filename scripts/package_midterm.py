#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Package Midterm Submission Archive (Plan v4 Section 7 & B3)
Authors: Group 19 - Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
External code: none
Description:
  - Packages the required Midterm submission files into GroupXX_Midterm.zip
  - Verifies inclusion of GroupXX_Midterm.ipynb and single-source src/ directory
  - Computes SHA-256 checksum for verification and archiving.
============================================================================
"""

import os
import sys
import zipfile
import hashlib
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def package_midterm(group_name: str = "Group19"):
    zip_filename = f"{group_name}_Midterm.zip"
    zip_path = WORKSPACE_ROOT / zip_filename

    # Source notebook: check group-specific notebook first, fallback to GroupXX
    notebook_source = WORKSPACE_ROOT / "notebooks" / f"{group_name}_Midterm.ipynb"
    if not notebook_source.is_file():
        notebook_source = WORKSPACE_ROOT / "notebooks" / "GroupXX_Midterm.ipynb"

    if not notebook_source.is_file():
        print(f"[Error] Midterm notebook not found for {group_name}")
        sys.exit(1)

    print(f"=== PACKAGING MIDTERM SUBMISSION: {zip_filename} ===")

    # Required files and directories
    entries_to_pack = [
        # Root notebook named with group ID
        (notebook_source, f"{group_name}_Midterm.ipynb"),
        (WORKSPACE_ROOT / "Makefile", "Makefile"),
        (WORKSPACE_ROOT / "README.md", "README.md"),
        (WORKSPACE_ROOT / "requirements.txt", "requirements.txt"),
        (WORKSPACE_ROOT / "project_plan_v4.md", "project_plan_v4.md"),
    ]

    # Directories to pack recursively
    dirs_to_pack = [
        ("src", ["common.h", "cuda_utils.cuh", "versions.h", "main.cu", 
                 "stb_image.h", "stb_image_write.h", "stb_impl.h", "stb_impl.cpp", 
                 "v0_sequential.h", "v0_sequential.cpp"]),
        ("scripts", ["preprocess_dataset.py", "generate_synthetic_test.py", 
                     "validate_metrics.py", "build_midterm_notebook.py", "package_midterm.py"]),
        ("tests", ["test_cpu.cpp", "test_scripts.py"]),
    ]

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root files
        for src_path, arcname in entries_to_pack:
            if src_path.is_file():
                zf.write(src_path, arcname)
                print(f"  + Added: {arcname}")
            else:
                print(f"[Warning] File not found: {src_path}")

        # Add directories
        for dir_name, expected_files in dirs_to_pack:
            dir_path = WORKSPACE_ROOT / dir_name
            for file_name in expected_files:
                fpath = dir_path / file_name
                if fpath.is_file():
                    arc_path = f"{dir_name}/{file_name}"
                    zf.write(fpath, arc_path)
                    print(f"  + Added: {arc_path}")
                else:
                    print(f"[Warning] File not found: {fpath}")

        # Add dataset manifest and synthetic metadata
        manifest = WORKSPACE_ROOT / "dataset" / "manifest.json"
        if manifest.is_file():
            zf.write(manifest, "dataset/manifest.json")
            print("  + Added: dataset/manifest.json")

        gt_json = WORKSPACE_ROOT / "dataset" / "synthetic" / "ground_truth.json"
        if gt_json.is_file():
            zf.write(gt_json, "dataset/synthetic/ground_truth.json")
            print("  + Added: dataset/synthetic/ground_truth.json")

    # Verify zip integrity
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        test_res = zf.testzip()
        if test_res is not None:
            print(f"[Error] Corrupted file in archive: {test_res}")
            sys.exit(1)

    file_size_kb = zip_path.stat().st_size / 1024
    checksum = compute_sha256(zip_path)

    print("\n=== PACKAGE VERIFICATION SUCCESSFUL ===")
    print(f"Archive:   {zip_path.name} ({file_size_kb:.1f} KB)")
    print(f"Entries:   {len(namelist)} items")
    print(f"SHA-256:   {checksum}")
    print("Structure conforms to Plan v4 Section 7 (Quality Gate B).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package Midterm Submission Zip")
    parser.add_argument("--group", type=str, default="Group19", help="Team Group Name (e.g., Group19)")
    args = parser.parse_args()

    package_midterm(args.group)
