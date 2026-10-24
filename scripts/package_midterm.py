#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Package Midterm Submission Archive (Plan v4 Section 7 & B3)
Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
         Dao Quoc Tuan (23120392)
External sources: Python standard-library documentation; no copied code.
Description:
  - Packages the required Midterm submission files into Group19_Midterm.zip
  - Verifies inclusion of Group19_Midterm.ipynb and single-source src/ directory
  - Computes SHA-256 checksum for verification and archiving.
============================================================================
"""

import os
import sys
import zipfile
import hashlib
import argparse
import json
import re
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def validate_notebook(notebook_path: Path, require_executed: bool = True) -> None:
    with open(notebook_path, "r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells = notebook.get("cells", [])
    if len(cells) < 20:
        raise ValueError(f"Notebook has only {len(cells)} cells; expected at least 20")
    source = "\n".join("".join(cell.get("source", [])) for cell in cells)
    if "git clone" in source or "git pull" in source:
        raise ValueError("Notebook depends on mutable Git state instead of packaged src/")
    required_text = [
        "TEAM MEMBERS AND WORKLOAD ALLOCATION",
        "THIRD-PARTY SOURCES",
        "Amdahl",
        "OpenCV",
        "challenging",
    ]
    missing = [item for item in required_text if item not in source]
    if missing:
        raise ValueError(f"Notebook is missing required content: {missing}")
    errors = [
        output for cell in cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise ValueError(f"Notebook contains {len(errors)} saved error output(s)")
    if require_executed:
        unexecuted = [
            index for index, cell in enumerate(cells)
            if cell.get("cell_type") == "code" and cell.get("execution_count") is None
        ]
        if unexecuted:
            raise ValueError(f"Notebook has unexecuted code cells: {unexecuted}")


def package_midterm(group_name: str = "Group19", require_executed: bool = True):
    if not re.fullmatch(r"Group\d+", group_name):
        raise ValueError("Group name must match 'Group' followed by digits")
    zip_filename = f"{group_name}_Midterm.zip"
    zip_path = WORKSPACE_ROOT / zip_filename

    # Source notebook must match the requested group exactly (search notebooks/ then root).
    notebook_source = WORKSPACE_ROOT / "notebooks" / f"{group_name}_Midterm.ipynb"
    if not notebook_source.is_file():
        notebook_source = WORKSPACE_ROOT / f"{group_name}_Midterm.ipynb"

    if not notebook_source.is_file():
        raise FileNotFoundError(f"Midterm notebook not found for {group_name}")

    validate_notebook(notebook_source, require_executed=require_executed)

    print(f"=== PACKAGING MIDTERM SUBMISSION: {zip_filename} ===")

    # Required files and directories
    entries_to_pack = [
        # Root notebook named with group ID
        (notebook_source, f"{group_name}_Midterm.ipynb"),
        (WORKSPACE_ROOT / "Makefile", "Makefile"),
        (WORKSPACE_ROOT / "README.md", "README.md"),
        (WORKSPACE_ROOT / "requirements.txt", "requirements.txt"),
        (WORKSPACE_ROOT / "project_plan_v4.md", "project_plan_v4.md"),
        (WORKSPACE_ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md"),
    ]

    # Directories to pack recursively
    dirs_to_pack = [
        ("src", ["common.h", "cuda_utils.cuh", "versions.h", "main.cu", 
                 "stb_image.h", "stb_image_write.h", "stb_impl.h", "stb_impl.cpp", 
                 "v0_sequential.h", "v0_sequential.cpp"]),
        ("scripts", ["preprocess_dataset.py", "generate_synthetic_test.py", 
                     "orb_pattern_data.py", "validate_metrics.py",
                     "validate_homography.py",
                     "build_midterm_notebook.py", "execute_notebook.py",
                     "package_midterm.py"]),
        ("tests", ["test_cpu.cpp", "test_scripts.py"]),
    ]

    missing_paths = [
        path for path, _ in entries_to_pack if not path.is_file()
    ]
    for dir_name, expected_files in dirs_to_pack:
        missing_paths.extend(
            WORKSPACE_ROOT / dir_name / file_name
            for file_name in expected_files
            if not (WORKSPACE_ROOT / dir_name / file_name).is_file()
        )
    if missing_paths:
        raise FileNotFoundError(f"Required package inputs are missing: {missing_paths}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add root files
        for src_path, arcname in entries_to_pack:
            zf.write(src_path, arcname)
            print(f"  + Added: {arcname}")

        # Add directories
        for dir_name, expected_files in dirs_to_pack:
            dir_path = WORKSPACE_ROOT / dir_name
            for file_name in expected_files:
                fpath = dir_path / file_name
                arc_path = f"{dir_name}/{file_name}"
                zf.write(fpath, arc_path)
                print(f"  + Added: {arc_path}")

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
    parser.add_argument(
        "--allow-unexecuted-notebook",
        action="store_true",
        help="Use only for the in-notebook preflight before the frontend saves outputs.",
    )
    args = parser.parse_args()

    package_midterm(args.group, require_executed=not args.allow_unexecuted_notebook)
