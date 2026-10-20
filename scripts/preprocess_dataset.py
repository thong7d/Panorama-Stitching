#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Parallel Programming
Script: Preprocess Dataset (PNG/JPEG/HEIC -> Standardized 16:9 Multi-resolution)
Authors: GroupXX (Members A, B, C)
Description:
  - Reads dataset/manifest.json
  - Registers pillow_heif opener if available
  - Corrects EXIF orientation
  - Center-crops images to 16:9 aspect ratio
  - Generates 3 resolutions: Small (1280x720), Medium (1920x1080), Large (3840x2160)
  - Validates output dimensions and returns 0 on success, non-zero on failure.
============================================================================
"""

import os
import sys
import json
from PIL import Image, ImageOps

# Register HEIF opener if pillow_heif is available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

RESOLUTIONS = {
    'small': (1280, 720),
    'medium': (1920, 1080),
    'large': (3840, 2160)
}

def crop_center_16_9(img: Image.Image) -> Image.Image:
    """Center crop image to exact 16:9 aspect ratio."""
    w, h = img.size
    target_ratio = 16.0 / 9.0
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is wider than 16:9 -> crop left and right margins
        new_w = int(round(h * target_ratio))
        offset = (w - new_w) // 2
        return img.crop((offset, 0, offset + new_w, h))
    else:
        # Image is taller than 16:9 -> crop top and bottom margins
        new_h = int(round(w / target_ratio))
        offset = (h - new_h) // 2
        return img.crop((0, offset, w, offset + new_h))

def process_file(raw_path: str, out_dir: str, prefix: str) -> bool:
    """Process a raw image: EXIF orientation, 16:9 center crop, and 3 resolutions export."""
    if not os.path.isfile(raw_path):
        print(f"[Error] Raw file not found: {raw_path}")
        return False

    try:
        with Image.open(raw_path) as img:
            # 1. Correct EXIF orientation
            img = ImageOps.exif_transpose(img)
            # 2. Ensure RGB mode (strip Alpha channel if present)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 3. Center crop to 16:9
            cropped = crop_center_16_9(img)

            # 4. Export 3 standard resolutions
            for res_name, (tw, th) in RESOLUTIONS.items():
                resized = cropped.resize((tw, th), Image.Resampling.LANCZOS)
                out_path = os.path.join(out_dir, f"{prefix}_{res_name}.jpg")
                resized.save(out_path, format='JPEG', quality=95)

                # Validate saved dimension
                with Image.open(out_path) as check_img:
                    if check_img.size != (tw, th):
                        print(f"[Error] Dimension mismatch for {out_path}: expected {(tw, th)}, got {check_img.size}")
                        return False

                print(f"  [OK] Saved: {out_path} ({tw}x{th})")

        return True
    except Exception as e:
        print(f"[Error] Failed to process {raw_path}: {e}")
        return False

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_path = os.path.join(repo_root, "dataset", "manifest.json")

    if not os.path.isfile(manifest_path):
        print(f"[Error] Manifest file not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    all_success = True
    processed_count = 0

    print("=== STARTING DATASET PREPROCESSING (16:9 Multi-Resolution) ===")
    for pair in manifest.get("pairs", []):
        pair_id = pair.get("id")
        kind = pair.get("kind")
        rel_dir = pair.get("directory")
        pair_dir = os.path.join(repo_root, rel_dir)

        if kind == "provided" or pair.get("status") == "external_blocker":
            print(f"\n[Pair: {pair_id}] Skipping provided dataset (External blocker - to be downloaded when released).")
            continue

        print(f"\n[Pair: {pair_id}] Processing directory: {pair_dir}")
        left_raw_name = pair.get("left_raw")
        right_raw_name = pair.get("right_raw")

        if not left_raw_name or not right_raw_name:
            print(f"[Error] Missing left_raw or right_raw in manifest for pair {pair_id}")
            all_success = False
            continue

        left_raw_path = os.path.join(pair_dir, left_raw_name)
        right_raw_path = os.path.join(pair_dir, right_raw_name)

        ok_left = process_file(left_raw_path, pair_dir, "left")
        ok_right = process_file(right_raw_path, pair_dir, "right")

        if ok_left and ok_right:
            processed_count += 1
        else:
            all_success = False

    print(f"\n=== PREPROCESSING COMPLETED: {processed_count} pairs processed successfully. ===")
    if not all_success:
        print("[Error] Some files failed to process!")
        sys.exit(1)
    else:
        print("[Success] All standard and challenging pairs preprocessed and validated.")

if __name__ == "__main__":
    main()
