#!/usr/bin/env python3
"""
============================================================================
CSC14120 - Lập trình song song
Kịch bản: Tiền xử lý tập dữ liệu (PNG/JPEG/HEIC -> Chuẩn hóa 16:9 Đa độ phân giải)
Tác giả: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
         Dao Quoc Tuan (23120392)
External sources: Tài liệu API Pillow và pillow-heif; không sao chép mã nguồn bên ngoài.
Mô tả:
  - Đọc dataset/manifest.json
  - Đăng ký bộ đọc HEIF nếu có sẵn thư viện
  - Hiệu chỉnh hướng xoay EXIF
  - Cắt trọng tâm theo tỷ lệ chuẩn 16:9
  - Xuất 3 độ phân giải: Small (1280x720), Medium (1920x1080), Large (3840x2160)
  - Xác thực kích thước ảnh đầu ra và trả về 0 nếu thành công.
============================================================================
"""

import os
import sys
import json
from PIL import Image, ImageOps

# Đăng ký bộ mở file HEIF nếu thư viện pillow_heif khả dụng
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
    """Cắt giữa ảnh theo tỷ lệ khung hình 16:9 chuẩn xác."""
    w, h = img.size
    target_ratio = 16.0 / 9.0
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Ảnh rộng hơn 16:9 -> Cắt bớt lề trái và phải
        new_w = int(round(h * target_ratio))
        offset = (w - new_w) // 2
        return img.crop((offset, 0, offset + new_w, h))
    else:
        # Ảnh cao hơn 16:9 -> Cắt bớt lề trên và dưới
        new_h = int(round(w / target_ratio))
        offset = (h - new_h) // 2
        return img.crop((0, offset, w, offset + new_h))

def process_file(raw_path: str, out_dir: str, prefix: str) -> bool:
    """Xử lý ảnh gốc: xoay EXIF, cắt tâm 16:9, và xuất 3 độ phân giải chuẩn."""
    if not os.path.isfile(raw_path):
        print(f"[Lỗi] Không tìm thấy file gốc: {raw_path}")
        return False

    try:
        with Image.open(raw_path) as img:
            # 1. Hiệu chỉnh góc xoay theo EXIF
            img = ImageOps.exif_transpose(img)
            # 2. Đảm bảo hệ màu RGB (bỏ kênh Alpha nếu có)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 3. Cắt trọng tâm 16:9
            cropped = crop_center_16_9(img)

            # 4. Xuất 3 độ phân giải chuẩn
            for res_name, (tw, th) in RESOLUTIONS.items():
                resized = cropped.resize((tw, th), Image.Resampling.LANCZOS)
                out_path = os.path.join(out_dir, f"{prefix}_{res_name}.jpg")
                resized.save(out_path, format='JPEG', quality=95)

                # Xác thực lại kích thước đã lưu
                with Image.open(out_path) as check_img:
                    if check_img.size != (tw, th):
                        print(f"[Lỗi] Sai lệch kích thước tại {out_path}: kỳ vọng {(tw, th)}, thực tế {check_img.size}")
                        return False

                print(f"  [OK] Đã lưu: {out_path} ({tw}x{th})")

        return True
    except Exception as e:
        print(f"[Lỗi] Không thể xử lý file {raw_path}: {e}")
        return False

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    manifest_path = os.path.join(repo_root, "dataset", "manifest.json")

    if not os.path.isfile(manifest_path):
        print(f"[Lỗi] Không tìm thấy tệp manifest tại {manifest_path}")
        sys.exit(1)

    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    all_success = True
    processed_count = 0

    print("=== BẮT ĐẦU TIỀN XỬ LÝ DATASET (Chuẩn hóa 16:9 Đa độ phân giải) ===")
    for pair in manifest.get("pairs", []):
        pair_id = pair.get("id")
        kind = pair.get("kind")
        rel_dir = pair.get("directory")
        pair_dir = os.path.join(repo_root, rel_dir)

        if kind == "provided" or pair.get("status") == "external_blocker":
            print(f"\n[Cặp: {pair_id}] Bỏ qua dataset giảng viên (External blocker - sẽ tải về khi được cung cấp).")
            continue

        print(f"\n[Cặp: {pair_id}] Đang xử lý thư mục: {pair_dir}")
        left_raw_name = pair.get("left_raw")
        right_raw_name = pair.get("right_raw")

        if not left_raw_name or not right_raw_name:
            print(f"[Lỗi] Thiếu left_raw hoặc right_raw trong manifest cho cặp {pair_id}")
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

    print(f"\n=== HOÀN TẤT TIỀN XỬ LÝ: {processed_count} cặp ảnh đã xử lý thành công. ===")
    if not all_success:
        print("[Lỗi] Một số tệp gặp sự cố khi xử lý!")
        sys.exit(1)
    else:
        print("[Thành công] Tất cả các cặp ảnh chuẩn và ảnh thử thách đã được tiền xử lý và xác thực.")

if __name__ == "__main__":
    main()

