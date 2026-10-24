// ============================================================================
// CSC14120 - Lập trình song song | Ghép ảnh Panorama
// Tác giả: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
//          Dao Quoc Tuan (23120392)
// External source: stb_image/stb_image_write bởi Sean Barrett và cộng sự,
// public domain hoặc MIT; xem THIRD_PARTY_NOTICES.md.
// ============================================================================

#pragma once

#include "common.h"
#include <string>

// Các hàm tiện ích tải và lưu cấu trúc Image
bool load_image(const std::string& path, Image& img);
bool save_image(const std::string& path, const Image& img, int quality = 95);
bool save_gray_image(const std::string& path, const GrayImage& gray_img);

