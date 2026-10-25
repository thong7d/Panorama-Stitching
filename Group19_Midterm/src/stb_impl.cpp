// ============================================================================
// CSC14120 - Lập trình song song | Ghép ảnh Panorama
// Tác giả: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369),
//          Dao Quoc Tuan (23120392)
// External source: stb_image/stb_image_write bởi Sean Barrett và cộng sự,
// public domain hoặc MIT; xem THIRD_PARTY_NOTICES.md.
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "stb_impl.h"
#include <iostream>
#include <cstring>
#include <algorithm>

bool load_image(const std::string& path, Image& img) {
    int w, h, channels;
    // Bắt buộc 4 kênh màu (RGBA)
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!raw) {
        std::cerr << "[Lỗi] Không thể tải ảnh: " << path 
                  << " (nguyên nhân: " << stbi_failure_reason() << ")\n";
        return false;
    }

    img.allocate(w, h);
    // Sao chép sang bộ nhớ uchar4
    std::memcpy(img.data, raw, (size_t)w * h * sizeof(uchar4));
    stbi_image_free(raw);
    return true;
}

bool save_image(const std::string& path, const Image& img, int quality) {
    if (!img.data || img.w <= 0 || img.h <= 0) {
        std::cerr << "[Lỗi] Không thể lưu ảnh rỗng vào " << path << "\n";
        return false;
    }

    int success = 0;
    // Kiểm tra định dạng phần mở rộng
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".png") {
        success = stbi_write_png(path.c_str(), img.w, img.h, 4, img.data, img.w * 4);
    } else {
        success = stbi_write_jpg(path.c_str(), img.w, img.h, 4, img.data, quality);
    }

    if (!success) {
        std::cerr << "[Lỗi] Lưu ảnh thất bại: " << path << "\n";
        return false;
    }
    return true;
}

bool save_gray_image(const std::string& path, const GrayImage& gray_img) {
    if (!gray_img.data || gray_img.w <= 0 || gray_img.h <= 0) return false;

    // Chuyển đổi float [0..255] sang uint8_t
    int total_pixels = gray_img.w * gray_img.h;
    std::vector<uint8_t> buf(total_pixels);
    for (int i = 0; i < total_pixels; i++) {
        float val = std::clamp(gray_img.data[i], 0.0f, 255.0f);
        buf[i] = static_cast<uint8_t>(val + 0.5f);
    }

    int success = 0;
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".png") {
        success = stbi_write_png(path.c_str(), gray_img.w, gray_img.h, 1, buf.data(), gray_img.w);
    } else {
        success = stbi_write_jpg(path.c_str(), gray_img.w, gray_img.h, 1, buf.data(), 90);
    }
    return success != 0;
}

