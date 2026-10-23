// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Image I/O Implementation using stb_image / stb_image_write
// Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// External sources: stb_image.h / stb_image_write.h — public domain, Sean T. Barrett
//   (https://github.com/nothings/stb, MIT / Unlicense)
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
    // Force 4 channels (RGBA)
    unsigned char* raw = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!raw) {
        std::cerr << "[Error] Failed to load image: " << path 
                  << " (reason: " << stbi_failure_reason() << ")\n";
        return false;
    }

    img.allocate(w, h);
    // Copy into uchar4 data
    std::memcpy(img.data, raw, (size_t)w * h * sizeof(uchar4));
    stbi_image_free(raw);
    return true;
}

bool save_image(const std::string& path, const Image& img, int quality) {
    if (!img.data || img.w <= 0 || img.h <= 0) {
        std::cerr << "[Error] Cannot save empty image to " << path << "\n";
        return false;
    }

    int success = 0;
    // Check extension
    if (path.size() >= 4 && path.substr(path.size() - 4) == ".png") {
        success = stbi_write_png(path.c_str(), img.w, img.h, 4, img.data, img.w * 4);
    } else {
        success = stbi_write_jpg(path.c_str(), img.w, img.h, 4, img.data, quality);
    }

    if (!success) {
        std::cerr << "[Error] Failed to save image: " << path << "\n";
        return false;
    }
    return true;
}

bool save_gray_image(const std::string& path, const GrayImage& gray_img) {
    if (!gray_img.data || gray_img.w <= 0 || gray_img.h <= 0) return false;

    // Convert float [0..255] to uint8_t
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
