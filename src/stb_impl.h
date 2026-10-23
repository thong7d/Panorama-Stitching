// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Image I/O Helper (load/save via stb_image)
// Authors: Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// External sources: stb_image.h / stb_image_write.h — public domain, Sean T. Barrett
//   (https://github.com/nothings/stb, MIT / Unlicense)
// ============================================================================

#pragma once

#include "common.h"
#include <string>

// Helper functions for loading and saving Image
bool load_image(const std::string& path, Image& img);
bool save_image(const std::string& path, const Image& img, int quality = 95);
bool save_gray_image(const std::string& path, const GrayImage& gray_img);
