// ============================================================================
// CSC14120 - Parallel Programming
// Project: Panorama Image Stitching
// Module: Image load/save helpers (declarations)
// Authors: Group 19 - Pham Minh Thong (23120364), Nguyen Le Tan Tien (23120369), Dao Quoc Tuan (23120392)
// External code: wraps stb_image / stb_image_write (public domain, Sean Barrett,
//   https://github.com/nothings/stb)
// ============================================================================

#pragma once

#include "common.h"
#include <string>

// Helper functions for loading and saving Image
bool load_image(const std::string& path, Image& img);
bool save_image(const std::string& path, const Image& img, int quality = 95);
bool save_gray_image(const std::string& path, const GrayImage& gray_img);
