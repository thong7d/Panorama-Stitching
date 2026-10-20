#pragma once

#include "common.h"
#include <string>

// Helper functions for loading and saving Image
bool load_image(const std::string& path, Image& img);
bool save_image(const std::string& path, const Image& img, int quality = 95);
bool save_gray_image(const std::string& path, const GrayImage& gray_img);
