#pragma once 
#include <string>
#include "ppm.hpp"

namespace filters {

bool readImage(const std::string& path, ImageU8& img);
bool writeImage(const std::string& path, const ImageU8& img);

} // namespace filters