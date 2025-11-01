#pragma once 
#include <vector>
#include <string>

namespace filters {
    // image structure
    struct ImageU8 { int w = 0, h = 0, c = 0; std::vector<unsigned char> data; };
    
    // PPM 
    bool readPPM(const std::string& path, ImageU8& img);
    bool writePPM(const std::string& path, const ImageU8& img);
}