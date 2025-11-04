#include "io.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace filters;

static std::string extLower(const std::string& p) {
    auto pos = p.find_last_of('.');
    std::string e = (pos == std::string::npos? "" : p.substr(pos + 1));
    for (auto& ch: e) ch = (char)std::tolower((unsigned char)ch);
    return e;
}

bool filters::readImage(const std::string& path, ImageU8& img) {
    const std::string e = extLower(path);
    if (e == "ppm" || e == "pnm") return readPPM(path, img);
    
    int w, h, n;
    stbi_uc* data = stbi_load(path.c_str(), &w, &h, &n, 0);
    if (!data) return false;

    img.w = w;
    img.h = h;
    img.c = n;
    img.data.assign(data, data + (size_t)w*h*n);
    stbi_image_free(data);
    return true;
}

bool filters::writeImage(const std::string& path, const ImageU8& img){
    const std::string e = extLower(path);
    if (e=="ppm" || e=="pnm") return writePPM(path, img);
    if (!(img.c==1 || img.c==3 || img.c==4)) return false;

    int ok = 0;
    if (e == "png") {
        ok = stbi_write_png(path.c_str(), img.w, img.h, img.c, img.data.data(), img.w * img.c);
    } else if (e == "jpg" || e == "jpeg") {
        ok = stbi_write_jpg(path.c_str(), img.w, img.h, img.c, img.data.data(), 90);
    } else if (e == "bmp") {
        ok = stbi_write_bmp(path.c_str(), img.w, img.h, img.c, img.data.data());
    } else {
        // fallback on png
        ok = stbi_write_png(path.c_str(), img.w, img.h, img.c, img.data.data(), img.w*img.c);
    }
    return ok!=0;
}
