#include "filters.hpp"
#include <algorithm>

using namespace filters;

// Filters
// Grayscale filter
void filters::grayscaleCPU(const ImageU8& in, ImageU8& out) {
    out = { in.w, in.h, 1, {} };
    out.data.resize(in.w * in.h);
    const unsigned char* src = in.data.data();
    unsigned char* dst = out.data.data();
    for (int i = 0; i < in.w * in.h; i++) {
        int r = src[3 * i+0];
        int g = src[3 * i+1];
        int b = src[3 * i+2];

        dst[i] = (unsigned char)std::clamp(int(0.299f * r + 0.587f * g + 0.114f * b + 0.5f), 0, 255);
    }
}

// Box Blur Filter
void filters::boxblurCPU(const ImageU8& in, ImageU8& out) {
    ImageU8 gray;
    if (in.c == 3) 
        grayscaleCPU(in, gray);
    else
        gray = in;

    out = { in.w, in.h, 1, {} };
    out.data.resize(in.w * in.h);

    auto idx = [&](int x, int y) { return y * in.w + x; };

    for (int y = 0; y < in.h; ++y) {
        for (int x = 0; x < in.w; ++x) {
            int sum = 0, count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = std::clamp(x + dx, 0, in.w - 1);
                    int yy = std::clamp(y + dy, 0, in.h - 1);
                    sum += gray.data[idx(xx, yy)];
                    ++count;
                }
            }
            out.data[idx(x, y)] = (unsigned char)((sum + count / 2) / count);
        }
    }
}

// Apply certain filter
bool filters::apply(Filter filter, Device device,
                    const ImageU8& in, ImageU8& out,
                    int tile, cudaStream_t stream) { 
    if (device == Device::CPU) {
        switch (filter) {
            case Filter::Grayscale:         grayscaleCPU(in, out); return true;
            case Filter::Boxblur:           boxblurCPU(in, out);   return true;
            case Filter::GaussianBlur:      //mock      
            case Filter::Sharpen:           //mock
            case Filter::SobelEdge:         //mock
            default:                        return false;
        }
    }

    // CUDA
    extern bool grayscaleCUDA(const ImageU8&, ImageU8&, int, cudaStream_t);
    extern bool boxblurCUDA(const ImageU8&, ImageU8&, int, cudaStream_t);

    switch (filter) {
        case Filter::Grayscale:             return grayscaleCUDA(in, out, tile, stream);
        case Filter::Boxblur:               return boxblurCUDA(in, out, tile, stream);
        case Filter::GaussianBlur:          // mock
        case Filter::Sharpen:               // mock
        case Filter::SobelEdge:             // mock
        default:                            return false;
    }
}