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
        int gray = src[3 * i+1];
        int b = src[3 * i+2];

        dst[i] = (unsigned char)std::clamp(int(0.299f * r + 0.587f * gray + 0.114f * b + 0.5f), 0, 255);
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

void filters::gaussCPU(const ImageU8& in, ImageU8& out) {
    ImageU8 gray = in;
    if (in.c == 3) { grayscaleCPU(in, gray); }
    out = {in.w, in.h, 1, {}};
    out.data.resize((size_t)in.w * in.h);

    static const int k[5] = {1, 4, 6, 4, 1};
    auto clamp = [&](int v, int lo, int hi) { return v<lo ? lo : v> hi ? hi : v; };
    
    std::vector<unsigned char> tmp((size_t)in.w * in.h);
    for (int y = 0; y < in.h; ++y) {
        for (int x = 0; x < in.w; ++x) {
            int s = 0, n = 0;
            for (int dx = -2; dx <= 2; ++dx) {
                int xx = clamp(x + dx, 0, in.w - 1);
                s += k[dx + 2] * (int)gray.data[y * in.w + xx];
                n += k[dx + 2];
            }
            tmp[y * in.w + x] = (unsigned char)((s + n / 2) / n);
        }
    }
    for (int y = 0; y < in.h; ++y){
        for (int x=0; x < in.w; ++x){
            int s=0, n=0;
            for (int dy= -2; dy <= 2; ++dy){
                int yy = clamp(y + dy, 0, in.h - 1);
                s += k[dy + 2]* (int)tmp[yy * in.w + x];
                n += k[dy + 2];
            }
            out.data[y * in.w + x] = (unsigned char)((s + n / 2) / n);
        }
    }
}

void filters::sobelCPU(const ImageU8& in, ImageU8& out){
    ImageU8 gray = in;
    if (in.c == 3) { grayscaleCPU(in, gray); }
    out = { in.w, in.h, 1, {} };
    out.data.resize((size_t)in.w * in.h);

    auto clamp = [&](int v,int lo,int hi){ return v < lo ? lo : v > hi ? hi : v; };
    for (int y=0; y < in.h; ++y){
        for (int x = 0; x < in.w; ++x){
            int x0 = clamp(x-1, 0, in.w -1), x1 = x, x2 = clamp(x +1, 0, in.w -1);
            int y0 = clamp(y-1, 0, in.h -1), y1 = y, y2 = clamp(y +1, 0, in.h -1);
            int p00 = gray.data[y0 * in.w + x0], p01=gray.data[y0 * in.w + x1], p02 = gray.data[y0 * in.w + x2];
            int p10 = gray.data[y1 * in.w + x0], p11=gray.data[y1 * in.w + x1], p12 = gray.data[y1 * in.w + x2];
            int p20 = gray.data[y2 * in.w + x0], p21=gray.data[y2 * in.w + x1], p22 = gray.data[y2 * in.w + x2];
            int gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
            int gy = (p00 + 2 * p01 + p02) - (p20 + 2 * p21 + p22);
            int mag = abs(gx) + abs(gy); // L1
            if (mag > 255) 
                mag = 255;
            out.data[y * in.w + x] = (unsigned char)mag;
        }
    }
}


void filters::unsharpCPU(const ImageU8& in, ImageU8& out, float amount){
    ImageU8 gray;
    if (in.c == 3) 
        grayscaleCPU(in, gray);
    else gray = in;
    ImageU8 blur; 
    gaussCPU(gray, blur);
    out = { in.w, in.h, 1, {} };
    out.data.resize((size_t)in.w * in.h);
    for (int i = 0; i < in.w * in.h; ++i){
        int base = (int)gray.data[i];
        int det = base - (int)blur.data[i];
        int val = base + (int)(amount * det + 0.5f);
        if (val < 0) val = 0; 
        if (val > 255) val = 255;
        out.data[i]=(unsigned char)val;
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
            case Filter::GaussianBlur:      gaussCPU (in, out);   return true;      
            case Filter::SobelEdge:         sobelCPU  (in, out);   return true;
            case Filter::Unsharp:           unsharpCPU(in, out, 1.0f); return true;
            default:                        return false;
        }
    }

    // CUDA
    extern bool grayscaleCUDA(const ImageU8&, ImageU8&, int, cudaStream_t);
    extern bool boxblurCUDA(const ImageU8&, ImageU8&, int, cudaStream_t);

    switch (filter) {
        case Filter::Grayscale:             return grayscaleCUDA(in, out, tile, stream);
        case Filter::Boxblur:               return boxblurCUDA(in, out, tile, stream);
        case Filter::GaussianBlur:          return gaussCUDA(in, out, tile, stream);
        case Filter::SobelEdge:             return sobelCUDA(in, out, tile, stream);
        case Filter::Unsharp:               return unsharpCUDA(in, out, 1.0f, tile, stream);
        default:                            return false;
    }
}