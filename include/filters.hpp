#include <vector>
#include "ppm.hpp"

// forward declaration 
#ifndef CIF_FWD_CUDA_STREAM_T
struct CUstream_st;
using cudaStream_t = CUstream_st*;  // global
#define CIF_FWD_CUDA_STREAM_T 1
#endif

namespace filters {

enum class Device { CPU, CUDA };
enum class Filter { 
    Grayscale, 
    Boxblur, 
    Sharpen, 
    GaussianBlur, 
    SobelEdge   
};

bool apply(
    Filter filter, 
    Device device, 
    const ImageU8& in, 
    ImageU8& out,
    int tile = 16, 
    ::cudaStream_t stream = nullptr);

// CPU References
void grayscaleCPU(const ImageU8& in, ImageU8& out);
void boxblurCPU(const ImageU8& in, ImageU8& out);

// CUDA References
bool grayscaleCUDA(const ImageU8& in, ImageU8& out, int tile, cudaStream_t stream);
bool boxblurCUDA(const ImageU8& in, ImageU8& out, int tile, cudaStream_t stream);
} // namespace filters