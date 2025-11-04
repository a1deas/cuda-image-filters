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
    Unsharp, 
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

    
// CUDA References
bool grayscaleCUDA(const ImageU8& in, ImageU8& out, int tile, cudaStream_t stream);
bool boxblurCUDA(const ImageU8& in, ImageU8& out, int tile, cudaStream_t stream);
bool gaussCUDA(const ImageU8& in, ImageU8& out, int tile, cudaStream_t stream);
bool sobelCUDA(const ImageU8& in, ImageU8& out, int tile, cudaStream_t stream);
bool unsharpCUDA(const ImageU8& in, ImageU8& out, float amount, int tile, cudaStream_t stream);

// CPU References
void grayscaleCPU(const ImageU8& in, ImageU8& out);
void boxblurCPU(const ImageU8& in, ImageU8& out);
void gaussCPU(const ImageU8& in, ImageU8& out);
void sobelCPU(const ImageU8& in, ImageU8& out);
void unsharpCPU(const ImageU8& in, ImageU8& out, float amount);

} // namespace filters