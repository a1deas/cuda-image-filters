#include <cuda_runtime.h>
#include "common.hpp"
#include "filters.hpp"

__global__ void kGrayscale(const unsigned char* __restrict__ in, unsigned char* out, int Width, int Height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= Width || y >= Height) {
        return;
    }

    int i = (y * Width + x);
    int r = in[3 * i + 0];
    int g = in[3 * i + 1];
    int b = in[3 * i + 2];

    out[i] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b + 0.5f);
}
namespace filters {

bool grayscaleCUDA(const ImageU8& in, ImageU8& out, int tile, ::cudaStream_t stream) {
  if (in.c != 3) return false;
  out = { in.w, in.h, 1, {} };
  out.data.resize(in.w * in.h);

  const size_t bytesIn  = (size_t)in.w * in.h * 3;
  const size_t bytesOut = (size_t)in.w * in.h;

  unsigned char *d_in = nullptr, *d_out = nullptr;
  CUDA_OK(cudaMalloc(&d_in,  bytesIn));
  CUDA_OK(cudaMalloc(&d_out, bytesOut));

  CUDA_OK(cudaMemcpyAsync(d_in, in.data.data(), bytesIn, cudaMemcpyHostToDevice, stream));

  int T = (tile <= 0 ? 16 : tile);
  dim3 blk(T, T);
  dim3 grd((in.w + T - 1) / T, (in.h + T - 1) / T);

  GPUTimer t; t.start(stream);
  kGrayscale<<<grd, blk, 0, stream>>>(d_in, d_out, in.w, in.h);
  CUDA_OK(cudaGetLastError());
  float ms = t.stop(stream);
  printf("grayscaleCUDA: %.3f ms\n", ms);

  CUDA_OK(cudaMemcpyAsync(out.data.data(), d_out, bytesOut, cudaMemcpyDeviceToHost, stream));
  CUDA_OK(cudaStreamSynchronize(stream));

  cudaFree(d_in);
  cudaFree(d_out);
  return true;
}

// Mock
bool boxblurCUDA(const ImageU8& in, ImageU8& out, int tile, ::cudaStream_t stream) {
  boxblurCPU(in, out);
  printf("Boxblur CUDA fallback on CPU.\n");
  return true;
}

}