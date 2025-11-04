#include <cuda_runtime.h>
#include "common.hpp"
#include "filters.hpp"

// Grayscale (RGB->Gray) 
__global__ void kGrayscale(const unsigned char* __restrict__ in,
                           unsigned char* out, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int i = y * W + x;
    int r = in[3 * i + 0];
    int g = in[3 * i + 1];
    int b = in[3 * i + 2];
    out[i] = (unsigned char)(0.299f*r + 0.587f*g + 0.114f*b + 0.5f);
}

// BoxBlur 3x3 (shared + halo)
template<int TILE>
__global__ void kBox3Gray(const unsigned char* __restrict__ in,
                          unsigned char* out, int W, int H) {
    constexpr int R = 1;0
    __shared__ unsigned char tile[TILE + 2*R][TILE + 2*R];

    int gx = blockIdx.x * TILE + threadIdx.x;
    int gy = blockIdx.y * TILE + threadIdx.y;
    int lx = threadIdx.x + R;
    int ly = threadIdx.y + R;

    // clamp
    int x = min(max(gx, 0), W - 1);
    int y = min(max(gy, 0), H - 1);
    int idx = y * W + x;

    tile[ly][lx] = (gx < W && gy < H) ? in[idx] : 0;

    if (threadIdx.x < R) {
        int xL = max(gx - R, 0);
        int xR = min(gx + TILE, W - 1);
        tile[ly][lx - R]     = in[y * W + xL];
        tile[ly][lx + TILE]  = in[y * W + xR];
    }

    if (threadIdx.y < R) {
        int yT = max(gy - R, 0);
        int yB = min(gy + TILE, H - 1);
        tile[ly - R][lx]     = in[yT * W + x];
        tile[ly + TILE][lx]  = in[yB * W + x];
    }

    if (threadIdx.x < R && threadIdx.y < R) {
        int xL = max(gx - R, 0), xR = min(gx + TILE, W - 1);
        int yT = max(gy - R, 0), yB = min(gy + TILE, H - 1);
        tile[ly - R][lx - R]       = in[yT * W + xL];
        tile[ly - R][lx + TILE]    = in[yT * W + xR];
        tile[ly + TILE][lx - R]    = in[yB * W + xL];
        tile[ly + TILE][lx + TILE] = in[yB * W + xR];
    }

    __syncthreads();

    if (gx >= W || gy >= H) return;

    int sum = 0;
    #pragma unroll
    for (int dy = -R; dy <= R; ++dy) {
        #pragma unroll
        for (int dx = -R; dx <= R; ++dx) {
            sum += tile[ly + dy][lx + dx];
        }
    }
    out[idx] = static_cast<unsigned char>((sum + 4) / 9);
}

namespace filters {

// CUDA Grayscale 
bool grayscaleCUDA(const ImageU8& in, ImageU8& out, int tile, cudaStream_t stream) {
    if (in.c != 3) return false;

    out = { in.w, in.h, 1, {} };
    out.data.resize((size_t)in.w * in.h);

    const size_t bytesIn  = (size_t)in.w * in.h * 3;
    const size_t bytesOut = (size_t)in.w * in.h;

    unsigned char *dIn = nullptr, *dOut = nullptr;
    CUDA_OK(cudaMalloc(&dIn,  bytesIn));
    CUDA_OK(cudaMalloc(&dOut, bytesOut));

    CUDA_OK(cudaMemcpyAsync(dIn, in.data.data(), bytesIn, cudaMemcpyHostToDevice, stream));

    int T = (tile <= 0 ? 16 : tile);
    dim3 blk(T, T), grd((in.w + T - 1) / T, (in.h + T - 1) / T);

    GPUTimer t; t.start(stream);
    kGrayscale<<<grd, blk, 0, stream>>>(dIn, dOut, in.w, in.h);
    CUDA_OK(cudaGetLastError());
    (void)t.stop(stream);

    CUDA_OK(cudaMemcpyAsync(out.data.data(), dOut, bytesOut, cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    cudaFree(dIn);
    cudaFree(dOut);
    return true;
}

// CUDA BoxBlur 3x3 
bool boxblurCUDA(const ImageU8& in, ImageU8& out, int tile, cudaStream_t stream) {
    ImageU8 gray;
    if (in.c == 3) {
        if (!grayscaleCUDA(in, gray, tile, stream)) return false;
    } else {
        gray = in;
    }

    out = { in.w, in.h, 1, {} };
    out.data.resize((size_t)in.w * in.h);

    const size_t bytes = (size_t)in.w * in.h;
    unsigned char *dIn = nullptr, *dOut = nullptr;
    CUDA_OK(cudaMalloc(&dIn,  bytes));
    CUDA_OK(cudaMalloc(&dOut, bytes));

    CUDA_OK(cudaMemcpyAsync(dIn, gray.data.data(), bytes, cudaMemcpyHostToDevice, stream));

    int T = (tile <= 0 ? 16 : tile);
    dim3 blk(T, T), grd((in.w + T - 1) / T, (in.h + T - 1) / T);

    GPUTimer timer; timer.start(stream);
    if      (T == 16) kBox3Gray<16><<<grd, blk, 0, stream>>>(dIn, dOut, in.w, in.h);
    else if (T == 32) kBox3Gray<32><<<grd, blk, 0, stream>>>(dIn, dOut, in.w, in.h);
    else              kBox3Gray<16><<<dim3((in.w+15)/16,(in.h+15)/16), dim3(16,16), 0, stream>>>(dIn, dOut, in.w, in.h);
    CUDA_OK(cudaGetLastError());
    (void)timer.stop(stream);

    CUDA_OK(cudaMemcpyAsync(out.data.data(), dOut, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    cudaFree(dIn);
    cudaFree(dOut);
    return true;
}

} // namespace filters
