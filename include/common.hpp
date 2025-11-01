#pragma once 
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

// Macro to check for CUDA errors
#define CUDA_OK(stmt) do {                                  \
    cudaError_t err = (stmt);                               \
    if (err != cudaSuccess) {                               \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n",    \
            cudaGetErrorString(err), __FILE__, __LINE__);   \
        std::abort();                                       \
    }                                                       \
} while(0)                                                  

// Kernel recording struct
struct GPUTimer {
    cudaEvent_t a{}, b{};
    float ms{};
    GPUTimer() { CUDA_OK(cudaEventCreate(&a)); CUDA_OK(cudaEventCreate(&b)); }
    ~GPUTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }

    // start recording
    void start(cudaStream_t stream = nullptr) { 
        CUDA_OK(cudaEventRecord(a, stream)); }
    // stop recording and syncrinize
    float stop(cudaStream_t stream = nullptr) { 
        CUDA_OK(cudaEventRecord(b, stream)); 
        CUDA_OK(cudaEventSynchronize(b)); 
        CUDA_OK(cudaEventElapsedTime(&ms, a, b)); 
        return ms; }
};

