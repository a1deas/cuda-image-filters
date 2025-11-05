# CUDA Image Filters
GPU-accelerated image filters in modern **CUDA** with clean CPU fallback, tidy CLI and benchmark harness.

This project demonstrates image filters, such as: `unsharp`, `gaussian blur`, `sobel edge`, `box blur`, `grayscale` and many others.

# Devices
- CPU 
- GPU(CUDA of course)

## Features
- Filters:
    - Grayscale
    - Boxblur
    - Gaussian Blur
    - Sobel Edge
    - Unsharp
- PPM support
- PNG, JPG, JPEG and BNP support

## PPM Dummy

![screenshot](demo/dummy.png)

## Grayscale Filtered Dummy 

![screenshot](demo/ppm_gray.jpg)

## Boxblur Filtered Dummy

![screenshot](demo/ppm_boxblur.jpg)

## Gaussian Blur Filtered Dummy

![screenshot](demo/ppm_gauss.jpg)

## Sobel Edge Filtered Dummy

![screenshot](demo/ppm_sobel_edge.jpg)

## Unsharp Filtered Dummy

![screenshot](demo/ppm_unsharp.png)

## Next Steps
- Add many other filters

## Tech: 
C++20, CUDA 13.0, stbi_image, CMake

