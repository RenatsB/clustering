#include <cuda_runtime.h>
#include "utilTypes.hpp"
#include <curand.h>
#include <curand_kernel.h>

// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__device__ float smoothNoiseP(const thrust::device_ptr<float> d_noise,
                              const size_t width,
                              const size_t height,
                              const float x,
                              const float y);

__device__ float turbulenceP(const thrust::device_ptr<float> d_noise,
                             const size_t noiseWidth,
                             const size_t noiseHeight,
                             const float x,
                             const float y,
                             const float size);

__global__ void generateNoiseP(thrust::device_ptr<float> d_noise,
                               const size_t data_size);

__global__ void assignColorsP(thrust::device_ptr<float> d_noise,
                              const size_t data_size,
                              const size_t noiseWidth,
                              const size_t noiseHeight,
                              const size_t imageWidth,
                              const float turbulence_size,
                              thrust::device_ptr<float> d_out);

DataFrame generate(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const uint numThreads);
std::vector<float> linear_generate(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const uint numThreads);
