#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include "gpuImageGen.h"

// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

__device__ float smoothNoiseP(const thrust::device_ptr<float> d_noise,
                             const size_t width,
                             const size_t height,
                             const float x,
                             const float y)
{
    //get fractional part of x and y
    float fractX = x - int(x);
    float fractY = y - int(y);

    //wrap around
    int x1 = (int(x) + width) % width;
    int y1 = (int(y) + height) % height;

    //neighbor values
    int x2 = (x1 + width - 1) % width;
    int y2 = (y1 + height - 1) % height;

    //smooth the noise with bilinear interpolation
    float value;
    value  = fractX       * fractY       * d_noise[y1*width+x1];
    value += (1 - fractX) * fractY       * d_noise[y1*width+x2];
    value += fractX       * (1 - fractY) * d_noise[y2*width+x1];
    value += (1 - fractX) * (1 - fractY) * d_noise[y2*width+x2];

    return value;
}

__device__ float turbulenceP(const thrust::device_ptr<float> d_noise,
                             const size_t noiseWidth,
                             const size_t noiseHeight,
                             const float x,
                             const float y,
                             const float size)
{
    float value = 0.0, initialSize = size, localSize = size;

    while(localSize >= 1)
    {
      value += smoothNoiseP(d_noise, noiseWidth, noiseHeight, x / localSize, y / localSize) * localSize;
      localSize /= 2.0;
    }

    return(128.0 * value / initialSize)/256.0;
}

__host__ void genNoiseCustom(thrust::device_ptr<float> d_noise,
                               const size_t data_size)
{
    float *ptr = thrust::raw_pointer_cast(d_noise);
    gpuRandFn::randFloatsInternal(ptr,data_size);
}

__global__ void assignColorsP(thrust::device_ptr<float> d_noise,
                             const size_t data_size,
                             const size_t noiseWidth,
                             const size_t noiseHeight,
                             const size_t imageWidth,
                             const float turbulence_size,
                             thrust::device_ptr<float> d_out)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    const uint x = index%imageWidth;
    const uint y = (index-x)/imageWidth;

    d_out[index*4]   = turbulenceP(d_noise, noiseWidth, noiseHeight, x, y, turbulence_size);
    d_out[index*4+1] = turbulenceP(d_noise, noiseWidth, noiseHeight, x, y+noiseHeight, turbulence_size/2);
    d_out[index*4+2] = turbulenceP(d_noise, noiseWidth, noiseHeight, x, y+noiseHeight*2, turbulence_size/2);
    d_out[index*4+3] = 1.f;
}

__global__ void assignColors4(thrust::device_ptr<float> d_noise,
                              const size_t data_size,
                              const size_t noiseWidth,
                              const size_t noiseHeight,
                              const size_t imageWidth,
                              const float turbulence_size,
                              thrust::device_ptr<float> d_outR,
                              thrust::device_ptr<float> d_outG,
                              thrust::device_ptr<float> d_outB,
                              thrust::device_ptr<float> d_outA)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    const uint x = index%imageWidth;
    const uint y = (index-x)/imageWidth;

    d_outR[index] = turbulenceP(d_noise, noiseWidth, noiseHeight, x, y, turbulence_size);
    d_outG[index] = turbulenceP(d_noise, noiseWidth, noiseHeight, x, y+noiseHeight, turbulence_size/2);
    d_outB[index] = turbulenceP(d_noise, noiseWidth, noiseHeight, x, y+noiseHeight*2, turbulence_size/2);
    d_outA[index] = 1.f;
}

ColorVector gpuImageGen::generate_parallel_CV(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const uint numThreads)
{
    int dataSize = w*h;
    ColorVector outData(dataSize);
    thrust::host_vector<float> h_transfer(dataSize*4);
    thrust::device_vector<float> d_noise(dataSize);
    thrust::device_vector<float> d_colors(dataSize*4);

    const int blocks = (dataSize + numThreads - 1) / numThreads;
    //generate the per-pixel noise
    /*generateNoiseP<<<blocks, numThreads>>>(d_noise.data(),
                                           dataSize);*/
    genNoiseCustom(d_noise.data(), dataSize);
    cudaDeviceSynchronize();
    //generate the map here
    assignColorsP<<<blocks, numThreads>>>(d_noise.data(),
                                          dataSize,
                                          w,
                                          h,
                                          w,
                                          turbulence_size,
                                          d_colors.data());
    //end of map generation
    cudaDeviceSynchronize();
    thrust::copy(d_colors.begin(), d_colors.end(), h_transfer.begin());
    for(uint i=0; i<dataSize; ++i)
    {
        outData.at(i).m_r = h_transfer[i*4];
        outData.at(i).m_g = h_transfer[i*4+1];
        outData.at(i).m_b = h_transfer[i*4+2];
        outData.at(i).m_a = h_transfer[i*4+3];
    }
    return outData;
}

ImageColors gpuImageGen::generate_parallel_IC(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const uint numThreads)
{
    int dataSize = w*h;
    ImageColors outData;
    //outData.resize(dataSize);
    //thrust::host_vector<float> h_transfer(dataSize*4);
    thrust::device_vector<float> d_noise(dataSize);
    thrust::device_vector<float> d_colorsR(dataSize);
    thrust::device_vector<float> d_colorsG(dataSize);
    thrust::device_vector<float> d_colorsB(dataSize);
    thrust::device_vector<float> d_colorsA(dataSize);

    const int blocks = (dataSize + numThreads - 1) / numThreads;
    genNoiseCustom(d_noise.data(), dataSize);
    cudaDeviceSynchronize();
    //generate the map here
    assignColors4<<<blocks, numThreads>>>(d_noise.data(),
                                          dataSize,
                                          w,
                                          h,
                                          w,
                                          turbulence_size,
                                          d_colorsR.data(),
                                          d_colorsG.data(),
                                          d_colorsB.data(),
                                          d_colorsA.data());
    //end of map generation
    cudaDeviceSynchronize();
    /*thrust::copy(d_colors.begin(), d_colors.end(), h_transfer.begin());
    for(uint i=0; i<dataSize; ++i)
    {
        outData.at(i).m_r = h_transfer[i*4];
        outData.at(i).m_g = h_transfer[i*4+1];
        outData.at(i).m_b = h_transfer[i*4+2];
        outData.at(i).m_a = h_transfer[i*4+3];
    }*/
    thrust::copy(d_colorsR.begin(), d_colorsR.end(), outData.m_r.begin());
    thrust::copy(d_colorsG.begin(), d_colorsG.end(), outData.m_g.begin());
    thrust::copy(d_colorsB.begin(), d_colorsB.end(), outData.m_b.begin());
    thrust::copy(d_colorsA.begin(), d_colorsA.end(), outData.m_a.begin());
    return outData;
}

std::vector<float> gpuImageGen::generate_parallel_LN(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const uint numThreads)
{
    int dataSize = w*h;
    std::vector<float> outData(dataSize*4);
    thrust::host_vector<float> h_transfer(dataSize*4);
    thrust::device_vector<float> d_noise(dataSize);
    thrust::device_vector<float> d_colors(dataSize*4);

    const int blocks = (dataSize + numThreads - 1) / numThreads;
    //generate the per-pixel noise
    /*generateNoiseP<<<blocks, numThreads>>>(d_noise.data(),
                                           dataSize);*/
    genNoiseCustom(d_noise.data(), dataSize);
    cudaDeviceSynchronize();
    //generate the map here
    assignColorsP<<<blocks, numThreads>>>(d_noise.data(),
                                          dataSize,
                                          w,
                                          h,
                                          w,
                                          turbulence_size,
                                          d_colors.data());
    //end of map generation
    cudaDeviceSynchronize();
    thrust::copy(d_colors.begin(), d_colors.end(), h_transfer.begin());
    for(uint i=0; i<dataSize*4; ++i)
    {
        outData.at(i) = h_transfer[i];
    }
    return outData;
}
