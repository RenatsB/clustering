#include "ImageGen.cuh"

__global__ void initCurand(curandState* ranDev, unsigned long seed )
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, index, 0, &ranDev[index]);
}

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

__global__ void generateNoiseP(thrust::device_ptr<float> d_noise,
                              const size_t data_size,
                              curandState* ranDev)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    curandState localState = ranDev[index];
    d_noise[index] = curand_uniform( &localState );
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

DataFrame generate(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const size_t noiseWidth,
                   const size_t noiseHeight,
                   const uint numThreads)
{
    uint dataSize = w*h;
    DataFrame outData(dataSize);
    thrust::device_vector<float> d_noise(dataSize);
    thrust::device_vector<float> d_colors(dataSize*4);

    curandState* randDevices;

    const int blocks = (dataSize + numThreads - 1) / numThreads;
    cudaMalloc ( &randDevices, numThreads*sizeof( curandState ) );
    initCurand<<<blocks, numThreads>>>(randDevices, time(NULL));
    cudaDeviceSynchronize();
    //generate the per-pixel noise
    generateNoiseP<<<blocks, numThreads>>>(d_noise.data(),
                                           dataSize,
                                           randDevices);
    cudaDeviceSynchronize();
    //generate the map here
    assignColorsP<<<blocks, numThreads>>>(d_noise.data(),
                                          dataSize,
                                          noiseWidth,
                                          noiseHeight,
                                          w,
                                          turbulence_size,
                                          d_colors.data());
    //end of map generation
    cudaDeviceSynchronize();
    for(uint i=0; i<dataSize; ++i)
    {
        outData.at(i).m_r = d_colors[i*4];
        outData.at(i).m_g = d_colors[i*4+1];
        outData.at(i).m_b = d_colors[i*4+2];
        outData.at(i).m_a = d_colors[i*4+3];
    }
    cudaFree(randDevices);
    return outData;
}

std::vector<float> linear_generate(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const size_t noiseWidth,
                   const size_t noiseHeight,
                   const uint numThreads)
{
    uint dataSize = w*h;
    std::vector<float> outData(dataSize*4);
    thrust::device_vector<float> d_noise(dataSize);
    thrust::device_vector<float> d_colors(dataSize*4);

    curandState* randDevices;

    const int blocks = (dataSize + numThreads - 1) / numThreads;
    cudaMalloc ( &randDevices, numThreads*sizeof( curandState ) );
    initCurand<<<blocks, numThreads>>>(randDevices, time(NULL));
    cudaDeviceSynchronize();
    //generate the per-pixel noise
    generateNoiseP<<<blocks, numThreads>>>(d_noise.data(),
                                           dataSize,
                                           randDevices);
    cudaDeviceSynchronize();
    //generate the map here
    assignColorsP<<<blocks, numThreads>>>(d_noise.data(),
                                          dataSize,
                                          noiseWidth,
                                          noiseHeight,
                                          w,
                                          turbulence_size,
                                          d_colors.data());
    //end of map generation
    cudaDeviceSynchronize();
    for(uint i=0; i<dataSize; ++i)
    {
        outData.at(i) = d_colors[i];
    }
    cudaFree(randDevices);
    return outData;
}
