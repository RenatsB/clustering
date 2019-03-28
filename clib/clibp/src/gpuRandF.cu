#include "gpuRandF.h"
// Cuda includes begin
#include <curand.h>
#include <curand_kernel.h>
#include "cpuRandomFn.hpp"
// cuda includes end

#define CURAND_CALL(x) {\
if((x)!=CURAND_STATUS_SUCCESS) {\
printf("CURAND failure at %s:%d\n",__FILE__,__LINE__);\
exit(0);\
}\
}

__global__ void setup_kernel(size_t seed, curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void generate_normal_kernel(curandState *state, float *result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    result[id] = curand_normal(&state[id]);
}



int gpuRandFn::randFloatsInternal(float *&devData,
                                  const size_t n,
                                  const size_t numThreads)
{
    RandomFn<float> rg;
    rg.setNumericLimitsL(0, std::numeric_limits<size_t>::max());
    const int blocks = (n + numThreads - 1) / numThreads;
    curandState *devStates;
    cudaMalloc((void **)&devStates, numThreads * blocks * sizeof(curandState));

    setup_kernel<<<blocks, numThreads>>>(rg.MT19937RandL(),devStates);
    cudaDeviceSynchronize();

    generate_normal_kernel<<<blocks, numThreads>>>(devStates, devData);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}


/*int gpuRandFn::randFloatsInternal(float *&devData, const size_t n)
{
  // The generator, used for random numbers
  curandGenerator_t gen;
  srand(time(NULL));
  int _seed = rand();

  // Create pseudo-random number generator
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

  // Set seed to be the current time (note that calls close together will have same seed!)
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, _seed));

    // Generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));
  // Cleanup
  CURAND_CALL(curandDestroyGenerator(gen));
  return EXIT_SUCCESS;
}*/
