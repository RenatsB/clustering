#include "gpuRandF.h"
// Cuda includes begin
#include <cuda_runtime.h>
#include <curand.h>
// cuda includes end

#define CURAND_CALL(x) {\
if((x)!=CURAND_STATUS_SUCCESS) {\
printf("CURAND failure at %s:%d\n",__FILE__,__LINE__);\
exit(0);\
}\
}


int GPUclib::randFloatsInternal(float *&devData, const size_t n)
{
  // The generator, used for random numbers
  curandGenerator_t gen;

  // Create pseudo-random number generator
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

  // Set seed to be the current time (note that calls close together will have same seed!)
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

    // Generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));
  // Cleanup
  CURAND_CALL(curandDestroyGenerator(gen));
  return EXIT_SUCCESS;
}
