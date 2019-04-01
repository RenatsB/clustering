#include "gpuRandF.h"
// Cuda includes begin
#include <curand.h>
#include <curand_kernel.h>
#include "cpuRandomFn.hpp"
#include <iostream>
// cuda includes end

#define CURAND_CALL(x) {\
if((x)!=CURAND_STATUS_SUCCESS) {\
printf("CURAND failure at %s:%d\n",__FILE__,__LINE__);\
exit(0);\
}\
}

int gpuRandFn::randFloatsInternal(float *&devData,
                                  const size_t n,
                                  const size_t numThreads)
{
    //CURAND_RNG_PSEUDO_MT19937




    RandomFn<float> rg;
    rg.setNumericLimitsL(0, std::numeric_limits<size_t>::max());
    // The generator, used for random numbers
    curandGenerator_t gen;

    // Create pseudo-random number generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Set seed to be the current time (note that calls close together will have same seed!)
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rg.MT19937RandL()));

    // Generate n floats on device
    CURAND_CALL(curandGenerateUniform(gen, devData, n));
    // Cleanup
    CURAND_CALL(curandDestroyGenerator(gen));
    return EXIT_SUCCESS;
}
