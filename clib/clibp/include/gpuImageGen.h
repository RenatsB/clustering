#include <vector>
#include "gpuRandF.h"
#include "utilTypes.hpp"

namespace gpuImageGen {
ColorVector generate_parallel_CV(const uint w,
                                 const uint h,
                                 const uint turbulence_size,
                                 const uint numThreads);
ImageColors generate_parallel_IC(const uint w,
                                 const uint h,
                                 const uint turbulence_size,
                                 const uint numThreads);
std::vector<float> generate_parallel_LN(const uint w,
                                        const uint h,
                                        const uint turbulence_size,
                                        const uint numThreads);
void generate_parallel_4SV(std::vector<float>* redChannel,
                           std::vector<float>* greenChannel,
                           std::vector<float>* blueChannel,
                           std::vector<float>* alphaChannel,
                           const uint w,
                           const uint h,
                           const uint turbulence_size,
                           const uint numThreads);
void generate_parallel_4LV(float* redChannel,
                           float* greenChannel,
                           float* blueChannel,
                           float* alphaChannel,
                           const uint w,
                           const uint h,
                           const uint turbulence_size,
                           const uint numThreads);
}


