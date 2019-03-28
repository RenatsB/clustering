#include <vector>
#include "gpuRandF.h"
#include "utilTypes.hpp"

namespace gpuImageGen {
ColorVector generate_parallel_CV(const size_t w,
                                 const size_t h,
                                 const size_t turbulence_size,
                                 const size_t numThreads);
ImageColors generate_parallel_IC(const size_t w,
                                 const size_t h,
                                 const size_t turbulence_size,
                                 const size_t numThreads);
std::vector<float> generate_parallel_LN(const size_t w,
                                        const size_t h,
                                        const size_t turbulence_size,
                                        const size_t numThreads);
void generate_parallel_4SV(std::vector<float>* redChannel,
                           std::vector<float>* greenChannel,
                           std::vector<float>* blueChannel,
                           std::vector<float>* alphaChannel,
                           const size_t w,
                           const size_t h,
                           const size_t turbulence_size,
                           const size_t numThreads);
void generate_parallel_4LV(float* redChannel,
                           float* greenChannel,
                           float* blueChannel,
                           float* alphaChannel,
                           const size_t w,
                           const size_t h,
                           const size_t turbulence_size,
                           const size_t numThreads);
}


