#include <vector>
#include "gpuRandF.h"
#include "utilTypes.hpp"

namespace GPUclib {
ColorVector generate(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const uint numThreads);
std::vector<float> linear_generate(const uint w,
                   const uint h,
                   const uint turbulence_size,
                   const uint numThreads);
}


