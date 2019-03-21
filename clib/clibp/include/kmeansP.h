#include "utilTypes.hpp"
#include "RandomFn.hpp"
#include <vector>

namespace GPUclib {
std::vector<float> kmeansP(const ColorVector &source,
                           size_t k,
                           size_t number_of_iterations,
                           RandomFn<float>* rfunc,
                           const uint numThreads);

std::vector<float> linear_kmeansP(const std::vector<float> &source,
                                  size_t k,
                                  size_t number_of_iterations,
                                  RandomFn<float>* rfunc,
                                  const uint numThreads);
}


