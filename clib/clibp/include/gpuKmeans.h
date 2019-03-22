#include "utilTypes.hpp"
#include "cpuRandomFn.hpp"
#include <vector>

namespace gpuKmeans {
std::vector<float> kmeans_parallel_CV(const ColorVector &source,
                                      size_t k,
                                      size_t number_of_iterations,
                                      const size_t numThreads,
                                      RandomFn<float>* rfunc=nullptr);
std::vector<float> kmeans_parallel_IC(const ImageColors &source,
                                      size_t k,
                                      size_t number_of_iterations,
                                      const size_t numThreads,
                                      RandomFn<float>* rfunc=nullptr);
std::vector<float> kmeans_parallel_LN(const std::vector<float> &source,
                                      size_t k,
                                      size_t number_of_iterations,
                                      const size_t numThreads,
                                      RandomFn<float>* rfunc=nullptr);
std::vector<float>  kmeans_serial_4SV(const std::vector<float>* _reds,
                       const std::vector<float>* _grns,
                       const std::vector<float>* _blus,
                       const std::vector<float>* _alps,
                       size_t k,
                       size_t number_of_iterations,
                       const size_t numThreads,
                       RandomFn<float>* rfunc=nullptr);
std::vector<float>  kmeans_serial_4LV(const float* _reds,
                       const float* _grns,
                       const float* _blus,
                       const float* _alps,
                       const size_t number_of_elements,
                       size_t k,
                       size_t number_of_iterations,
                       const size_t numThreads,
                       RandomFn<float>* rfunc=nullptr);
}


