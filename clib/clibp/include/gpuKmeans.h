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
void  kmeans_serial_4SV(const std::vector<float>* _inreds,
                        const std::vector<float>* _ingrns,
                        const std::vector<float>* _inblus,
                        const std::vector<float>* _inalps,
                        std::vector<float>* _outreds,
                        std::vector<float>* _outgrns,
                        std::vector<float>* _outblus,
                        std::vector<float>* _outalps,
                        size_t k,
                        size_t number_of_iterations,
                        const size_t numThreads,
                        RandomFn<float>* rfunc=nullptr);
void  kmeans_serial_4LV(const float* _inreds,
                        const float* _ingrns,
                        const float* _inblus,
                        const float* _inalps,
                        float* _outreds,
                        float* _outgrns,
                        float* _outblus,
                        float* _outalps,
                        const size_t number_of_elements,
                        size_t k,
                        size_t number_of_iterations,
                        const size_t numThreads,
                        RandomFn<float>* rfunc=nullptr);
}


