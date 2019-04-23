#include "utilTypes.hpp"
#include "cpuRandomFn.hpp"
#include <vector>
//-------------------------------------------------------------------------------------------------------
/// @author Renats Bikmajevs
/// @note GPU K-Means clustering solver
//-------------------------------------------------------------------------------------------------------
namespace gpuKmeans {
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using ColorVector data type
/// @param [in]data Source image
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
/// @param [in]numThreads number of threads to use on device
//-----------------------------------------------------------------------------------------------------
ColorVector kmeans_parallel_CV(const ColorVector &source,
                               size_t k,
                               size_t number_of_iterations,
                               const size_t numThreads);
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using ImageColors data type
/// @param [in]data Source image
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
/// @param [in]numThreads number of threads to use on device
//-----------------------------------------------------------------------------------------------------
ImageColors kmeans_parallel_IC(const ImageColors &source,
                               size_t k,
                               size_t number_of_iterations,
                               const size_t numThreads);
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using single vector of float values
/// @param [in]data Source image
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
/// @param [in]numThreads number of threads to use on device
//-----------------------------------------------------------------------------------------------------
std::vector<float> kmeans_parallel_LN(const std::vector<float> &source,
                                      size_t k,
                                      size_t number_of_iterations,
                                      const size_t numThreads);
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using 4 float vectors
/// @param [in]_inreds Source image red channel
/// @param [in]_ingrns Source image green channel
/// @param [in]_inblus Source image blue channel
/// @param [in]_inalps Source image alpha channel
/// @param [out]_outreds filtered image red output channel
/// @param [out]_outgrns filtered image green output channel
/// @param [out]_outblus filtered image blue output channel
/// @param [out]_outalps filtered image alpha output channel
/// @param [in]number_of_elements number of pixels to process (= input and output vector sizes)
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
/// @param [in]numThreads number of threads to use on device
//-----------------------------------------------------------------------------------------------------
void  kmeans_parallel_4SV(const std::vector<float>* _inreds,
                          const std::vector<float>* _ingrns,
                          const std::vector<float>* _inblus,
                          const std::vector<float>* _inalps,
                          std::vector<float>* _outreds,
                          std::vector<float>* _outgrns,
                          std::vector<float>* _outblus,
                          std::vector<float>* _outalps,
                          const size_t number_of_elements,
                          size_t k,
                          size_t number_of_iterations,
                          const size_t numThreads);
//-----------------------------------------------------------------------------------------------------
/// @brief Clustering method using 4 float vectors, accessing elements through [] operator
/// @param [in]_inreds Pointer to source image red channel
/// @param [in]_ingrns Pointer to source image green channel
/// @param [in]_inblus Pointer to source image blue channel
/// @param [in]_inalps Pointer to source image alpha channel
/// @param [out]_outreds Pointer to filtered image red output channel
/// @param [out]_outgrns Pointer to filtered image green output channel
/// @param [out]_outblus Pointer to filtered image blue output channel
/// @param [out]_outalps Pointer to filtered image alpha output channel
/// @param [in]number_of_elements number of pixels to process (= input and output vector sizes)
/// @param [in]k number of clusters
/// @param [in]number_of_iterations number of clustering iterations to perform
/// @param [in]numThreads number of threads to use on device
//-----------------------------------------------------------------------------------------------------
void  kmeans_parallel_4LV(const float* _inreds,
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
                          const size_t numThreads);
}


