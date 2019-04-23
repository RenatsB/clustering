#include <vector>
#include "gpuRandF.h"
#include "utilTypes.hpp"
//-------------------------------------------------------------------------------------------------------
/// @author Renats Bikmajevs
/// @note GPU noise map generator methods
//-------------------------------------------------------------------------------------------------------
namespace gpuImageGen {
//-----------------------------------------------------------------------------------------------------
/// @brief Noisemap generator method using ColorVector (std::vector<Color>) data type
/// @param [in]w image width
/// @param [in]h image height
/// @param [in]turbulence_size base noise square dimension (in pixels)
/// @param [in]numThreads number of threads to use on device
/// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
//-----------------------------------------------------------------------------------------------------
ColorVector generate_parallel_CV(const size_t w,
                                 const size_t h,
                                 const size_t turbulence_size,
                                 const size_t numThreads,
                                 const bool randAlpha=false);
//-----------------------------------------------------------------------------------------------------
/// @brief Noisemap generator method using ImageColors data type
/// @param [in]w image width
/// @param [in]h image height
/// @param [in]turbulence_size base noise square dimension (in pixels)
/// @param [in]numThreads number of threads to use on device
/// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
//-----------------------------------------------------------------------------------------------------
ImageColors generate_parallel_IC(const size_t w,
                                 const size_t h,
                                 const size_t turbulence_size,
                                 const size_t numThreads,
                                 const bool randAlpha=false);
//-----------------------------------------------------------------------------------------------------
/// @brief Noisemap generator method using std::vector<float> data type
/// @param [in]w image width
/// @param [in]h image height
/// @param [in]turbulence_size base noise square dimension (in pixels)
/// @param [in]numThreads number of threads to use on device
/// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
//-----------------------------------------------------------------------------------------------------
std::vector<float> generate_parallel_LN(const size_t w,
                                        const size_t h,
                                        const size_t turbulence_size,
                                        const size_t numThreads,
                                        const bool randAlpha=false);
//-----------------------------------------------------------------------------------------------------
/// @brief Noisemap generator method using 4 float vectors
/// @param [io]redChannel red channel vector
/// @param [io]greenChannel green channel vector
/// @param [io]blueChannel blue channel vector
/// @param [io]alphaChannel alpha channel vector
/// @param [in]w image width
/// @param [in]h image height
/// @param [in]turbulence_size base noise square dimension (in pixels)
/// @param [in]numThreads number of threads to use on device
/// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
//-----------------------------------------------------------------------------------------------------
void generate_parallel_4SV(std::vector<float>* redChannel,
                           std::vector<float>* greenChannel,
                           std::vector<float>* blueChannel,
                           std::vector<float>* alphaChannel,
                           const size_t w,
                           const size_t h,
                           const size_t turbulence_size,
                           const size_t numThreads,
                           const bool randAlpha=false);
//-----------------------------------------------------------------------------------------------------
/// @brief Noisemap generator method using 4 float vectors, but accessing data through [] operator
/// @param [io]redChannel raw pointer to red channel vector
/// @param [io]greenChannel raw pointer to green channel vector
/// @param [io]blueChannel raw pointer to blue channel vector
/// @param [io]alphaChannel raw pointer to alpha channel vector
/// @param [in]w image width
/// @param [in]h image height
/// @param [in]turbulence_size base noise square dimension (in pixels)
/// @param [in]numThreads number of threads to use on device
/// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
//-----------------------------------------------------------------------------------------------------
void generate_parallel_4LV(float* redChannel,
                           float* greenChannel,
                           float* blueChannel,
                           float* alphaChannel,
                           const size_t w,
                           const size_t h,
                           const size_t turbulence_size,
                           const size_t numThreads,
                           const bool randAlpha=false);
}


