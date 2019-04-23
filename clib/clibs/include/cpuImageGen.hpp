#ifndef CLUSTERING_IMAGEGENFN_HPP_
#define CLUSTERING_IMAGEGENFN_HPP_

#include "cpuRandomFn.hpp"
#include "utilTypes.hpp"
//-------------------------------------------------------------------------------------------------------
/// @author Renats Bikmajevs
/// @note CPU noise map generator class
//-------------------------------------------------------------------------------------------------------
class ImageGenFn
{
public:
    //-----------------------------------------------------------------------------------------------------
    /// @brief Noisemap generator method using ColorVector (std::vector<Color>) data type
    /// @param [in]w image width
    /// @param [in]h image height
    /// @param [in]turbulence_size base noise square dimension (in pixels)
    /// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
    //-----------------------------------------------------------------------------------------------------
    ColorVector generate_serial_CV(const uint w,
                       const uint h,
                       const uint turbulence_size,
                       const bool randomAlpha=false);
    //-----------------------------------------------------------------------------------------------------
    /// @brief Noisemap generator method using ImageColors data type
    /// @param [in]w image width
    /// @param [in]h image height
    /// @param [in]turbulence_size base noise square dimension (in pixels)
    /// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
    //-----------------------------------------------------------------------------------------------------
    ImageColors generate_serial_IC(const uint w,
                       const uint h,
                       const uint turbulence_size,
                       const bool randomAlpha=false);
    //-----------------------------------------------------------------------------------------------------
    /// @brief Noisemap generator method using std::vector<float> data type
    /// @param [in]w image width
    /// @param [in]h image height
    /// @param [in]turbulence_size base noise square dimension (in pixels)
    /// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
    //-----------------------------------------------------------------------------------------------------
    std::vector<float> generate_serial_LN(const uint w,
                       const uint h,
                       const uint turbulence_size,
                       const bool randomAlpha=false);
    //-----------------------------------------------------------------------------------------------------
    /// @brief Noisemap generator method using 4 float vectors
    /// @param [in]w image width
    /// @param [in]h image height
    /// @param [in]turbulence_size base noise square dimension (in pixels)
    /// @param [io]redChannel red channel vector
    /// @param [io]greenChannel green channel vector
    /// @param [io]blueChannel blue channel vector
    /// @param [io]alphaChannel alpha channel vector
    /// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
    //-----------------------------------------------------------------------------------------------------
    void generate_serial_4SV(const uint w,
                       const uint h,
                       const uint turbulence_size,
                       std::vector<float>* redChannel,
                       std::vector<float>* greenChannel,
                       std::vector<float>* blueChannel,
                       std::vector<float>* alphaChannel,
                       const bool randomAlpha=false);
    //-----------------------------------------------------------------------------------------------------
    /// @brief Noisemap generator method using 4 float vectors, but accessing data through [] operator
    /// @param [in]w image width
    /// @param [in]h image height
    /// @param [in]turbulence_size base noise square dimension (in pixels)
    /// @param [io]redChannel raw pointer to red channel vector
    /// @param [io]greenChannel raw pointer to green channel vector
    /// @param [io]blueChannel raw pointer to blue channel vector
    /// @param [io]alphaChannel raw pointer to alpha channel vector
    /// @param [in]randomAlpha a toggle to use alpha channel randomization, or set it to 1 (false)
    //-----------------------------------------------------------------------------------------------------
    void generate_serial_4LV(const uint w,
                       const uint h,
                       const uint turbulence_size,
                       float* redChannel,
                       float* greenChannel,
                       float* blueChannel,
                       float* alphaChannel,
                       const bool randomAlpha=false);

private:
    //-----------------------------------------------------------------------------------------------------
    /// @brief Per-pixel noise smoothing function
    /// @param [in]x horizontal position of current point of interest
    /// @param [in]y vertical position of current point of interest
    //-----------------------------------------------------------------------------------------------------
    float smoothNoise(float x, float y);
    //-----------------------------------------------------------------------------------------------------
    /// @brief Layered noisemap accumulator. Combines different size noises
    /// @param [in]x horizontal position of current point of interest
    /// @param [in]y vertical position of current point of interest
    /// @param [in]size maximum size of noise layer
    //-----------------------------------------------------------------------------------------------------
    float turbulence(float x, float y, float size);
    //-----------------------------------------------------------------------------------------------------
    /// @brief Base per-pixel noise generation method. Fills m_noise using m_rand
    //-----------------------------------------------------------------------------------------------------
    void generateNoise();
private:
    //-----------------------------------------------------------------------------------------------------
    /// @brief Base noise map horizontal dimensions, effectively horizontal image size
    //-----------------------------------------------------------------------------------------------------
    size_t m_noiseWidth=1;
    //-----------------------------------------------------------------------------------------------------
    /// @brief Base noise map vertical dimensions, effectively vertical image size
    //-----------------------------------------------------------------------------------------------------
    size_t m_noiseHeight=1;
    //-----------------------------------------------------------------------------------------------------
    /// @brief Base noise map. Each pixel represented by a single float value
    //-----------------------------------------------------------------------------------------------------
    std::vector<float> m_noise;
    //-----------------------------------------------------------------------------------------------------
    /// @brief Custom random number generator wrapper. This project is using MT19937 methods
    //-----------------------------------------------------------------------------------------------------
    RandomFn<float> m_rand;
};

#endif //CLUSTERING_IMAGEGENFN_HPP_
