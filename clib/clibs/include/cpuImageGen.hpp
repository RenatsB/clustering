#ifndef CLUSTERING_IMAGEGENFN_HPP_
#define CLUSTERING_IMAGEGENFN_HPP_

#include "cpuRandomFn.hpp"
#include "utilTypes.hpp"

class ImageGenFn
{
public:
    ColorVector generate_serial_CV(const uint w,
                       const uint h,
                       const uint turbulence_size);
    ImageColors generate_serial_IC(const uint w,
                       const uint h,
                       const uint turbulence_size);
    std::vector<float> generate_serial_LN(const uint w,
                       const uint h,
                       const uint turbulence_size);
    void generate_serial_4SV(const uint w,
                       const uint h,
                       const uint turbulence_size,
                       std::vector<float>* redChannel,
                       std::vector<float>* greenChannel,
                       std::vector<float>* blueChannel,
                       std::vector<float>* alphaChannel);
    void generate_serial_4LV(const uint w,
                       const uint h,
                       const uint turbulence_size,
                       float* redChannel,
                       float* greenChannel,
                       float* blueChannel,
                       float* alphaChannel);

private:
    float smoothNoise(float x, float y);
    float turbulence(float x, float y, float size);
    void generateNoise();
private:
    size_t m_noiseWidth=1;
    size_t m_noiseHeight=1;
    std::vector<float> m_noise;
    RandomFn<float> m_rand;
};

#endif //CLUSTERING_IMAGEGENFN_HPP_
