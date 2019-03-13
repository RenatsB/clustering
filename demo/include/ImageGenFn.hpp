#ifndef CLUSTERING_IMAGEGENFN_HPP_
#define CLUSTERING_IMAGEGENFN_HPP_

#include "ImageGenFn.hpp"
#include "RandomFn.hpp"
#include "utilTypes.hpp"

class ImageGenFn
{
public:
    DataFrame generate(uint w, uint h, uint size);
private:
    float smoothNoise(float x, float y);
    float turbulence(float x, float y, float size);
    void generateNoise();
private:
    size_t m_noiseWidth=1;
    size_t m_noiseHeight=1;
    std::vector<std::vector<float>> m_noise;
    RandomFn<float> m_rand;
};

#endif //CLUSTERING_IMAGEGENFN_HPP_
