#ifndef CLUSTERING_GEN_HPP_
#define CLUSTERING_GEN_HPP_

#include "ran.hpp"
#include "utilTypes.hpp"

class Gen
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
    Ran<float> m_rand;
};

#endif //CLUSTERING_GEN_HPP_
