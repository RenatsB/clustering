#ifndef CLUSTERING_GEN_HPP_
#define CLUSTERING_GEN_HPP_

#include "ran.hpp"
#include "utilTypes.hpp"

class Gen
{
public:
    DataFrame generate(uint w, uint h);
private:
    double smoothNoise(double x, double y);
    double turbulence(double x, double y, double size);
    void generateNoise();
private:
    size_t m_noiseWidth;
    size_t m_noiseHeight;
    std::vector<std::vector<double>> m_noise;
    Ran<double> m_rand;
};

#endif //CLUSTERING_GEN_HPP_
