#ifndef CLUSTERING_GEN_HPP_
#define CLUSTERING_GEN_HPP_

#include "ran.hpp"
#include "utilTypes.hpp"

class Gen
{
public:
    DataFrame generate(uint w, uint h, uint size);
private:
    double smoothNoise(double x, double y);
    double turbulence(double x, double y, double size);
    void generateNoise();
private:
    size_t m_noiseWidth=1;
    size_t m_noiseHeight=1;
    std::vector<std::vector<double>> m_noise;
    Ran<double> m_rand;
};

#endif //CLUSTERING_GEN_HPP_