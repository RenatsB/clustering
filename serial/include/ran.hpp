#ifndef CLUSTERING_RANDOMFN_HPP_
#define CLUSTERING_RANDOMFN_HPP_

#include <random>
#include <cstdlib>
class Ran
{
public:
    Ran();
    ~Ran()=default;
    int randi(int r_low, int r_high, uint _t=0);
    float randf(float r_low, float r_high, uint _t=0);
    double randd(double r_low, double r_high, uint _t=0);
private:
    int SimpleRandI(int r_low, int r_high);
    float SimpleRandF(float r_low, float r_high);
    double SimpleRandD(double r_low, double r_high);
    int UniformRandI(int r_low, int r_high);
    float UniformRandF(float r_low, float r_high);
    double UniformRandD(double r_low, double r_high);
    std::default_random_engine m_generator;
};

#endif //CLUSTERING_RANDOMFN_HPP_
