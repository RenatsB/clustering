#ifndef GEN_RANDOMFN_H_
#define GEN_RANDOMFN_H_

#include <random>
#include <cstdlib>
class Ran
{
private:
    Ran();
public:
    static Ran* instance();
    Ran(const Ran&) = delete;
    Ran& operator=(const Ran&) = delete;

    int randi(int r_low, int r_high, uint _t=0);
    float randf(float r_low, float r_high, uint _t=0);
private:
    int SimpleRandI(int r_low, int r_high);
    float SimpleRandF(float r_low, float r_high);
    int UniformRandI(int r_low, int r_high);
    float UniformRandF(float r_low, float r_high);
    Ran* r_instance;
    std::default_random_engine m_generator;
};

#endif //GEN_RANDOMFN_H_
