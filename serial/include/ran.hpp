#ifndef CLUSTERING_RANDOMFN_HPP_
#define CLUSTERING_RANDOMFN_HPP_

#include <random>
#include <cstdlib>
template <typename T>
class Ran
{
public:
    Ran()
    {
        srand((unsigned)time(NULL));
        m_generator.seed((seed()));
        m_MTgenerator=std::mt19937(seed());
    }
    ~Ran()=default;
    T SimpleRand(T r_low, T r_high)
    {
        double r = rand()/(1.0 + RAND_MAX);
        T range = r_high - r_low +1;
        T r_scaled = (r * range) + r_low;
        return r_scaled;
    }
    int UniformRandI()
    {
        return distroI(m_generator);
    }
    size_t UniformRandL()
    {
        return distroLong(m_generator);
    }
    T UniformRandU()
    {
        return distroU(m_generator);
    }
    int MT19937RandI()
    {
        return distroI(m_MTgenerator);
    }
    size_t MT19937RandL()
    {
        return distroLong(m_MTgenerator);
    }
    T MT19937RandU()
    {
        return distroU(m_MTgenerator);
    }
    void setNumericLimits(T r_low, T r_high)
    {
        distroU=std::uniform_real_distribution<T>(r_low,r_high);
    }
    void setNumericLimitsI(int r_low, int r_high)
    {
        distroI=std::uniform_int_distribution<int>(r_low,r_high);
    }
    void setNumericLimitsL(size_t r_low, size_t r_high)
    {
        distroLong=std::uniform_int_distribution<size_t>(r_low,r_high);
    }
private:
    std::uniform_int_distribution<size_t> distroLong;
    std::uniform_int_distribution<int> distroI;
    std::uniform_real_distribution<T> distroU;
    std::default_random_engine m_generator;
    std::mt19937 m_MTgenerator;
    std::random_device seed;
};

#endif //CLUSTERING_RANDOMFN_HPP_
