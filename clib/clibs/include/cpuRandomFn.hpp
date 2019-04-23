#ifndef CLUSTERING_RANDOMFN_HPP_
#define CLUSTERING_RANDOMFN_HPP_

#include <random>
#include <cstdlib>
//-------------------------------------------------------------------------------------------------------
/// @author Renats Bikmajevs
/// @note Random number generator wrapper.
/// @note Includes standard pseudo-random rand(), default_random_engine and mt19937.
/// @note Due to wrapper design, templated data type must not be an integer of any kind
//-------------------------------------------------------------------------------------------------------
template <typename T>
class RandomFn
{
public:
    //-----------------------------------------------------------------------------------------------------
    /// @brief Constructor. Inits core modules.
    //-----------------------------------------------------------------------------------------------------
    RandomFn()
    {
        srand((unsigned)time(NULL));
        m_generator.seed((seed()));
        m_MTgenerator=std::mt19937(seed());
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Destructor
    //-----------------------------------------------------------------------------------------------------
    ~RandomFn()=default;
    //-----------------------------------------------------------------------------------------------------
    /// @brief Standard rand() wrapper, which uses limits.
    /// @param [in]r_low lower bounds
    /// @param [in]r_high upper bounds
    //-----------------------------------------------------------------------------------------------------
    T SimpleRand(T r_low, T r_high)
    {
        double r = rand()/(1.0 + RAND_MAX);
        T range = r_high - r_low +1;
        T r_scaled = (r * range) + r_low;
        return r_scaled;
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Wrapper for default_random_engine with uniform distribution. INTEGER ONLY
    //-----------------------------------------------------------------------------------------------------
    int UniformRandI()
    {
        return distroI(m_generator);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Wrapper for default_random_engine with uniform distribution. LONG INT ONLY
    //-----------------------------------------------------------------------------------------------------
    size_t UniformRandL()
    {
        return distroLong(m_generator);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Wrapper for default_random_engine with uniform distribution. Templated data type only.
    //-----------------------------------------------------------------------------------------------------
    T UniformRandU()
    {
        return distroU(m_generator);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Wrapper for mt19937 with uniform distribution. INTEGER ONLY
    //-----------------------------------------------------------------------------------------------------
    int MT19937RandI()
    {
        return distroI(m_MTgenerator);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Wrapper for default_random_engine with uniform distribution. LONG INT ONLY
    //-----------------------------------------------------------------------------------------------------
    size_t MT19937RandL()
    {
        return distroLong(m_MTgenerator);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Wrapper for default_random_engine with uniform distribution. Templated data type only.
    //-----------------------------------------------------------------------------------------------------
    T MT19937RandU()
    {
        return distroU(m_MTgenerator);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Sets numeric limits for uniform_real_distribution (templated only)
    /// @param [in]r_low lower bounds
    /// @param [in]r_high upper bounds
    //-----------------------------------------------------------------------------------------------------
    void setNumericLimits(T r_low, T r_high)
    {
        distroU=std::uniform_real_distribution<T>(r_low,r_high);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Sets numeric limits for uniform_int_distribution (int only)
    /// @param [in]r_low lower bounds
    /// @param [in]r_high upper bounds
    //-----------------------------------------------------------------------------------------------------
    void setNumericLimitsI(int r_low, int r_high)
    {
        distroI=std::uniform_int_distribution<int>(r_low,r_high);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief Sets numeric limits for uniform_int_distribution (long int only)
    /// @param [in]r_low lower bounds
    /// @param [in]r_high upper bounds
    //-----------------------------------------------------------------------------------------------------
    void setNumericLimitsL(size_t r_low, size_t r_high)
    {
        distroLong=std::uniform_int_distribution<size_t>(r_low,r_high);
    }
    //-----------------------------------------------------------------------------------------------------
    /// @brief A wrapper for probability vector based discrete distribution
    /// @param [in]w probability (weight) vector reference
    //-----------------------------------------------------------------------------------------------------
    size_t weightedRand(std::vector<size_t>& w)
    {
        weightedDistr=std::discrete_distribution<size_t>(w.begin(), w.end());
        return weightedDistr(m_MTgenerator);
    }
private:
    //-----------------------------------------------------------------------------------------------------
    /// @brief Size_t uniform distribution
    //-----------------------------------------------------------------------------------------------------
    std::uniform_int_distribution<size_t> distroLong;
    //-----------------------------------------------------------------------------------------------------
    /// @brief int uniform distribution
    //-----------------------------------------------------------------------------------------------------
    std::uniform_int_distribution<int> distroI;
    //-----------------------------------------------------------------------------------------------------
    /// @brief templated real uniform distribution
    //-----------------------------------------------------------------------------------------------------
    std::uniform_real_distribution<T> distroU;
    //-----------------------------------------------------------------------------------------------------
    /// @brief Weighted distribution
    //-----------------------------------------------------------------------------------------------------
    std::discrete_distribution<size_t> weightedDistr;
    //-----------------------------------------------------------------------------------------------------
    /// @brief Default generator
    //-----------------------------------------------------------------------------------------------------
    std::default_random_engine m_generator;
    //-----------------------------------------------------------------------------------------------------
    /// @brief MT generator
    //-----------------------------------------------------------------------------------------------------
    std::mt19937 m_MTgenerator;
    //-----------------------------------------------------------------------------------------------------
    /// @brief Seed necessary for MT19937 generator
    //-----------------------------------------------------------------------------------------------------
    std::random_device seed;
};

#endif //CLUSTERING_RANDOMFN_HPP_
