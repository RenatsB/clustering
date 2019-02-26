#include "ran.hpp"
#include <time.h>

Ran::Ran()
{
    srand((unsigned)time(NULL));
}

int Ran::randi(int r_low, int r_high, uint _t)
{
    switch(_t)
    {
        case 0:
        {
            return SimpleRandI(r_low, r_high);
        }
        case 1:
        {
            return UniformRandI(r_low, r_high);
        }
        default:
        {
            return 0.f;
        }
    }
}

float Ran::randf(float r_low, float r_high, uint _t)
{
    switch(_t)
    {
        case 0:
        {
            return SimpleRandF(r_low, r_high);
        }
        case 1:
        {
            return UniformRandF(r_low, r_high);
        }
        default:
        {
            return 0.f;
        }
    }
}

double Ran::randd(double r_low, double r_high, uint _t)
{
    switch(_t)
    {
        case 0:
        {
            return SimpleRandD(r_low, r_high);
        }
        case 1:
        {
            return UniformRandD(r_low, r_high);
        }
        default:
        {
            return 0.f;
        }
    }
}

int Ran::SimpleRandI(int r_low, int r_high)
{
    double r = rand()/(1.0 + RAND_MAX);
    int range = r_high - r_low +1;
    int r_scaled = (r * range) + r_low;
    return r_scaled;
}

float Ran::SimpleRandF(float r_low, float r_high)
{
    float r = r_low + (rand()/(RAND_MAX / (r_high-r_low)));
    return r;
}

double Ran::SimpleRandD(double r_low, double r_high)
{
    double r = r_low + (rand()/(RAND_MAX / (r_high-r_low)));
    return r;
}

int Ran::UniformRandI(int r_low, int r_high)
{
    std::uniform_int_distribution distribution(r_low,r_high);
    return distribution(m_generator);
}

float Ran::UniformRandF(float r_low, float r_high)
{
    std::uniform_real_distribution<float> distribution(r_low,r_high);
    return distribution(m_generator);
}

double Ran::UniformRandD(double r_low, double r_high)
{
    std::uniform_real_distribution<double> distribution(r_low,r_high);
    return distribution(m_generator);
}
