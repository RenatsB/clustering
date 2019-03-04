#ifndef CLUSTERING_UTILTYPES_HPP_
#define CLUSTERING_UTILTYPES_HPP_
#include <vector>
#include <array>

struct Color
{
    double m_r=1.0;
    double m_g=1.0;
    double m_b=1.0;
    double m_a=1.0;
    std::array<double,4> getData() const
    {
        return std::array{m_r,m_g,m_b,m_a};
    }
};

using DataFrame = std::vector<Color>;

#endif //CLUSTERING_UTILTYPES_HPP_
