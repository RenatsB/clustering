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
    void setData(double _r, double _g, double _b, double _a=1.0)
    {
        m_r = _r;
        m_g = _g;
        m_b = _b;
        m_a = _a;
    }
    void setData(double _v)
    {
        m_r = _v;
        m_g = _v;
        m_b = _v;
        m_a = 1.0;
    }
};

using DataFrame = std::vector<Color>;
struct uinteger2
{
    uint x;
    uint y;
};

#endif //CLUSTERING_UTILTYPES_HPP_
