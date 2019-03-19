#ifndef CLUSTERING_UTILTYPES_HPP_
#define CLUSTERING_UTILTYPES_HPP_
#include <array>
#include <vector>

struct Color
{
    Color(){}
    Color(float x, float y, float z, float w):
        m_r(x),
        m_g(y),
        m_b(z),
        m_a(w)
    {}
    float m_r=1.0;
    float m_g=1.0;
    float m_b=1.0;
    float m_a=1.0;
    void setData(float _r, float _g, float _b, float _a=1.0)
    {
        m_r = _r;
        m_g = _g;
        m_b = _b;
        m_a = _a;
    }
    void setData(float _v)
    {
        m_r = _v;
        m_g = _v;
        m_b = _v;
        m_a = 1.0;
    }
    Color operator+(const Color& rhs){return Color{m_r+=rhs.m_r,
                                                   m_g+=rhs.m_g,
                                                   m_b+=rhs.m_b,
                                                   m_a+=rhs.m_a};}
    Color operator/(const float& rhs){return Color{m_r/rhs,
                                                   m_g/rhs,
                                                   m_b/rhs,
                                                   m_a/rhs};}
    Color operator/(const int& rhs){return Color{m_r/rhs,
                                                 m_g/rhs,
                                                 m_b/rhs,
                                                 m_a/rhs};}
};
using ColorVector = std::vector<Color>;
struct uinteger2
{
    uint x;
    uint y;
};

#endif //CLUSTERING_UTILTYPES_HPP_
