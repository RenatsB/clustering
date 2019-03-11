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
    Color operator+(const Color& rhs){return Color{this->m_r+=rhs.m_r,
                                                   this->m_g+=rhs.m_g,
                                                   this->m_b+=rhs.m_b,
                                                   this->m_a+=rhs.m_a};}
    Color operator+=(const Color& rhs){return Color{this->m_r+=rhs.m_r,
                                                    this->m_g+=rhs.m_g,
                                                    this->m_b+=rhs.m_b,
                                                    this->m_a+=rhs.m_a};}
    Color operator-(const Color& rhs){return Color{this->m_r-=rhs.m_r,
                                                   this->m_g-=rhs.m_g,
                                                   this->m_b-=rhs.m_b,
                                                   this->m_a-=rhs.m_a};}
    Color operator-=(const Color& rhs){return Color{this->m_r-=rhs.m_r,
                                                    this->m_g-=rhs.m_g,
                                                    this->m_b-=rhs.m_b,
                                                    this->m_a-=rhs.m_a};}
    Color operator/(const double& rhs){return Color{this->m_r/rhs,
                                                    this->m_g/rhs,
                                                    this->m_b/rhs,
                                                    this->m_a/rhs};}
    Color operator/(const float& rhs){return Color{this->m_r/(double)rhs,
                                                   this->m_g/(double)rhs,
                                                   this->m_b/(double)rhs,
                                                   this->m_a/(double)rhs};}
    Color operator/(const int& rhs){return Color{this->m_r/(double)rhs,
                                                   this->m_g/(double)rhs,
                                                   this->m_b/(double)rhs,
                                                   this->m_a/(double)rhs};}
};

using DataFrame = std::vector<Color>;
struct uinteger2
{
    uint x;
    uint y;
};

#endif //CLUSTERING_UTILTYPES_HPP_
