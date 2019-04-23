#ifndef CLUSTERING_UTILTYPES_HPP_
#define CLUSTERING_UTILTYPES_HPP_
#include <array>
#include <vector>
//-------------------------------------------------------------------------------------------------------
/// @author Renats Bikmajevs
/// @note A few custom structure definitions required by the clib project
//-------------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------------
/// @brief A structure of vectors containing 4 colour channels
//-------------------------------------------------------------------------------------------------------
struct ImageColors
{
    std::vector<float> m_r;
    std::vector<float> m_g;
    std::vector<float> m_b;
    std::vector<float> m_a;
    void resize(size_t size)
    {
        m_r.resize(size);
        m_g.resize(size);
        m_b.resize(size);
        m_a.resize(size);
    }
    //Note that for performance purposes the following function does not perform a range check
    void setData(size_t idX, float _r, float _g, float _b, float _a)
    {
        m_r.at(idX) = _r;
        m_g.at(idX) = _g;
        m_b.at(idX) = _b;
        m_a.at(idX) = _a;
    }
    void setData(std::vector<float>* _rArr, std::vector<float>* _gArr, std::vector<float>* _bArr)
    {
        m_r=*_rArr;
        m_g=*_gArr;
        m_b=*_bArr;
        m_a.resize(m_r.size());
        std::fill(m_a.begin(),m_a.end(),1.f);
    }
    size_t getSize() const
    {
        return std::min(std::min(std::min(m_r.size(), m_g.size()),m_b.size()),m_a.size());
    }
};
//-------------------------------------------------------------------------------------------------------
/// @brief A 4 colour value structure
//-------------------------------------------------------------------------------------------------------
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
    Color operator+(const Color& rhs) const {return Color{m_r+rhs.m_r,
                                                   m_g+rhs.m_g,
                                                   m_b+rhs.m_b,
                                                   m_a+rhs.m_a};}
    Color operator/(const float& rhs) const {return Color{m_r/rhs,
                                                   m_g/rhs,
                                                   m_b/rhs,
                                                   m_a/rhs};}
    Color operator/(const int& rhs) const {return Color{m_r/rhs,
                                                 m_g/rhs,
                                                 m_b/rhs,
                                                 m_a/rhs};}
};
//-------------------------------------------------------------------------------------------------------
/// @brief convenience alias
//-------------------------------------------------------------------------------------------------------
using ColorVector = std::vector<Color>;
//-------------------------------------------------------------------------------------------------------
/// @brief was planning to use this for reading files, but now deprecated
//-------------------------------------------------------------------------------------------------------
/*struct uinteger2
{
    uint x;
    uint y;
};*/

#endif //CLUSTERING_UTILTYPES_HPP_
