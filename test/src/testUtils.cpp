#include "testUtils.h"

::testing::AssertionResult clibTutils::testRange(const float value,
                                                 const float min,
                                                 const float max,
                                                 const float precision)
{
    if(value >= min-precision && value <= max+precision)
        return ::testing::AssertionSuccess();
    else
        return ::testing::AssertionFailure()
                << value << " is outside the range " << min << " to " << max;
}


::testing::AssertionResult clibTutils::compareColors(const Color &A,
                                                     const Color &B)
{
    if(A.m_r!=B.m_r||A.m_g!=B.m_g||A.m_b!=B.m_b||A.m_a!=B.m_a)
        return ::testing::AssertionFailure();
    return ::testing::AssertionSuccess();
}


::testing::AssertionResult clibTutils::compareColors(const ImageColors& A,
                                                     const ImageColors& B,
                                                     const size_t id)
{
    if(A.m_r.at(id)!=B.m_r.at(id)||A.m_g.at(id)!=B.m_g.at(id)||
       A.m_b.at(id)!=B.m_b.at(id)||A.m_a.at(id)!=B.m_a.at(id))
        return ::testing::AssertionFailure();
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult clibTutils::compareColors(const std::vector<float> &A,
                                                     const std::vector<float> &B,
                                                     const size_t id)
{
    //WARNING: this does not check wether the data beyond id is accessible. I am assuming that it is, because I know how this will be used.
    if(A.at(id)!=B.at(id)    ||A.at(id+1)!=B.at(id+1)||
       A.at(id+2)!=B.at(id+2)||A.at(id+3)!=B.at(id+3))
        return ::testing::AssertionFailure();
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult clibTutils::compareColors(const float Fr,
                                                     const float Fg,
                                                     const float Fb,
                                                     const float Fa,
                                                     const float Sr,
                                                     const float Sg,
                                                     const float Sb,
                                                     const float Sa)
{
    if(Fr!=Sr||Fg!=Sg||Fb!=Sb||Fa!=Sa)
        return ::testing::AssertionFailure();
    return ::testing::AssertionSuccess();
}
