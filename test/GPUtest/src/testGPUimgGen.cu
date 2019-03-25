#include "gtest/gtest.h"
#include "testUtils.h"
#include "gpuImageGen.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( GPUImageGen, COlorVectorBased )
{
    ColorVector cv = gpuImageGen::generate_parallel_CV(64,64,64,64);

    for(auto &c : cv)
    {
        EXPECT_GE(c.m_r, 0.f);
        EXPECT_LE(c.m_r, 1.f);
        EXPECT_GE(c.m_g, 0.f);
        EXPECT_LE(c.m_g, 1.f);
        EXPECT_GE(c.m_b, 0.f);
        EXPECT_LE(c.m_b, 1.f);
        EXPECT_GE(c.m_a, 0.f);
        EXPECT_LE(c.m_a, 1.f);
    }
}


TEST( GPUImageGen, linearBased)
{
    std::vector<float> fv = gpuImageGen::generate_parallel_LN(64,64,64,64);
    for(auto &f : fv)
    {
        EXPECT_GE(f, 0.f);
        EXPECT_LE(f, 1.f);
    }
}

////----------------------------------------------------------------------------------------------------------------------
