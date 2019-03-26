#ifndef _CLIB_TESTIMGGEN_H
#define _CLIB_TESTIMGGEN_H

#include <gtest/gtest.h>
#include "testUtils.hpp"
#include "cpuImageGen.hpp"

////----------------------------------------------------------------------------------------------------------------------

TEST( CPUImgGenerator, ColorVectorBased)
{
    ImageGenFn gen;
    ColorVector cv = gen.generate_serial_CV(64,64,64);
    for(auto &c : cv)
    {
        EXPECT_TRUE(testRange(c.m_r, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(c.m_g, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(c.m_b, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(c.m_a, 0.f, 1.f, 0.001f));
    }
}

TEST( CPUImgGenerator, linearBased)
{
    ImageGenFn gen;
    std::vector<float> fv = gen.generate_serial_LN(64,64,64);
    for(auto &f : fv)
    {
        EXPECT_TRUE(testRange(f, 0.f, 1.f, 0.001f));
    }
}

TEST( CPUImgGenerator, ImageColorsStructure )
{
    ImageGenFn gen;
    ImageColors ic = gen.generate_serial_IC(64,64,64);
    for(size_t i=0; i<ic.m_r.size(); ++i)
    {
        EXPECT_TRUE(testRange(ic.m_r.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(ic.m_g.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(ic.m_b.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(ic.m_a.at(i), 0.f, 1.f, 0.001f));
    }
}


TEST( CPUImgGenerator, 4VectorPointers )
{
    ImageGenFn gen;
    std::vector<float> rv(4096);
    std::vector<float> gv(4096);
    std::vector<float> bv(4096);
    std::vector<float> av(4096);
    gen.generate_serial_4SV(64,64,64,&rv,&gv,&bv,&av);
    for(size_t i=0; i<4096; ++i)
    {
        EXPECT_TRUE(testRange(rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(av.at(i), 0.f, 1.f, 0.001f));
    }
}

TEST( CPUImgGenerator, 4FloatPointers )
{
    ImageGenFn gen;
    std::vector<float> rv(4096);
    std::vector<float> gv(4096);
    std::vector<float> bv(4096);
    std::vector<float> av(4096);
    gen.generate_serial_4LV(64,64,64,rv.data(),gv.data(),bv.data(),av.data());
    for(size_t i=0; i<4096; ++i)
    {
        EXPECT_TRUE(testRange(rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(av.at(i), 0.f, 1.f, 0.001f));
    }
}

TEST( CPUImgGenerator, 4FloatPointersDirectAssignment )
{
    ImageGenFn gen;
    std::vector<float> rv(4096);
    std::vector<float> gv(4096);
    std::vector<float> bv(4096);
    std::vector<float> av(4096);
    gen.generate_serial_4LL(64,64,64,rv.data(),gv.data(),bv.data(),av.data());
    for(size_t i=0; i<4096; ++i)
    {
        EXPECT_TRUE(testRange(rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(testRange(av.at(i), 0.f, 1.f, 0.001f));
    }
}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CLIB_TESTIMGGEN_H
