#ifndef _CLIB_TESTIMGGEN_H
#define _CLIB_TESTIMGGEN_H

#include <gtest/gtest.h>
#include "testParams.h"
#include "testUtils.h"
#include "cpuImageGen.hpp"

////----------------------------------------------------------------------------------------------------------------------

TEST( CPUImgGenerator, ColorVectorBased)
{
    ImageGenFn gen;
    ColorVector cv = gen.generate_serial_CV(CLIB_TEST_DIMX,
                                            CLIB_TEST_DIMY,
                                            CLIB_TEST_NOISE);
    for(auto &c : cv)
    {
        EXPECT_TRUE(clibTutils::testRange(c.m_r, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(c.m_g, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(c.m_b, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(c.m_a, 0.f, 1.f, 0.001f));
    }
}

TEST( CPUImgGenerator, linearBased)
{
    ImageGenFn gen;
    std::vector<float> fv = gen.generate_serial_LN(CLIB_TEST_DIMX,
                                                   CLIB_TEST_DIMY,
                                                   CLIB_TEST_NOISE);
    for(auto &f : fv)
    {
        EXPECT_TRUE(clibTutils::testRange(f, 0.f, 1.f, 0.001f));
    }
}

TEST( CPUImgGenerator, ImageColorsStructure )
{
    ImageGenFn gen;
    ImageColors ic = gen.generate_serial_IC(CLIB_TEST_DIMX,
                                            CLIB_TEST_DIMY,
                                            CLIB_TEST_NOISE);
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(ic.m_r.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(ic.m_g.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(ic.m_b.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(ic.m_a.at(i), 0.f, 1.f, 0.001f));
    }
}


TEST( CPUImgGenerator, 4VectorPointers )
{
    ImageGenFn gen;
    std::vector<float> rv(CLIB_TEST_ITEMS);
    std::vector<float> gv(CLIB_TEST_ITEMS);
    std::vector<float> bv(CLIB_TEST_ITEMS);
    std::vector<float> av(CLIB_TEST_ITEMS);
    gen.generate_serial_4SV(CLIB_TEST_DIMX,
                            CLIB_TEST_DIMY,
                            CLIB_TEST_NOISE,
                            &rv,&gv,&bv,&av);
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(av.at(i), 0.f, 1.f, 0.001f));
    }
}

TEST( CPUImgGenerator, 4FloatPointers )
{
    ImageGenFn gen;
    std::vector<float> rv(CLIB_TEST_ITEMS);
    std::vector<float> gv(CLIB_TEST_ITEMS);
    std::vector<float> bv(CLIB_TEST_ITEMS);
    std::vector<float> av(CLIB_TEST_ITEMS);
    gen.generate_serial_4LV(CLIB_TEST_DIMX,
                            CLIB_TEST_DIMY,
                            CLIB_TEST_NOISE,
                            rv.data(),gv.data(),bv.data(),av.data());
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(av.at(i), 0.f, 1.f, 0.001f));
    }
}

TEST( CPUImgGenerator, 4FloatPointersDirectAssignment )
{
    ImageGenFn gen;
    std::vector<float> rv(CLIB_TEST_ITEMS);
    std::vector<float> gv(CLIB_TEST_ITEMS);
    std::vector<float> bv(CLIB_TEST_ITEMS);
    std::vector<float> av(CLIB_TEST_ITEMS);
    gen.generate_serial_4LL(CLIB_TEST_DIMX,
                            CLIB_TEST_DIMY,
                            CLIB_TEST_NOISE,
                            rv.data(),gv.data(),bv.data(),av.data());
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(av.at(i), 0.f, 1.f, 0.001f));
    }
}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CLIB_TESTIMGGEN_H
