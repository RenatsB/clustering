#include <gtest/gtest.h>
#include "testParams.h"
#include "testUtils.h"
#include "gpuImageGen.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( GPUImageGen, COlorVectorBased )
{
    ColorVector cv = gpuImageGen::generate_parallel_CV(CLIB_TEST_DIMX,
                                                       CLIB_TEST_DIMY,
                                                       CLIB_TEST_NOISE,
                                                       CLIB_TEST_THREADS);

    for(auto &c : cv)
    {
        EXPECT_TRUE(clibTutils::testRange(c.m_r, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(c.m_g, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(c.m_b, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(c.m_a, 0.f, 1.f, 0.001f));
    }
}

TEST( GPUImageGen, linearBased)
{
    std::vector<float> fv = gpuImageGen::generate_parallel_LN(CLIB_TEST_DIMX,
                                                              CLIB_TEST_DIMY,
                                                              CLIB_TEST_NOISE,
                                                              CLIB_TEST_THREADS);
    for(auto &f : fv)
    {
        EXPECT_TRUE(clibTutils::testRange(f, 0.f, 1.f, 0.001f));
    }
}

TEST( GPUImageGen, ImageColorsStructure )
{
    ImageColors ic = gpuImageGen::generate_parallel_IC(CLIB_TEST_DIMX,
                                                     CLIB_TEST_DIMY,
                                                     CLIB_TEST_NOISE,
                                                     CLIB_TEST_THREADS);
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(ic.m_r.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(ic.m_g.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(ic.m_b.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(ic.m_a.at(i), 0.f, 1.f, 0.001f));
    }
}


TEST( GPUImageGen, 4VectorPointers )
{
    std::vector<float> rv(CLIB_TEST_ITEMS);
    std::vector<float> gv(CLIB_TEST_ITEMS);
    std::vector<float> bv(CLIB_TEST_ITEMS);
    std::vector<float> av(CLIB_TEST_ITEMS);
    gpuImageGen::generate_parallel_4SV(&rv,&gv,&bv,&av,
                                       CLIB_TEST_DIMX,
                                       CLIB_TEST_DIMY,
                                       CLIB_TEST_NOISE,
                                       CLIB_TEST_THREADS);
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(av.at(i), 0.f, 1.f, 0.001f));
    }
}

TEST( GPUImageGen, 4FloatPointers )
{
    std::vector<float> rv(CLIB_TEST_ITEMS);
    std::vector<float> gv(CLIB_TEST_ITEMS);
    std::vector<float> bv(CLIB_TEST_ITEMS);
    std::vector<float> av(CLIB_TEST_ITEMS);
    gpuImageGen::generate_parallel_4LV(rv.data(),
                                       gv.data(),
                                       bv.data(),
                                       av.data(),
                                       CLIB_TEST_DIMX,
                                       CLIB_TEST_DIMY,
                                       CLIB_TEST_NOISE,
                                       CLIB_TEST_THREADS);
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(av.at(i), 0.f, 1.f, 0.001f));
    }
}

////----------------------------------------------------------------------------------------------------------------------
