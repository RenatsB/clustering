#include <gtest/gtest.h>
#include "testParams.h"
#include "testUtils.h"
#include "gpuImageGen.h"
#include "gpuKmeans.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( GPUkmeans, ColorVectorBased )
{
    RandomFn<float> rg;
    ColorVector cv = gpuImageGen::generate_parallel_CV(CLIB_TEST_DIMX,
                                                       CLIB_TEST_DIMY,
                                                       CLIB_TEST_NOISE,
                                                       CLIB_TEST_THREADS);

    ColorVector flt1 = gpuKmeans::kmeans_parallel_CV(cv,
                                                     CLIB_TEST_CLUSTERS,
                                                     CLIB_TEST_ITER,
                                                     CLIB_TEST_THREADS);
    for(auto c=0; c<CLIB_TEST_ITEMS; ++c)
    {
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c).m_r, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c).m_g, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c).m_b, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c).m_a, 0.f, 1.f, 0.001f));
        EXPECT_FALSE(clibTutils::compareColors(flt1.at(c),cv.at(c)));
    }
}

TEST( GPUkmeans, ImageColorsBased)
{
    RandomFn<float> rg;
    ImageColors ic = gpuImageGen::generate_parallel_IC(CLIB_TEST_DIMX,
                                                       CLIB_TEST_DIMY,
                                                       CLIB_TEST_NOISE,
                                                       CLIB_TEST_THREADS);
    ImageColors flt1 = gpuKmeans::kmeans_parallel_IC(ic,
                                                     CLIB_TEST_CLUSTERS,
                                                     CLIB_TEST_ITER,
                                                     CLIB_TEST_THREADS);
    for(auto c=0; c<CLIB_TEST_ITEMS; ++c)
    {
        EXPECT_TRUE(clibTutils::testRange(flt1.m_r.at(c), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.m_g.at(c), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.m_b.at(c), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.m_a.at(c), 0.f, 1.f, 0.001f));
        EXPECT_FALSE(clibTutils::compareColors(flt1.m_r.at(c),
                                               flt1.m_g.at(c),
                                               flt1.m_b.at(c),
                                               flt1.m_a.at(c),
                                               ic.m_r.at(c),
                                               ic.m_g.at(c),
                                               ic.m_b.at(c),
                                               ic.m_a.at(c)));
    }
}

TEST( GPUkmeans, linearBased)
{
    RandomFn<float> rg;
    std::vector<float> ln = gpuImageGen::generate_parallel_LN(CLIB_TEST_DIMX,
                                                              CLIB_TEST_DIMY,
                                                              CLIB_TEST_NOISE,
                                                              CLIB_TEST_THREADS);
    std::vector<float> flt1 = gpuKmeans::kmeans_parallel_LN(ln,
                                                            CLIB_TEST_CLUSTERS,
                                                            CLIB_TEST_ITER,
                                                            CLIB_TEST_THREADS);
    for(auto c=0; c<CLIB_TEST_ITEMS; ++c)
    {
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c*4), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c*4+1), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c*4+2), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c*4+3), 0.f, 1.f, 0.001f));
        EXPECT_FALSE(clibTutils::compareColors(flt1, ln, c*4));
    }
}

TEST( GPUkmeans, 4VectorPointers )
{
    RandomFn<float> rg;
    std::vector<float> rv(CLIB_TEST_ITEMS);
    std::vector<float> gv(CLIB_TEST_ITEMS);
    std::vector<float> bv(CLIB_TEST_ITEMS);
    std::vector<float> av(CLIB_TEST_ITEMS);
    gpuImageGen::generate_parallel_4SV(&rv,&gv,&bv,&av,
                                       CLIB_TEST_DIMX,
                                       CLIB_TEST_DIMY,
                                       CLIB_TEST_NOISE,
                                       CLIB_TEST_THREADS);
    std::vector<float> flt1_rv(CLIB_TEST_ITEMS);
    std::vector<float> flt1_gv(CLIB_TEST_ITEMS);
    std::vector<float> flt1_bv(CLIB_TEST_ITEMS);
    std::vector<float> flt1_av(CLIB_TEST_ITEMS);
    gpuKmeans::kmeans_parallel_4SV(&rv,&gv,&bv,&av,
                                   &flt1_rv,&flt1_gv,&flt1_bv,&flt1_av,
                                   CLIB_TEST_ITEMS,
                                   CLIB_TEST_CLUSTERS,
                                   CLIB_TEST_ITER,
                                   CLIB_TEST_THREADS);
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(flt1_rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1_gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1_bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1_av.at(i), 0.f, 1.f, 0.001f));
        EXPECT_FALSE(clibTutils::compareColors(flt1_rv.at(i),
                                               flt1_gv.at(i),
                                               flt1_bv.at(i),
                                               flt1_av.at(i),
                                               rv.at(i),
                                               gv.at(i),
                                               bv.at(i),
                                               av.at(i)));
    }
}

TEST( GPUkmeans, 4FloatPointers )
{
    RandomFn<float> rg;
    std::vector<float> rv(CLIB_TEST_ITEMS);
    std::vector<float> gv(CLIB_TEST_ITEMS);
    std::vector<float> bv(CLIB_TEST_ITEMS);
    std::vector<float> av(CLIB_TEST_ITEMS);
    gpuImageGen::generate_parallel_4LV(rv.data(),gv.data(),bv.data(),av.data(),
                                       CLIB_TEST_DIMX,
                                       CLIB_TEST_DIMY,
                                       CLIB_TEST_NOISE,
                                       CLIB_TEST_THREADS);
    std::vector<float> flt1_rv(CLIB_TEST_ITEMS);
    std::vector<float> flt1_gv(CLIB_TEST_ITEMS);
    std::vector<float> flt1_bv(CLIB_TEST_ITEMS);
    std::vector<float> flt1_av(CLIB_TEST_ITEMS);
    gpuKmeans::kmeans_parallel_4LV(rv.data(),gv.data(),bv.data(),av.data(),
                                   flt1_rv.data(),flt1_gv.data(),flt1_bv.data(),flt1_av.data(),
                                   CLIB_TEST_ITEMS,
                                   CLIB_TEST_CLUSTERS,
                                   CLIB_TEST_ITER,
                                   CLIB_TEST_THREADS);
    for(size_t i=0; i<CLIB_TEST_ITEMS; ++i)
    {
        EXPECT_TRUE(clibTutils::testRange(flt1_rv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1_gv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1_bv.at(i), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1_av.at(i), 0.f, 1.f, 0.001f));
        EXPECT_FALSE(clibTutils::compareColors(flt1_rv.at(i),
                                               flt1_gv.at(i),
                                               flt1_bv.at(i),
                                               flt1_av.at(i),
                                               rv.at(i),
                                               gv.at(i),
                                               bv.at(i),
                                               av.at(i)));
    }
}

////----------------------------------------------------------------------------------------------------------------------
