#ifndef _CLIB_TESTKMEANS_H
#define _CLIB_TESTKMEANS_H

#include <gtest/gtest.h>
#include "testParams.h"
#include "testUtils.h"
#include "cpuImageGen.hpp"
#include "cpuKmeans.hpp"

////----------------------------------------------------------------------------------------------------------------------

TEST( CPUkmeans, ColorVectorBased)
{
    cpuKmeans km;
    ImageGenFn gen;
    ColorVector cv = gen.generate_serial_CV(CLIB_TEST_DIMX,
                                            CLIB_TEST_DIMY,
                                            CLIB_TEST_NOISE);
    ColorVector flt1 = km.kmeans_serial_CV(cv, CLIB_TEST_CLUSTERS, CLIB_TEST_ITER);
    for(auto c=0; c<cv.size(); ++c)
    {
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c).m_r, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c).m_g, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c).m_b, 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c).m_a, 0.f, 1.f, 0.001f));
        EXPECT_FALSE(clibTutils::compareColors(flt1.at(c), cv.at(c)));
    }
}

TEST( CPUkmeans, ImageColorsBased)
{
    cpuKmeans km;
    ImageGenFn gen;
    size_t items = CLIB_TEST_ITEMS;
    ImageColors ic = gen.generate_serial_IC(CLIB_TEST_DIMX,
                                            CLIB_TEST_DIMY,
                                            CLIB_TEST_NOISE);
    ImageColors flt1 = km.kmeans_serial_IC(ic,
                                           CLIB_TEST_CLUSTERS,
                                           CLIB_TEST_ITER);
    for(auto c=0; c<items; ++c)
    {
        EXPECT_TRUE(clibTutils::testRange(flt1.m_r.at(c), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.m_g.at(c), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.m_b.at(c), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.m_a.at(c), 0.f, 1.f, 0.001f));
        EXPECT_FALSE(clibTutils::compareColors(flt1, ic, c));
    }
}

TEST( CPUkmeans, linearBased)
{
    cpuKmeans km;
    ImageGenFn gen;
    size_t items = CLIB_TEST_ITEMS;
    std::vector<float> ln = gen.generate_serial_LN(CLIB_TEST_DIMX,
                                                   CLIB_TEST_DIMY,
                                                   CLIB_TEST_NOISE);
    std::vector<float> flt1 = km.kmeans_serial_LN(ln,
                                                  CLIB_TEST_CLUSTERS,
                                                  CLIB_TEST_ITER);
    for(auto c=0; c<items; ++c)
    {
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c*4), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c*4+1), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c*4+2), 0.f, 1.f, 0.001f));
        EXPECT_TRUE(clibTutils::testRange(flt1.at(c*4+3), 0.f, 1.f, 0.001f));
        EXPECT_FALSE(clibTutils::compareColors(flt1, ln, c*4));
    }
}

TEST( CPUkmeans, 4VectorPointers )
{
    cpuKmeans km;
    ImageGenFn gen;
    size_t items = CLIB_TEST_ITEMS;
    std::vector<float> rv(items);
    std::vector<float> gv(items);
    std::vector<float> bv(items);
    std::vector<float> av(items);
    gen.generate_serial_4SV(CLIB_TEST_DIMX,CLIB_TEST_DIMY,CLIB_TEST_NOISE,
                            &rv,&gv,&bv,&av);
    std::vector<float> flt1_rv(items);
    std::vector<float> flt1_gv(items);
    std::vector<float> flt1_bv(items);
    std::vector<float> flt1_av(items);
    km.kmeans_serial_4SV(&rv,&gv,&bv,&av,
                         &flt1_rv,&flt1_gv,&flt1_bv,&flt1_av,
                         items,CLIB_TEST_CLUSTERS, CLIB_TEST_ITER);
    for(size_t i=0; i<items; ++i)
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

TEST( CPUkmeans, 4FloatPointers )
{
    cpuKmeans km;
    ImageGenFn gen;
    size_t items = CLIB_TEST_ITEMS;
    std::vector<float> rv(items);
    std::vector<float> gv(items);
    std::vector<float> bv(items);
    std::vector<float> av(items);
    gen.generate_serial_4LV(CLIB_TEST_DIMX,CLIB_TEST_DIMY,CLIB_TEST_NOISE,
                            rv.data(),gv.data(),bv.data(),av.data());
    std::vector<float> flt1_rv(items);
    std::vector<float> flt1_gv(items);
    std::vector<float> flt1_bv(items);
    std::vector<float> flt1_av(items);
    km.kmeans_serial_4LV(rv.data(),gv.data(),bv.data(),av.data(),
                         flt1_rv.data(),flt1_gv.data(),flt1_bv.data(),flt1_av.data(),
                         items,CLIB_TEST_CLUSTERS, CLIB_TEST_ITER);
    for(size_t i=0; i<items; ++i)
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

#endif // _CLIB_TESTKMEANS_H
