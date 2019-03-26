#ifndef _CLIB_TESTKMEANS_H
#define _CLIB_TESTKMEANS_H

#include <gtest/gtest.h>
#include "testUtils.hpp"
#include "cpuImageGen.hpp"
#include "cpuKmeans.hpp"

////----------------------------------------------------------------------------------------------------------------------

TEST( CPUkmeans, ColorVectorBased)
{
    cpuKmeans km;
    //ColorVector cv(512*512);
    ImageGenFn gen;
    ColorVector cv = gen.generate_serial_CV(512,512,128);
    ColorVector flt1 = km.kmeans_serial_CV(cv, 8, 1);
    for(auto c=0; c<cv.size(); ++c)
    {
        EXPECT_GE(flt1.at(c).m_r, 0.f);
        EXPECT_LE(flt1.at(c).m_r, 1.f);

        EXPECT_GE(flt1.at(c).m_g, 0.f);
        EXPECT_LE(flt1.at(c).m_g, 1.f);
        EXPECT_GE(flt1.at(c).m_b, 0.f);
        EXPECT_LE(flt1.at(c).m_b, 1.f);
        EXPECT_GE(flt1.at(c).m_a, 0.f);
        EXPECT_LE(flt1.at(c).m_a, 1.f);
        bool equal = true;
        if(flt1.at(c).m_r!=cv.at(c).m_r||flt1.at(c).m_g!=cv.at(c).m_g||
           flt1.at(c).m_b!=cv.at(c).m_b||flt1.at(c).m_a!=cv.at(c).m_a)
            equal = false;
        EXPECT_FALSE(equal);
    }
    /*ColorVector flt2 = km.kmeans_serial_CV(cv, 8, 1);
    for(auto c=0; c<cv.size(); ++c)
    {
        EXPECT_GE(flt2.at(c).m_r, 0.f);
        EXPECT_LE(flt2.at(c).m_r, 1.f);
        EXPECT_GE(flt2.at(c).m_g, 0.f);
        EXPECT_LE(flt2.at(c).m_g, 1.f);
        EXPECT_GE(flt2.at(c).m_b, 0.f);
        EXPECT_LE(flt2.at(c).m_b, 1.f);
        EXPECT_GE(flt2.at(c).m_a, 0.f);
        EXPECT_LE(flt2.at(c).m_a, 1.f);
        bool equal = true;
        if(flt2.at(c).m_r!=flt1.at(c).m_r||flt2.at(c).m_g!=flt1.at(c).m_g||
           flt2.at(c).m_b!=flt1.at(c).m_b||flt2.at(c).m_a!=flt1.at(c).m_a)
            equal = false;
        EXPECT_FALSE(equal);
    }
    ColorVector flt3 = km.kmeans_serial_CV(cv, 4, 2);
    for(auto c=0; c<cv.size(); ++c)
    {
        EXPECT_GE(flt2.at(c).m_r, 0.f);
        EXPECT_LE(flt2.at(c).m_r, 1.f);
        EXPECT_GE(flt2.at(c).m_g, 0.f);
        EXPECT_LE(flt2.at(c).m_g, 1.f);
        EXPECT_GE(flt2.at(c).m_b, 0.f);
        EXPECT_LE(flt2.at(c).m_b, 1.f);
        EXPECT_GE(flt2.at(c).m_a, 0.f);
        EXPECT_LE(flt2.at(c).m_a, 1.f);
        bool equal = true;
        if(flt3.at(c).m_r!=flt2.at(c).m_r||flt3.at(c).m_g!=flt2.at(c).m_g||
           flt3.at(c).m_b!=flt2.at(c).m_b||flt3.at(c).m_a!=flt2.at(c).m_a)
            equal = false;
        EXPECT_FALSE(equal);
    }*/
}

TEST( CPUkmeans, linearBased)
{

}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CLIB_TESTKMEANS_H
