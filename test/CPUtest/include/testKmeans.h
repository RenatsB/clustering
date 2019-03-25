#ifndef _CLIB_TESTKMEANS_H
#define _CLIB_TESTKMEANS_H

#include <gtest/gtest.h>
#include "cpuImageGen.hpp"
#include "cpuKmeans.hpp"

////----------------------------------------------------------------------------------------------------------------------

TEST( CPUkmeans, ColorVectorBased)
{
    /*cpuKmeans km;
    ColorVector cv(512*512);
    ImageGenFn gen;
    cv = gen.generate_serial_CV(512,512,128);
    ColorVector flt1 = km.kmeans_serial_CV(cv, 4, 1);
    for(auto c=0; c<cv.size(); ++c)
    {
        EXPECT_GE(flt1.at(c*4),   0.f);
        EXPECT_LE(flt1.at(c*4),   1.f);
        EXPECT_GE(flt1.at(c*4+1), 0.f);
        EXPECT_LE(flt1.at(c*4+1), 1.f);
        EXPECT_GE(flt1.at(c*4+2), 0.f);
        EXPECT_LE(flt1.at(c*4+2), 1.f);
        EXPECT_GE(flt1.at(c*4+3), 0.f);
        EXPECT_LE(flt1.at(c*4+3), 1.f);
        bool equal = true;
        if(flt1.at(c*4)!=cv.at(c).m_r||flt1.at(c*4+1)!=cv.at(c).m_g||
           flt1.at(c*4+2)!=cv.at(c).m_b||flt1.at(c*4+3)!=cv.at(c).m_a)
            equal = false;
        EXPECT_FALSE(equal);
    }
    ColorVector flt2 = km.kmeans_serial_CV(cv, 8, 1);
    for(auto c=0; c<cv.size(); ++c)
    {
        EXPECT_GE(flt2.at(c*4),   0.f);
        EXPECT_LE(flt2.at(c*4),   1.f);
        EXPECT_GE(flt2.at(c*4+1), 0.f);
        EXPECT_LE(flt2.at(c*4+1), 1.f);
        EXPECT_GE(flt2.at(c*4+2), 0.f);
        EXPECT_LE(flt2.at(c*4+2), 1.f);
        EXPECT_GE(flt2.at(c*4+3), 0.f);
        EXPECT_LE(flt2.at(c*4+3), 1.f);
        bool equal = true;
        if(flt2.at(c)!=flt1.at(c)||flt2.at(c*4+1)!=flt1.at(c*4+1)||
           flt2.at(c)!=flt1.at(c)||flt2.at(c*4+3)!=flt1.at(c*4+3))
            equal = false;
        EXPECT_FALSE(equal);
    }
    ColorVector flt3 = km.kmeans_serial_CV(cv, 4, 2);
    for(auto c=0; c<cv.size(); ++c)
    {
        EXPECT_GE(flt2.at(c*4),   0.f);
        EXPECT_LE(flt2.at(c*4),   1.f);
        EXPECT_GE(flt2.at(c*4+1), 0.f);
        EXPECT_LE(flt2.at(c*4+1), 1.f);
        EXPECT_GE(flt2.at(c*4+2), 0.f);
        EXPECT_LE(flt2.at(c*4+2), 1.f);
        EXPECT_GE(flt2.at(c*4+3), 0.f);
        EXPECT_LE(flt2.at(c*4+3), 1.f);
        bool equal = true;
        if(flt3.at(c*4)!=flt2.at(c*4)||flt3.at(c*4+1)!=flt2.at(c*4+1)||
           flt3.at(c*4+2)!=flt2.at(c*4+2)||flt3.at(c*4+3)!=flt2.at(c*4+3))
            equal = false;
        EXPECT_FALSE(equal);
    }*/
}

TEST( CPUkmeans, linearBased)
{

}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CLIB_TESTKMEANS_H
