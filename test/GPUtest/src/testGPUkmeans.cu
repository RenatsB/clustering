#include "gtest/gtest.h"
#include "testUtils.h"
#include "ImageGen.h"
#include "kmeansP.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( GPUkmeans, COlorVectorBased )
{
    ColorVector cv = GPUclib::generate(64,64,64,64);
    RandomFn<float> rg;
    std::vector<float> fl = GPUclib::kmeansP(cv,4,1,&rg,64);
    for(auto c=0; c<cv.size(); ++c)
    {
        EXPECT_GE(fl.at(c*4),   0.f);
        EXPECT_LE(fl.at(c*4),   1.f);
        EXPECT_GE(fl.at(c*4+1), 0.f);
        EXPECT_LE(fl.at(c*4+1), 1.f);
        EXPECT_GE(fl.at(c*4+2), 0.f);
        EXPECT_LE(fl.at(c*4+2), 1.f);
        EXPECT_GE(fl.at(c*4+3), 0.f);
        EXPECT_LE(fl.at(c*4+3), 1.f);
        bool equal = true;
        if(fl.at(c*4)!=cv.at(c).m_r||fl.at(c*4+1)!=cv.at(c).m_g||
           fl.at(c*4+2)!=cv.at(c).m_b||fl.at(c*4+3)!=cv.at(c).m_a)
            equal = false;
        EXPECT_FALSE(equal);
    }
}


TEST( GPUkmeans, linearBased)
{
    std::vector<float> fv = GPUclib::linear_generate(64,64,64,64);
    RandomFn<float> rg;
    std::vector<float> fl = GPUclib::linear_kmeansP(fv,4,1,&rg,64);
    for(auto f=0; f<fv.size(); ++f)
    {
        EXPECT_GE(fl.at(f), 0.f);
        EXPECT_LE(fl.at(f), 1.f);
    }
    for(auto c=0; c<fv.size()/4; ++c)
    {
        bool equal = true;
        if(fv.at(c*4)!=fl.at(c*4)||fv.at(c*4+1)!=fl.at(c*4+1)||
           fv.at(c*4+2)!=fl.at(c*4+2)||fv.at(c*4+3)!=fl.at(c*4+3))
            equal = false;
        EXPECT_FALSE(equal);
    }
}

////----------------------------------------------------------------------------------------------------------------------
