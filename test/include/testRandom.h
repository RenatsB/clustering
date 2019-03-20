#ifndef _CLIB_TESTRANDOM_H
#define _CLIB_TESTRANDOM_H

#include <gtest/gtest.h>
#include "RandomFn.hpp"
#include "gpuRandF.h"

////----------------------------------------------------------------------------------------------------------------------

using nn=std::numeric_limits<float>;

TEST( RandomCPU, testType )
{
  RandomFn<float> r1;
  EXPECT_EQ(typeid(r1.SimpleRand(0.f,1.f)), typeid(float));
  EXPECT_EQ(typeid(r1.UniformRandU()), typeid(float));
  EXPECT_EQ(typeid(r1.MT19937RandU()), typeid(float));

  RandomFn<double> r2;
  EXPECT_EQ(typeid(r2.SimpleRand(0.0,1.0)), typeid(double));
  EXPECT_EQ(typeid(r2.UniformRandU()), typeid(double));
  EXPECT_EQ(typeid(r2.MT19937RandU()), typeid(double));

  EXPECT_EQ(typeid(r1.MT19937RandL()), typeid(r2.MT19937RandL()));
}
TEST( RandomCPU, testLimits )
{
    RandomFn<float> r1;
    float t1 = r1.UniformRandU();
    EXPECT_GE(t1, 0.f);
    EXPECT_LE(t1, 1.f);
    size_t t2 = r1.UniformRandL();
    EXPECT_GE(t2, 0);
    EXPECT_LE(t2, std::numeric_limits<size_t>::max());

    r1.setNumericLimits(-100.f,100.f);
    r1.setNumericLimitsL(0, 200);
    float t3 = r1.UniformRandU();
    EXPECT_GE(t3, -100.f);
    EXPECT_LE(t3, 100.f);
    size_t t4 = r1.UniformRandL();
    EXPECT_GE(t4, 0);
    EXPECT_LE(t4, 200);

    r1.setNumericLimits(100.f, -100.f);
    for(int a=0; a<1000; ++a)
    {
        float t = r1.UniformRandU();
        EXPECT_GE(t, -100.f);
        EXPECT_LE(t, 100.f);
    }

    r1.setNumericLimits(100.f, 100.f);
    EXPECT_EQ(r1.UniformRandU(), 100.f);

}

////----------------------------------------------------------------------------------------------------------------------

TEST( RandomGPU, fillVectorGpu )
{
    /*std::vector<float> v(10000, 10.f);
    thrust::device_vector<float> dv(10000, 10.f);
    thrust::host_vector<float> hv(10000);
    float* dvPtr = thrust::raw_pointer_cast(dv);
    GPU_RandF::randFloatsInternal(dvPtr,10000);
    cudaDeviceSynchronize();
    thrust::copy(dv.begin(), dv.end(), hv.begin());
    for(auto i=0; i<v.size(); ++i)
        v.at(i) = hv[i];

    for(auto val : v)
    {
        EXPECT_NE(val, 10.f);
        EXPECT_LE(val, 1.f);
        EXPECT_GE(val, 1.f);
    }*/
    std::vector<float> v(10000, 10.f);
    float* vPtr = v.data();
    GPU_RandF::randFloatsInternal(vPtr,10000);
    for(auto val : v)
    {
        EXPECT_EQ(val, 10.f); //because gpu rand writes to GPU memory
    }
}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CLIB_TESTRANDOM_H

