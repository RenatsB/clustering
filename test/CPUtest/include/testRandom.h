#ifndef _CLIB_TESTRANDOM_H
#define _CLIB_TESTRANDOM_H

#include <gtest/gtest.h>
#include "testUtils.h"
#include "cpuRandomFn.hpp"

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
    EXPECT_TRUE(clibTutils::testRange(t1, 0.f, 1.f, 0.001f));
    size_t t2 = r1.UniformRandL();
    EXPECT_GE(t2, 0);
    EXPECT_LE(t2, std::numeric_limits<size_t>::max());

    r1.setNumericLimits(-100.f,100.f);
    r1.setNumericLimitsL(0, 200);
    float t3 = r1.UniformRandU();
    EXPECT_TRUE(clibTutils::testRange(t3, -100.f, 100.f, 0.001f));
    size_t t4 = r1.UniformRandL();
    EXPECT_GE(t4, 0);
    EXPECT_LE(t4, 200);

    r1.setNumericLimits(100.f, -100.f);
    for(int a=0; a<1000; ++a)
    {
        float t = r1.UniformRandU();
        EXPECT_TRUE(clibTutils::testRange(t, -100.f, 100.f, 0.001f));
    }

    r1.setNumericLimits(100.f, 100.f);
    EXPECT_EQ(r1.UniformRandU(), 100.f);

}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CLIB_TESTRANDOM_H

