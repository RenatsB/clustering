#ifndef _CLIB_TESTCLIBTYPES_H
#define _CLIB_TESTCLIBTYPES_H

#include <gtest/gtest.h>
#include "utilTypes.hpp"

////----------------------------------------------------------------------------------------------------------------------

using nn=std::numeric_limits<float>;

TEST( ColorConstructor, unspecified )
{
  Color c1;
  EXPECT_EQ(c1.m_r, 1.f);
  EXPECT_EQ(c1.m_g, 1.f);
  EXPECT_EQ(c1.m_b, 1.f);
  EXPECT_EQ(c1.m_a, 1.f);
}
TEST( ColorConstructor, specified )
{
  Color c1(0.15f,0.23f,0.78f,0.3f);
  EXPECT_EQ(c1.m_r, 0.15f);
  EXPECT_EQ(c1.m_g, 0.23f);
  EXPECT_EQ(c1.m_b, 0.78f);
  EXPECT_EQ(c1.m_a, 0.3f);
  Color c2{0.225f,0.347f,0.1f,0.f};
  EXPECT_EQ(c2.m_r, 0.225f);
  EXPECT_EQ(c2.m_g, 0.347f);
  EXPECT_EQ(c2.m_b, 0.1f);
  EXPECT_EQ(c2.m_a, 0.f);

  Color c3(nn::quiet_NaN(),nn::quiet_NaN(),nn::quiet_NaN(),nn::quiet_NaN());
  EXPECT_NE(c3.m_r, c3.m_r);
  EXPECT_NE(c3.m_g, c3.m_g);
  EXPECT_NE(c3.m_b, c3.m_b);
  EXPECT_NE(c3.m_a, c3.m_a);
}

////----------------------------------------------------------------------------------------------------------------------

TEST( ColorValueModification, direct )
{
    Color c1;
    c1.m_r+=253.f;
    EXPECT_EQ(c1.m_r, 254.f);
    c1.m_g/=5.f;
    EXPECT_EQ(c1.m_g, 0.2f);
    c1.m_b=8.f;
    EXPECT_EQ(c1.m_b, 8.f);
    c1.m_a = c1.m_g;
    EXPECT_EQ(c1.m_a, c1.m_g);
}

TEST( ColorValueModification, methodBased )
{
    Color c1(0.f,0.f,0.f,0.f);
    EXPECT_EQ(c1.m_r, 0.f);
    EXPECT_EQ(c1.m_g, 0.f);
    EXPECT_EQ(c1.m_b, 0.f);
    EXPECT_EQ(c1.m_a, 0.f);

    c1.setData(1.f,2.f,3.f,4.f);
    EXPECT_EQ(c1.m_r, 1.f);
    EXPECT_EQ(c1.m_g, 2.f);
    EXPECT_EQ(c1.m_b, 3.f);
    EXPECT_EQ(c1.m_a, 4.f);

    c1.setData(1.f);
    EXPECT_EQ(c1.m_r, 1.f);
    EXPECT_EQ(c1.m_g, 1.f);
    EXPECT_EQ(c1.m_b, 1.f);
    EXPECT_EQ(c1.m_a, 1.f);
}

TEST( ColorValueModification, operatorBased )
{
    Color c1(0.5f,0.5f,0.5f,0.5f);
    Color c2(2.f,2.f,2.f,2.f);
    float d = 10.f;
    int di = 2;
    Color c3 = c1+c2;
    Color c4 = c1/d;
    Color c5 = c2/di;
    EXPECT_EQ(c3.m_r, c1.m_r+c2.m_r);
    EXPECT_EQ(c3.m_g, c1.m_g+c2.m_g);
    EXPECT_EQ(c3.m_b, c1.m_b+c2.m_b);
    EXPECT_EQ(c3.m_a, c1.m_a+c2.m_a);

    EXPECT_EQ(c4.m_r, c1.m_r/d);
    EXPECT_EQ(c4.m_g, c1.m_g/d);
    EXPECT_EQ(c4.m_b, c1.m_b/d);
    EXPECT_EQ(c4.m_a, c1.m_a/d);

    EXPECT_EQ(c5.m_r, c2.m_r/di);
    EXPECT_EQ(c5.m_g, c2.m_g/di);
    EXPECT_EQ(c5.m_b, c2.m_b/di);
    EXPECT_EQ(c5.m_a, c2.m_a/di);
}

////----------------------------------------------------------------------------------------------------------------------

TEST( ImageColorsConstructor, initialData )
{
    ImageColors img1;
    EXPECT_EQ(img1.m_r.size(), 0);
    EXPECT_EQ(img1.m_g.size(), 0);
    EXPECT_EQ(img1.m_b.size(), 0);
    EXPECT_EQ(img1.m_a.size(), 0);
}

////----------------------------------------------------------------------------------------------------------------------

TEST( ImageColorsMethods, resizing )
{
    ImageColors img1;
    img1.resize(20);
    EXPECT_EQ(img1.m_r.size(), 20);
    EXPECT_EQ(img1.m_g.size(), 20);
    EXPECT_EQ(img1.m_b.size(), 20);
    EXPECT_EQ(img1.m_a.size(), 20);
}

TEST( ImageColorsMethods, setAtIndex )
{
    ImageColors img1;
    img1.resize(3);
    img1.setData(0, 1.f, 0.f, 0.f, 1.f);
    img1.setData(2, 0.f, 0.f, 1.f, 1.f);
    img1.setData(1, 0.f, 1.f, 0.f, 1.f);
    EXPECT_EQ(img1.m_r.size(), 3);
    EXPECT_EQ(img1.m_g.size(), 3);
    EXPECT_EQ(img1.m_b.size(), 3);
    EXPECT_EQ(img1.m_a.size(), 3);

    EXPECT_EQ(img1.m_r.at(0), 1.f);
    EXPECT_EQ(img1.m_g.at(0), 0.f);
    EXPECT_EQ(img1.m_b.at(0), 0.f);
    EXPECT_EQ(img1.m_a.at(0), 1.f);

    EXPECT_EQ(img1.m_r.at(1), 0.f);
    EXPECT_EQ(img1.m_g.at(1), 1.f);
    EXPECT_EQ(img1.m_b.at(1), 0.f);
    EXPECT_EQ(img1.m_a.at(1), 1.f);

    EXPECT_EQ(img1.m_r.at(2), 0.f);
    EXPECT_EQ(img1.m_g.at(2), 0.f);
    EXPECT_EQ(img1.m_b.at(2), 1.f);
    EXPECT_EQ(img1.m_a.at(2), 1.f);
}

TEST( ImageColorsMethods, setData )
{
    ImageColors img1;
    std::vector<float> reds{0.f,1.f,2.f,3.f,4.f,5.f};
    std::vector<float> greens{5.f,4.f,3.f,2.f,1.f,nn::quiet_NaN()};
    std::vector<float> blues{3.f,1.f,4.f,0.f};
    //6
    img1.setData(&reds, &greens, &blues);
    EXPECT_EQ(img1.m_r.size(), reds.size());
    EXPECT_EQ(img1.m_g.size(), greens.size());
    EXPECT_EQ(img1.m_b.size(), blues.size());
    EXPECT_EQ(img1.m_a.size(), reds.size());
    EXPECT_NE(img1.m_a.size(), blues.size());

    EXPECT_EQ(img1.m_b.at(0), blues.at(0));
    EXPECT_EQ(img1.m_r.at(3), reds.at(3));
    EXPECT_EQ(img1.m_g.at(1), greens.at(1));
    EXPECT_NE(greens.at(5), greens.at(5));
}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CLIB_TESTCLIBTYPES_H

