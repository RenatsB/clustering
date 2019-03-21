#ifndef _CLIB_TESTKMEANS_H
#define _CLIB_TESTKMEANS_H

#include <gtest/gtest.h>
#include "kmeans.hpp"
#include "kmeansP.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( CPUkmeans, ColorVectorBased)
{
    kmeans km;
    ColorVector cv(512*512);
}

TEST( CPUkmeans, linearBased)
{

}

////----------------------------------------------------------------------------------------------------------------------

#endif // _CLIB_TESTKMEANS_H
