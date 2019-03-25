#include "gtest/gtest.h"

#include "testClibTypes.h"
#include "testImgGen.h"
#include "testKmeans.h"
#include "testRandom.h"

int main( int argc, char **argv )
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
