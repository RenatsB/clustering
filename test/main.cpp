#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <memory>

//tests
#include "testClibTypes.h"
#include "testRandom.h"
#include "testKmeans.h"
#include "testImgGen.h"
// end tests

int main( int argc, char **argv )
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
