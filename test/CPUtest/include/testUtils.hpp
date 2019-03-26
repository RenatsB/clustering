#include <gtest/gtest.h>

namespace clibTutils
{
::testing::AssertionResult testRange(const float value, const float min, const float max, const float precision)
{
    if(value >= min-precision && value <= max+precision)
        return ::testing::AssertionSuccess();
    else
        return ::testing::AssertionFailure()
                << value << " is outside the range " << min << " to " << max;
}
}
