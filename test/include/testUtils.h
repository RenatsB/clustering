#include <gtest/gtest.h>
#include "utilTypes.hpp"

namespace clibTutils
{
::testing::AssertionResult testRange(const float value,
                                     const float min,
                                     const float max,
                                     const float precision);

::testing::AssertionResult compareColors(const Color& A,
                                         const Color& B);

::testing::AssertionResult compareColors(const ImageColors& A,
                                         const ImageColors& B,
                                         const size_t id);

::testing::AssertionResult compareColors(const std::vector<float>& A,
                                         const std::vector<float>& B,
                                         const size_t id);

::testing::AssertionResult compareColors(const float Fr,
                                         const float Fg,
                                         const float Fb,
                                         const float Fa,
                                         const float Sr,
                                         const float Sg,
                                         const float Sb,
                                         const float Sa);
}
