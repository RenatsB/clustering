#include "gtest/gtest.h"
#include "testUtils.h"
#include "gpuRandF.h"
#include <thrust/device_vector.h>

////----------------------------------------------------------------------------------------------------------------------

TEST( RandomGPU, fillVectorGpu )
{
    thrust::device_vector<float> dv(10000, 10.f);
    float* dvPtr = thrust::raw_pointer_cast(dv.data());
    gpuRandFn::randFloatsInternal(dvPtr,10000);
    cudaDeviceSynchronize();
    std::vector<float> v(dv.size());
    thrust::copy(dv.begin(), dv.end(), v.begin());

    for(auto val : v)
    {
        EXPECT_TRUE(clibTutils::testRange(val, 0.f, 1.f, 0.001f));
    }
}

////----------------------------------------------------------------------------------------------------------------------
