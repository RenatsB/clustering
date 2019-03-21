#include "gtest/gtest.h"
#include "testUtils.h"
#include "gpuRandF.h"

////----------------------------------------------------------------------------------------------------------------------

TEST( RandomGPU, fillVectorGpu )
{
    thrust::device_vector<float> dv(10000, 10.f);
    float* dvPtr = thrust::raw_pointer_cast(dv.data());
    GPUclib::randFloatsInternal(dvPtr,10000);
    cudaDeviceSynchronize();
    std::vector<float> v = device_vector_to_host(dv);

    for(auto val : v)
    {
        EXPECT_NE(val, 10.f);
        EXPECT_LE(val, 1.f);
        EXPECT_GE(val, 0.f);
    }
}

////----------------------------------------------------------------------------------------------------------------------
