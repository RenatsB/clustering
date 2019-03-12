#include <cuda_runtime.h>
#include <iostream>

// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// For thrust routines (e.g. stl-like operators and algorithms on vectors)
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// Include
#include "../../serial/src/gen.cpp" //this includes generator, random and utilTypes
#include "../../serial/include/img.hpp"

// Needed to get matrix functions working in CUDA
//#include <eigen-nvcc/Dense>

// Note that the resolution has to be at least 4 because we are writing the values into shared
// memory in parallel: if the RES is less than 4 not all the values will be written to the shared
// matrix structure.
// NOTE: My system fails to handle a RES higher than 30 - it _should_ theoretically handle 32. This
//       could be a hardware fault. The result of RES being too high is that your output will just
//       be full of zeros.
#define RES 30
#define XRESOLUTION 512
#define YRESOLUTION 512

__device__ float sq_Parallel(float ref)
{
    return ref*ref;
}

__device__ float sq_Col_l2_Dist_Parallel(float4 first, float4 second)
{
    return sq_Parallel(first.x*first.w - second.x*second.w)
         + sq_Parallel(first.y*first.w - second.y*second.w)
         + sq_Parallel(first.z*first.w - second.z*second.w);
}

// In the assignment step, each point (thread) computes its distance to each
// cluster centroid and adds its x and y values to the sum of its closest
// centroid, as well as incrementing that centroid's count of assigned points.

//const float4* __restrict__ data this is waaaaaaay faster
__global__ void assign_clusters(thrust::device_ptr<float4> data,
                                size_t data_size,
                                const thrust::device_ptr<float4> means,
                                thrust::device_ptr<float4> new_sums,
                                size_t k,
                                thrust::device_ptr<int> counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    // Make global loads once.
    const float4 current = data[index];

    float best_distance = FLT_MAX;
    int best_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster) {
        const float distance = sq_Col_l2_Dist_Parallel(current, means[cluster]);
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
        }
    }

    atomicAdd(&thrust::raw_pointer_cast(new_sums + best_cluster)->x, current.x);
    atomicAdd(&thrust::raw_pointer_cast(new_sums + best_cluster)->y, current.y);
    atomicAdd(&thrust::raw_pointer_cast(new_sums + best_cluster)->z, current.z);
    atomicAdd(&thrust::raw_pointer_cast(new_sums + best_cluster)->w, current.w);
    atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(thrust::device_ptr<float4> means,
                                const thrust::device_ptr<float4> new_sum,
                                const thrust::device_ptr<int> counts) {
    const int cluster = threadIdx.x;
    const int count = max(1, counts[cluster]);
    float4 temp = new_sum[cluster];
    temp.x/=count;
    temp.y/=count;
    temp.z/=count;
    temp.w/=count;
    means[cluster] = temp;
}

int main(int argc, const char* argv[]) {

    Ran<float> rfunc;
    Gen picGen;
    size_t k = 4;
    size_t number_of_iterations = 1;

    DataFrame source = picGen.generate(XRESOLUTION,YRESOLUTION,128);
    thrust::host_vector<float4> h_source(source.size());
    for(uint x=0; x<source.size(); ++x)
    {
        h_source[x].x=source.at(x).m_r;
        h_source[x].y=source.at(x).m_g;
        h_source[x].z=source.at(x).m_b;
        h_source[x].w=source.at(x).m_a;
    }
    const size_t number_of_elements = h_source.size();
    //thrust::fill(h_source.begin(), h_source.end(), source.begin());
    thrust::device_vector<float4> d_means(k);
    rfunc.setNumericLimitsL(0, number_of_elements - 1);
    for(uint cluster=0; cluster<k; ++cluster)
    {
        d_means[cluster] = h_source[rfunc.MT19937RandL()];
    }

    thrust::device_vector<float4> d_source = h_source;
    thrust::host_vector<float4> h_filtered = h_source;

    thrust::device_vector<float4> d_sums(k);
    thrust::device_vector<int> d_counts(k, 0);

    const int threads = 1024;
    const int blocks = (number_of_elements + threads - 1) / threads;

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sums.begin(), d_sums.end(), float4{0.0,0.0,0.0,0.0});
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assign_clusters<<<blocks, threads>>>(d_source.data(),
                     number_of_elements,
                     d_means.data(),
                     d_sums.data(),
                     k,
                     d_counts.data());
        cudaDeviceSynchronize();
        //__syncthreads();

        compute_new_means<<<1, k>>>(d_means.data(),
            d_sums.data(),
            d_counts.data());
        cudaDeviceSynchronize();
        //__syncthreads();
    }
}
