#include "kmeansP.h"

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
                                thrust::device_ptr<int> counts,
                                thrust::device_ptr<int> h_assign)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    // Make global loads once.
    const float4 current = data[index];

    float best_distance = FLT_MAX;
    int best_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster)
    {
        const float distance = sq_Col_l2_Dist_Parallel(current, means[cluster]);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_cluster = cluster;
        }
    }

    atomicAdd(&thrust::raw_pointer_cast(new_sums + best_cluster)->x, current.x);
    atomicAdd(&thrust::raw_pointer_cast(new_sums + best_cluster)->y, current.y);
    atomicAdd(&thrust::raw_pointer_cast(new_sums + best_cluster)->z, current.z);
    atomicAdd(&thrust::raw_pointer_cast(new_sums + best_cluster)->w, current.w);
    atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);
    h_assign[index]=best_cluster;
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(thrust::device_ptr<float4> means,
                                const thrust::device_ptr<float4> new_sum,
                                const thrust::device_ptr<int> counts)
{
    const int cluster = threadIdx.x;
    const int count = max(1, counts[cluster]);
    float4 temp = new_sum[cluster];
    temp.x/=count;
    temp.y/=count;
    temp.z/=count;
    temp.w/=count;
    means[cluster] = temp;
}

__global__ void write_new_mean_colors(thrust::device_ptr<float4> means,
                                      size_t data_size,
                                      thrust::device_ptr<int> assignment,
                                      thrust::device_ptr<float4> newOut)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;
    newOut[index] = means[assignment[index]];
}
std::vector<float> kmeansP(const ColorVector &source,
                           size_t k,
                           size_t number_of_iterations,
                           RandomFn<float>* rfunc,
                           const uint numThreads)
{
    thrust::host_vector<float4> h_source(source.size());
    for(uint x=0; x<source.size(); ++x)
    {
        h_source[x].x=source.at(x).m_r;
        h_source[x].y=source.at(x).m_g;
        h_source[x].z=source.at(x).m_b;
        h_source[x].w=source.at(x).m_a;
    }
    const size_t number_of_elements = source.size();
    //thrust::fill(h_source.begin(), h_source.end(), source.begin());
    thrust::device_vector<float4> d_means(k);
    rfunc->setNumericLimitsL(0, number_of_elements - 1);
    for(uint cluster=0; cluster<k; ++cluster)
    {
        float4 assignment;
        Color c = source[rfunc->MT19937RandL()];
        assignment.x = c.m_r;
        assignment.y = c.m_g;
        assignment.z = c.m_b;
        assignment.w = c.m_a;
        d_means[cluster] = assignment;
    }

    thrust::device_vector<float4> d_source(source.size());
    thrust::copy(h_source.begin(), h_source.end(), d_source.begin());
    thrust::device_vector<int> d_assignments(source.size()); //for cluster assignments
    thrust::device_vector<float4> d_filtered(source.size()); //to copy back and return
    thrust::host_vector<float4> h_filtered(source.size());

    thrust::device_vector<float4> d_sums(k);
    thrust::device_vector<int> d_counts(k, 0);

    const int blocks = (number_of_elements + numThreads - 1) / numThreads;

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sums.begin(), d_sums.end(), float4{0.0,0.0,0.0,0.0});
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assign_clusters<<<blocks, numThreads>>>(d_source.data(),
                                             number_of_elements,
                                             d_means.data(),
                                             d_sums.data(),
                                             k,
                                             d_counts.data(),
                                             d_assignments.data());
        cudaDeviceSynchronize();

        compute_new_means<<<1, k>>>(d_means.data(),
                                    d_sums.data(),
                                    d_counts.data());
        cudaDeviceSynchronize();
    }
    write_new_mean_colors<<<blocks, numThreads>>>(d_means.data(),
                                               number_of_elements,
                                               d_assignments.data(),
                                               d_filtered.data());
    cudaDeviceSynchronize();
    thrust::copy(d_filtered.begin(), d_filtered.end(), h_filtered.begin());

    //h_source = d_source;
    std::vector<float> ret(source.size()*4);
    for(uint i=0; i<source.size(); ++i)
    {
        ret.at(i*4)   = h_filtered[i].x;
        ret.at(i*4+1) = h_filtered[i].y;
        ret.at(i*4+2) = h_filtered[i].z;
        ret.at(i*4+3) = h_filtered[i].w;
    }
    return ret;
}

//=======================================================================
//--------------------------|   LINEAR   |-------------------------------
//=======================================================================

__device__ float linear_sq_Col_l2_Dist_Parallel(float FR,float FG,float FB, float SR,float SG, float SB)
{
    return sq_Parallel(FR - SR) + sq_Parallel(FG - SG) + sq_Parallel(FB - SB);
}

__global__ void linear_assign_clusters(thrust::device_ptr<float> data,
                                size_t data_size,
                                const thrust::device_ptr<float> means,
                                thrust::device_ptr<float> new_sums,
                                size_t k,
                                thrust::device_ptr<int> counts,
                                thrust::device_ptr<int> h_assign)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    float best_distance = FLT_MAX;
    int best_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster)
    {
        const float distance = linear_sq_Col_l2_Dist_Parallel(data[index*4],data[index*4+1],data[index*4+2],means[cluster*4],means[cluster*4+1],means[cluster*4+2]);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_cluster = cluster;
        }
    }

    atomicAdd(thrust::raw_pointer_cast(new_sums + best_cluster*4), data[index*4]);
    atomicAdd(thrust::raw_pointer_cast(new_sums + best_cluster*4+1), data[index*4+1]);
    atomicAdd(thrust::raw_pointer_cast(new_sums + best_cluster*4+2), data[index*4+2]);
    atomicAdd(thrust::raw_pointer_cast(new_sums + best_cluster*4+3), data[index*4+3]);
    atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);
    h_assign[index]=best_cluster;
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void linear_compute_new_means(thrust::device_ptr<float> means,
                                const thrust::device_ptr<float> new_sum,
                                const thrust::device_ptr<int> counts)
{
    const int cluster = threadIdx.x;
    const int count = max(1, counts[cluster]);
    means[cluster*4]   = new_sum[cluster*4]  /count;
    means[cluster*4+1] = new_sum[cluster*4+1]/count;
    means[cluster*4+2] = new_sum[cluster*4+2]/count;
    means[cluster*4+3] = new_sum[cluster*4+3]/count;
}

__global__ void linear_write_new_mean_colors(thrust::device_ptr<float> means,
                                      size_t data_size,
                                      thrust::device_ptr<int> assignment,
                                      thrust::device_ptr<float> newOut)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;
    newOut[index*4]   = means[assignment[index]*4];
    newOut[index*4+1] = means[assignment[index]*4+1];
    newOut[index*4+2] = means[assignment[index]*4+2];
    newOut[index*4+3] = means[assignment[index]*4+3];
}
std::vector<float> linear_kmeansP(const std::vector<float> &source,
                                  size_t k,
                                  size_t number_of_iterations,
                                  RandomFn<float>* rfunc,
                                  const uint numThreads)
{
    const size_t number_of_elements = source.size()/4;
    thrust::device_vector<float> d_means(k*4);
    rfunc->setNumericLimitsL(0, number_of_elements - 1);
    for(uint cluster=0; cluster<k; ++cluster)
    {
        size_t cID = rfunc->MT19937RandL();
        d_means[cluster*4]   = source[cID*4];
        d_means[cluster*4+1] = source[cID*4+1];
        d_means[cluster*4+2] = source[cID*4+2];
        d_means[cluster*4+3] = source[cID*4+3];
    }

    thrust::device_vector<float> d_source(source.size());
    thrust::copy(source.begin(), source.end(), d_source.begin());
    thrust::device_vector<int> d_assignments(source.size()/4); //for cluster assignments
    thrust::device_vector<float> d_filtered(source.size()); //to copy back and return
    thrust::host_vector<float> h_filtered(source.size());

    thrust::device_vector<float> d_sums(k*4);
    thrust::device_vector<int> d_counts(k, 0);

    const int blocks = (number_of_elements + numThreads - 1) / numThreads;

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sums.begin(), d_sums.end(), 0.f);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        linear_assign_clusters<<<blocks, numThreads>>>(d_source.data(),
                                             number_of_elements,
                                             d_means.data(),
                                             d_sums.data(),
                                             k,
                                             d_counts.data(),
                                             d_assignments.data());
        cudaDeviceSynchronize();

        linear_compute_new_means<<<1, k>>>(d_means.data(),
                                    d_sums.data(),
                                    d_counts.data());
        cudaDeviceSynchronize();
    }
    linear_write_new_mean_colors<<<blocks, numThreads>>>(d_means.data(),
                                               number_of_elements,
                                               d_assignments.data(),
                                               d_filtered.data());
    cudaDeviceSynchronize();
    thrust::copy(d_filtered.begin(), d_filtered.end(), h_filtered.begin());

    //h_source = d_source;
    std::vector<float> ret(source.size());
    for(uint i=0; i<source.size(); ++i)
    {
        ret.at(i)   = h_filtered[i];
    }
    return ret;
}
