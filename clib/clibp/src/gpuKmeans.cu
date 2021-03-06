#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "gpuKmeans.h"
#include "helper_math.h"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif




__device__ float sq_device(float ref)
{
    return ref*ref;
}

__device__ float sqL2Dist_device_CL(float4 first, float4 second)
{
    return sq_device(first.x*first.w - second.x*second.w)
         + sq_device(first.y*first.w - second.y*second.w)
         + sq_device(first.z*first.w - second.z*second.w);
}

__device__ float sqL2Dist_device_LN(float FR,
                                    float FG,
                                    float FB,
                                    float FA,
                                    float SR,
                                    float SG,
                                    float SB,
                                    float SA)
{
    return sq_device(FR*FA - SR*SA) + sq_device(FG*FA - SG*SA) + sq_device(FB*FA - SB*SA);
}

// In the assignment step, each point (thread) computes its distance to each
// cluster centroid and adds its value to the sum of its closest
// centroid, as well as incrementing that centroid's count of assigned points.

__global__ void assignClusters_parallel_CL(thrust::device_ptr<float4> data,
                                size_t data_size,
                                const thrust::device_ptr<float4> means,
                                thrust::device_ptr<float4> new_sums,
                                size_t k,
                                thrust::device_ptr<int> counts,
                                thrust::device_ptr<int> d_assign)
{
    //extern __shared__ float4 shared_means[];

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    // Let the first k threads copy over the cluster means.
    /*if (threadIdx.x < k) {
      // Using a flat array
      shared_means[threadIdx.x] = means[threadIdx.x];

    }*/
    //thrust::copy(thrust::raw_pointer_cast(means), thrust::raw_pointer_cast(means+k), &shared_means[0]);
    //__syncthreads();

    // Make global loads once.
    const float4 current = data[index];

    float best_distance = FLT_MAX;
    int best_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster)
    {
        const float distance = sqL2Dist_device_CL(current, means[cluster]);
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
    d_assign[index]=best_cluster;
}



__global__ void assignClusters_parallel_4F(thrust::device_ptr<float> inRed,
                                 thrust::device_ptr<float> inGrn,
                                 thrust::device_ptr<float> inBlu,
                                 thrust::device_ptr<float> inAlp,
                                 size_t data_size,
                                 const thrust::device_ptr<float> meansR,
                                 const thrust::device_ptr<float> meansG,
                                 const thrust::device_ptr<float> meansB,
                                 const thrust::device_ptr<float> meansA,
                                 thrust::device_ptr<float> new_sumsR,
                                 thrust::device_ptr<float> new_sumsG,
                                 thrust::device_ptr<float> new_sumsB,
                                 thrust::device_ptr<float> new_sumsA,
                                 size_t k,
                                 thrust::device_ptr<int> counts,
                                 thrust::device_ptr<int> d_assign)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    // Make global loads once.
    const float currentR = inRed[index];
    const float currentG = inGrn[index];
    const float currentB = inBlu[index];
    const float currentA = inAlp[index];

    float best_distance = FLT_MAX;
    int best_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster)
    {
        const float distance = sqL2Dist_device_LN(currentR,currentG,currentB,currentA,
                                                  meansR[cluster],
                                                  meansG[cluster],
                                                  meansB[cluster],
                                                  meansA[cluster]);
        if (distance < best_distance)
        {
            best_distance = distance;
            best_cluster = cluster;
        }
    }

    atomicAdd(thrust::raw_pointer_cast(new_sumsR + best_cluster), currentR);
    atomicAdd(thrust::raw_pointer_cast(new_sumsG + best_cluster), currentG);
    atomicAdd(thrust::raw_pointer_cast(new_sumsB + best_cluster), currentB);
    atomicAdd(thrust::raw_pointer_cast(new_sumsA + best_cluster), currentA);
    atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);
    d_assign[index]=best_cluster;
}

__global__ void assignClusters_parallel_LN(thrust::device_ptr<float> data,
                                size_t data_size,
                                const thrust::device_ptr<float> means,
                                thrust::device_ptr<float> new_sums,
                                size_t k,
                                thrust::device_ptr<int> counts,
                                thrust::device_ptr<int> d_assign)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;

    float best_distance = FLT_MAX;
    int best_cluster = 0;
    for (int cluster = 0; cluster < k; ++cluster)
    {
        const float distance = sqL2Dist_device_LN(data[index*4],
                                                  data[index*4+1],
                                                  data[index*4+2],
                                                  data[index*4+3],
                                                  means[cluster*4],
                                                  means[cluster*4+1],
                                                  means[cluster*4+2],
                                                  means[cluster*4+3]);
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
    d_assign[index]=best_cluster;
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void computeNewMeans_parallel_CL(thrust::device_ptr<float4> means,
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

__global__ void computeNewMeans_parallel_4F(thrust::device_ptr<float> meansR,
                                            thrust::device_ptr<float> meansG,
                                            thrust::device_ptr<float> meansB,
                                            thrust::device_ptr<float> meansA,
                                            const thrust::device_ptr<float> new_sumR,
                                            const thrust::device_ptr<float> new_sumG,
                                            const thrust::device_ptr<float> new_sumB,
                                            const thrust::device_ptr<float> new_sumA,
                                            const thrust::device_ptr<int> counts)
{
    const int cluster = threadIdx.x;
    const int count = max(1, counts[cluster]);
    meansR[cluster] = new_sumR[cluster]/count;
    meansG[cluster] = new_sumG[cluster]/count;
    meansB[cluster] = new_sumB[cluster]/count;
    meansA[cluster] = new_sumA[cluster]/count;
}

__global__ void computeNewMeans_parallel_LN(thrust::device_ptr<float> means,
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

//===================================================
//tried using shared memory for mean calculation and color assignment
/*
__global__ void fineReduce_parallel_CL(const thrust::device_ptr<float4> data,
                                       const size_t data_size,
                                       const thrust::device_ptr<float4> means,
                                       const thrust::device_ptr<int> d_assign,
                                       const thrust::device_ptr<float4> new_sums,
                                       const size_t k,
                                       const thrust::device_ptr<int> counts) {
  extern __shared__ uint8_t shared_memory[];
  float4* shared_means = (float4*)(shared_memory);
  int* shared_counts = (int*)(shared_means+k);

  const int local_index = threadIdx.x;
  const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_index >= data_size) return;

  // Load the mean values into shared memory.
  if (local_index < k) {
    shared_means[local_index] = means[local_index];
  }

  __syncthreads();

  // Assignment step.

  // Load once here.
  const float4 value = data[global_index];

  float best_distance = FLT_MAX;
  int best_cluster = -1;
  for (int cluster = 0; cluster < k; ++cluster) {
    const float distance = sqL2Dist_device_CL(value, shared_means[cluster]);
    if (distance < best_distance) {
      best_distance = distance;
      best_cluster = cluster;
    }
  }
  d_assign[global_index]=best_cluster;
  __syncthreads();

  // Reduction step.

    const int count = local_index;
    //const float4 zeroF4(0.f,0.f,0.f,0.f);

    for (int cluster = 0; cluster < k; ++cluster) {
      // Zeros if this point (thread) is not assigned to the cluster, else the
      // values of the point.
      shared_means[local_index] = (best_cluster == cluster) ? value : float4{0.f,0.f,0.f,0.f};
      shared_counts[count] = (best_cluster == cluster) ? 1 : 0;
      __syncthreads();

      // Tree-reduction for this cluster.
      for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (local_index < stride) {
          shared_means[local_index] += shared_means[local_index + stride];
          shared_counts[count] += shared_counts[count + stride];
        }
        __syncthreads();
      }

      // Now shared_data[0] holds the sum for x.

      if (local_index == 0) {
        const int cluster_index = blockIdx.x * k + cluster;
        new_sums[cluster_index] = shared_means[local_index];
        counts[cluster_index] = shared_counts[count];
      }
      __syncthreads();
    }
  }


__global__ void coarseReduce_parallel_CL(const thrust::device_ptr<float4> means,
                                         const thrust::device_ptr<float4> new_sums,
                                         const size_t k,
                                         const thrust::device_ptr<int> counts) {
    extern __shared__ float4 shared_means[];

  const int index = threadIdx.x;
  //const int y_offset = blockDim.x;
  if(index < k){
  // Load into shared memory for more efficient reduction.
  shared_means[index] = new_sums[index];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride >= k; stride /= 2) {
    if (index < stride) {
      shared_means[index] += shared_means[index + stride];
    }
    __syncthreads();
  }

  // The first k threads can recompute their clusters' means now.
  if (index < k) {
    const int count = max(1, counts[index]);
    means[index] = new_sums[index] / count;
    new_sums[index] = float4{0.f};
    counts[index] = 0;
  }
}
*/
//tried using shared memory for mean calculation and color assignment
//===================================================

__global__ void writeNewColors_parallel_CL(thrust::device_ptr<float4> means,
                                      size_t data_size,
                                      thrust::device_ptr<int> assignment,
                                      thrust::device_ptr<float4> newOut)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;
    newOut[index] = means[assignment[index]];
}

__global__ void writeNewColors_parallel_4F(thrust::device_ptr<float> meansR,
                                           thrust::device_ptr<float> meansG,
                                           thrust::device_ptr<float> meansB,
                                           thrust::device_ptr<float> meansA,
                                           size_t data_size,
                                           thrust::device_ptr<int> assignment,
                                           thrust::device_ptr<float> newOutR,
                                           thrust::device_ptr<float> newOutG,
                                           thrust::device_ptr<float> newOutB,
                                           thrust::device_ptr<float> newOutA)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;
    newOutR[index] = meansR[assignment[index]];
    newOutG[index] = meansG[assignment[index]];
    newOutB[index] = meansB[assignment[index]];
    newOutA[index] = meansA[assignment[index]];
}

__global__ void writeNewColors_parallel_LN(thrust::device_ptr<float> means,
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

__global__ void calculateDistancesToCentroids_4F(thrust::device_ptr<float4> d_source,
                                                 const size_t data_size,
                                                 thrust::device_ptr<float4> d_means,
                                                 const size_t current,
                                                 thrust::device_ptr<float> d_dist,
                                                 thrust::device_ptr<size_t> d_weights)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;
    for(auto c=0; c<current; ++c)
    {
        d_dist[index]=sqL2Dist_device_CL(d_source[index], d_means[c]);
        d_weights[index]=d_weights[index]+(size_t)(d_dist[index]*1000.f);
    }
}

__global__ void calculateDistancesToCentroids_LN(thrust::device_ptr<float> d_source,
                                                 const size_t data_size,
                                                 thrust::device_ptr<float> d_means,
                                                 const size_t current,
                                                 thrust::device_ptr<float> d_dist,
                                                 thrust::device_ptr<size_t> d_weights)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;
    for(auto c=0; c<current; ++c)
    {
        d_dist[index]=sqL2Dist_device_LN(d_source[index*4],d_source[index*4+1],
                                         d_source[index*4+2],d_source[index*4+3],
                                         d_means[c*4],d_means[c*4+1],
                                         d_means[c*4+2],d_means[c*4+3]);
        d_weights[index]=d_weights[index]+(size_t)(d_dist[index]*1000.f);
    }
}

__global__ void calculateDistancesToCentroids_4V(thrust::device_ptr<float> d_sourceR,
                                                 thrust::device_ptr<float> d_sourceG,
                                                 thrust::device_ptr<float> d_sourceB,
                                                 thrust::device_ptr<float> d_sourceA,
                                                 const size_t data_size,
                                                 thrust::device_ptr<float> d_meansR,
                                                 thrust::device_ptr<float> d_meansG,
                                                 thrust::device_ptr<float> d_meansB,
                                                 thrust::device_ptr<float> d_meansA,
                                                 const size_t current,
                                                 thrust::device_ptr<float> d_dist,
                                                 thrust::device_ptr<size_t> d_weights)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= data_size) return;
    for(auto c=0; c<current; ++c)
    {
        d_dist[index]=sqL2Dist_device_LN(d_sourceR[index],d_sourceG[index],
                                         d_sourceB[index],d_sourceA[index],
                                         d_meansR[c],d_meansG[c],
                                         d_meansB[c],d_meansA[c]);
        d_weights[index]=d_weights[index]+(size_t)(d_dist[index]*1000.f);
    }
}

///=================================================================================
///----------------------------|  END UTILITY  |------------------------------------
///=================================================================================

ColorVector gpuKmeans::kmeans_parallel_CV(const ColorVector &source,
                           size_t k,
                           size_t number_of_iterations,
                           const size_t numThreads)
{
    RandomFn<float> rfunc;
    const size_t number_of_elements = source.size();
    const int blocks = (number_of_elements + numThreads - 1) / numThreads;
    thrust::host_vector<float4> h_source(number_of_elements);
    for(auto i=0; i<number_of_elements; ++i)
    {
        h_source[i].x = source.at(i).m_r;
        h_source[i].y = source.at(i).m_g;
        h_source[i].z = source.at(i).m_b;
        h_source[i].w = source.at(i).m_a;
    }
    thrust::device_vector<float4> d_source(source.size());
    thrust::copy(h_source.begin(), h_source.end(), d_source.begin());
    thrust::device_vector<float4> d_means(k);
    rfunc.setNumericLimitsL(0, number_of_elements - 1);
    // Pick centroids as random points from the dataset.
    /*for(uint cluster=0; cluster<k; ++cluster)
    {
        float4 assignment;
        Color c = source[rfunc->MT19937RandL()];
        assignment.x = c.m_r;
        assignment.y = c.m_g;
        assignment.z = c.m_b;
        assignment.w = c.m_a;
        d_means[cluster] = assignment;
    }*/

    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    d_means[0] = d_source[number]; //first mean is random
    thrust::device_vector<float> d_distances(number_of_elements, 0.f);
    thrust::device_vector<size_t> d_weights(number_of_elements, 0);
    //thrust::device_vector<float> d_totalDistance(k, 0.f);
    std::vector<size_t> tWeights(number_of_elements);
    for(auto centroid=1; centroid<k; ++centroid)
    {
        calculateDistancesToCentroids_4F<<<blocks, numThreads>>>(d_source.data(),
                                                                 number_of_elements,
                                                                 d_means.data(),
                                                                 centroid,
                                                                 d_distances.data(),
                                                                 d_weights.data());
        cudaDeviceSynchronize();
        thrust::copy(d_weights.begin(),
                     d_weights.end(),
                     tWeights.begin());
        size_t randomIDx = rfunc.weightedRand(tWeights);
        d_means[centroid]=d_source[randomIDx];
        thrust::fill(d_distances.begin(), d_distances.end(), 0.f);
    }
    //end of centoid picking

    thrust::device_vector<int> d_assignments(source.size()); //for cluster assignments
    thrust::device_vector<float4> d_filtered(source.size()); //to copy back and return
    thrust::host_vector<float4> h_filtered(source.size());

    thrust::device_vector<float4> d_sums(k);
    thrust::device_vector<int> d_counts(k, 0);

    //const int shared_data1 = (sizeof(int)+sizeof(float4))*numThreads;
    //const int shared_data2 = sizeof(float4)*k*blocks;
    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sums.begin(), d_sums.end(), float4{0.0,0.0,0.0,0.0});
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assignClusters_parallel_CL<<<blocks, numThreads>>>(d_source.data(),
                                             number_of_elements,
                                             d_means.data(),
                                             d_sums.data(),
                                             k,
                                             d_counts.data(),
                                             d_assignments.data());
        /*fineReduce_parallel_CL<<<blocks, numThreads, shared_data1>>>(
                                                         d_source.data(),
                                                         number_of_elements,
                                                         d_means.data(),
                                                         d_assignments.data(),
                                                         d_sums.data(),
                                                         k,
                                                         d_counts.data());*/
        cudaDeviceSynchronize();

        computeNewMeans_parallel_CL<<<1, k>>>(d_means.data(),
                                    d_sums.data(),
                                    d_counts.data());
        /*const int num = k*blocks;
        coarseReduce_parallel_CL<<<1, num, shared_data2>>>(
                                                        d_means.data(),
                                                        d_sums.data(),
                                                        k,
                                                        d_counts.data());*/
        cudaDeviceSynchronize();
    }
    writeNewColors_parallel_CL<<<blocks, numThreads>>>(d_means.data(),
                                               number_of_elements,
                                               d_assignments.data(),
                                               d_filtered.data());
    cudaDeviceSynchronize();
    thrust::copy(d_filtered.begin(), d_filtered.end(), h_filtered.begin());

    ColorVector ret(source.size());
    for(uint i=0; i<source.size(); ++i)
    {
        ret.at(i).m_r = h_filtered[i].x;
        ret.at(i).m_g = h_filtered[i].y;
        ret.at(i).m_b = h_filtered[i].z;
        ret.at(i).m_a = h_filtered[i].w;
    }
    return ret;
}

ImageColors gpuKmeans::kmeans_parallel_IC(const ImageColors &source,
                           size_t k,
                           size_t number_of_iterations,
                           const size_t numThreads)
{
    RandomFn<float> rfunc;
    const size_t number_of_elements = source.m_r.size();
    const int blocks = (number_of_elements + numThreads - 1) / numThreads;
    thrust::host_vector<float4> h_source(number_of_elements);
    for(uint x=0; x<number_of_elements; ++x)
    {
        h_source[x].x=source.m_r.at(x);
        h_source[x].y=source.m_g.at(x);
        h_source[x].z=source.m_b.at(x);
        h_source[x].w=source.m_a.at(x);
    }
    thrust::device_vector<float4> d_source(number_of_elements);
    thrust::copy(h_source.begin(), h_source.end(), d_source.begin());
    thrust::device_vector<float4> d_means(k);
    rfunc.setNumericLimitsL(0, number_of_elements - 1);
    /*for(uint cluster=0; cluster<k; ++cluster)
    {
        size_t num = rfunc.MT19937RandL();
        float4 assignment;
        assignment.x = source.m_r.at(num);
        assignment.y = source.m_g.at(num);
        assignment.z = source.m_b.at(num);
        assignment.w = source.m_a.at(num);
        d_means[cluster] = assignment;
    }*/


    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    d_means[0] = d_source[number]; //first mean is random
    thrust::device_vector<float> d_distances(number_of_elements, 0.f);
    thrust::device_vector<size_t> d_weights(number_of_elements, 0);
    std::vector<size_t> tWeights(number_of_elements);
    for(auto centroid=1; centroid<k; ++centroid)
    {
        calculateDistancesToCentroids_4F<<<blocks, numThreads>>>(d_source.data(),
                                                                 number_of_elements,
                                                                 d_means.data(),
                                                                 centroid,
                                                                 d_distances.data(),
                                                                 d_weights.data());
        cudaDeviceSynchronize();
        thrust::copy(d_weights.begin(),
                     d_weights.end(),
                     tWeights.begin());
        size_t randomIDx = rfunc.weightedRand(tWeights);
        d_means[centroid]=d_source[randomIDx];
        thrust::fill(d_distances.begin(), d_distances.end(), 0.f);
    }
    //end of centoid picking

    thrust::device_vector<int> d_assignments(number_of_elements); //for cluster assignments
    thrust::device_vector<float4> d_filtered(number_of_elements); //to copy back and return
    thrust::host_vector<float4> h_filtered(number_of_elements);

    thrust::device_vector<float4> d_sums(k);
    thrust::device_vector<int> d_counts(k, 0);

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sums.begin(), d_sums.end(), float4{0.0,0.0,0.0,0.0});
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assignClusters_parallel_CL<<<blocks, numThreads>>>(d_source.data(),
                                             number_of_elements,
                                             d_means.data(),
                                             d_sums.data(),
                                             k,
                                             d_counts.data(),
                                             d_assignments.data());
        cudaDeviceSynchronize();

        computeNewMeans_parallel_CL<<<1, k>>>(d_means.data(),
                                    d_sums.data(),
                                    d_counts.data());
        cudaDeviceSynchronize();
    }
    writeNewColors_parallel_CL<<<blocks, numThreads>>>(d_means.data(),
                                               number_of_elements,
                                               d_assignments.data(),
                                               d_filtered.data());
    cudaDeviceSynchronize();
    thrust::copy(d_filtered.begin(), d_filtered.end(), h_filtered.begin());

    //h_source = d_source;
    ImageColors ret;
    ret.resize(number_of_elements);
    for(uint i=0; i<number_of_elements; ++i)
    {
        ret.m_r.at(i) = h_filtered[i].x;
        ret.m_g.at(i) = h_filtered[i].y;
        ret.m_b.at(i) = h_filtered[i].z;
        ret.m_a.at(i) = h_filtered[i].w;
    }
    return ret;
}

std::vector<float> gpuKmeans::kmeans_parallel_LN(const std::vector<float> &source,
                                  size_t k,
                                  size_t number_of_iterations,
                                  const size_t numThreads)
{
    RandomFn<float> rfunc;
    const size_t number_of_elements = source.size()/4;
    const int blocks = (number_of_elements + numThreads - 1) / numThreads;
    thrust::device_vector<float> d_source(source.size());
    thrust::copy(source.begin(), source.end(), d_source.begin());
    thrust::device_vector<float> d_means(k*4);
    rfunc.setNumericLimitsL(0, number_of_elements - 1);
    /*for(uint cluster=0; cluster<k; ++cluster)
    {
        size_t cID = rfunc.MT19937RandL();
        d_means[cluster*4]   = source[cID*4];
        d_means[cluster*4+1] = source[cID*4+1];
        d_means[cluster*4+2] = source[cID*4+2];
        d_means[cluster*4+3] = source[cID*4+3];
    }*/


    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    d_means[0] = d_source[number*4]; //first mean is random
    d_means[1] = d_source[number*4+1];
    d_means[2] = d_source[number*4+2];
    d_means[3] = d_source[number*4+3];
    thrust::device_vector<float> d_distances(number_of_elements, 0.f);
    thrust::device_vector<size_t> d_weights(number_of_elements, 0);
    std::vector<size_t> tWeights(number_of_elements);
    for(auto centroid=1; centroid<k; ++centroid)
    {
        calculateDistancesToCentroids_LN<<<blocks, numThreads>>>(d_source.data(),
                                                                 number_of_elements,
                                                                 d_means.data(),
                                                                 centroid,
                                                                 d_distances.data(),
                                                                 d_weights.data());
        cudaDeviceSynchronize();
        thrust::copy(d_weights.begin(),
                     d_weights.end(),
                     tWeights.begin());
        size_t randomIDx = rfunc.weightedRand(tWeights);
        d_means[centroid*4]=d_source[randomIDx*4];
        d_means[centroid*4+1]=d_source[randomIDx*4+1];
        d_means[centroid*4+2]=d_source[randomIDx*4+2];
        d_means[centroid*4+3]=d_source[randomIDx*4+3];
        thrust::fill(d_distances.begin(), d_distances.end(), 0.f);
    }

    //end of centoid picking

    thrust::device_vector<int> d_assignments(source.size()/4); //for cluster assignments
    thrust::device_vector<float> d_filtered(source.size()); //to copy back and return
    thrust::host_vector<float> h_filtered(source.size());

    thrust::device_vector<float> d_sums(k*4);
    thrust::device_vector<int> d_counts(k, 0);

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sums.begin(), d_sums.end(), 0.f);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assignClusters_parallel_LN<<<blocks, numThreads>>>(d_source.data(),
                                             number_of_elements,
                                             d_means.data(),
                                             d_sums.data(),
                                             k,
                                             d_counts.data(),
                                             d_assignments.data());
        cudaDeviceSynchronize();

        computeNewMeans_parallel_LN<<<1, k>>>(d_means.data(),
                                    d_sums.data(),
                                    d_counts.data());
        cudaDeviceSynchronize();
    }
    writeNewColors_parallel_LN<<<blocks, numThreads>>>(d_means.data(),
                                               number_of_elements,
                                               d_assignments.data(),
                                               d_filtered.data());
    cudaDeviceSynchronize();
    thrust::copy(d_filtered.begin(), d_filtered.end(), h_filtered.begin());

    std::vector<float> ret(source.size());
    for(uint i=0; i<source.size(); ++i)
    {
        ret.at(i)   = h_filtered[i];
    }
    return ret;
}

void gpuKmeans::kmeans_parallel_4SV(const std::vector<float>* _inreds,
                                    const std::vector<float>* _ingrns,
                                    const std::vector<float>* _inblus,
                                    const std::vector<float>* _inalps,
                                    std::vector<float>* _outreds,
                                    std::vector<float>* _outgrns,
                                    std::vector<float>* _outblus,
                                    std::vector<float>* _outalps,
                                    const size_t number_of_elements,
                                    size_t k,
                                    size_t number_of_iterations,
                                    const size_t numThreads)
{
    RandomFn<float> rfunc;
    const int blocks = (number_of_elements + numThreads - 1) / numThreads;
    thrust::device_vector<float> d_sourceR(number_of_elements);
    thrust::device_vector<float> d_sourceG(number_of_elements);
    thrust::device_vector<float> d_sourceB(number_of_elements);
    thrust::device_vector<float> d_sourceA(number_of_elements);
    thrust::copy(_inreds->begin(), _inreds->end(), d_sourceR.begin());
    thrust::copy(_ingrns->begin(), _ingrns->end(), d_sourceG.begin());
    thrust::copy(_inblus->begin(), _inblus->end(), d_sourceB.begin());
    thrust::copy(_inalps->begin(), _inalps->end(), d_sourceA.begin());
    thrust::device_vector<float> d_meansR(k);
    thrust::device_vector<float> d_meansG(k);
    thrust::device_vector<float> d_meansB(k);
    thrust::device_vector<float> d_meansA(k);
    rfunc.setNumericLimitsL(0, number_of_elements - 1);
    /*for(auto cluster=0; cluster<k; ++cluster)
    {
        size_t num = rfunc.MT19937RandL();
        d_meansR[cluster] = _inreds->at(num);
        d_meansG[cluster] = _ingrns->at(num);
        d_meansB[cluster] = _inblus->at(num);
        d_meansA[cluster] = _inalps->at(num);
    }*/


    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    d_meansR[0] = d_sourceR[number]; //first mean is random
    d_meansG[0] = d_sourceG[number];
    d_meansB[0] = d_sourceB[number];
    d_meansA[0] = d_sourceA[number];
    thrust::device_vector<float> d_distances(number_of_elements, 0.f);
    thrust::device_vector<size_t> d_weights(number_of_elements, 0);
    //thrust::device_vector<float> d_totalDistance(k, 0.f);
    std::vector<size_t> tWeights(number_of_elements);
    for(auto centroid=1; centroid<k; ++centroid)
    {
        calculateDistancesToCentroids_4V<<<blocks, numThreads>>>(d_sourceR.data(),
                                                                 d_sourceG.data(),
                                                                 d_sourceB.data(),
                                                                 d_sourceA.data(),
                                                                 number_of_elements,
                                                                 d_meansR.data(),
                                                                 d_meansG.data(),
                                                                 d_meansB.data(),
                                                                 d_meansA.data(),
                                                                 centroid,
                                                                 d_distances.data(),
                                                                 d_weights.data());
        cudaDeviceSynchronize();
        thrust::copy(d_weights.begin(),
                     d_weights.end(),
                     tWeights.begin());
        size_t randomIDx = rfunc.weightedRand(tWeights);
        d_meansR[centroid]=d_sourceR[randomIDx];
        d_meansG[centroid]=d_sourceG[randomIDx];
        d_meansB[centroid]=d_sourceB[randomIDx];
        d_meansA[centroid]=d_sourceA[randomIDx];
        thrust::fill(d_distances.begin(), d_distances.end(), 0.f);
    }
    //end of centoid picking


    thrust::device_vector<int> d_assignments(number_of_elements);
    thrust::device_vector<float> d_filteredR(number_of_elements);
    thrust::device_vector<float> d_filteredG(number_of_elements);
    thrust::device_vector<float> d_filteredB(number_of_elements);
    thrust::device_vector<float> d_filteredA(number_of_elements);

    thrust::device_vector<float> d_sumsR(k);
    thrust::device_vector<float> d_sumsG(k);
    thrust::device_vector<float> d_sumsB(k);
    thrust::device_vector<float> d_sumsA(k);
    thrust::device_vector<int> d_counts(k, 0);

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sumsR.begin(), d_sumsR.end(), float{0.0f});
        thrust::fill(d_sumsG.begin(), d_sumsG.end(), float{0.0f});
        thrust::fill(d_sumsB.begin(), d_sumsB.end(), float{0.0f});
        thrust::fill(d_sumsA.begin(), d_sumsA.end(), float{0.0f});

        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assignClusters_parallel_4F<<<blocks, numThreads>>>(d_sourceR.data(),
                                                           d_sourceG.data(),
                                                           d_sourceB.data(),
                                                           d_sourceA.data(),
                                                           number_of_elements,
                                                           d_meansR.data(),
                                                           d_meansG.data(),
                                                           d_meansB.data(),
                                                           d_meansA.data(),
                                                           d_sumsR.data(),
                                                           d_sumsG.data(),
                                                           d_sumsB.data(),
                                                           d_sumsA.data(),
                                                           k,
                                                           d_counts.data(),
                                                           d_assignments.data());
        cudaDeviceSynchronize();

        computeNewMeans_parallel_4F<<<1, k>>>(d_meansR.data(),
                                              d_meansG.data(),
                                              d_meansB.data(),
                                              d_meansA.data(),
                                              d_sumsR.data(),
                                              d_sumsG.data(),
                                              d_sumsB.data(),
                                              d_sumsA.data(),
                                              d_counts.data());
        cudaDeviceSynchronize();
    }
    writeNewColors_parallel_4F<<<blocks, numThreads>>>(d_meansR.data(),
                                                       d_meansG.data(),
                                                       d_meansB.data(),
                                                       d_meansA.data(),
                                                       number_of_elements,
                                                       d_assignments.data(),
                                                       d_filteredR.data(),
                                                       d_filteredG.data(),
                                                       d_filteredB.data(),
                                                       d_filteredA.data());
    cudaDeviceSynchronize();
    thrust::copy(d_filteredR.begin(), d_filteredR.end(), _outreds->begin());
    thrust::copy(d_filteredG.begin(), d_filteredG.end(), _outgrns->begin());
    thrust::copy(d_filteredB.begin(), d_filteredB.end(), _outblus->begin());
    thrust::copy(d_filteredA.begin(), d_filteredA.end(), _outalps->begin());

    return;
}

void gpuKmeans::kmeans_parallel_4LV(const float* _inreds,
                                    const float* _ingrns,
                                    const float* _inblus,
                                    const float* _inalps,
                                    float* _outreds,
                                    float* _outgrns,
                                    float* _outblus,
                                    float* _outalps,
                                    const size_t number_of_elements,
                                    size_t k,
                                    size_t number_of_iterations,
                                    const size_t numThreads)
{
    RandomFn<float> rfunc;
    const int blocks = (number_of_elements + numThreads - 1) / numThreads;
    thrust::device_vector<float> d_sourceR(number_of_elements);
    thrust::device_vector<float> d_sourceG(number_of_elements);
    thrust::device_vector<float> d_sourceB(number_of_elements);
    thrust::device_vector<float> d_sourceA(number_of_elements);
    thrust::copy(_inreds, _inreds+number_of_elements, d_sourceR.begin());
    thrust::copy(_ingrns, _ingrns+number_of_elements, d_sourceG.begin());
    thrust::copy(_inblus, _inblus+number_of_elements, d_sourceB.begin());
    thrust::copy(_inalps, _inalps+number_of_elements, d_sourceA.begin());
    thrust::device_vector<float> d_meansR(k);
    thrust::device_vector<float> d_meansG(k);
    thrust::device_vector<float> d_meansB(k);
    thrust::device_vector<float> d_meansA(k);
    rfunc.setNumericLimitsL(0, number_of_elements - 1);
    /*for(auto cluster=0; cluster<k; ++cluster)
    {
        size_t num = rfunc.MT19937RandL();
        d_meansR[cluster] = _inreds[num];
        d_meansG[cluster] = _ingrns[num];
        d_meansB[cluster] = _inblus[num];
        d_meansA[cluster] = _inalps[num];
    }*/

    //Pick Centroids according to kmeans++ method by getting distances to all points
    size_t number = rfunc.MT19937RandL();
    d_meansR[0] = d_sourceR[number]; //first mean is random
    d_meansG[0] = d_sourceG[number];
    d_meansB[0] = d_sourceB[number];
    d_meansA[0] = d_sourceA[number];
    thrust::device_vector<float> d_distances(number_of_elements, 0.f);
    thrust::device_vector<size_t> d_weights(number_of_elements, 0);
    //thrust::device_vector<float> d_totalDistance(k, 0.f);
    std::vector<size_t> tWeights(number_of_elements);
    for(auto centroid=1; centroid<k; ++centroid)
    {
        calculateDistancesToCentroids_4V<<<blocks, numThreads>>>(d_sourceR.data(),
                                                                 d_sourceG.data(),
                                                                 d_sourceB.data(),
                                                                 d_sourceA.data(),
                                                                 number_of_elements,
                                                                 d_meansR.data(),
                                                                 d_meansG.data(),
                                                                 d_meansB.data(),
                                                                 d_meansA.data(),
                                                                 centroid,
                                                                 d_distances.data(),
                                                                 d_weights.data());
        cudaDeviceSynchronize();
        thrust::copy(d_weights.begin(),
                     d_weights.end(),
                     tWeights.begin());
        size_t randomIDx = rfunc.weightedRand(tWeights);
        d_meansR[centroid]=d_sourceR[randomIDx];
        d_meansG[centroid]=d_sourceG[randomIDx];
        d_meansB[centroid]=d_sourceB[randomIDx];
        d_meansA[centroid]=d_sourceA[randomIDx];
        thrust::fill(d_distances.begin(), d_distances.end(), 0.f);
    }
    //end of centoid picking


    thrust::device_vector<int> d_assignments(number_of_elements);
    thrust::device_vector<float> d_filteredR(number_of_elements);
    thrust::device_vector<float> d_filteredG(number_of_elements);
    thrust::device_vector<float> d_filteredB(number_of_elements);
    thrust::device_vector<float> d_filteredA(number_of_elements);

    thrust::device_vector<float> d_sumsR(k);
    thrust::device_vector<float> d_sumsG(k);
    thrust::device_vector<float> d_sumsB(k);
    thrust::device_vector<float> d_sumsA(k);
    thrust::device_vector<int> d_counts(k, 0);

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sumsR.begin(), d_sumsR.end(), float{0.0f});
        thrust::fill(d_sumsG.begin(), d_sumsG.end(), float{0.0f});
        thrust::fill(d_sumsB.begin(), d_sumsB.end(), float{0.0f});
        thrust::fill(d_sumsA.begin(), d_sumsA.end(), float{0.0f});

        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assignClusters_parallel_4F<<<blocks, numThreads>>>(d_sourceR.data(),
                                                           d_sourceG.data(),
                                                           d_sourceB.data(),
                                                           d_sourceA.data(),
                                                           number_of_elements,
                                                           d_meansR.data(),
                                                           d_meansG.data(),
                                                           d_meansB.data(),
                                                           d_meansA.data(),
                                                           d_sumsR.data(),
                                                           d_sumsG.data(),
                                                           d_sumsB.data(),
                                                           d_sumsA.data(),
                                                           k,
                                                           d_counts.data(),
                                                           d_assignments.data());
        cudaDeviceSynchronize();

        computeNewMeans_parallel_4F<<<1, k>>>(d_meansR.data(),
                                              d_meansG.data(),
                                              d_meansB.data(),
                                              d_meansA.data(),
                                              d_sumsR.data(),
                                              d_sumsG.data(),
                                              d_sumsB.data(),
                                              d_sumsA.data(),
                                              d_counts.data());
        cudaDeviceSynchronize();
    }
    writeNewColors_parallel_4F<<<blocks, numThreads>>>(d_meansR.data(),
                                                       d_meansG.data(),
                                                       d_meansB.data(),
                                                       d_meansA.data(),
                                                       number_of_elements,
                                                       d_assignments.data(),
                                                       d_filteredR.data(),
                                                       d_filteredG.data(),
                                                       d_filteredB.data(),
                                                       d_filteredA.data());
    cudaDeviceSynchronize();
    thrust::copy(d_filteredR.begin(), d_filteredR.end(), _outreds);
    thrust::copy(d_filteredG.begin(), d_filteredG.end(), _outgrns);
    thrust::copy(d_filteredB.begin(), d_filteredB.end(), _outblus);
    thrust::copy(d_filteredA.begin(), d_filteredA.end(), _outalps);

    return;
}
