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
#include <eigen-nvcc/Dense>

// Note that the resolution has to be at least 4 because we are writing the values into shared
// memory in parallel: if the RES is less than 4 not all the values will be written to the shared
// matrix structure.
// NOTE: My system fails to handle a RES higher than 30 - it _should_ theoretically handle 32. This
//       could be a hardware fault. The result of RES being too high is that your output will just 
//       be full of zeros.
#define RES 30
#define XRESOLUTION 512
#define YRESOLUTION 512

__device__ double sq_Parallel(double ref)
{
	return ref*ref;
}

__device__ double sq_Col_l2_Dist_Parallel(Color first, Color second)
{
    return sq_Parallel(first.m_r*first.m_a - second.m_r*second.m_a)
         + sq_Parallel(first.m_g*first.m_a - second.m_g*second.m_a)
         + sq_Parallel(first.m_b*first.m_a - second.m_b*second.m_a);
}

// In the assignment step, each point (thread) computes its distance to each
// cluster centroid and adds its x and y values to the sum of its closest
// centroid, as well as incrementing that centroid's count of assigned points.
__global__ void assign_clusters(const thrust::device_ptr<Color> data,
								size_t data_size,
								const thrust::device_ptr<Color> means,
								thrust::device_ptr<Color> new_sums,
								size_t k,
								thrust::device_ptr<int> counts) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= data_size) return;

	// Make global loads once.
	const Color current = data[index];

	double best_distance = DBL_MAX;
	int best_cluster = 0;
	for (int cluster = 0; cluster < k; ++cluster) {
		const double distance = sq_Col_l2_Dist_Parallel(current, means[cluster]);
		if (distance < best_distance) {
			best_distance = distance;
			best_cluster = cluster;
		}
	}

	atomicAdd(thrust::raw_pointer_cast(new_sums_x + best_cluster), x);
	atomicAdd(thrust::raw_pointer_cast(new_sums_y + best_cluster), y);
	atomicAdd(thrust::raw_pointer_cast(counts + best_cluster), 1);
}

// Each thread is one cluster, which just recomputes its coordinates as the mean
// of all points assigned to it.
__global__ void compute_new_means(thrust::device_ptr<Color> means,
  								const thrust::device_ptr<Color> new_sum,
  								const thrust::device_ptr<int> counts) {
	const int cluster = threadIdx.x;
	const int count = max(1, counts[cluster]);
	means[cluster] = new_sum[cluster] / count;
}

int main(int argc, const char* argv[]) {

	Ran<size_t> rfunc;
	Gen picGen;
	size_t k = 4;
    thrust::host_vector<float> h_x;
    thrust::host_vector<float> h_y;
    DataFrame source = picGen.generate(XRESOLUTION,YRESOLUTION,128);

    thrust::device_vector<double> means(k);
	randF.setNumericLimitsL(0, data.size() - 1);

	thrust::device_vector<Color> means(k);
	for(auto &cluster : means)
	{
		cluster = data[randF.MT19937RandL()];
	}

    // Load x and y into host vectors ... (omitted)

    const size_t number_of_elements = h_x.size();

    thrust::device_vector<float> d_x = h_x;
    thrust::device_vector<float> d_y = h_y;

    std::mt19937 rng(std::random_device{}());
    std::shuffle(h_x.begin(), h_x.end(), rng);
    std::shuffle(h_y.begin(), h_y.end(), rng);
    thrust::device_vector<float> d_mean_x(h_x.begin(), h_x.begin() + k);
    thrust::device_vector<float> d_mean_y(h_y.begin(), h_y.begin() + k);

    thrust::device_vector<float> d_sums_x(k);
    thrust::device_vector<float> d_sums_y(k);
    thrust::device_vector<int> d_counts(k, 0);

    const int threads = 1024;
    const int blocks = (number_of_elements + threads - 1) / threads;

    for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
        thrust::fill(d_sums_x.begin(), d_sums_x.end(), 0);
        thrust::fill(d_sums_y.begin(), d_sums_y.end(), 0);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        assign_clusters<<<blocks, threads>>>(d_x.data(),
                     d_y.data(),
                     number_of_elements,
                     d_mean_x.data(),
                     d_mean_y.data(),
                     d_sums_x.data(),
                     d_sums_y.data(),
                     k,
                     d_counts.data());
        //cudaDeviceSynchronize();
        __syncthreads();

        compute_new_means<<<1, k>>>(d_mean_x.data(),
            d_mean_y.data(),
            d_sums_x.data(),
            d_sums_y.data(),
            d_counts.data());
        //cudaDeviceSynchronize();
        __syncthreads();
    }
}
