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
#include <../../serial/src/gen.cpp> //this includes generator, random and utilTypes
#include <../../serial/include/img.hpp>

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

double squared_Colour_l2_Distance(Color first, Color second)
{
    return square(first.m_r*first.m_a - second.m_r*second.m_a)
       + square(first.m_g*first.m_a - second.m_g*second.m_a)
       + square(first.m_b*first.m_a - second.m_b*second.m_a);
}

__global__ DataFrame kmeansParallel(DataFrame &data,
                                    thrust::device_vector<float> *means,
                                    Ran *rfun,
                                    size_t k,
                                    size_t numIterations) {
    // Here is where YOU make it happen
    rfunc.setNumericLimitsL(0, data.size() - 1);
    DataFrame correctedImage(data.size());

	Eigen::Map<Eigen::Matrix<float,4,4,Eigen::RowMajor> > M(Mf)

	Eigen::Matrix4f Px, Py, Pz;
	__syncthreads();

	if((ThreadIdx.x < 4) && (threadIdx.y < 4))
	{
		uint idx = patches[blockIdx.x * 16 + threadIdx.x*4 + threadIdx.y];
		Px(threadIdx.x,threadIdx.y) = vertices[idx * 3 + 0];
		Py(threadIdx.x,threadIdx.y) = vertices[idx * 3 + 1];
		Pz(threadIdx.x,threadIdx.y) = vertices[idx * 3 + 2];
	}
	__syncthreads();

	float posU = (float)(threadIdx.x)/(float)(blockDim.x-1);
	float posV = (float)(threadIdx.y)/(float)(blockDim.y-1);
	Eigen::Vector4f U(1.f, posU, posU*posU, posU*posU*posU);
	Eigen::Vector4f V(1.f, posV, posV*posV, posV*posV*posV);
	
	uint output_idx = blockIdx.x * blockDim.x * blockDim.y +
	threadIdx.x * blockDim.y + threadIdx.y;
	output_vertices[output_idx * 3] = U.dot(M*Px*M.transpose()*V);
	output_vertices[output_idx * 3+1] = U.dot(M*Py*M.transpose()*V);
	output_vertices[output_idx * 3+2] = U.dot(M*Pz*M.transpose()*V);
}

/**
 * Host main routine. In this function we'll basically stash the teapot data onto the device using
 * thrust iterators, call the kernel, and then output the result to the screen, formatted for an
 * obj file.
 */
int main(void) {
	
    Ran<double> rfunc;
	// Create our vectors for the device to hold our point data by using
	// array iterators (like STL)
	thrust::device_vector<float> d_vertices(teapot::vertices, teapot::vertices + 3*teapot::num_vertices);
	thrust::copy(d_vertices.begin(), d_vertices.end(), std::ostream_iterator<float>(std::cout, " "));
	// Create an array for our patch indices
	thrust::device_vector<unsigned int> d_patches(teapot::patches, teapot::patches + 16*teapot::num_patches);

	// Create output data structure (there will be (x,y,z) values in both the u and v directions along the patch)
	thrust::device_vector<float> d_output_vertices(RES*RES*3*teapot::num_patches);

	float M[] = 1.f, 0.f, 0.f, 0.f, -3.f, 3.f, 0.f, 0.f, 3.f, -6.f, 3.f, 0.f, -1.f, 3.f, -3.f, 1.f;
	//std::copy(M, M+16, std::ostream_iterator<float>)
	if(cudaMemCpyToSymbol(Mf, M, sizof(M)) != cudaSuccess)
	{
		return EXIT_FAILURE;
	}

	// Define the dimension of the block - the u and v coordinates of the sampled point are given
	// by the threadIdx.x/y.
	dim3 blockDim(RES, RES, 1);

	// Call our kernel function to sample points on the bezier patch
	tesselate<<<teapot::num_patches, blockDim>>>(thrust::raw_pointer_cast(&d_output_vertices[0]),
	  					thrust::raw_pointer_cast(&d_patches[0]),
						thrust::raw_pointer_cast(&d_vertices[0]),
						teapot::num_patches);

	// Dump the data in obj format to cout (use the pipe ">" command to make an obj file)
	thrust::device_vector<float>::iterator dit = d_output_vertices.begin();
	unsigned int i;
	for (i=0; i<RES*RES*teapot::num_patches; ++i) {
		std::cout << "v ";
		thrust::copy(dit, dit+3, std::ostream_iterator<float>(std::cout, " "));
		std::cout << "\n";
		dit += 3;
	}

	// Exit politely
	return 0;
}

