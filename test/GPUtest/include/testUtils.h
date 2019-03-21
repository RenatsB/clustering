#ifndef _CLIB_TESTUTILS_H
#define _CLIB_TESTUTILS_H

#include <thrust/device_vector.h>
#include <vector>

template <typename T>
std::vector<T> device_vector_to_host(const thrust::device_vector<T>& di_vec)
{
    std::vector<T> h_vec(di_vec.size());
    thrust::copy(di_vec.begin(), di_vec.end(), h_vec.begin());
    return h_vec;
}

#endif //_CLIB_TESTUTILS_H
