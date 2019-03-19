#ifndef _GPURANDF_H
#define _GPURANDF_H

#include <thrust/device_vector.h>

namespace GPU_RandF
{

/// Fill up a vector on the device with n floats. Memory is assumed already preallocated.
int randFloatsInternal(float *&/*devData*/, const size_t /*n*/);
//int randFloatsInternal(int *&/*devData*/, const size_t /*n*/);

}
#endif //_GPURANDF_H
