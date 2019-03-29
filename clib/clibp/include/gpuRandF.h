#ifndef _GPURANDF_H
#define _GPURANDF_H

namespace gpuRandFn
{

/// Fill up a vector on the device with n floats. Memory is assumed already preallocated.
//int randFloatsInternal(float *&/*devData*/, const size_t /*n*/);
int randFloatsInternal(float *&devData,
                       const size_t n,
                       const size_t numThreads);
//int randFloatsInternal(int *&/*devData*/, const size_t /*n*/);

}
#endif //_GPURANDF_H
