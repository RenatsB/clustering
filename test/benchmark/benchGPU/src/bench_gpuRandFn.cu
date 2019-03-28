#include <benchmark/benchmark.h>
#include "benchParams.h"
#include "gpuRandF.h"
#include <thrust/device_vector.h>

  static void Bench_Ran_Device(benchmark::State& state)
  {
    thrust::device_vector<float> dv(CLIB_BENCH_RANDNUM);
    float *ptr = thrust::raw_pointer_cast(dv.data());
    for (auto _ : state)
    {
      benchmark::DoNotOptimize(
        gpuRandFn::randFloatsInternal(ptr, CLIB_BENCH_RANDNUM, CLIB_BENCH_RANDNUM));
    }
  }
  BENCHMARK(Bench_Ran_Device);
