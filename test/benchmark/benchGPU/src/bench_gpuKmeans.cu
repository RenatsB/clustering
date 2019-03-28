#include <benchmark/benchmark.h>
#include "gpuKmeans.h"

#define DEVICE_BM_KM_CV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                   \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      benchmark::DoNotOptimize(gpuImageGen::generate_parallel_CV(HORIZ_DIM,VERT_DIM,TURB, \
                                                                 NUMTHREADS));            \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_IG(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                   \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      benchmark::DoNotOptimize(gpuImageGen::generate_parallel_IC(HORIZ_DIM,VERT_DIM,TURB, \
                                                                 NUMTHREADS));            \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_LN(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                   \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      benchmark::DoNotOptimize(gpuImageGen::generate_parallel_LN(HORIZ_DIM,VERT_DIM,TURB, \
                                                                 NUMTHREADS));            \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_4SV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                  \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    std::vector<float> r(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> g(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> b(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> a(HORIZ_DIM*VERT_DIM);                                             \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      gpuImageGen::generate_parallel_4SV(&r,&g,&b,&a,HORIZ_DIM,VERT_DIM,TURB,NUMTHREADS); \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_4LV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                  \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    std::vector<float> r(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> g(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> b(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> a(HORIZ_DIM*VERT_DIM);                                             \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      gpuImageGen::generate_parallel_4LV(r.data(), g.data(), b.data(), a.data(),          \
                                         HORIZ_DIM,VERT_DIM,TURB, NUMTHREADS);            \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_4LL(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                  \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    std::vector<float> r(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> g(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> b(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> a(HORIZ_DIM*VERT_DIM);                                             \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      gpuImageGen::generate_parallel_4LV(r.data(), g.data(), b.data(), a.data(),          \
                                         HORIZ_DIM,VERT_DIM,TURB, NUMTHREADS);            \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

DEVICE_BM_KM_CV(Bench_GenGPU_ColorVectorS, 1024, 1024, 128, 128);
DEVICE_BM_KM_CV(Bench_GenGPU_ColorVectorM, 1024, 1024, 128, 512);
DEVICE_BM_KM_CV(Bench_GenGPU_ColorVectorL, 1024, 1024, 128, 1024);

DEVICE_BM_KM_IG(Bench_GenGPU_ImageColorsS, 1024, 1024, 128, 128);
DEVICE_BM_KM_IG(Bench_GenGPU_ImageColorsM, 1024, 1024, 128, 512);
DEVICE_BM_KM_IG(Bench_GenGPU_ImageColorsL, 1024, 1024, 128, 1024);

DEVICE_BM_KM_LN(Bench_GenGPU_LinearS, 1024, 1024, 128, 128);
DEVICE_BM_KM_LN(Bench_GenGPU_LinearM, 1024, 1024, 128, 512);
DEVICE_BM_KM_LN(Bench_GenGPU_LinearL, 1024, 1024, 128, 1024);

DEVICE_BM_KM_4SV(Bench_GenGPU_StdVecS, 1024, 1024, 128, 128);
DEVICE_BM_KM_4SV(Bench_GenGPU_StdVecM, 1024, 1024, 128, 512);
DEVICE_BM_KM_4SV(Bench_GenGPU_StdVecL, 1024, 1024, 128, 1024);

DEVICE_BM_KM_4LV(Bench_GenGPU_VecS, 1024, 1024, 128, 128);
DEVICE_BM_KM_4LV(Bench_GenGPU_VecM, 1024, 1024, 128, 512);
DEVICE_BM_KM_4LV(Bench_GenGPU_VecL, 1024, 1024, 128, 1024);

DEVICE_BM_KM_4LL(Bench_GenGPU_VecDirS, 1024, 1024, 128, 128);
DEVICE_BM_KM_4LL(Bench_GenGPU_VecDirM, 1024, 1024, 128, 512);
DEVICE_BM_KM_4LL(Bench_GenGPU_VecDirL, 1024, 1024, 128, 1024);
