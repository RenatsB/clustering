#include <benchmark/benchmark.h>
#include "benchParams.h"
#include "gpuImageGen.h"

#define DEVICE_BM_IG_CV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                   \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      benchmark::DoNotOptimize(gpuImageGen::generate_parallel_CV(HORIZ_DIM,VERT_DIM,TURB, \
                                                                 NUMTHREADS));            \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_IG_IG(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                   \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      benchmark::DoNotOptimize(gpuImageGen::generate_parallel_IC(HORIZ_DIM,VERT_DIM,TURB, \
                                                                 NUMTHREADS));            \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_IG_LN(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                   \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    for (auto _ : state)                                                                  \
    {                                                                                     \
      benchmark::DoNotOptimize(gpuImageGen::generate_parallel_LN(HORIZ_DIM,VERT_DIM,TURB, \
                                                                 NUMTHREADS));            \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_IG_4SV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                  \
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

#define DEVICE_BM_IG_4LV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                  \
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

#define DEVICE_BM_IG_4LL(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, NUMTHREADS)                  \
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

DEVICE_BM_IG_CV(Bench_GenGPU_ColorVectorS, CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S);
DEVICE_BM_IG_CV(Bench_GenGPU_ColorVectorM, CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M);
DEVICE_BM_IG_CV(Bench_GenGPU_ColorVectorL, CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L);

DEVICE_BM_IG_IG(Bench_GenGPU_ImageColorsS, CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S);
DEVICE_BM_IG_IG(Bench_GenGPU_ImageColorsM, CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M);
DEVICE_BM_IG_IG(Bench_GenGPU_ImageColorsL, CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L);

DEVICE_BM_IG_LN(Bench_GenGPU_LinearS, CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S);
DEVICE_BM_IG_LN(Bench_GenGPU_LinearM, CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M);
DEVICE_BM_IG_LN(Bench_GenGPU_LinearL, CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L);

DEVICE_BM_IG_4SV(Bench_GenGPU_StdVecS, CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S);
DEVICE_BM_IG_4SV(Bench_GenGPU_StdVecM, CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M);
DEVICE_BM_IG_4SV(Bench_GenGPU_StdVecL, CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L);

DEVICE_BM_IG_4LV(Bench_GenGPU_VecS, CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S);
DEVICE_BM_IG_4LV(Bench_GenGPU_VecM, CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M);
DEVICE_BM_IG_4LV(Bench_GenGPU_VecL, CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L);

DEVICE_BM_IG_4LL(Bench_GenGPU_VecDirS, CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S);
DEVICE_BM_IG_4LL(Bench_GenGPU_VecDirM, CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M);
DEVICE_BM_IG_4LL(Bench_GenGPU_VecDirL, CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L, CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L);
