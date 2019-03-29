#include <benchmark/benchmark.h>
#include "benchParams.h"
#include "gpuImageGen.h"
#include "gpuKmeans.h"
#include "cpuRandomFn.hpp"

#define DEVICE_BM_KM_CV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, THREADS, K, ITER)             \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    RandomFn<float> rg;                                                                   \
    ColorVector cv=gpuImageGen::generate_parallel_CV(HORIZ_DIM,VERT_DIM,TURB,THREADS);    \
    for (auto _ : state)                                                                  \
    {                                                                                     \
        benchmark::DoNotOptimize(gpuKmeans::kmeans_parallel_CV(cv,K,ITER,THREADS,&rg));   \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_IG(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, THREADS, K, ITER)             \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    RandomFn<float> rg;                                                                   \
    ImageColors ic=gpuImageGen::generate_parallel_IC(HORIZ_DIM,VERT_DIM,TURB,THREADS);    \
    for (auto _ : state)                                                                  \
    {                                                                                     \
        benchmark::DoNotOptimize(gpuKmeans::kmeans_parallel_IC(ic,K,ITER,THREADS,&rg));   \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_LN(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, THREADS, K, ITER)             \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    RandomFn<float> rg;                                                                   \
    std::vector<float> ln=gpuImageGen::generate_parallel_LN(HORIZ_DIM,VERT_DIM,           \
                                                            TURB,THREADS);                \
    for (auto _ : state)                                                                  \
    {                                                                                     \
        benchmark::DoNotOptimize(gpuKmeans::kmeans_parallel_LN(ln,K,ITER,THREADS,&rg));   \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_4SV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, THREADS, K, ITER)            \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    RandomFn<float> rg;                                                                   \
    std::vector<float> r(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> g(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> b(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> a(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> fr(HORIZ_DIM*VERT_DIM);                                            \
    std::vector<float> fg(HORIZ_DIM*VERT_DIM);                                            \
    std::vector<float> fb(HORIZ_DIM*VERT_DIM);                                            \
    std::vector<float> fa(HORIZ_DIM*VERT_DIM);                                            \
    gpuImageGen::generate_parallel_4SV(&r,&g,&b,&a,HORIZ_DIM,VERT_DIM,TURB,THREADS);      \
    for (auto _ : state)                                                                  \
    {                                                                                     \
        gpuKmeans::kmeans_parallel_4SV(&r,&g,&b,&a,&fr,&fg,&fb,&fa,HORIZ_DIM*VERT_DIM,    \
                                                                   K,ITER,THREADS,&rg);   \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

#define DEVICE_BM_KM_4LV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, THREADS, K, ITER)            \
  static void BM_NAME(benchmark::State& state)                                            \
  {                                                                                       \
    RandomFn<float> rg;                                                                   \
    std::vector<float> r(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> g(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> b(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> a(HORIZ_DIM*VERT_DIM);                                             \
    std::vector<float> fr(HORIZ_DIM*VERT_DIM);                                            \
    std::vector<float> fg(HORIZ_DIM*VERT_DIM);                                            \
    std::vector<float> fb(HORIZ_DIM*VERT_DIM);                                            \
    std::vector<float> fa(HORIZ_DIM*VERT_DIM);                                            \
    gpuImageGen::generate_parallel_4LV(r.data(), g.data(), b.data(), a.data(),            \
                                       HORIZ_DIM,VERT_DIM,TURB, THREADS);                 \
    for (auto _ : state)                                                                  \
    {                                                                                     \
        gpuKmeans::kmeans_parallel_4LV(r.data(),g.data(),b.data(),a.data(),               \
                                       fr.data(),fg.data(),fb.data(),fa.data(),           \
                                       HORIZ_DIM*VERT_DIM,                                \
                                       K,ITER,THREADS,&rg);                               \
    }                                                                                     \
  }                                                                                       \
  BENCHMARK(BM_NAME)

DEVICE_BM_KM_CV(Bench_GenGPU_ColorVectorS,
                CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S,
                CLIB_BENCH_KM_CLUSTERS_S, CLIB_BENCH_KM_ITER_A);
DEVICE_BM_KM_CV(Bench_GenGPU_ColorVectorM,
                CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M,
                CLIB_BENCH_KM_CLUSTERS_M, CLIB_BENCH_KM_ITER_B);
DEVICE_BM_KM_CV(Bench_GenGPU_ColorVectorL,
                CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L,
                CLIB_BENCH_KM_CLUSTERS_L, CLIB_BENCH_KM_ITER_C);

DEVICE_BM_KM_IG(Bench_GenGPU_ImageColorsS,
                CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S,
                CLIB_BENCH_KM_CLUSTERS_S, CLIB_BENCH_KM_ITER_A);
DEVICE_BM_KM_IG(Bench_GenGPU_ImageColorsM,
                CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M,
                CLIB_BENCH_KM_CLUSTERS_M, CLIB_BENCH_KM_ITER_B);
DEVICE_BM_KM_IG(Bench_GenGPU_ImageColorsL,
                CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L,
                CLIB_BENCH_KM_CLUSTERS_L, CLIB_BENCH_KM_ITER_C);

DEVICE_BM_KM_LN(Bench_GenGPU_LinearS,
                CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S,
                CLIB_BENCH_KM_CLUSTERS_S, CLIB_BENCH_KM_ITER_A);
DEVICE_BM_KM_LN(Bench_GenGPU_LinearM,
                CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M,
                CLIB_BENCH_KM_CLUSTERS_M, CLIB_BENCH_KM_ITER_B);
DEVICE_BM_KM_LN(Bench_GenGPU_LinearL,
                CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L,
                CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L,
                CLIB_BENCH_KM_CLUSTERS_L, CLIB_BENCH_KM_ITER_C);

DEVICE_BM_KM_4SV(Bench_GenGPU_StdVecS,
                 CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S,
                 CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S,
                 CLIB_BENCH_KM_CLUSTERS_S, CLIB_BENCH_KM_ITER_A);
DEVICE_BM_KM_4SV(Bench_GenGPU_StdVecM,
                 CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M,
                 CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M,
                 CLIB_BENCH_KM_CLUSTERS_M, CLIB_BENCH_KM_ITER_B);
DEVICE_BM_KM_4SV(Bench_GenGPU_StdVecL,
                 CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L,
                 CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L,
                 CLIB_BENCH_KM_CLUSTERS_L, CLIB_BENCH_KM_ITER_C);

DEVICE_BM_KM_4LV(Bench_GenGPU_VecS,
                 CLIB_BENCH_GENDIM_X_S, CLIB_BENCH_GENDIM_Y_S,
                 CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_S,
                 CLIB_BENCH_KM_CLUSTERS_S, CLIB_BENCH_KM_ITER_A);
DEVICE_BM_KM_4LV(Bench_GenGPU_VecM,
                 CLIB_BENCH_GENDIM_X_M, CLIB_BENCH_GENDIM_Y_M,
                 CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_M,
                 CLIB_BENCH_KM_CLUSTERS_M, CLIB_BENCH_KM_ITER_B);
DEVICE_BM_KM_4LV(Bench_GenGPU_VecL,
                 CLIB_BENCH_GENDIM_X_L, CLIB_BENCH_GENDIM_Y_L,
                 CLIB_BENCH_GENDIM_NOISE, CLIB_BENCH_GENP_THREADS_L,
                 CLIB_BENCH_KM_CLUSTERS_L, CLIB_BENCH_KM_ITER_C);

