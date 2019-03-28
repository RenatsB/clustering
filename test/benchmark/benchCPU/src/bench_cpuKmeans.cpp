#include <benchmark/benchmark.h>
#include "benchParams.h"
#include "cpuImageGen.hpp"
#include "cpuKmeans.hpp"

#define HOST_BM_KM_CV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, K, NUMIT)             \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    ColorVector src = ig.generate_serial_CV(HORIZ_DIM,VERT_DIM,TURB);           \
    cpuKmeans km;                                                               \
    for (auto _ : state)                                                        \
    {                                                                           \
      benchmark::DoNotOptimize(km.kmeans_serial_CV(src, K, NUMIT));             \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

#define HOST_BM_KM_IG(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, K, NUMIT)             \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    ImageColors src = ig.generate_serial_IC(HORIZ_DIM,VERT_DIM,TURB);           \
    cpuKmeans km;                                                               \
    for (auto _ : state)                                                        \
    {                                                                           \
      benchmark::DoNotOptimize(km.kmeans_serial_IC(src, K, NUMIT));             \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

#define HOST_BM_KM_LN(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, K, NUMIT)             \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    std::vector<float> src = ig.generate_serial_LN(HORIZ_DIM,VERT_DIM,TURB);    \
    cpuKmeans km;                                                               \
    for (auto _ : state)                                                        \
    {                                                                           \
      benchmark::DoNotOptimize(km.kmeans_serial_LN(src, K, NUMIT));             \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

#define HOST_BM_KM_4SV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, K, NUMIT)            \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    size_t N = HORIZ_DIM*VERT_DIM;                                              \
    std::vector<float> r(N);                                                    \
    std::vector<float> g(N);                                                    \
    std::vector<float> b(N);                                                    \
    std::vector<float> a(N);                                                    \
    std::vector<float> outr(N);                                                 \
    std::vector<float> outg(N);                                                 \
    std::vector<float> outb(N);                                                 \
    std::vector<float> outa(N);                                                 \
    ig.generate_serial_4SV(HORIZ_DIM,VERT_DIM,TURB, &r, &g, &b, &a);            \
    cpuKmeans km;                                                               \
    for (auto _ : state)                                                        \
    {                                                                           \
      km.kmeans_serial_4SV(&r, &g, &b, &a, &outr, &outg, &outb, &outa,          \
                                                    N, K, NUMIT);               \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

#define HOST_BM_KM_4LV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB, K, NUMIT)            \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    size_t N = HORIZ_DIM*VERT_DIM;                                              \
    std::vector<float> r(N);                                                    \
    std::vector<float> g(N);                                                    \
    std::vector<float> b(N);                                                    \
    std::vector<float> a(N);                                                    \
    std::vector<float> outr(N);                                                 \
    std::vector<float> outg(N);                                                 \
    std::vector<float> outb(N);                                                 \
    std::vector<float> outa(N);                                                 \
    ig.generate_serial_4LV(HORIZ_DIM,VERT_DIM,TURB,                             \
                           r.data(), g.data(), b.data(), a.data());             \
    cpuKmeans km;                                                               \
    for (auto _ : state)                                                        \
    {                                                                           \
      km.kmeans_serial_4LV(r.data(),g.data(),b.data(), a.data(),outr.data(),    \
                           outg.data(),outb.data(), outa.data(), N, K, NUMIT);  \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

HOST_BM_KM_CV(Bench_Kmn_ColorVectorS,
              CLIB_BENCH_GENDIM_X_S,
              CLIB_BENCH_GENDIM_Y_S,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_S,
              CLIB_BENCH_KM_ITER_A);
HOST_BM_KM_CV(Bench_Kmn_ColorVectorM,
              CLIB_BENCH_GENDIM_X_M,
              CLIB_BENCH_GENDIM_Y_M,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_M,
              CLIB_BENCH_KM_ITER_B);
HOST_BM_KM_CV(Bench_Kmn_ColorVectorL,
              CLIB_BENCH_GENDIM_X_L,
              CLIB_BENCH_GENDIM_Y_L,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_L,
              CLIB_BENCH_KM_ITER_C);

HOST_BM_KM_IG(Bench_Kmn_ImageColorsS,
              CLIB_BENCH_GENDIM_X_S,
              CLIB_BENCH_GENDIM_Y_S,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_S,
              CLIB_BENCH_KM_ITER_A);
HOST_BM_KM_IG(Bench_Kmn_ImageColorsM,
              CLIB_BENCH_GENDIM_X_M,
              CLIB_BENCH_GENDIM_Y_M,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_M,
              CLIB_BENCH_KM_ITER_B);
HOST_BM_KM_IG(Bench_Kmn_ImageColorsL,
              CLIB_BENCH_GENDIM_X_L,
              CLIB_BENCH_GENDIM_Y_L,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_L,
              CLIB_BENCH_KM_ITER_C);

HOST_BM_KM_LN(Bench_Kmn_LinearS,
              CLIB_BENCH_GENDIM_X_S,
              CLIB_BENCH_GENDIM_Y_S,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_S,
              CLIB_BENCH_KM_ITER_A);
HOST_BM_KM_LN(Bench_Kmn_LinearM,
              CLIB_BENCH_GENDIM_X_M,
              CLIB_BENCH_GENDIM_Y_M,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_M,
              CLIB_BENCH_KM_ITER_B);
HOST_BM_KM_LN(Bench_Kmn_LinearL,
              CLIB_BENCH_GENDIM_X_L,
              CLIB_BENCH_GENDIM_Y_L,
              CLIB_BENCH_GENDIM_NOISE,
              CLIB_BENCH_KM_CLUSTERS_L,
              CLIB_BENCH_KM_ITER_C);

HOST_BM_KM_4SV(Bench_Kmn_StdVecS,
               CLIB_BENCH_GENDIM_X_S,
               CLIB_BENCH_GENDIM_Y_S,
               CLIB_BENCH_GENDIM_NOISE,
               CLIB_BENCH_KM_CLUSTERS_S,
               CLIB_BENCH_KM_ITER_A);
HOST_BM_KM_4SV(Bench_Kmn_StdVecM,
               CLIB_BENCH_GENDIM_X_M,
               CLIB_BENCH_GENDIM_Y_M,
               CLIB_BENCH_GENDIM_NOISE,
               CLIB_BENCH_KM_CLUSTERS_M,
               CLIB_BENCH_KM_ITER_B);
HOST_BM_KM_4SV(Bench_Kmn_StdVecL,
               CLIB_BENCH_GENDIM_X_L,
               CLIB_BENCH_GENDIM_Y_L,
               CLIB_BENCH_GENDIM_NOISE,
               CLIB_BENCH_KM_CLUSTERS_L,
               CLIB_BENCH_KM_ITER_C);

HOST_BM_KM_4LV(Bench_Kmn_VecS,
               CLIB_BENCH_GENDIM_X_S,
               CLIB_BENCH_GENDIM_Y_S,
               CLIB_BENCH_GENDIM_NOISE,
               CLIB_BENCH_KM_CLUSTERS_S,
               CLIB_BENCH_KM_ITER_A);
HOST_BM_KM_4LV(Bench_Kmn_VecM,
               CLIB_BENCH_GENDIM_X_M,
               CLIB_BENCH_GENDIM_Y_M,
               CLIB_BENCH_GENDIM_NOISE,
               CLIB_BENCH_KM_CLUSTERS_M,
               CLIB_BENCH_KM_ITER_B);
HOST_BM_KM_4LV(Bench_Kmn_VecL,
               CLIB_BENCH_GENDIM_X_L,
               CLIB_BENCH_GENDIM_Y_L,
               CLIB_BENCH_GENDIM_NOISE,
               CLIB_BENCH_KM_CLUSTERS_L,
               CLIB_BENCH_KM_ITER_C);
