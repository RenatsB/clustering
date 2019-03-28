#include <benchmark/benchmark.h>
#include "benchParams.h"
#include "cpuImageGen.hpp"

#define HOST_BM_IG_CV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB)                       \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    for (auto _ : state)                                                        \
    {                                                                           \
      benchmark::DoNotOptimize(ig.generate_serial_CV(HORIZ_DIM,VERT_DIM,TURB)); \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

#define HOST_BM_IG_IG(BM_NAME, HORIZ_DIM, VERT_DIM, TURB)                       \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    for (auto _ : state)                                                        \
    {                                                                           \
      benchmark::DoNotOptimize(ig.generate_serial_IC(HORIZ_DIM,VERT_DIM,TURB)); \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

#define HOST_BM_IG_LN(BM_NAME, HORIZ_DIM, VERT_DIM, TURB)                       \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    for (auto _ : state)                                                        \
    {                                                                           \
      benchmark::DoNotOptimize(ig.generate_serial_LN(HORIZ_DIM,VERT_DIM,TURB)); \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

#define HOST_BM_IG_4SV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB)                      \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    std::vector<float> r(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> g(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> b(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> a(HORIZ_DIM*VERT_DIM);                                   \
    for (auto _ : state)                                                        \
    {                                                                           \
      ig.generate_serial_4SV(HORIZ_DIM,VERT_DIM,TURB, &r, &g, &b, &a);          \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

#define HOST_BM_IG_4LV(BM_NAME, HORIZ_DIM, VERT_DIM, TURB)                      \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    std::vector<float> r(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> g(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> b(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> a(HORIZ_DIM*VERT_DIM);                                   \
    for (auto _ : state)                                                        \
    {                                                                           \
      ig.generate_serial_4LV(HORIZ_DIM,VERT_DIM,TURB,                           \
                             r.data(), g.data(), b.data(), a.data());           \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

HOST_BM_IG_CV(Bench_Gen_ColorVectorS,
              CLIB_BENCH_GENDIM_X_S,
              CLIB_BENCH_GENDIM_Y_S,
              CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_CV(Bench_Gen_ColorVectorM,
              CLIB_BENCH_GENDIM_X_M,
              CLIB_BENCH_GENDIM_Y_M,
              CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_CV(Bench_Gen_ColorVectorL,
              CLIB_BENCH_GENDIM_X_L,
              CLIB_BENCH_GENDIM_Y_L,
              CLIB_BENCH_GENDIM_NOISE);

HOST_BM_IG_IG(Bench_Gen_ImageColorsS,
              CLIB_BENCH_GENDIM_X_S,
              CLIB_BENCH_GENDIM_Y_S,
              CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_IG(Bench_Gen_ImageColorsM,
              CLIB_BENCH_GENDIM_X_M,
              CLIB_BENCH_GENDIM_Y_M,
              CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_IG(Bench_Gen_ImageColorsL,
              CLIB_BENCH_GENDIM_X_L,
              CLIB_BENCH_GENDIM_Y_L,
              CLIB_BENCH_GENDIM_NOISE);

HOST_BM_IG_LN(Bench_Gen_LinearS,
              CLIB_BENCH_GENDIM_X_S,
              CLIB_BENCH_GENDIM_Y_S,
              CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_LN(Bench_Gen_LinearM,
              CLIB_BENCH_GENDIM_X_M,
              CLIB_BENCH_GENDIM_Y_M,
              CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_LN(Bench_Gen_LinearL,
              CLIB_BENCH_GENDIM_X_L,
              CLIB_BENCH_GENDIM_Y_L,
              CLIB_BENCH_GENDIM_NOISE);

HOST_BM_IG_4SV(Bench_Gen_StdVecS,
               CLIB_BENCH_GENDIM_X_S,
               CLIB_BENCH_GENDIM_Y_S,
               CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_4SV(Bench_Gen_StdVecM,
               CLIB_BENCH_GENDIM_X_M,
               CLIB_BENCH_GENDIM_Y_M,
               CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_4SV(Bench_Gen_StdVecL,
               CLIB_BENCH_GENDIM_X_L,
               CLIB_BENCH_GENDIM_Y_L,
               CLIB_BENCH_GENDIM_NOISE);

HOST_BM_IG_4LV(Bench_Gen_VecS,
               CLIB_BENCH_GENDIM_X_S,
               CLIB_BENCH_GENDIM_Y_S,
               CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_4LV(Bench_Gen_VecM,
               CLIB_BENCH_GENDIM_X_M,
               CLIB_BENCH_GENDIM_Y_M,
               CLIB_BENCH_GENDIM_NOISE);
HOST_BM_IG_4LV(Bench_Gen_VecL,
               CLIB_BENCH_GENDIM_X_L,
               CLIB_BENCH_GENDIM_Y_L,
               CLIB_BENCH_GENDIM_NOISE);
