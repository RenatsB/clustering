#include <benchmark/benchmark.h>
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

#define HOST_BM_IG_4LL(BM_NAME, HORIZ_DIM, VERT_DIM, TURB)                      \
  static void BM_NAME(benchmark::State& state)                                  \
  {                                                                             \
    ImageGenFn ig;                                                              \
    std::vector<float> r(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> g(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> b(HORIZ_DIM*VERT_DIM);                                   \
    std::vector<float> a(HORIZ_DIM*VERT_DIM);                                   \
    for (auto _ : state)                                                        \
    {                                                                           \
      ig.generate_serial_4LL(HORIZ_DIM,VERT_DIM,TURB,                           \
                               r.data(), g.data(), b.data(), a.data());         \
    }                                                                           \
  }                                                                             \
  BENCHMARK(BM_NAME)

HOST_BM_IG_CV(Bench_Gen_ColorVectorS, 128, 128, 64);
HOST_BM_IG_CV(Bench_Gen_ColorVectorM, 512, 512, 128);
HOST_BM_IG_CV(Bench_Gen_ColorVectorL, 1024, 1024, 256);

HOST_BM_IG_IG(Bench_Gen_ImageColorsS, 128, 128, 64);
HOST_BM_IG_IG(Bench_Gen_ImageColorsM, 512, 512, 128);
HOST_BM_IG_IG(Bench_Gen_ImageColorsL, 1024, 1024, 256);

HOST_BM_IG_LN(Bench_Gen_LinearS, 128, 128, 64);
HOST_BM_IG_LN(Bench_Gen_LinearM, 512, 512, 128);
HOST_BM_IG_LN(Bench_Gen_LinearL, 1024, 1024, 256);

HOST_BM_IG_4SV(Bench_Gen_StdVecS, 128, 128, 64);
HOST_BM_IG_4SV(Bench_Gen_StdVecM, 512, 512, 128);
HOST_BM_IG_4SV(Bench_Gen_StdVecL, 1024, 1024, 256);

HOST_BM_IG_4LV(Bench_Gen_VecS, 128, 128, 64);
HOST_BM_IG_4LV(Bench_Gen_VecM, 512, 512, 128);
HOST_BM_IG_4LV(Bench_Gen_VecL, 1024, 1024, 256);

HOST_BM_IG_4LL(Bench_Gen_VecDirS, 128, 128, 64);
HOST_BM_IG_4LL(Bench_Gen_VecDirM, 512, 512, 128);
HOST_BM_IG_4LL(Bench_Gen_VecDirL, 1024, 1024, 256);
