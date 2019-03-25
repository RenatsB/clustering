#include <benchmark/benchmark.h>
#include "cpuRandomFn.hpp"

#define HOST_BM_RG_SF(BM_NAME, LOW, HI)                          \
  static void BM_NAME(benchmark::State& state)                   \
  {                                                              \
    RandomFn<float> rg;                                          \
    for (auto _ : state)                                         \
    {                                                            \
      benchmark::DoNotOptimize(rg.SimpleRand(LOW, HI));          \
    }                                                            \
  }                                                              \
  BENCHMARK(BM_NAME)

#define HOST_BM_RG_UF(BM_NAME, LOW, HI)                          \
  static void BM_NAME(benchmark::State& state)                   \
  {                                                              \
    RandomFn<float> rg;                                          \
    rg.setNumericLimits(LOW,HI);                                 \
    for (auto _ : state)                                         \
    {                                                            \
      benchmark::DoNotOptimize(rg.UniformRandU());               \
    }                                                            \
  }                                                              \
  BENCHMARK(BM_NAME)

#define HOST_BM_RG_MF(BM_NAME, LOW, HI)                          \
  static void BM_NAME(benchmark::State& state)                   \
  {                                                              \
    RandomFn<float> rg;                                          \
    rg.setNumericLimits(LOW,HI);                                 \
    for (auto _ : state)                                         \
    {                                                            \
      benchmark::DoNotOptimize(rg.MT19937RandU());               \
    }                                                            \
  }                                                              \
  BENCHMARK(BM_NAME)

HOST_BM_RG_SF(Bench_Ran_Simple, -1000.f, 1000.f);
HOST_BM_RG_UF(Bench_Ran_Uniform, -1000.f, 1000.f);
HOST_BM_RG_MF(Bench_Ran_MT, -1000.f, 1000.f);
