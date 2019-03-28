#include <benchmark/benchmark.h>
#include "benchParams.h"
#include "cpuRandomFn.hpp"

bool fillVectorS()
{
    RandomFn<float> rg;
    std::vector<float> numbers(CLIB_BENCH_RANDNUM);
    for(auto &n : numbers)
        n = rg.SimpleRand(CLIB_BENCH_RANDLOW, CLIB_BENCH_RANDHI);
    return true;
}

bool fillVectorU()
{
    RandomFn<float> rg;
    std::vector<float> numbers(CLIB_BENCH_RANDNUM);
    rg.setNumericLimits(CLIB_BENCH_RANDLOW,CLIB_BENCH_RANDHI);
    for(auto &n : numbers)
        n = rg.UniformRandU();
    return true;
}

bool fillVectorM()
{
    RandomFn<float> rg;
    std::vector<float> numbers(CLIB_BENCH_RANDNUM);
    rg.setNumericLimits(CLIB_BENCH_RANDLOW,CLIB_BENCH_RANDHI);
    for(auto &n : numbers)
        n = rg.MT19937RandU();
    return true;
}

static void Bench_Ran_Simple(benchmark::State& state)
{
    for (auto _ : state)
    {
      benchmark::DoNotOptimize(fillVectorS());
    }
}
BENCHMARK(Bench_Ran_Simple);

static void Bench_Ran_Uniform(benchmark::State& state)
{
    for (auto _ : state)
    {
      benchmark::DoNotOptimize(fillVectorU());
    }
}
BENCHMARK(Bench_Ran_Uniform);

static void Bench_Ran_MT(benchmark::State& state)
{
    for (auto _ : state)
    {
      benchmark::DoNotOptimize(fillVectorM());
    }
}
BENCHMARK(Bench_Ran_MT);
