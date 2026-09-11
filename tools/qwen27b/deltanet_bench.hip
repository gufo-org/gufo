// Exact decode/verification/replay ablation using the production kernel.
// Build: nix develop -c tools/bench/build.sh tools/qwen27b/deltanet_bench.hip
// Run: /tmp/deltanet_bench [rows=8] [iterations=48] [seed=1]
// Rotating 48 recurrent states models the 144 MiB production working set.
#define ENGINE_ENABLE_HIP 1
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kernels/deltanet_decode.hpp"

namespace {
void Check(hipError_t error) {
  if (error != hipSuccess)
    throw std::runtime_error(hipGetErrorString(error));
}

struct DeviceFree {
  void operator()(float* pointer) const { (void)hipFree(pointer); }
};
using Buffer = std::unique_ptr<float, DeviceFree>;

Buffer Allocate(std::size_t count) {
  float* pointer = nullptr;
  Check(hipMalloc(&pointer, count * sizeof(float)));
  return Buffer(pointer);
}

unsigned ParseUnsigned(const char* text) {
  unsigned value = 0;
  const char* const end = text + std::strlen(text);
  const auto [parsed, error] = std::from_chars(text, end, value);
  if (error != std::errc{} || parsed != end)
    throw std::invalid_argument("expected an unsigned integer");
  return value;
}

struct Events {
  hipEvent_t begin{}, end{};
  Events() {
    Check(hipEventCreate(&begin));
    Check(hipEventCreate(&end));
  }
  ~Events() {
    (void)hipEventDestroy(begin);
    (void)hipEventDestroy(end);
  }
  Events(const Events&) = delete;
  Events& operator=(const Events&) = delete;
};

__global__ void Fill(float* values, std::size_t count, unsigned seed) {
  const auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  unsigned hash = index + seed;
  hash = (hash ^ (hash >> 16)) * 0x7feb352dU;
  hash = (hash ^ (hash >> 15)) * 0x846ca68bU;
  hash ^= hash >> 16;
  values[index] = (static_cast<int>(hash % 20001) - 10000) * 0.0001F;
}

std::vector<float> Download(const float* data, std::size_t count) {
  std::vector<float> result(count);
  Check(hipMemcpy(result.data(), data, count * sizeof(float),
                  hipMemcpyDeviceToHost));
  return result;
}

void ExpectExact(const std::vector<float>& reference,
                 const std::vector<float>& candidate) {
  if (!std::ranges::all_of(candidate,
                           [](float x) { return std::isfinite(x); }) ||
      std::memcmp(reference.data(), candidate.data(),
                  reference.size() * sizeof(float)) != 0) {
    throw std::runtime_error("recurrence changed output/state bits");
  }
}

void Run(unsigned rows, unsigned iterations, unsigned seed) {
  using gufo::hip::DeltaNetRecurrenceKernel;
  constexpr unsigned heads = 48, key_heads = 16, dim = 128, layers = 48;
  constexpr std::size_t inner = heads * dim;
  constexpr std::size_t conv = 2 * key_heads * dim + inner;
  constexpr std::size_t state_size = heads * dim * dim;
  auto x = Allocate(rows * conv), alpha = Allocate(rows * heads),
       beta = Allocate(rows * heads), a = Allocate(heads), dt = Allocate(heads),
       norm = Allocate(dim), gate = Allocate(rows * inner),
       initial = Allocate(state_size), states = Allocate(layers * state_size),
       output = Allocate(rows * inner);
  const auto fill = [&](float* data, std::size_t count) {
    Fill<<<(count + 255) / 256, 256>>>(data, count, seed);
  };
  fill(x.get(), rows * conv);
  fill(alpha.get(), rows * heads);
  fill(beta.get(), rows * heads);
  fill(dt.get(), heads);
  fill(norm.get(), dim);
  fill(gate.get(), rows * inner);
  fill(initial.get(), state_size);
  fill(states.get(), layers * state_size);
  const std::vector<float> decay(heads, -0.15F);
  Check(hipMemcpy(a.get(), decay.data(), heads * sizeof(float),
                  hipMemcpyHostToDevice));
  const auto launch = [&]<bool Resident, bool WriteOutput>(float* state) {
    DeltaNetRecurrenceKernel<float, Resident, WriteOutput>
        <<<heads, Resident ? 128 : 256>>>(
            x.get(), state, alpha.get(), beta.get(), a.get(), dt.get(),
            norm.get(), gate.get(), WriteOutput ? output.get() : nullptr, 0,
            key_heads, heads, dim, dim, {}, rows, conv, heads, inner);
  };
  const auto reset = [&] {
    Check(hipMemcpy(states.get(), initial.get(), state_size * sizeof(float),
                    hipMemcpyDeviceToDevice));
  };
  reset();
  launch.template operator()<false, true>(states.get());
  const auto expected_state = Download(states.get(), state_size);
  const auto expected_output = Download(output.get(), rows * inner);
  struct Geometry {
    const char* name;
    bool writes_output;
    std::function<void(float*)> launch;
  };
  const std::vector<Geometry> geometries{
      {"original", true,
       [&](float* state) { launch.template operator()<false, true>(state); }},
      {"verification", true,
       [&](float* state) {
         if (rows > 1)
           launch.template operator()<true, true>(state);
         else
           launch.template operator()<false, true>(state);
       }},
      {"state_only", false,
       [&](float* state) {
         if (rows > 1)
           launch.template operator()<true, false>(state);
         else
           launch.template operator()<false, false>(state);
       }},
  };
  Events events;
  std::puts("rows,round,variant,ms,state_exact,output_exact");
  for (unsigned round = 0; round < 4; ++round) {
    for (std::size_t i = 0; i < geometries.size(); ++i) {
      const auto& geometry =
          geometries[round % 2 ? geometries.size() - i - 1 : i];
      reset();
      geometry.launch(states.get());
      Check(hipGetLastError());
      ExpectExact(expected_state, Download(states.get(), state_size));
      if (geometry.writes_output)
        ExpectExact(expected_output, Download(output.get(), rows * inner));
      Check(hipEventRecord(events.begin));
      for (unsigned iteration = 0; iteration < iterations; ++iteration)
        geometry.launch(states.get() + (iteration % layers) * state_size);
      Check(hipEventRecord(events.end));
      Check(hipEventSynchronize(events.end));
      float ms = 0;
      Check(hipEventElapsedTime(&ms, events.begin, events.end));
      std::printf("%u,%u,%s,%.6f,true,%s\n", rows, round, geometry.name,
                  ms / iterations, geometry.writes_output ? "true" : "unused");
    }
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc > 4)
      throw std::invalid_argument(
          "usage: deltanet_bench [rows] [iterations] [seed]");
    const unsigned rows = argc > 1 ? ParseUnsigned(argv[1]) : 8;
    const unsigned iterations = argc > 2 ? ParseUnsigned(argv[2]) : 48;
    const unsigned seed = argc > 3 ? ParseUnsigned(argv[3]) : 1;
    if (rows < 1 || rows > 16 || iterations == 0)
      throw std::invalid_argument("require 1..16 rows and positive iterations");
    Run(rows, iterations, seed);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
