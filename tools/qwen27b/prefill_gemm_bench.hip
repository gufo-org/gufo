// Production prefill GEMM + SwiGLU/quantization ablation. No copied kernels.
// Build: nix develop -c tools/bench/build.sh
// tools/qwen27b/prefill_gemm_bench.hip Run: /tmp/prefill_gemm_bench
// [batch=2048] [iterations=12] Component timings do not replace release model
// A/B measurements.
#define ENGINE_ENABLE_HIP 1
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hip"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {
using gufo::test::DeviceBuffer;

__device__ unsigned Hash(unsigned value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__global__ void FillWeights(unsigned char* data, std::size_t size,
                            std::size_t block_bytes) {
  const std::size_t block = blockIdx.x * blockDim.x + threadIdx.x;
  if (block >= size / block_bytes)
    return;
  auto* bytes = data + block * block_bytes;
  for (std::size_t i = 0; i < block_bytes; ++i)
    bytes[i] = Hash(block * block_bytes + i);
  auto* scales = reinterpret_cast<__half*>(bytes);
  scales[0] = __float2half(0.007F + (Hash(block) % 1000) * 0.00001F);
}

__global__ void FillValues(float* values, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    values[i] = (static_cast<int>(Hash(i) % 20001) - 10000) * 0.00027F;
}

void Run(std::size_t batch, unsigned iterations) {
  using namespace gufo::hip;
  using gufo::core::GgmlType;
  constexpr GgmlType type = GgmlType::kQ8_0;
  constexpr std::size_t m = 17408, k = 5120;
  const std::size_t output_bytes =
      Q8ActTiledBytes(batch, m) + Q8ActSumBytes(batch, m);
  if (iterations == 0 ||
      !IsFusedSwiGluGemmEpilogueSupported(type, batch, m, output_bytes)) {
    throw std::invalid_argument(
        "require a supported batch and positive iterations");
  }
  constexpr std::size_t block_bytes = 34;
  constexpr std::size_t bytes = m * k / 32 * block_bytes;
  DeviceBuffer<unsigned char> weights(bytes),
      act(Q8ActTiledBytes(batch, k) + Q8ActSumBytes(batch, k)),
      reference(output_bytes), candidate(output_bytes);
  DeviceBuffer<float> input(batch * k), gate(batch * m), up(batch * m);
  FillWeights<<<(bytes / block_bytes + 255) / 256, 256>>>(weights.data(), bytes,
                                                          block_bytes);
  FillValues<<<(input.size() + 255) / 256, 256>>>(input.data(), input.size());
  LaunchQuantizeActivationQ8_1FromFp32(input.data(), act.data(), batch, k);
  HIP_CHECK(hipMemset(reference.data(), 0x5A, output_bytes));
  HIP_CHECK(hipMemset(candidate.data(), 0x5A, output_bytes));
  const auto separate = [&] {
    LaunchBatchedDualQuantGEMMPreQuantized(type, weights.data(), weights.data(),
                                           act.data(), gate.data(), up.data(),
                                           batch, m, k);
    LaunchBatchedFusedSwiGLUQuantizeQ8_1(gate.data(), up.data(),
                                         reference.data(), batch, m);
  };
  const auto fused = [&] {
    LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1(
        type, weights.data(), weights.data(), act.data(), gate.data(),
        candidate.data(), batch, m, k);
  };
  separate();
  fused();
  if (reference.CopyToHost() != candidate.CopyToHost())
    throw std::runtime_error("prefill epilogue bytes differ");

  hipEvent_t begin{}, end{};
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  const auto time = [&](auto launch) {
    for (unsigned i = 0; i < 3; ++i)
      launch();
    HIP_CHECK(hipEventRecord(begin));
    for (unsigned i = 0; i < iterations; ++i)
      launch();
    HIP_CHECK(hipEventRecord(end));
    HIP_CHECK(hipEventSynchronize(end));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, begin, end));
    return ms / iterations;
  };
  for (unsigned repetition = 0; repetition < 3; ++repetition) {
    float baseline_ms, candidate_ms;
    if (repetition % 2 == 0) {
      baseline_ms = time(separate);
      candidate_ms = time(fused);
    } else {
      candidate_ms = time(fused);
      baseline_ms = time(separate);
    }
    std::printf(
        "q8 batch=%zu M=%zu K=%zu rep=%u separate_ms=%.6f "
        "fused_ms=%.6f exact\n",
        batch, m, k, repetition, baseline_ms, candidate_ms);
  }
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Run(argc > 1 ? std::stoul(argv[1]) : 2048,
        argc > 2 ? std::stoul(argv[2]) : 12);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
