// Production prefill GEMM ablations. Build inside Nix with
// tools/bench/build.sh. Run: prefill_gemm_bench q4|q5|q6|q8|iq4 M K
// [batch=2048] [iterations=12]
//      prefill_gemm_bench q8-swiglu [batch=2048] [iterations=12]
// Synthetic device weights; component timings do not replace release model
// measurements with the actual mapped GGUF artifacts.
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
                            std::size_t block_bytes,
                            std::size_t scale_offset = 0, bool affine = false) {
  const std::size_t block = blockIdx.x * blockDim.x + threadIdx.x;
  if (block >= size / block_bytes)
    return;
  auto* bytes = data + block * block_bytes;
  for (std::size_t i = 0; i < block_bytes; ++i)
    bytes[i] = Hash(block * block_bytes + i);
  auto* scales = reinterpret_cast<__half*>(bytes + scale_offset);
  scales[0] = __float2half(0.007F + (Hash(block) % 1000) * 0.00001F);
  if (affine)
    scales[1] = __float2half(0.004F + (Hash(block + 19) % 1000) * 0.00001F);
}

__global__ void FillValues(float* values, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    values[i] = (static_cast<int>(Hash(i) % 20001) - 10000) * 0.00027F;
}

__global__ void Compare(const float* reference, const float* candidate,
                        std::size_t count, unsigned* mismatches) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count &&
      (!isfinite(reference[i]) || !isfinite(candidate[i]) ||
       __float_as_uint(reference[i]) != __float_as_uint(candidate[i])))
    atomicAdd(mismatches, 1U);
}

class Event {
public:
  Event() { HIP_CHECK(hipEventCreate(&event_)); }
  ~Event() { (void)hipEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  hipEvent_t Get() const { return event_; }

private:
  hipEvent_t event_{};
};

template<gufo::core::GgmlType Type>
void RunQuant(std::size_t m, std::size_t k, std::size_t batch,
              unsigned iterations) {
  using namespace gufo::hip;
  using gufo::core::GgmlType;
  if (m < 1024 || k < 1024 || k % 256 || batch < 96 || iterations == 0)
    throw std::invalid_argument(
        "require M/K >= 1024, K % 256 = 0, batch >= 96");
  constexpr bool affine = Type == GgmlType::kQ4_K || Type == GgmlType::kQ5_K;
  constexpr std::size_t block_bytes = Type == GgmlType::kQ4_K   ? 144
                                      : Type == GgmlType::kQ5_K ? 176
                                      : Type == GgmlType::kQ6_K ? 210
                                      : Type == GgmlType::kQ8_0 ? 34
                                                                : 136;
  const std::size_t weight_bytes =
      m * k / (Type == GgmlType::kQ8_0 ? 32 : 256) * block_bytes;
  DeviceBuffer<unsigned char> weights(weight_bytes),
      act(Q8ActTiledBytes(batch, k) + Q8ActSumBytes(batch, k));
  DeviceBuffer<float> input(batch * k), reference(batch * m),
      candidate(batch * m);
  DeviceBuffer<unsigned> mismatches(1);
  FillWeights<<<(weight_bytes / block_bytes + 255) / 256, 256>>>(
      weights.data(), weight_bytes, block_bytes,
      Type == GgmlType::kQ6_K ? 208 : 0, affine);
  FillValues<<<(input.size() + 255) / 256, 256>>>(input.data(), input.size());
  LaunchQuantizeActivationQ8_1FromFp32(input.data(), act.data(), batch, k);
  const auto control = [&] {
    const dim3 grid((batch + 127) / 128, (m + 127) / 128);
    if constexpr (Type == GgmlType::kQ8_0) {
      W8A8BlockedWmmaGEMMKernel<128, 128, 2, 4, 2><<<grid, 256>>>(
          weights.data(), act.data(), reference.data(), batch, m, k);
    } else {
      WKQuantA8BlockedWmmaGEMMKernel<128, 128, 2, 4, 2, Type><<<grid, 256>>>(
          weights.data(), act.data(), reference.data(), batch, m, k);
    }
  };
  const auto production = [&] {
    LaunchBatchedQuantGEMMPreQuantized(Type, weights.data(), act.data(),
                                       candidate.data(), batch, m, k);
  };
  control();
  production();
  HIP_CHECK(hipMemset(mismatches.data(), 0, sizeof(unsigned)));
  Compare<<<(reference.size() + 255) / 256, 256>>>(
      reference.data(), candidate.data(), reference.size(), mismatches.data());
  if (mismatches.CopyToHost()[0] != 0)
    throw std::runtime_error("production prefill differs from wave32 control");
  Event begin, end;
  const auto time = [&](auto launch) {
    for (unsigned i = 0; i < 3; ++i)
      launch();
    HIP_CHECK(hipEventRecord(begin.Get()));
    for (unsigned i = 0; i < iterations; ++i)
      launch();
    HIP_CHECK(hipEventRecord(end.Get()));
    HIP_CHECK(hipEventSynchronize(end.Get()));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, begin.Get(), end.Get()));
    return ms / iterations;
  };
  for (unsigned repeat = 0; repeat < 3; ++repeat) {
    float control_ms, production_ms;
    if (repeat % 2 == 0) {
      control_ms = time(control);
      production_ms = time(production);
    } else {
      production_ms = time(production);
      control_ms = time(control);
    }
    std::printf(
        "type=%u batch=%zu M=%zu K=%zu rep=%u control_ms=%.6f "
        "production_ms=%.6f exact\n",
        static_cast<unsigned>(Type), batch, m, k, repeat, control_ms,
        production_ms);
  }
}

void RunEpilogue(std::size_t batch, unsigned iterations) {
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
    const std::string type = argc > 1 ? argv[1] : "";
    if (type == "q8-swiglu" && argc <= 4) {
      RunEpilogue(argc > 2 ? std::stoul(argv[2]) : 2048,
                  argc > 3 ? std::stoul(argv[3]) : 12);
    } else if (argc >= 4 && argc <= 6) {
      const auto m = std::stoul(argv[2]), k = std::stoul(argv[3]);
      const auto batch = argc > 4 ? std::stoul(argv[4]) : 2048;
      const auto iterations = argc > 5 ? std::stoul(argv[5]) : 12;
      using gufo::core::GgmlType;
      if (type == "q4")
        RunQuant<GgmlType::kQ4_K>(m, k, batch, iterations);
      else if (type == "q5")
        RunQuant<GgmlType::kQ5_K>(m, k, batch, iterations);
      else if (type == "q6")
        RunQuant<GgmlType::kQ6_K>(m, k, batch, iterations);
      else if (type == "q8")
        RunQuant<GgmlType::kQ8_0>(m, k, batch, iterations);
      else if (type == "iq4")
        RunQuant<GgmlType::kIQ4_XS>(m, k, batch, iterations);
      else
        throw std::invalid_argument(
            "expected q4, q5, q6, q8, iq4 or q8-swiglu");
    } else {
      throw std::invalid_argument(
          "usage: prefill_gemm_bench q4|q5|q6|q8|iq4 M K [batch] [iterations], "
          "or q8-swiglu [batch] [iterations]");
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
