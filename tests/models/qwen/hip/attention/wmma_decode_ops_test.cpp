// Oracle test for the matrix-core decode and verification attention
// (LaunchQwenWmmaDecodeAttention) on the Qwen3.6-35B-A3B head shape: sixteen
// query heads over two 256-wide KV heads.
//
// Queries and probabilities are FP16 operands, so the result agrees with a
// double-precision reference to FP16 precision, not exactly. What must hold
// exactly is that a row does not depend on the rows verified with it: every
// case also computes each row alone, with different values in the cache past
// that row's position, as single-token decoding would see, and requires the
// outputs to be byte-identical. Covered: one partition, many partitions,
// one to eight rows, and rows on both sides of a partition-size step.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

constexpr std::uint32_t kHeads = 16;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::size_t kKvWidth = std::size_t{kKvHeads} * kHeadDim;
constexpr std::size_t kAttnWidth = std::size_t{kHeads} * kHeadDim;

class Rng {
public:
  explicit Rng(std::uint32_t seed) : state_(seed) {}
  float Uniform(float lo, float hi) {
    state_ = (state_ * 1664525U) + 1013904223U;
    const float u = static_cast<float>((state_ >> 8) & 0xFFFFU) / 65535.0F;
    return lo + (u * (hi - lo));
  }

private:
  std::uint32_t state_;
};

void Fail(const char* message) {
  std::cerr << "FAIL: " << message << '\n';
  std::exit(1);
}

// Runs `positions` rows starting at `start_pos` over a cache whose rows at or
// past `valid_end` hold `filler` instead of their values.
std::vector<float> Run(const std::vector<float>& q,
                       const std::vector<float>& gate,
                       const std::vector<__half>& keys,
                       const std::vector<__half>& values, std::uint32_t context,
                       std::uint32_t start_pos, std::uint32_t positions,
                       std::uint32_t valid_end) {
  auto k = keys;
  auto v = values;
  for (std::size_t i = std::size_t{valid_end} * kKvWidth; i < k.size(); ++i) {
    k[i] = __float2half(50.0F);
    v[i] = __float2half(-9.0F);
  }
  const std::size_t row_scratch =
      gufo::hip::detail::DecodeAttentionScratchElements(kHeads, kHeadDim);
  gufo::test::DeviceBuffer<__half> d_k(k.size()), d_v(v.size());
  gufo::test::DeviceBuffer<float> d_q(positions * kAttnWidth),
      d_gate(positions * kAttnWidth), d_out(positions * kAttnWidth),
      d_scratch(positions * row_scratch);
  HIP_CHECK(hipMemcpy(d_k.data(), k.data(), k.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v.data(), v.data(), v.size() * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_q.data(), q.data(),
                      positions * kAttnWidth * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gate.data(), gate.data(),
                      positions * kAttnWidth * sizeof(float),
                      hipMemcpyHostToDevice));
  if (!gufo::hip::LaunchQwenWmmaDecodeAttention(
          d_q.data(), d_gate.data(), d_k.data(), d_v.data(), d_out.data(), 0,
          start_pos, positions, context, kHeads, kKvHeads, kHeadDim, nullptr,
          {d_scratch.data(), positions * row_scratch}))
    Fail("the matrix-core decode attention rejected a supported shape");
  std::vector<float> out(positions * kAttnWidth);
  HIP_CHECK(hipMemcpy(out.data(), d_out.data(), out.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  return out;
}

void RunCase(std::uint32_t start_pos, std::uint32_t positions) {
  const std::uint32_t context = start_pos + positions + 32;
  Rng rng(start_pos * 31U + positions);
  std::vector<__half> keys(std::size_t{context} * kKvWidth),
      values(std::size_t{context} * kKvWidth);
  for (auto& x : keys)
    x = __float2half(rng.Uniform(-1.0F, 1.0F));
  for (auto& x : values)
    x = __float2half(rng.Uniform(-1.0F, 1.0F));
  std::vector<float> q(positions * kAttnWidth), gate(positions * kAttnWidth);
  for (auto& x : q)
    x = rng.Uniform(-3.0F, 3.0F);
  for (auto& x : gate)
    x = rng.Uniform(-2.0F, 2.0F);

  const auto batched = Run(q, gate, keys, values, context, start_pos, positions,
                           start_pos + positions);

  double max_error = 0.0;
  double magnitude = 0.0;
  for (std::uint32_t row = 0; row < positions; ++row) {
    const std::uint32_t pos = start_pos + row;
    for (std::uint32_t head = 0; head < kHeads; ++head) {
      const std::uint32_t kv_head = head / (kHeads / kKvHeads);
      const float* qh = &q[row * kAttnWidth + head * kHeadDim];
      std::vector<double> scores(pos + 1);
      double max_score = -INFINITY;
      for (std::uint32_t key = 0; key <= pos; ++key) {
        double dot = 0.0;
        for (std::uint32_t d = 0; d < kHeadDim; ++d)
          dot += static_cast<double>(qh[d]) *
                 __half2float(keys[key * kKvWidth + kv_head * kHeadDim + d]);
        scores[key] = dot / 16.0;
        max_score = std::max(max_score, scores[key]);
      }
      double sum = 0.0;
      for (auto& s : scores)
        sum += (s = std::exp(s - max_score));
      for (std::uint32_t d = 0; d < kHeadDim; ++d) {
        double acc = 0.0;
        for (std::uint32_t key = 0; key <= pos; ++key)
          acc += scores[key] *
                 __half2float(values[key * kKvWidth + kv_head * kHeadDim + d]);
        const std::size_t at = row * kAttnWidth + head * kHeadDim + d;
        const double want = acc / sum / (1.0 + std::exp(-gate[at]));
        max_error = std::max(max_error, std::fabs(want - batched[at]));
        magnitude = std::max(magnitude, std::fabs(want));
      }
    }
  }
  if (!(max_error <= 4e-3 * std::max(magnitude, 1e-3)))
    Fail("the matrix-core decode attention is far from the reference");

  for (std::uint32_t row = 0; row < positions; ++row) {
    const std::vector<float> q_row(q.begin() + row * kAttnWidth,
                                   q.begin() + (row + 1) * kAttnWidth);
    const std::vector<float> gate_row(gate.begin() + row * kAttnWidth,
                                      gate.begin() + (row + 1) * kAttnWidth);
    const auto alone = Run(q_row, gate_row, keys, values, context,
                           start_pos + row, 1, start_pos + row + 1);
    if (std::memcmp(alone.data(), &batched[row * kAttnWidth],
                    kAttnWidth * sizeof(float)) != 0)
      Fail("a verification row differs from the same position decoded alone");
  }
  std::cout << "WMMA decode attention start " << start_pos << " rows "
            << positions << ": max error " << max_error << " of " << magnitude
            << ", rows identical alone\n";
}

}  // namespace
#endif

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen WMMA decode attention ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }
  // One partition, a tile boundary inside the rows.
  RunCase(140, 1);
  RunCase(150, 3);
  // Many partitions.
  RunCase(5000, 1);
  RunCase(5000, 8);
  RunCase(20000, 3);
  // Rows on both sides of the 256-to-512 partition step.
  RunCase(8190, 4);
  std::cout << "Qwen WMMA decode attention ops test passed on gfx1151.\n";
#endif
  return 0;
}
