// Oracle test for the split prefill attention (opt-c165-attn-split).
//
// The split path computes attention for a prefill chunk at depth as two
// partial softmaxes -- an AOTriton non-causal pass over the fully visible
// prefix and the tiled causal kernel over the N x N diagonal -- then merges
// them by log-sum-exp. The merge is exact in real arithmetic, so the whole path
// must agree with the unsplit tiled kernel over the same key range to within
// the FP16 precision the two halves share.
//
// This exercises production shapes (24 query heads over 4 KV heads, head
// dimension 256) at several depths, including a depth that is not a multiple of
// the kernel's 64-key tile.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/ops.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

namespace {

constexpr std::uint32_t kNumHeads = 24;
constexpr std::uint32_t kNumKvHeads = 4;
constexpr std::uint32_t kHeadDim = 256;
constexpr std::uint32_t kMaxContext = 16384;

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

struct Buffers {
  float* q = nullptr;
  float* k = nullptr;
  float* v = nullptr;
  float* gate = nullptr;
  float* kv_cache = nullptr;
  void* kv_f16 = nullptr;
  float* out = nullptr;
  void* q_f16 = nullptr;
  void* prefix_f16 = nullptr;
  float* diag_out = nullptr;
  float* lse_prefix = nullptr;
  float* lse_diag = nullptr;
};

void Compare(const char* label, const std::vector<float>& got,
             const std::vector<float>& want, double tolerance) {
  double max_abs = 0.0;
  double magnitude = 0.0;
  std::size_t non_finite = 0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i])) {
      ++non_finite;
      continue;
    }
    max_abs = std::fmax(max_abs, std::fabs(static_cast<double>(got[i]) -
                                           static_cast<double>(want[i])));
    magnitude = std::fmax(magnitude, std::fabs(static_cast<double>(want[i])));
  }
  const double rel = (magnitude > 0.0) ? (max_abs / magnitude) : max_abs;
  std::cout << "  " << label << ": max_abs=" << max_abs
            << " magnitude=" << magnitude << " rel=" << rel
            << " non_finite=" << non_finite << "\n";
  if (non_finite != 0 || rel > tolerance) {
    std::cerr << label
              << ": split prefill attention disagrees with the "
                 "unsplit tiled kernel\n";
    std::abort();
  }
}

void RunCase(std::uint32_t start_pos, std::size_t batch_size) {
  std::cout << "split prefill attention: start_pos=" << start_pos
            << " batch=" << batch_size << "\n";

  const std::size_t attention_size =
      static_cast<std::size_t>(kNumHeads) * kHeadDim;
  const std::size_t kv_size = static_cast<std::size_t>(kNumKvHeads) * kHeadDim;
  const std::size_t total_kv =
      static_cast<std::size_t>(kNumKvHeads) * kMaxContext * kHeadDim;
  const std::size_t q_elements = batch_size * attention_size;

  Rng rng(0xC0FFEEU ^ (start_pos * 31U) ^
          static_cast<std::uint32_t>(batch_size));
  std::vector<float> h_q(q_elements);
  std::vector<float> h_gate(q_elements);
  std::vector<float> h_k(batch_size * kv_size);
  std::vector<float> h_v(batch_size * kv_size);
  for (auto& x : h_q) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_gate) {
    x = rng.Uniform(-2.0F, 2.0F);
  }
  for (auto& x : h_k) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  for (auto& x : h_v) {
    x = rng.Uniform(-1.0F, 1.0F);
  }
  // The prefix keys live in the FP32 cache; the tiled launcher mirrors them to
  // FP16 when start_pos > 0, which is also what AOTriton reads.
  std::vector<float> h_cache(total_kv * 2);
  for (auto& x : h_cache) {
    x = rng.Uniform(-1.0F, 1.0F);
  }

  Buffers b;
  HIP_CHECK(hipMalloc(&b.q, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.gate, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.k, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.v, batch_size * kv_size * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.kv_cache, total_kv * 2 * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.kv_f16, total_kv * 2 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&b.out, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.q_f16, q_elements * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&b.prefix_f16, q_elements * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&b.diag_out, q_elements * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.lse_prefix, batch_size * kNumHeads * sizeof(float)));
  HIP_CHECK(hipMalloc(&b.lse_diag, batch_size * kNumHeads * sizeof(float)));

  HIP_CHECK(hipMemcpy(b.q, h_q.data(), q_elements * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.gate, h_gate.data(), q_elements * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.k, h_k.data(), batch_size * kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.v, h_v.data(), batch_size * kv_size * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b.kv_cache, h_cache.data(), total_kv * 2 * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(b.kv_f16, 0, total_kv * 2 * sizeof(std::uint16_t)));

  auto* v_cache = b.kv_cache + total_kv;
  auto* kv_f16_v = static_cast<std::uint16_t*>(b.kv_f16) + total_kv;

  // Reference: the unsplit tiled kernel over the whole visible range. This
  // test supplies both representations, so the call packs both and refreshes
  // the FP16 prefix.
  HIP_CHECK(hipMemset(b.out, 0, q_elements * sizeof(float)));
  if (!gufo::hip::LaunchBatchedAttentionTile(
          b.q, b.k, b.v, b.gate, b.kv_cache, v_cache, b.kv_f16, kv_f16_v, b.out,
          /*layer_idx=*/0, start_pos, batch_size, kMaxContext, kNumHeads,
          kNumKvHeads, kHeadDim)) {
    std::cerr << "tiled attention rejected the production shape\n";
    std::abort();
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> want(q_elements);
  HIP_CHECK(hipMemcpy(want.data(), b.out, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Candidate: diagonal through the tiled kernel with its partial statistics,
  // prefix through AOTriton, merged by log-sum-exp. Both supplied KV
  // representations are already packed by the reference call above, so
  // suppress the write.
  gufo::hip::LaunchConvertQueriesToHalf(b.q, b.q_f16, q_elements);
  HIP_CHECK(hipMemset(b.diag_out, 0, q_elements * sizeof(float)));
  if (!gufo::hip::LaunchBatchedAttentionTile(
          b.q, b.k, b.v, b.gate, b.kv_cache, v_cache, b.kv_f16, kv_f16_v,
          b.diag_out, /*layer_idx=*/0, start_pos, batch_size, kMaxContext,
          kNumHeads, kNumKvHeads, kHeadDim, nullptr, b.lse_diag, start_pos,
          /*skip_kv_write=*/true)) {
    std::cerr << "tiled attention rejected the diagonal half\n";
    std::abort();
  }
  if (!gufo::hip::LaunchQwenAotritonPrefixAttention(
          static_cast<const __half*>(b.q_f16), b.kv_f16, kv_f16_v,
          static_cast<__half*>(b.prefix_f16), b.lse_prefix, batch_size,
          start_pos, kNumHeads, kNumKvHeads, kHeadDim)) {
    std::cerr << "AOTriton rejected the prefix shape\n";
    std::abort();
  }
  HIP_CHECK(hipMemset(b.out, 0, q_elements * sizeof(float)));
  gufo::hip::LaunchMergeSplitAttention(b.prefix_f16, b.lse_prefix, b.diag_out,
                                       b.lse_diag, b.gate, b.out, batch_size,
                                       kNumHeads, kHeadDim);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(q_elements);
  HIP_CHECK(hipMemcpy(got.data(), b.out, q_elements * sizeof(float),
                      hipMemcpyDeviceToHost));

  // Both halves carry FP16 operands, and the prefix result itself round-trips
  // through FP16, so the tolerance is set by FP16 output precision rather than
  // by the merge algebra.
  Compare("split vs unsplit", got, want, 5e-3);

  HIP_CHECK(hipFree(b.q));
  HIP_CHECK(hipFree(b.gate));
  HIP_CHECK(hipFree(b.k));
  HIP_CHECK(hipFree(b.v));
  HIP_CHECK(hipFree(b.kv_cache));
  HIP_CHECK(hipFree(b.kv_f16));
  HIP_CHECK(hipFree(b.out));
  HIP_CHECK(hipFree(b.q_f16));
  HIP_CHECK(hipFree(b.prefix_f16));
  HIP_CHECK(hipFree(b.diag_out));
  HIP_CHECK(hipFree(b.lse_prefix));
  HIP_CHECK(hipFree(b.lse_diag));
}

}  // namespace

#endif  // defined(ENGINE_ENABLE_HIP)

int main() {
#if defined(ENGINE_ENABLE_HIP)
  const int device_status =
      gufo::test::GateHipDevice(gufo::test::HipDeviceRequirement::kOptional,
                                "Qwen split prefill attention ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  RunCase(1024, 256);
  RunCase(4096, 512);
  // A depth that is not a multiple of the 64-key tile, so the diagonal half
  // starts mid-tile.
  RunCase(1500, 128);

  std::cout << "Qwen split prefill attention ops test passed on gfx1151.\n";
  return 0;
#else
  std::cout
      << "HIP disabled, skipping Qwen split prefill attention ops test.\n";
  return 77;
#endif
}
