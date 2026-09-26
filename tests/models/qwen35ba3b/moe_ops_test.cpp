// Synthetic op-level checks for the qwen35moe GPU path: router top-k,
// routed expert projections (MMQ and the per-slot fallback), the shared
// expert and the combine epilogue, each compared against the CPU reference
// on fixed pseudo-random weights. No model file required.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/model_config.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/hip/ops/moe.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#include "src/models/qwen/modules/moe.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"
#include "tests/models/qwen/hip/support/device.hpp"

#include "qfn_mmq.h"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::models::QwenTensorRef;
using Q8_0Block = gufo::quant::block_q8_0;

std::uint32_t seed = 20260926U;
std::uint32_t Rnd() {
  seed = seed * 1664525U + 1013904223U;
  return seed;
}
float RndFloat(float lo, float hi) {
  const float u = static_cast<float>(Rnd() & 0xFFFFU) / 65535.0F;
  return lo + u * (hi - lo);
}

std::uint16_t FloatToFp16Bits(float v) {
  // Round-to-nearest fp32 -> fp16 for the test's block scales.
  const std::uint32_t bits = [&] {
    std::uint32_t b;
    std::memcpy(&b, &v, sizeof(b));
    return b;
  }();
  const std::uint32_t sign = (bits >> 16) & 0x8000U;
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xFF) - 127;
  std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent > 15) {
    return static_cast<std::uint16_t>(sign | 0x7BFFU);
  }
  if (exponent < -14) {
    return static_cast<std::uint16_t>(sign);
  }
  std::uint16_t out = 0;
  if (exponent >= -14) {
    out = static_cast<std::uint16_t>((exponent + 15) << 10);
  }
  const std::uint32_t rounded =
      (mantissa >> 13) + ((mantissa >> 12) & 1U);
  return static_cast<std::uint16_t>(sign | out | rounded);
}

/// Quantizes a dense row-major matrix into Q8_0 blocks.
std::vector<Q8_0Block> QuantizeQ8_0(const float* w, std::size_t rows,
                                    std::size_t cols) {
  const std::size_t blocks_per_row = cols / 32;
  std::vector<Q8_0Block> out(rows * blocks_per_row);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0; b < blocks_per_row; ++b) {
      float max_abs = 1e-8F;
      for (int i = 0; i < 32; ++i) {
        max_abs = std::max(max_abs, std::fabs(w[r * cols + b * 32 + i]));
      }
      const float scale = max_abs / 127.0F;
      out[r * blocks_per_row + b].d = FloatToFp16Bits(scale);
      for (int i = 0; i < 32; ++i) {
        const float v = w[r * cols + b * 32 + i] / scale;
        out[r * blocks_per_row + b].qs[i] =
            static_cast<std::int8_t>(std::lround(v));
      }
    }
  }
  return out;
}

/// Decodes one row of a Q8_0 matrix.
float DequantRowDot(const std::vector<Q8_0Block>& q, std::size_t row,
                    std::size_t cols, const std::vector<float>& x) {
  const std::size_t blocks_per_row = cols / 32;
  float sum = 0.0F;
  for (std::size_t b = 0; b < blocks_per_row; ++b) {
    const Q8_0Block& blk = q[row * blocks_per_row + b];
    const float d = gufo::quant::Fp16ToFloat(blk.d);
    float dot = 0.0F;
    for (int i = 0; i < 32; ++i) {
      dot += static_cast<float>(blk.qs[i]) * x[b * 32 + i];
    }
    sum += d * dot;
  }
  return sum;
}

void Expect(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Fixture {
  static constexpr std::size_t kHidden = 2048;
  static constexpr std::size_t kExpertFf = 512;
  static constexpr std::size_t kSharedFf = 512;
  static constexpr std::uint32_t kExperts = 256;
  static constexpr std::uint32_t kUsed = 8;

  std::vector<float> x;              // [hidden]
  std::vector<float> router;         // [experts x hidden]
  std::vector<float> shexp_gate_inp; // [hidden]
  std::vector<float> gate_src;       // [experts x ff x hidden]
  std::vector<float> up_src;         // [experts x ff x hidden]
  std::vector<float> down_src;       // [experts x hidden x ff]
  std::vector<float> shexp_src;      // dense shexp trio source
  std::vector<Q8_0Block> gate_q, up_q, down_q, shexp_gate_q, shexp_up_q,
      shexp_down_q;

  gufo::core::ModelConfig config;
  std::vector<gufo::models::QwenLayerWeights> layers;

  Fixture()
      : x(kHidden),
        router(kExperts * kHidden),
        shexp_gate_inp(kHidden),
        gate_src(kExperts * kExpertFf * kHidden),
        up_src(kExperts * kExpertFf * kHidden),
        down_src(kExperts * kHidden * kExpertFf),
        shexp_src(3 * kSharedFf * kHidden) {
    for (auto& v : x) v = RndFloat(-1.0F, 1.0F);
    for (auto& v : router) v = RndFloat(-1.0F, 1.0F) * 0.05F;
    for (auto& v : shexp_gate_inp) v = RndFloat(-1.0F, 1.0F) * 0.05F;
    for (auto& v : gate_src) v = RndFloat(-1.0F, 1.0F) * 0.1F;
    for (auto& v : up_src) v = RndFloat(-1.0F, 1.0F) * 0.1F;
    for (auto& v : down_src) v = RndFloat(-1.0F, 1.0F) * 0.1F;
    for (auto& v : shexp_src) v = RndFloat(-1.0F, 1.0F) * 0.1F;
    gate_q = QuantizeQ8_0(gate_src.data(), kExperts * kExpertFf, kHidden);
    up_q = QuantizeQ8_0(up_src.data(), kExperts * kExpertFf, kHidden);
    down_q = QuantizeQ8_0(down_src.data(), kExperts * kHidden, kExpertFf);
    shexp_gate_q = QuantizeQ8_0(shexp_src.data(), kSharedFf, kHidden);
    shexp_up_q = QuantizeQ8_0(shexp_src.data() + kSharedFf * kHidden,
                              kSharedFf, kHidden);
    shexp_down_q =
        QuantizeQ8_0(shexp_src.data() + 2 * kSharedFf * kHidden, kHidden,
                     kSharedFf);

    config.hidden_size = kHidden;
    config.expert_count = kExperts;
    config.expert_used_count = kUsed;
    config.expert_ff_length = kExpertFf;
    config.expert_shared_ff_length = kSharedFf;
    config.intermediate_size = kSharedFf;
    layers.resize(1);
    auto& l = layers.front();
    l.ffn_gate_inp = {.data = router.data(),
                      .type = GgmlType::kF32,
                      .num_elements = router.size()};
    l.ffn_gate_inp_shexp = {.data = shexp_gate_inp.data(),
                            .type = GgmlType::kF32,
                            .num_elements = shexp_gate_inp.size()};
    l.ffn_gate_exps = {.data = gate_q.data(),
                       .type = GgmlType::kQ8_0,
                       .num_elements = kExperts * kExpertFf * kHidden};
    l.ffn_up_exps = {.data = up_q.data(),
                     .type = GgmlType::kQ8_0,
                     .num_elements = kExperts * kExpertFf * kHidden};
    l.ffn_down_exps = {.data = down_q.data(),
                       .type = GgmlType::kQ8_0,
                       .num_elements = kExperts * kHidden * kExpertFf};
    l.ffn_gate_shexp = {.data = shexp_gate_q.data(),
                        .type = GgmlType::kQ8_0,
                        .num_elements = kSharedFf * kHidden};
    l.ffn_up_shexp = {.data = shexp_up_q.data(),
                      .type = GgmlType::kQ8_0,
                      .num_elements = kSharedFf * kHidden};
    l.ffn_down_shexp = {.data = shexp_down_q.data(),
                        .type = GgmlType::kQ8_0,
                        .num_elements = kHidden * kSharedFf};
  }
};

void CpuRouterTopK(const std::vector<float>& logits, std::uint32_t n_experts,
                   std::uint32_t k, std::vector<std::int32_t>& ids,
                   std::vector<float>& weights) {
  std::vector<float> probs(n_experts);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (std::uint32_t e = 0; e < n_experts; ++e)
    max_logit = std::max(max_logit, logits[e]);
  float denom = 0.0F;
  for (std::uint32_t e = 0; e < n_experts; ++e) {
    probs[e] = std::exp(logits[e] - max_logit);
    denom += probs[e];
  }
  for (std::uint32_t e = 0; e < n_experts; ++e) probs[e] /= denom;
  ids.assign(k, 0);
  weights.assign(k, 0.0F);
  for (std::uint32_t slot = 0; slot < k; ++slot) {
    float best = -1.0F;
    std::uint32_t index = n_experts;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      if (probs[e] > best) {
        best = probs[e];
        index = e;
      }
    }
    ids[slot] = static_cast<std::int32_t>(index);
    weights[slot] = best;
    probs[index] = -1.0F;
  }
  float sum = 0.0F;
  for (std::uint32_t slot = 0; slot < k; ++slot) sum += weights[slot];
  sum = std::max(sum, 6.103515625e-5F);
  for (std::uint32_t slot = 0; slot < k; ++slot) weights[slot] /= sum;
}

void CheckRouterTopK(const Fixture& f, hipStream_t stream) {
  std::vector<float> logits(f.kExperts);
  for (auto& v : logits) v = RndFloat(-3.0F, 3.0F);
  gufo::test::DeviceBuffer<float> d_logits(logits);
  gufo::test::DeviceBuffer<std::int32_t> d_ids(
      std::vector<std::int32_t>(f.kUsed, -1));
  gufo::test::DeviceBuffer<float> d_weights(
      std::vector<float>(f.kUsed, -1.0F));
  gufo::hip::LaunchMoeRouterTopK(d_logits.data(), f.kExperts, d_ids.data(),
                                 d_weights.data(), 1, f.kExperts, f.kUsed,
                                 stream);
  const auto h_ids = d_ids.CopyToHost();
  const auto h_weights = d_weights.CopyToHost();
  std::vector<std::int32_t> ref_ids;
  std::vector<float> ref_weights;
  CpuRouterTopK(logits, f.kExperts, f.kUsed, ref_ids, ref_weights);
  for (std::uint32_t s = 0; s < f.kUsed; ++s) {
    Expect(h_ids[s] == ref_ids[s],
           "top-k id mismatch at slot " + std::to_string(s) + ": gpu " +
               std::to_string(h_ids[s]) + " ref " +
               std::to_string(ref_ids[s]));
    Expect(std::fabs(h_weights[s] - ref_weights[s]) < 1e-5F,
           "top-k weight mismatch at slot " + std::to_string(s));
  }
  std::cout << "router top-k: OK\n";
}

void CheckDecodeMoe(const Fixture& f, hipStream_t stream,
                    bool force_fallback) {
  const auto view =
      gufo::models::qwen::MakeMoeView(f.layers.front(), f.config);

  gufo::test::DeviceBuffer<float> d_x(f.x);
  gufo::test::DeviceBuffer<float> d_router_w(f.router);
  gufo::test::DeviceBuffer<float> d_shexp_gi(f.shexp_gate_inp);
  gufo::test::DeviceBuffer<float> d_logits(
      std::vector<float>(f.kExperts, 0.0F));
  gufo::test::DeviceBuffer<float> d_shexp_gate(std::vector<float>(1, 0.0F));
  gufo::test::DeviceBuffer<std::int32_t> d_ids(
      std::vector<std::int32_t>(f.kUsed, 0));
  gufo::test::DeviceBuffer<float> d_weights(
      std::vector<float>(f.kUsed, 0.0F));
  gufo::test::DeviceBuffer<float> d_gate_e(
      std::vector<float>(f.kUsed * f.kExpertFf, 0.0F));
  gufo::test::DeviceBuffer<float> d_down_e(
      std::vector<float>(f.kUsed * f.kHidden, 0.0F));
  gufo::test::DeviceBuffer<float> d_shexp_act(
      std::vector<float>(f.kSharedFf, 0.0F));
  gufo::test::DeviceBuffer<float> d_shexp_out(
      std::vector<float>(f.kHidden, 0.0F));
  gufo::test::DeviceBuffer<float> d_out(std::vector<float>(f.kHidden, 0.0F));
  gufo::test::DeviceBuffer<Q8_0Block> d_gate_q(f.gate_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_up_q(f.up_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_down_q(f.down_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_shexp_gate_q(f.shexp_gate_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_shexp_up_q(f.shexp_up_q);
  gufo::test::DeviceBuffer<Q8_0Block> d_shexp_down_q(f.shexp_down_q);

  // Router logits and the shared gate scalar.
  gufo::hip::LaunchGEMV(d_router_w.data(), GgmlType::kF32, d_x.data(),
                        d_logits.data(), f.kExperts, f.kHidden, stream);
  gufo::hip::LaunchGEMV(d_shexp_gi.data(), GgmlType::kF32, d_x.data(),
                        d_shexp_gate.data(), 1, f.kHidden, stream);
  gufo::hip::LaunchMoeRouterTopK(d_logits.data(), f.kExperts, d_ids.data(),
                                 d_weights.data(), 1, f.kExperts, f.kUsed,
                                 stream);
  const auto gpu_logits = d_logits.CopyToHost();
  for (std::uint32_t e = 0; e < f.kExperts; ++e) {
    float ref = 0.0F;
    for (std::size_t i = 0; i < f.kHidden; ++i)
      ref += f.router[e * f.kHidden + i] * f.x[i];
    Expect(std::fabs(gpu_logits[e] - ref) < 1e-2F,
           "router logit mismatch at expert " + std::to_string(e) + ": gpu " +
               std::to_string(gpu_logits[e]) + " ref " + std::to_string(ref));
  }
  const auto h_ids = d_ids.CopyToHost();
  std::vector<std::int32_t> ref_ids;
  std::vector<float> ref_weights;
  CpuRouterTopK(gpu_logits, f.kExperts, f.kUsed, ref_ids, ref_weights);
  for (std::uint32_t s = 0; s < f.kUsed; ++s) {
    Expect(h_ids[s] == ref_ids[s], "decode top-k id mismatch at slot " +
                                       std::to_string(s) + ": gpu " +
                                       std::to_string(h_ids[s]) + " ref " +
                                       std::to_string(ref_ids[s]));
  }
  std::cout << "decode router: OK\n";

  // Shared expert.
  gufo::hip::LaunchFusedSwiGLUGEMV(
      d_shexp_gate_q.data(), GgmlType::kQ8_0, d_shexp_up_q.data(),
      GgmlType::kQ8_0, d_x.data(), d_shexp_act.data(), f.kSharedFf, f.kHidden,
      stream);
  gufo::hip::LaunchGEMV(d_shexp_down_q.data(), GgmlType::kQ8_0,
                        d_shexp_act.data(), d_shexp_out.data(), f.kHidden,
                        f.kSharedFf, stream);

  // Routed experts.
  if (!force_fallback) {
    const int rc = qfn_mmq_moe_gated_vec(
        static_cast<int>(GgmlType::kQ8_0), d_gate_q.data(), d_up_q.data(),
        d_x.data(), d_ids.data(), d_gate_e.data(),
        static_cast<int>(f.kExpertFf), static_cast<int>(f.kHidden), 1,
        static_cast<int>(f.kExperts), static_cast<int>(f.kUsed), stream);
    Expect(rc == 0, "qfn_mmq_moe_gated_vec failed");
  } else {
    gufo::hip::LaunchMoeSlotSwigluGemv(
        d_gate_q.data(), GgmlType::kQ8_0, d_up_q.data(), GgmlType::kQ8_0,
        d_x.data(), d_ids.data(), d_gate_e.data(), f.kExpertFf, f.kHidden,
        f.kUsed, f.kUsed, stream);
  }
  const auto gate_e = d_gate_e.CopyToHost();
  for (std::uint32_t s = 0; s < f.kUsed; ++s) {
    const std::size_t expert = static_cast<std::size_t>(h_ids[s]);
    for (std::size_t r = 0; r < f.kExpertFf; ++r) {
      const float g = DequantRowDot(f.gate_q, expert * f.kExpertFf + r,
                                    f.kHidden, f.x);
      const float u = DequantRowDot(f.up_q, expert * f.kExpertFf + r,
                                    f.kHidden, f.x);
      const float ref = (g / (1.0F + std::exp(-g))) * u;
      const float got =
          gate_e[static_cast<std::size_t>(s) * f.kExpertFf + r];
      Expect(std::fabs(got - ref) < 0.02F + 0.05F * std::fabs(ref),
             "expert activation mismatch slot " + std::to_string(s) +
                 " row " + std::to_string(r) + ": gpu " + std::to_string(got) +
                 " ref " + std::to_string(ref));
    }
  }
  std::cout << (force_fallback ? "fallback" : "mmq") << " gated experts: OK\n";

  if (!force_fallback) {
    const int rc = qfn_mmq_moe_vec(
        static_cast<int>(GgmlType::kQ8_0), d_down_q.data(), d_gate_e.data(),
        d_ids.data(), d_down_e.data(), static_cast<int>(f.kHidden),
        static_cast<int>(f.kExpertFf), static_cast<int>(f.kUsed),
        static_cast<int>(f.kExperts), 1, stream, nullptr, nullptr);
    Expect(rc == 0, "qfn_mmq_moe_vec failed");
  } else {
    gufo::hip::LaunchMoeSlotGemv(d_down_q.data(), GgmlType::kQ8_0,
                                 d_gate_e.data(), d_ids.data(),
                                 d_down_e.data(), f.kHidden, f.kExpertFf,
                                 f.kUsed, 1, stream);
  }

  gufo::hip::LaunchMoeEpilogue(d_down_e.data(), d_weights.data(),
                               d_shexp_out.data(), d_shexp_gate.data(),
                               d_out.data(), 1, f.kUsed, f.kHidden, stream);
  const auto gpu_out = d_out.CopyToHost();

  gufo::models::QwenScratchArena arena(f.config);
  std::vector<float> cpu_out(f.kHidden, 0.0F);
  const gufo::models::qwen::CpuModuleContext ctx;
  gufo::models::qwen::MoeForward(ctx, view, f.x, arena, cpu_out);

  double max_abs = 0.0;
  double sum_sq = 0.0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < f.kHidden; ++i) {
    const double d = std::fabs(static_cast<double>(gpu_out[i]) - cpu_out[i]);
    if (d > max_abs) {
      max_abs = d;
      worst = i;
    }
    sum_sq += d * d;
  }
  std::cout << "decode MoE (" << (force_fallback ? "fallback" : "mmq")
            << ") vs CPU: max_abs=" << max_abs << " rmse="
            << std::sqrt(sum_sq / f.kHidden) << " worst_idx=" << worst
            << " gpu=" << gpu_out[worst] << " cpu=" << cpu_out[worst] << "\n";
  Expect(max_abs < 0.02 + 0.05 * std::fabs(cpu_out[worst]),
         "decode MoE output mismatch");
}


std::uint16_t FloatToBf16Bits(float v) {
  std::uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  u += 0x7FFFU + ((u >> 16) & 1U);
  return static_cast<std::uint16_t>(u >> 16);
}

/// Grouped BF16 expert GEMM (prefill) against the per-slot GEMV, which reads
/// the same BF16 weights with full FP32 activations. Routing is skewed so hot
/// experts span several row tiles while others receive no slot.
void CheckGroupedBf16(hipStream_t stream) {
  constexpr std::uint32_t kExperts = 64;
  constexpr std::uint32_t kUsed = 8;
  struct Shape {
    const char* name;
    std::size_t m;
    std::size_t k;
    bool per_slot_input;
  };
  for (const Shape shape : {Shape{"gate/up", 512, 2048, false},
                            Shape{"down", 2048, 512, true}}) {
    Expect(gufo::hip::IsMoeGroupedBf16GemmSupported(shape.m, shape.k,
                                                    kExperts),
           "grouped BF16 shape rejected");
    std::vector<std::uint16_t> w(static_cast<std::size_t>(kExperts) *
                                 shape.m * shape.k);
    for (auto& v : w) {
      v = FloatToBf16Bits(RndFloat(-0.05F, 0.05F));
    }
    gufo::test::DeviceBuffer<std::uint16_t> d_w(w);
    for (const std::uint32_t tokens : {1U, 7U, 100U, 512U}) {
      const std::uint32_t slots = tokens * kUsed;
      std::vector<std::int32_t> ids(slots);
      for (std::uint32_t s = 0; s < slots; ++s) {
        ids[s] = static_cast<std::int32_t>(
            (Rnd() % 3 == 0) ? (s % 3) : (3 + Rnd() % (kExperts / 2)));
      }
      const std::size_t x_rows = shape.per_slot_input ? slots : tokens;
      std::vector<float> x(x_rows * shape.k);
      for (auto& v : x) {
        v = RndFloat(-1.0F, 1.0F);
      }
      gufo::test::DeviceBuffer<std::int32_t> d_ids(ids);
      gufo::test::DeviceBuffer<float> d_x(x);
      const std::vector<float> nan_fill(
          static_cast<std::size_t>(slots) * shape.m,
          std::numeric_limits<float>::quiet_NaN());
      gufo::test::DeviceBuffer<float> d_ref(nan_fill);
      gufo::test::DeviceBuffer<float> d_got(nan_fill);
      gufo::test::DeviceBuffer<std::int32_t> d_sorted(slots);
      gufo::test::DeviceBuffer<std::int32_t> d_tiles(slots);
      gufo::test::DeviceBuffer<std::int32_t> d_bounds(kExperts + 1);
      const gufo::hip::MoeGroupedScratch scratch{
          .sorted_slots = d_sorted.data(),
          .tiles = d_tiles.data(),
          .expert_bounds = d_bounds.data()};
      const std::uint32_t divisor = shape.per_slot_input ? 1U : kUsed;

      gufo::hip::LaunchMoeSlotGemv(d_w.data(), GgmlType::kBF16, d_x.data(),
                                   d_ids.data(), d_ref.data(), shape.m,
                                   shape.k, slots, divisor, stream);
      gufo::hip::LaunchMoeGroupSlots(d_ids.data(), slots, kExperts, scratch,
                                     stream);
      gufo::hip::LaunchMoeGroupedBf16Gemm(d_w.data(), d_x.data(),
                                          d_got.data(), shape.m, shape.k,
                                          slots, kExperts, divisor, scratch,
                                          stream);
      HIP_CHECK(hipStreamSynchronize(stream));
      const auto ref = d_ref.CopyToHost();
      const auto got = d_got.CopyToHost();

      double max_abs = 0.0;
      double magnitude = 0.0;
      std::size_t non_finite = 0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) {
          ++non_finite;
          continue;
        }
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) -
                                              static_cast<double>(ref[i])));
        magnitude = std::max(magnitude, std::fabs(static_cast<double>(ref[i])));
      }
      const double rel = magnitude > 0.0 ? max_abs / magnitude : max_abs;
      std::cout << "grouped BF16 " << shape.name << " tokens=" << tokens
                << ": max_abs=" << max_abs << " rel=" << rel
                << " non_finite=" << non_finite << "\n";
      Expect(non_finite == 0, "grouped BF16 GEMM left slots unwritten");
      // Only the activations are rounded to BF16 (8-bit mantissa).
      Expect(rel < 1e-2, "grouped BF16 GEMM disagrees with the slot GEMV");

      // The expert-major slot order is built with atomics; the result must
      // not depend on it.
      gufo::test::DeviceBuffer<float> d_again(nan_fill);
      gufo::hip::LaunchMoeGroupSlots(d_ids.data(), slots, kExperts, scratch,
                                     stream);
      gufo::hip::LaunchMoeGroupedBf16Gemm(d_w.data(), d_x.data(),
                                          d_again.data(), shape.m, shape.k,
                                          slots, kExperts, divisor, scratch,
                                          stream);
      HIP_CHECK(hipStreamSynchronize(stream));
      const auto again = d_again.CopyToHost();
      Expect(std::memcmp(again.data(), got.data(),
                         got.size() * sizeof(float)) == 0,
             "grouped BF16 GEMM is not deterministic");
    }
  }
  std::cout << "grouped BF16 experts: OK\n";
}


/// Dense mode of the same WMMA kernel (BF16 dense projections in prefill),
/// against the per-slot GEMV with every row routed to one matrix.
void CheckDenseBf16(hipStream_t stream) {
  constexpr std::size_t kM = 4096;
  constexpr std::size_t kK = 2048;
  std::vector<std::uint16_t> w(kM * kK);
  for (auto& v : w) {
    v = FloatToBf16Bits(RndFloat(-0.05F, 0.05F));
  }
  gufo::test::DeviceBuffer<std::uint16_t> d_w(w);
  for (const std::uint32_t rows : {1U, 100U, 512U}) {
    std::vector<float> x(rows * kK);
    for (auto& v : x) {
      v = RndFloat(-1.0F, 1.0F);
    }
    gufo::test::DeviceBuffer<float> d_x(x);
    gufo::test::DeviceBuffer<std::int32_t> d_ids(
        std::vector<std::int32_t>(rows, 0));
    const std::vector<float> nan_fill(rows * kM,
                                      std::numeric_limits<float>::quiet_NaN());
    gufo::test::DeviceBuffer<float> d_ref(nan_fill);
    gufo::test::DeviceBuffer<float> d_got(nan_fill);
    gufo::hip::LaunchMoeSlotGemv(d_w.data(), GgmlType::kBF16, d_x.data(),
                                 d_ids.data(), d_ref.data(), kM, kK, rows, 1,
                                 stream);
    gufo::hip::LaunchBf16WmmaGemm(d_w.data(), d_x.data(), d_got.data(), rows,
                                  kM, kK, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto ref = d_ref.CopyToHost();
    const auto got = d_got.CopyToHost();
    double max_abs = 0.0;
    double magnitude = 0.0;
    std::size_t non_finite = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
      if (!std::isfinite(got[i])) {
        ++non_finite;
        continue;
      }
      max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) -
                                            static_cast<double>(ref[i])));
      magnitude = std::max(magnitude, std::fabs(static_cast<double>(ref[i])));
    }
    const double rel = magnitude > 0.0 ? max_abs / magnitude : max_abs;
    std::cout << "dense BF16 WMMA rows=" << rows << ": max_abs=" << max_abs
              << " rel=" << rel << " non_finite=" << non_finite << "\n";
    Expect(non_finite == 0, "dense BF16 WMMA GEMM left rows unwritten");
    Expect(rel < 1e-2, "dense BF16 WMMA GEMM disagrees with the GEMV");
  }
  std::cout << "dense BF16 WMMA: OK\n";
}



/// Flash-Next's routed F16 expert GEMM on this model's Q8_0 expert shapes,
/// against the per-slot GEMV (FP32 activations, same weights). The difference
/// is the F16 rounding of the activation rows.
void CheckRoutedF16(hipStream_t stream) {
  namespace routed = gufo::models::qwen38_flash_next::rocm;
  constexpr std::uint32_t kExperts = 64;
  constexpr std::uint32_t kUsed = 8;
  struct Shape {
    const char* name;
    std::size_t m;
    std::size_t k;
    bool per_slot_input;
  };
  for (const Shape shape : {Shape{"gate/up", 512, 2048, false},
                            Shape{"down", 2048, 512, true}}) {
    std::vector<float> w_src(static_cast<std::size_t>(kExperts) * shape.m *
                             shape.k);
    for (auto& v : w_src) {
      v = RndFloat(-0.05F, 0.05F);
    }
    const auto w_q = QuantizeQ8_0(w_src.data(), kExperts * shape.m, shape.k);
    gufo::test::DeviceBuffer<Q8_0Block> d_w(w_q);
    for (const std::uint32_t tokens : {1U, 7U, 100U, 512U}) {
      const std::uint32_t slots = tokens * kUsed;
      std::vector<std::int32_t> ids(slots);
      for (std::uint32_t t = 0; t < tokens; ++t) {
        // Distinct experts per token, skewed toward the first few.
        for (std::uint32_t s = 0; s < kUsed; ++s) {
          ids[t * kUsed + s] = static_cast<std::int32_t>(
              (Rnd() % 3 == 0) ? s : (kUsed + (t + s * 5) % (kExperts - kUsed)));
        }
      }
      const std::size_t x_rows = shape.per_slot_input ? slots : tokens;
      std::vector<float> x(x_rows * shape.k);
      for (auto& v : x) {
        v = RndFloat(-1.0F, 1.0F);
      }
      gufo::test::DeviceBuffer<std::int32_t> d_ids(ids);
      gufo::test::DeviceBuffer<float> d_x(x);
      const std::vector<float> nan_fill(
          static_cast<std::size_t>(slots) * shape.m,
          std::numeric_limits<float>::quiet_NaN());
      gufo::test::DeviceBuffer<float> d_ref(nan_fill);
      gufo::test::DeviceBuffer<float> d_got(nan_fill);
      gufo::hip::LaunchMoeSlotGemv(d_w.data(), GgmlType::kQ8_0, d_x.data(),
                                   d_ids.data(), d_ref.data(), shape.m,
                                   shape.k, slots,
                                   shape.per_slot_input ? 1U : kUsed, stream);

      gufo::test::DeviceBuffer<std::uint32_t> d_counts(kExperts);
      const std::size_t rows = gufo::hip::RoutedRows(slots, kExperts);
      gufo::test::DeviceBuffer<std::int32_t> d_bounds(kExperts + 1);
      gufo::test::DeviceBuffer<std::int32_t> d_cursors(kExperts);
      gufo::test::DeviceBuffer<std::int32_t> d_rows_token(rows);
      gufo::test::DeviceBuffer<std::int32_t> d_rows_slot(rows);
      gufo::test::DeviceBuffer<std::uint16_t> d_x_half(x_rows * shape.k);
      routed::ExpertCounts(d_ids.data(), d_counts.data(), tokens, kExperts,
                           kUsed, stream);
      const auto counts = d_counts.CopyToHost();
      for (const std::uint32_t tile_rows : {16U, 48U}) {
        std::vector<std::int32_t> tiles;
        for (std::uint32_t e = 0; e < kExperts; ++e) {
          const std::uint32_t padded = (counts[e] + 15U) / 16U * 16U;
          for (std::uint32_t j = 0; j * tile_rows < padded; ++j) {
            tiles.push_back(static_cast<std::int32_t>(e | (j << 16)));
          }
        }
        gufo::test::DeviceBuffer<std::int32_t> d_tiles(tiles);
        d_got.CopyFrom(std::span<const float>(nan_fill));
        routed::RoutedCompact(d_ids.data(), d_counts.data(), d_bounds.data(),
                              d_cursors.data(), d_rows_token.data(),
                              d_rows_slot.data(), tokens, kUsed, kExperts,
                              stream);
        routed::NarrowActivations(d_x.data(), d_x_half.data(), false,
                                  x_rows * shape.k, stream);
        const auto* rows_in = shape.per_slot_input ? d_rows_slot.data()
                                                   : d_rows_token.data();
        Expect(routed::RoutedF16Gemm(
                   d_w.data(), routed::WeightType::kQ8_0,
                   reinterpret_cast<const __half*>(d_x_half.data()),
                   d_tiles.data(), static_cast<std::uint32_t>(tiles.size()),
                   tile_rows, d_bounds.data(), rows_in, d_rows_slot.data(),
                   nullptr, d_got.data(), nullptr, shape.m, shape.k, stream),
               "routed F16 GEMM rejected the shape");
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto ref = d_ref.CopyToHost();
        const auto got = d_got.CopyToHost();
        double max_abs = 0.0;
        double magnitude = 0.0;
        std::size_t non_finite = 0;
        for (std::size_t i = 0; i < ref.size(); ++i) {
          if (!std::isfinite(got[i])) {
            ++non_finite;
            continue;
          }
          max_abs = std::max(max_abs, std::fabs(static_cast<double>(got[i]) -
                                                static_cast<double>(ref[i])));
          magnitude =
              std::max(magnitude, std::fabs(static_cast<double>(ref[i])));
        }
        const double rel = magnitude > 0.0 ? max_abs / magnitude : max_abs;
        std::cout << "routed F16 " << shape.name << " tokens=" << tokens
                  << " tile_rows=" << tile_rows << ": rel=" << rel
                  << " non_finite=" << non_finite << "\n";
        Expect(non_finite == 0, "routed F16 GEMM left slots unwritten");
        Expect(rel < 1e-2, "routed F16 GEMM disagrees with the slot GEMV");
      }
    }
  }
  std::cout << "routed F16 experts: OK\n";
}

}  // namespace

int main() {
  const int gate = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kRequired, "qwen35ba3b_moe_ops_test");
  if (gate != gufo::test::kHipTestSuccess) {
    return gate;
  }
  try {
    Expect(qfn_mmq_init(0) == 0, "qfn_mmq_init failed");
    Fixture f;
    CheckRouterTopK(f, nullptr);
    CheckDecodeMoe(f, nullptr, /*force_fallback=*/false);
    CheckDecodeMoe(f, nullptr, /*force_fallback=*/true);
    CheckGroupedBf16(nullptr);
    CheckDenseBf16(nullptr);
    CheckRoutedF16(nullptr);
  } catch (const std::exception& e) {
    std::cerr << "qwen35ba3b_moe_ops_test: " << e.what() << "\n";
    return 1;
  }
  std::cout << "qwen35ba3b_moe_ops_test: PASS\n";
  return 0;
}
