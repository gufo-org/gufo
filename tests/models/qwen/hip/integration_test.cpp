// GPU module-seam integration test.
//
// Wire a short multi-layer forward through the Qwen module seam on the HIP
// backend end-to-end: typed HipModuleContext, real synthetic weights pulled
// from build_synthetic_qwen_weights (unit-scaled, seeded), a few layers, and
// assert the pipeline produces finite, non-trivial logits and that the module
// functions return non-zero activations across layers.
//
// SCOPE NOTE (module seam reality, recorded so this gate is honest):
//   On the HIP module seam only NormForward / QuantGemm / ResidualAdd (and
//   FfnForward, which is BF16/quantized-only on HIP) carry a HIP branch.
//   EmbedForward / UnembedForward / SampleForward are CPU-only today, and
//   Attention/SSM are fused or not yet moduleized. Consequently a pure-GPU
//   decode that emits *real* logits through the module seam alone is not yet
//   complete. This L2 test therefore drives the HIP-complete modules across a
//   few layers (hidden -> intermediate -> hidden, mirroring the FFN body's
//   projection cycle) and emits a logit vector through a head QuantGemm, which
//   is the module-seam surface the composition layer will call. It exercises
//   the exact typed HIP overload path.

#include "src/core/hip/hip_utils.hpp"  // HIP_CHECK
#include "src/models/qwen/modules/modules.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"
#include "tests/testing/test_common.hpp"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

using strix::core::GgmlType;
using strix::core::ModelConfig;
using strix::models::QwenTensorRef;
using strix::models::qwen::HipModuleContext;
using strix::models::qwen::NormForward;
using strix::models::qwen::NormLayerView;
using strix::models::qwen::QuantGemm;
using strix::models::qwen::ResidualAdd;
using strix::models::qwen::build_synthetic_qwen_weights;
using strix::models::qwen::make_small_qwen_config;
using strix::test::make_random_tensor;
using strix::test::make_seeded_rng;

// Copies a host F32 QwenTensorRef into a freshly-allocated device buffer.
void ToDevice(const QwenTensorRef& host, float** dptr) {
  HIP_CHECK(hipMalloc(dptr, host.num_elements * sizeof(float)));
  HIP_CHECK(hipMemcpy(*dptr, host.data, host.num_elements * sizeof(float),
                      hipMemcpyHostToDevice));
}

float MaxAbs(std::span<const float> v) {
  float m = 0.0F;
  for (float x : v) m = std::max(m, std::abs(x));
  return m;
}

// Runs a handful of "layers" through the HIP module seam. Each layer applies
//   pre-norm(hidden) -> ffn_gate GEMV (hidden -> intermediate) ->
//   ffn_down GEMV (intermediate -> hidden) -> residual add(hidden),
//   then a head QuantGemm (output @ hidden -> logits). Returns logits.
std::vector<float> RunMultiLayerForward() {
  const ModelConfig config = make_small_qwen_config();
  static const auto sw = build_synthetic_qwen_weights(config);
  const auto& w = sw.weights;
  const std::size_t hidden = config.hidden_size;
  const std::size_t inter = config.intermediate_size;
  const std::size_t vocab = config.vocab_size;
  const std::size_t nlayers =
      std::min<std::size_t>(3, config.num_layers);

  const HipModuleContext ctx;  // null hipStream_t = default stream

  // Deterministic seed hidden state.
  std::mt19937 rng = make_seeded_rng(0x1BADB0B0u);
  std::vector<float> h_hidden = make_random_tensor(hidden, rng, -1.0F, 1.0F);

  float* d_hidden = nullptr;
  float* d_normed = nullptr;
  float* d_gate = nullptr;
  float* d_proj = nullptr;
  float* d_logits = nullptr;
  HIP_CHECK(hipMalloc(&d_hidden, hidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normed, hidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_gate, inter * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_proj, hidden * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_logits, vocab * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_hidden, h_hidden.data(), hidden * sizeof(float),
                      hipMemcpyHostToDevice));

  for (std::uint32_t layer_idx = 0; layer_idx < nlayers; ++layer_idx) {
    const auto& layer = w.layers[layer_idx];

    float* d_nw = nullptr;  // attn_norm (pre-norm) weight
    float* d_gw = nullptr;  // ffn_gate (inter, hidden)
    float* d_dw = nullptr;  // ffn_down (hidden, inter)
    ToDevice(layer.attn_norm, &d_nw);
    ToDevice(layer.ffn_gate, &d_gw);
    ToDevice(layer.ffn_down, &d_dw);

    // pre-norm: hidden -> normed (RMSNorm, F32)
    NormLayerView nv;
    nv.weight.data = d_nw;
    nv.weight.type = GgmlType::kF32;
    nv.weight.num_elements = hidden;
    nv.eps = 1e-6F;
    NormForward(ctx, nv, std::span<const float>(d_hidden, hidden),
                std::span<float>(d_normed, hidden));

    // gate = ffn_gate @ normed   (inter, hidden) -> (inter)
    QwenTensorRef gate_w{d_gw, GgmlType::kF32, inter * hidden};
    QuantGemm(ctx, gate_w, std::span<const float>(d_normed, hidden), inter,
              hidden, std::span<float>(d_gate, inter));

    // proj = ffn_down @ gate     (hidden, inter) -> (hidden)
    QwenTensorRef down_w{d_dw, GgmlType::kF32, hidden * inter};
    QuantGemm(ctx, down_w, std::span<const float>(d_gate, inter), hidden, inter,
              std::span<float>(d_proj, hidden));

    // residual: hidden += proj
    ResidualAdd(ctx, std::span<float>(d_hidden, hidden),
                std::span<const float>(d_proj, hidden));

    HIP_CHECK(hipDeviceSynchronize());

    // Module outputs must be finite + non-trivial at every stage.
    std::vector<float> normed(hidden), gate(inter), proj(hidden),
        hidden_o(hidden);
    HIP_CHECK(hipMemcpy(normed.data(), d_normed, hidden * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gate.data(), d_gate, inter * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(proj.data(), d_proj, hidden * sizeof(float),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hidden_o.data(), d_hidden, hidden * sizeof(float),
                        hipMemcpyDeviceToHost));
    for (float v : normed) assert(std::isfinite(v));
    for (float v : gate) assert(std::isfinite(v));
    for (float v : proj) assert(std::isfinite(v));
    for (float v : hidden_o) assert(std::isfinite(v));
    assert(MaxAbs(normed) > 1e-4F);         // norm did a real transform
    assert(MaxAbs(gate) > 1e-4F);           // ffn_gate GEMV non-trivial
    assert(MaxAbs(proj) > 1e-4F);           // ffn_down GEMV non-trivial
    assert(MaxAbs(hidden_o) > 1e-4F);       // residual preserved a signal
    // MaxAbs(proj) above implies the residual add changed hidden non-trivially.

    HIP_CHECK(hipFree(d_nw));
    HIP_CHECK(hipFree(d_gw));
    HIP_CHECK(hipFree(d_dw));
  }

  // Head: logits = output (tied token_embd) @ hidden  (vocab, hidden) -> vocab.
  float* d_head = nullptr;
  ToDevice(w.output, &d_head);
  QwenTensorRef head_w{d_head, GgmlType::kF32, vocab * hidden};
  QuantGemm(ctx, head_w, std::span<const float>(d_hidden, hidden), vocab, hidden,
            std::span<float>(d_logits, vocab));
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> logits(vocab, 0.0F);
  HIP_CHECK(hipMemcpy(logits.data(), d_logits, vocab * sizeof(float),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(d_hidden));
  HIP_CHECK(hipFree(d_normed));
  HIP_CHECK(hipFree(d_gate));
  HIP_CHECK(hipFree(d_proj));
  HIP_CHECK(hipFree(d_logits));
  HIP_CHECK(hipFree(d_head));

  return logits;
}

void TestGpuL2MultiLayerModuleForward() {
  std::vector<float> logits = RunMultiLayerForward();

  // Finite, non-trivial logits (a real head projection happened).
  for (float v : logits) assert(std::isfinite(v));
  assert(MaxAbs(logits) > 1e-4F);
}

}  // namespace

int main() {
  TestGpuL2MultiLayerModuleForward();
  std::cout << "All Qwen L2 GPU module integration tests passed.\n";
  return 0;
}
