// CPU module contract test for the SwiGLU FFN module (FfnForward).
//
// Drives the public typed CPU seam and explicit scratch spans over
// deterministic synthetic weights. Asserts the module
//   (1) reproduces an independent oracle (ReferenceGEMV + ReferenceSwiGLU for
//       gate/up/down projections — a DIFFERENT GEMV implementation than the
//       module's TensorGEMV path, so this is a genuine cross-check), and
//   (2) emits finite, non-trivial output.
//
// CPU-only; no HIP dependency. Shared deterministic data and comparisons live
// in the Qwen support builder and tests/testing/test_common.hpp.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "src/models/qwen/modules/modules.hpp"
#include "src/models/qwen/oracles.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"
#include "tests/testing/test_common.hpp"

namespace {

using strix::core::ModelConfig;
using strix::models::QwenLayerWeights;
using strix::models::qwen::build_synthetic_qwen_weights;
using strix::models::qwen::CpuModuleContext;
using strix::models::qwen::FfnForward;
using strix::models::qwen::FfnLayerView;
using strix::models::qwen::make_small_qwen_config;
using strix::models::qwen::MakeFfnView;
using strix::models::qwen::ReferenceGEMV;
using strix::models::qwen::ReferenceSwiGLU;

void TestFfnModuleMatchesOracle() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& weights = sw.weights;
  std::mt19937 rng = strix::test::make_seeded_rng(0xFF12u);
  const std::size_t hidden = config.hidden_size;
  const std::size_t inter = config.intermediate_size;
  const auto& layer = weights.layers[0];

  std::vector<float> x =
      strix::test::make_random_tensor(hidden, rng, -0.5F, 0.5F);

  // Module output.
  FfnLayerView view = MakeFfnView(layer, config);
  std::vector<float> gate_s(inter, 0.0F), up_s(inter, 0.0F), act_s(inter, 0.0F);
  std::vector<float> out(hidden, 0.0F);
  const CpuModuleContext ctx;
  FfnForward(ctx, view, x, gate_s, up_s, act_s, out);

  // Independent oracle: ReferenceGEMV + ReferenceSwiGLU (FP64 reference GEMV).
  std::vector<float> g(inter, 0.0F), u(inter, 0.0F), a(inter, 0.0F);
  std::vector<float> ref(hidden, 0.0F);
  ReferenceGEMV(layer.ffn_gate.AsFloatSpan(), x, inter, hidden, g);
  ReferenceGEMV(layer.ffn_up.AsFloatSpan(), x, inter, hidden, u);
  ReferenceSwiGLU(g, u, a);
  ReferenceGEMV(layer.ffn_down.AsFloatSpan(), a, hidden, inter, ref);

  auto res = strix::test::compare_module_logits(ref, out);
  assert(res.match);
  assert(res.finite);
  float max_abs = 0.0F;
  for (float v : out) {
    assert(std::isfinite(v));
    max_abs = std::max(max_abs, std::abs(v));
  }
  assert(max_abs > 1e-6F);
}

}  // namespace

int main() {
  TestFfnModuleMatchesOracle();
  std::cout << "All Qwen L1 ffn module tests passed.\n";
  return 0;
}
