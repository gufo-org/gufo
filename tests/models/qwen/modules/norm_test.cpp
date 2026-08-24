// CPU module contract test for RMSNorm (NormForward).
//
// Drives the public typed CPU seam over deterministic synthetic weights without
// a GgufReader. Asserts the module:
//   (1) reproduces an independent F32 RMSNorm reference (a different code path
//       than the module's FP64 ReferenceRMSNorm oracle), for both the attn
//       pre-norm and ffn pre-norm weight slices (validates MakeAttnNormView /
//       MakeFfnNormView seam wiring), and
//   (2) emits finite, non-trivial output.
//
// CPU-only; no HIP dependency. Shared deterministic data and comparisons live
// in the Qwen support builder and tests/testing/test_common.hpp.

#include "src/models/qwen/modules/modules.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"
#include "tests/testing/test_common.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

using strix::core::ModelConfig;
using strix::models::QwenLayerWeights;
using strix::models::qwen::build_synthetic_qwen_weights;
using strix::models::qwen::make_small_qwen_config;
using strix::models::qwen::MakeAttnNormView;
using strix::models::qwen::MakeFfnNormView;
using strix::models::qwen::CpuModuleContext;
using strix::models::qwen::NormForward;
using strix::models::qwen::NormLayerView;

void RunNormModule(const ModelConfig& c, const NormLayerView& view,
                   std::span<const float> x, std::vector<float>& out) {
  (void)c;
  const CpuModuleContext ctx;
  out.assign(x.size(), 0.0F);
  NormForward(ctx, view, x, out);
}

// Independent F32 RMSNorm (different path than the module's FP64 oracle):
//   ref[i] = (x[i] / sqrt(mean(x^2) + eps)) * w[i]
void ComputeManualRmsNorm(std::span<const float> x, std::span<const float> w,
                          float eps, std::span<float> out) {
  double sum = 0.0;
  for (float v : x) {
    sum += static_cast<double>(v) * static_cast<double>(v);
  }
  const double rms =
      std::sqrt(sum / static_cast<double>(x.size()) +
                static_cast<double>(eps));
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = static_cast<float>(static_cast<double>(x[i]) / rms *
                                static_cast<double>(w[i]));
  }
}

void TestNormModuleMatchesIndependentReference() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& weights = sw.weights;
  std::mt19937 rng = strix::test::make_seeded_rng(0xA117u);
  const std::size_t hidden = config.hidden_size;
  const auto& layer = weights.layers[0];

  std::vector<float> x = strix::test::make_random_tensor(hidden, rng, -1.0F, 1.0F);

  // attn pre-norm slice (MakeAttnNormView).
  {
    NormLayerView view = MakeAttnNormView(layer, config);
    std::vector<float> out;
    RunNormModule(config, view, x, out);

    std::vector<float> ref(hidden, 0.0F);
    ComputeManualRmsNorm(x, layer.attn_norm.AsFloatSpan(), view.eps, ref);

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

  // ffn pre-norm slice (MakeFfnNormView).
  {
    NormLayerView view = MakeFfnNormView(layer, config);
    std::vector<float> out;
    RunNormModule(config, view, x, out);

    std::vector<float> ref(hidden, 0.0F);
    ComputeManualRmsNorm(x, layer.ffn_norm.AsFloatSpan(), view.eps, ref);

    auto res = strix::test::compare_module_logits(ref, out);
    assert(res.match);
    assert(res.finite);
  }
}

}  // namespace

int main() {
  TestNormModuleMatchesIndependentReference();
  std::cout << "All Qwen L1 norm module tests passed.\n";
  return 0;
}
