// CPU module contract test for the SSM (gated DeltaNet linear attention).
//
// Drives the public typed CPU layer seam over deterministic synthetic weights
// without a GgufReader. Asserts the module:
//   (1) reproduces exactly the production ForwardSSM result (validates the
//       view.source / arena-slice / config / layer_idx seam wiring), and
//   (2) is deterministic across a fresh arena+cache (the SSM recurrent cache is
//       restored to its zero state), and
//   (3) emits finite, non-trivial output.
//
// CPU-only; no HIP dependency. Shared deterministic data and comparisons live
// in the Qwen support builder and tests/testing/test_common.hpp.

#include "src/models/qwen/modules/modules.hpp"
#include "src/models/qwen/ssm.hpp"
#include "src/models/qwen/state.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"
#include "tests/testing/test_common.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace {

using strix::core::ModelConfig;
using strix::models::ForwardSSM;
using strix::models::QwenLayerWeights;
using strix::models::QwenScratchArena;
using strix::models::QwenSsmCache;
using strix::models::qwen::build_synthetic_qwen_weights;
using strix::models::qwen::make_small_qwen_config;
using strix::models::qwen::MakeSsmView;
using strix::models::qwen::CpuLayerContext;
using strix::models::qwen::SsmForward;
using strix::models::qwen::SsmLayerView;

QwenSsmCache MakeCache(const ModelConfig& c) {
  return QwenSsmCache(c.num_layers, c.SsmQkvSize(), c.ssm_conv_kernel,
                      c.ssm_time_step_rank, c.ssm_state_size, c.SsmValueSize());
}

// Runs the module once over a freshly constructed arena + cache (each call is
// an independent decode step, i.e. position 0 with a zeroed recurrent state).
void RunSsmModule(const ModelConfig& config,
                  const QwenLayerWeights& layer, std::uint32_t layer_idx,
                  std::span<const float> x, std::vector<float>& out) {
  QwenScratchArena arena(config);
  QwenSsmCache cache = MakeCache(config);
  cache.Reset();

  const CpuLayerContext ctx(config, arena, layer_idx);

  SsmLayerView view = MakeSsmView(layer, config);
  out.assign(x.size(), 0.0F);
  SsmForward(ctx, view, x, cache, out);
}

void TestSsmModuleMatchesProduction() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& weights = sw.weights;

  std::mt19937 rng = strix::test::make_seeded_rng(0x5EEDu);
  const std::size_t hidden = config.hidden_size;

  // In this config (full_attention_interval==4) layers 0..2 are SSM layers.
  const std::uint32_t layer_idx = 0;
  const auto& layer = weights.layers[layer_idx];
  if (layer.is_full_attention) return;  // no SSM layer to test (unexpected)

  std::vector<float> x = strix::test::make_random_tensor(hidden, rng, -1.0F, 1.0F);

  // Module output.
  std::vector<float> out;
  RunSsmModule(config, layer, layer_idx, x, out);

  // Direct production reference.
  QwenScratchArena arena_ref(config);
  QwenSsmCache cache_ref = MakeCache(config);
  cache_ref.Reset();
  std::vector<float> ref(hidden, 0.0F);
  ForwardSSM(x, layer, config, cache_ref, layer_idx, arena_ref.ssm_qkv,
             arena_ref.ssm_gate, arena_ref.ssm_out_buf, ref);

  auto res = strix::test::compare_module_logits(ref, out);
  assert(res.match);
  assert(res.finite);
  assert(res.max_abs_diff < 1e-5F);

  // Every output finite + non-trivial (a real transform happened).
  float max_abs = 0.0F;
  for (float v : out) {
    assert(std::isfinite(v));
    max_abs = std::max(max_abs, std::abs(v));
  }
  assert(max_abs > 1e-6F);
}

void TestSsmModuleDeterministicAcrossReset() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& weights = sw.weights;

  std::mt19937 rng = strix::test::make_seeded_rng(123u);
  const std::size_t hidden = config.hidden_size;
  const std::uint32_t layer_idx = 0;
  const auto& layer = weights.layers[layer_idx];
  if (layer.is_full_attention) return;

  std::vector<float> x = strix::test::make_random_tensor(hidden, rng, -1.0F, 1.0F);

  std::vector<float> a, b;
  RunSsmModule(config, layer, layer_idx, x, a);
  RunSsmModule(config, layer, layer_idx, x, b);  // fresh arena+cache -> independent

  assert(a.size() == b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    assert(a[i] == b[i]);
  }
}

float MaxAbsDiff(std::span<const float> a, std::span<const float> b) {
  assert(a.size() == b.size());
  float m = 0.0F;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

// Sensitivity/adversarial check on the SSM module contract.
// Proves the module is not trivially constant and that a shared component (the
// per-head output RMSNorm, layer.ssm_norm) is genuinely exercised by the
// module seam. (a) perturbing the input must move the output meaningfully;
// (b) perturbing the norm must move the downstream output. Either check would
// fail if the module were a passthrough or ignored the norm.
void TestSsmModuleSensitivity() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& weights = sw.weights;
  std::mt19937 rng = strix::test::make_seeded_rng(0xA11u);
  const std::size_t hidden = config.hidden_size;
  const std::uint32_t layer_idx = 0;
  const auto& layer = weights.layers[layer_idx];
  if (layer.is_full_attention || layer.ssm_norm.empty()) return;

  std::vector<float> x = strix::test::make_random_tensor(hidden, rng, -1.0F, 1.0F);
  std::vector<float> out_base;
  RunSsmModule(config, layer, layer_idx, x, out_base);

  // (a) Input perturbation: a real transform must move the output. 0.1 is a
  // meaningful fraction of the U(-1,1) input range.
  std::vector<float> xp = x;
  for (float& v : xp) v += 0.1F;
  std::vector<float> out_input;
  RunSsmModule(config, layer, layer_idx, xp, out_input);
  const float input_delta = MaxAbsDiff(out_base, out_input);
  std::cerr << "[SENS] input_delta=" << input_delta << "\n";
  assert(input_delta > 1e-2F);

  // (b) Shared-component (per-head output RMSNorm) perturbation: perturbing
  // layer.ssm_norm must move the downstream output, proving the module wires
  // the norm into its computation.
  std::vector<float> norm_new(layer.ssm_norm.num_elements);
  for (std::size_t i = 0; i < layer.ssm_norm.num_elements; ++i) {
    norm_new[i] = layer.ssm_norm.Get(i) * 1.5F;
  }
  QwenLayerWeights layer_mod = layer;
  layer_mod.ssm_norm.data = norm_new.data();
  layer_mod.ssm_norm.type = strix::core::GgmlType::kF32;
  layer_mod.ssm_norm.num_elements = norm_new.size();
  std::vector<float> out_norm;
  RunSsmModule(config, layer_mod, layer_idx, x, out_norm);
  const float norm_delta = MaxAbsDiff(out_base, out_norm);
  std::cerr << "[SENS] norm_delta=" << norm_delta << "\n";
  assert(norm_delta > 1e-3F);
}

}  // namespace

int main() {
  TestSsmModuleMatchesProduction();
  TestSsmModuleDeterministicAcrossReset();
  TestSsmModuleSensitivity();
  std::cout << "All Qwen L1 SSM module tests passed.\n";
  return 0;
}
