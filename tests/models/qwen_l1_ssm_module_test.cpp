// L1 CPU module e2e test for the SSM module (gated DeltaNet linear attention).
//
// Phase 3 (#18): drive the module through its public seam — a ModuleCtx +
// SsmLayerView over a small synthetic model built by build_synthetic_qwen_weights
// (no GgufReader). Asserts the module:
//   (1) reproduces exactly the production ForwardSSM result (validates the
//       view.source / arena-slice / config / layer_idx seam wiring), and
//   (2) is deterministic across a fresh arena+cache (the SSM recurrent cache is
//       restored to its zero state), and
//   (3) emits finite, non-trivial output.
//
// CPU-only; no HIP dependency. Uses the Phase 1 test_common + synthetic-weights
// builder headers (both header-only inline).

#include "src/models/qwen/modules/modules.hpp"
#include "src/models/qwen/qwen_ssm.hpp"
#include "src/models/qwen/qwen_state.hpp"
#include "tests/testing/synthetic_qwen_weights.hpp"
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
using strix::models::qwen::Backend;
using strix::models::qwen::build_synthetic_qwen_weights;
using strix::models::qwen::make_small_qwen_config;
using strix::models::qwen::MakeSsmView;
using strix::models::qwen::ModuleCtx;
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

  ModuleCtx ctx;
  ctx.config = &config;
  ctx.arena = &arena;
  ctx.backend = Backend::Cpu;
  ctx.layer_idx = layer_idx;

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

}  // namespace

int main() {
  TestSsmModuleMatchesProduction();
  TestSsmModuleDeterministicAcrossReset();
  std::cout << "All Qwen L1 SSM module tests passed.\n";
  return 0;
}
