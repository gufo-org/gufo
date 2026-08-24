// CPU module contract test for the SSM (gated DeltaNet linear attention).
//
// Drives the public typed CPU layer seam over deterministic synthetic weights
// without a GgufReader. Asserts the module:
//   (1) reproduces exactly the whole-layer ForwardSSM compatibility wrapper,
//       proving both paths use the same typed tensor slice implementation;
//   (2) is deterministic across a fresh arena+cache (the SSM recurrent cache is
//       restored to its zero state);
//   (3) safely zero-fills invalid typed views; and
//   (4) emits finite, non-trivial output.
//
// CPU-only; no HIP dependency. Shared deterministic data and comparisons live
// in the Qwen support builder and tests/testing/test_common.hpp.

#include "src/models/qwen/modules/modules.hpp"
#include "src/models/qwen/ssm.hpp"
#include "src/models/qwen/state.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"
#include "tests/testing/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

using strix::core::ModelConfig;
using strix::models::ForwardSSM;
using strix::models::QwenLayerWeights;
using strix::models::QwenScratchArena;
using strix::models::QwenSsmCache;
using strix::models::QwenSsmParameters;
using strix::models::qwen::build_synthetic_qwen_weights;
using strix::models::qwen::make_small_qwen_config;
using strix::models::qwen::MakeSsmView;
using strix::models::qwen::CpuLayerContext;
using strix::models::qwen::SsmForward;
using strix::models::qwen::SsmLayerView;

static_assert(std::is_same_v<SsmLayerView, QwenSsmParameters>);

void Check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Qwen SSM module test failure: " << message << '\n';
    std::abort();
  }
}

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

  const SsmLayerView view = MakeSsmView(layer, config);
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
  Check(!layer.is_full_attention, "synthetic layer 0 must be an SSM layer");

  std::vector<float> x = strix::test::make_random_tensor(hidden, rng, -1.0F, 1.0F);

  // Module output.
  std::vector<float> out;
  RunSsmModule(config, layer, layer_idx, x, out);

  // Whole-layer compatibility-wrapper reference. Both calls must converge on
  // the same typed-slice implementation.
  QwenScratchArena arena_ref(config);
  QwenSsmCache cache_ref = MakeCache(config);
  cache_ref.Reset();
  std::vector<float> ref(hidden, 0.0F);
  ForwardSSM(x, layer, config, cache_ref, layer_idx, arena_ref.ssm_qkv,
             arena_ref.ssm_gate, arena_ref.ssm_out_buf, ref);

  auto res = strix::test::compare_module_logits(ref, out);
  Check(res.match, "typed SSM module must match compatibility wrapper");
  Check(res.finite, "typed SSM module output must be finite");
  Check(res.max_abs_diff < 1e-5F,
        "typed SSM module compatibility difference exceeds tolerance");

  // Every output finite + non-trivial (a real transform happened).
  float max_abs = 0.0F;
  for (float v : out) {
    Check(std::isfinite(v), "SSM module output contains a non-finite value");
    max_abs = std::max(max_abs, std::abs(v));
  }
  Check(max_abs > 1e-6F, "SSM module output must be non-trivial");
}

void TestSsmModuleDeterministicAcrossReset() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& weights = sw.weights;

  std::mt19937 rng = strix::test::make_seeded_rng(123u);
  const std::size_t hidden = config.hidden_size;
  const std::uint32_t layer_idx = 0;
  const auto& layer = weights.layers[layer_idx];
  Check(!layer.is_full_attention, "synthetic layer 0 must be an SSM layer");

  std::vector<float> x = strix::test::make_random_tensor(hidden, rng, -1.0F, 1.0F);

  std::vector<float> a, b;
  RunSsmModule(config, layer, layer_idx, x, a);
  RunSsmModule(config, layer, layer_idx, x, b);  // fresh arena+cache -> independent

  Check(a == b, "fresh SSM cache runs must be deterministic");
}

struct CacheState {
  std::vector<float> conv;
  std::vector<float> deltanet;
};

bool SameBytes(std::span<const float> lhs, std::span<const float> rhs) {
  return lhs.size() == rhs.size() &&
         std::memcmp(lhs.data(), rhs.data(), lhs.size_bytes()) == 0;
}

bool SameCacheState(const CacheState& lhs, const CacheState& rhs) {
  return SameBytes(lhs.conv, rhs.conv) &&
         SameBytes(lhs.deltanet, rhs.deltanet);
}

CacheState CaptureCacheState(QwenSsmCache& cache, std::uint32_t layer_idx) {
  const auto conv = cache.GetConvState(layer_idx);
  CacheState state{
      .conv = std::vector<float>(conv.begin(), conv.end()),
      .deltanet = {}};
  for (std::uint32_t head = 0; head < cache.NumHeads(); ++head) {
    const auto matrix = cache.GetDeltaNetState(layer_idx, head);
    state.deltanet.insert(state.deltanet.end(), matrix.begin(), matrix.end());
  }
  return state;
}

void SeedCacheState(QwenSsmCache& cache, std::uint32_t layer_idx) {
  auto conv = cache.GetConvState(layer_idx);
  for (std::size_t index = 0; index < conv.size(); ++index) {
    conv[index] = static_cast<float>((index % 17U) + 1U) * 0.001F;
  }
  for (std::uint32_t head = 0; head < cache.NumHeads(); ++head) {
    auto matrix = cache.GetDeltaNetState(layer_idx, head);
    for (std::size_t index = 0; index < matrix.size(); ++index) {
      matrix[index] =
          static_cast<float>(((index + head) % 19U) + 1U) * 0.0001F;
    }
  }
}

void CheckZeroOutput(std::span<const float> out, std::string_view message) {
  for (const float value : out) {
    Check(value == 0.0F, message);
  }
}

void CheckRejectedSsmParameters(const ModelConfig& config,
                                const QwenSsmParameters& parameters,
                                std::string_view message) {
  constexpr std::uint32_t layer_idx = 0;
  QwenScratchArena arena(config);
  QwenSsmCache cache = MakeCache(config);
  SeedCacheState(cache, layer_idx);
  const CacheState before = CaptureCacheState(cache, layer_idx);
  const CpuLayerContext ctx(config, arena, layer_idx);
  std::vector<float> input(config.hidden_size, 1.0F);
  std::vector<float> out(config.hidden_size, 7.0F);

  SsmForward(ctx, parameters, input, cache, out);

  CheckZeroOutput(out, message);
  Check(SameCacheState(CaptureCacheState(cache, layer_idx), before),
        "rejected SSM parameters must not mutate recurrent cache state");
}

void TestInvalidSsmViewZeroFills() {
  const ModelConfig config = make_small_qwen_config();
  CheckRejectedSsmParameters(config, SsmLayerView{},
                             "invalid SSM view must zero-fill output");
}

void TestMalformedSsmParametersRejectBeforeStateMutation() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& layer = sw.weights.layers[0];
  Check(!layer.is_full_attention, "synthetic layer 0 must be an SSM layer");
  const QwenSsmParameters valid = MakeSsmView(layer, config);

  QwenSsmParameters truncated = valid;
  Check(truncated.qkv.EncodedSizeBytes() > 0,
        "synthetic QKV tensor must have encoded storage");
  truncated.qkv.available_bytes = truncated.qkv.EncodedSizeBytes() - 1U;
  CheckRejectedSsmParameters(config, truncated,
                             "truncated SSM tensor must zero-fill output");

  QwenSsmParameters unsupported = valid;
  unsupported.qkv.type = strix::core::GgmlType::kQ4_0;
  CheckRejectedSsmParameters(config, unsupported,
                             "unsupported SSM projection format must reject");

  QwenSsmParameters wrong_norm = valid;
  Check(wrong_norm.val_dim > 1U && wrong_norm.val_dim % 2U == 0U,
        "synthetic SSM value width must support divisible-width test");
  wrong_norm.norm.num_elements = wrong_norm.val_dim / 2U;
  CheckRejectedSsmParameters(config, wrong_norm,
                             "wrong divisible SSM norm width must reject");

  QwenSsmParameters undersized = valid;
  Check(undersized.gate.num_elements > 0,
        "synthetic SSM gate projection must be nonempty");
  --undersized.gate.num_elements;
  CheckRejectedSsmParameters(config, undersized,
                             "undersized SSM tensor must zero-fill output");

  QwenSsmParameters cache_mismatch = valid;
  ++cache_mismatch.conv_kernel;
  CheckRejectedSsmParameters(config, cache_mismatch,
                             "SSM cache shape mismatch must zero-fill output");
}

std::vector<float> RunSsmStep(const ModelConfig& config,
                              const QwenLayerWeights& layer,
                              QwenScratchArena& arena, QwenSsmCache& cache,
                              std::span<const float> input) {
  constexpr std::uint32_t layer_idx = 0;
  const CpuLayerContext ctx(config, arena, layer_idx);
  const SsmLayerView view = MakeSsmView(layer, config);
  std::vector<float> out(config.hidden_size, 0.0F);
  SsmForward(ctx, view, input, cache, out);
  return out;
}

void TestSsmCacheEvolutionIsDeterministicAcrossReset() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& layer = sw.weights.layers[0];
  Check(!layer.is_full_attention, "synthetic layer 0 must be an SSM layer");
  std::mt19937 rng = strix::test::make_seeded_rng(0xCA5EU);
  const std::vector<float> input =
      strix::test::make_random_tensor(config.hidden_size, rng, -1.0F, 1.0F);

  QwenScratchArena arena_a(config);
  QwenSsmCache cache_a = MakeCache(config);
  cache_a.Reset();
  const std::vector<float> first_a =
      RunSsmStep(config, layer, arena_a, cache_a, input);
  const CacheState first_state = CaptureCacheState(cache_a, 0);
  const std::vector<float> second_a =
      RunSsmStep(config, layer, arena_a, cache_a, input);
  const CacheState second_state = CaptureCacheState(cache_a, 0);
  Check(!SameCacheState(first_state, second_state),
        "second SSM step must evolve cache state");

  QwenScratchArena arena_b(config);
  QwenSsmCache cache_b = MakeCache(config);
  cache_b.Reset();
  const std::vector<float> first_b =
      RunSsmStep(config, layer, arena_b, cache_b, input);
  const std::vector<float> second_b =
      RunSsmStep(config, layer, arena_b, cache_b, input);
  Check(first_a == first_b && second_a == second_b,
        "two-step SSM outputs must be deterministic");
  Check(SameCacheState(CaptureCacheState(cache_b, 0), second_state),
        "two-step SSM cache evolution must be deterministic");

  cache_a.Reset();
  const std::vector<float> reset_first =
      RunSsmStep(config, layer, arena_a, cache_a, input);
  Check(reset_first == first_a,
        "reset SSM cache must reproduce the first-step output");
  Check(SameCacheState(CaptureCacheState(cache_a, 0), first_state),
        "reset SSM cache must reproduce the first-step state");
}

float MaxAbsDiff(std::span<const float> a, std::span<const float> b) {
  Check(a.size() == b.size(), "SSM comparison size mismatch");
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
  Check(!layer.is_full_attention, "synthetic layer 0 must be an SSM layer");
  Check(!layer.ssm_norm.empty(), "synthetic SSM norm must be present");

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
  Check(input_delta > 1e-2F, "SSM output must respond to input changes");

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
  Check(norm_delta > 1e-3F, "SSM output must use the norm tensor");
}

}  // namespace

int main() {
  TestSsmModuleMatchesProduction();
  TestSsmModuleDeterministicAcrossReset();
  TestInvalidSsmViewZeroFills();
  TestMalformedSsmParametersRejectBeforeStateMutation();
  TestSsmCacheEvolutionIsDeterministicAcrossReset();
  TestSsmModuleSensitivity();
  std::cout << "All Qwen L1 SSM module tests passed.\n";
  return 0;
}
