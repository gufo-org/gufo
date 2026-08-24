// Fusion-route (#44): verify the detail:: toggles that route the per-token decode
// between fused kernels (owned by the ExecuteStep composition layer, Option B) and
// the unfused module fallback (modules are pure).
//
// CPU-only. The production defaults and immutable executor policy are host-side,
// so a CPU test asserts the routing decisions directly. For the cross-module fusions that are
// currently DISABLED, the decode falls back to the module path (NormForward /
// ResidualAdd / RopeForward) — this test drives those module fallbacks over small
// synthetic data and checks they produce correct output, so a false toggle cannot
// silently strand a broken fallback. The literal fused-kernel-vs-module GPU
// comparison is covered by the L2-GPU integration test (orchestrator-gated).

#include "src/models/qwen/modules/modules.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "tests/models/qwen/support/synthetic_weights.hpp"
#include "tests/testing/test_common.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <type_traits>
#include <vector>

namespace {

using strix::core::ModelConfig;
using strix::models::qwen::build_synthetic_qwen_weights;
using strix::models::qwen::make_small_qwen_config;
using strix::models::qwen::MakeAttnNormView;
using strix::models::qwen::CpuLayerContext;
using strix::models::qwen::CpuModuleContext;
using strix::models::qwen::HipModuleContext;
using strix::models::qwen::NormForward;
using strix::models::qwen::NormLayerView;
using strix::models::qwen::ResidualAdd;
using strix::models::qwen::RopeForward;
using strix::models::qwen::RopeLayerView;

static_assert(std::is_empty_v<CpuModuleContext>);
static_assert(!std::is_default_constructible_v<CpuLayerContext>);
static_assert(!std::is_convertible_v<CpuModuleContext, HipModuleContext>);

// --- 1. Fusion-toggle routing values (strix::hip::detail, host constexpr) ---
void TestFusionToggleValues() {
  // Q/K norm + RoPE + KV write: ENABLED -> decode uses the fused kernel.
  assert(strix::hip::detail::ShouldFuseQKNormRoPEKvWrite() == true);

  // The cross-module fusions below are DISABLED -> decode uses the unfused module
  // fallback. Each must stay consistent with the composition layer's choice.
  assert(strix::hip::detail::ShouldFuseResidualAddRMSNorm() == false);
  assert(strix::hip::detail::ShouldFuseSSMGateResidual() == false);
  assert(strix::hip::detail::ShouldFuseRMSNormProjection() == false);
  assert(strix::hip::detail::ShouldFuseFFNSwiGLU() == false);
  assert(strix::hip::detail::ShouldPrefetchNextLayer() == false);

  auto candidate = strix::hip::QwenExecutionPolicy::Production();
  candidate.fuse_decode_ssm_output_residual = true;
  candidate.prefetch_next_layer = true;
  assert(strix::hip::detail::ShouldFuseDecodeSSMOutputResidual(candidate));
  assert(strix::hip::detail::ShouldPrefetchNextLayer(candidate));
  assert(candidate.Fingerprint() == ((1ULL << 0U) | (1ULL << 4U) |
                                     (1ULL << 7U)));

  const auto decode_plan = strix::hip::ResolveQwenLayerRoute(
      candidate, strix::hip::QwenExecutionMode::kDecode, false);
  const auto prefill_plan = strix::hip::ResolveQwenLayerRoute(
      candidate, strix::hip::QwenExecutionMode::kPrefill, false);
  assert(decode_plan.fuse_ssm_epilogue);
  assert(decode_plan.prefetch_next_layer);
  assert(!prefill_plan.fuse_ssm_epilogue);
  assert(!prefill_plan.prefetch_next_layer);
  assert(decode_plan.Fingerprint() != prefill_plan.Fingerprint());

  auto ffn_candidate = strix::hip::QwenExecutionPolicy::Production();
  ffn_candidate.fuse_decode_rmsnorm_projection = true;
  auto ffn_plan = strix::hip::ResolveQwenLayerRoute(
      ffn_candidate, strix::hip::QwenExecutionMode::kDecode, true);
  assert(ffn_plan.fuse_rmsnorm_projection);
  assert(!ffn_plan.fuse_ffn_swiglu);

  ffn_candidate.fuse_decode_rmsnorm_swiglu = true;
  ffn_plan = strix::hip::ResolveQwenLayerRoute(
      ffn_candidate, strix::hip::QwenExecutionMode::kDecode, true);
  assert(ffn_plan.fuse_rmsnorm_projection);
  assert(ffn_plan.fuse_ffn_swiglu);

  const auto production_ssm = strix::hip::ResolveQwenLayerRouteWithReasons(
      strix::hip::QwenExecutionPolicy::Production(),
      strix::hip::QwenExecutionMode::kDecode, false);
  assert(strix::hip::HasQwenRouteRejection(
      production_ssm.rejected,
      strix::hip::QwenRouteRejection::kQkNormRopeKvRequiresAttention));

  auto all_routes = strix::hip::QwenExecutionPolicy::Production();
  all_routes.fuse_prefill_ffn_swiglu = true;
  all_routes.fuse_decode_rmsnorm_swiglu = true;
  all_routes.fuse_decode_ssm_output_residual = true;
  all_routes.fuse_prefill_ssm_post_norm_gate = true;
  all_routes.fuse_decode_rmsnorm_projection = true;
  all_routes.prefetch_next_layer = true;
  const auto decode_attention = strix::hip::ResolveQwenLayerRouteWithReasons(
      all_routes, strix::hip::QwenExecutionMode::kDecode, true);
  assert(strix::hip::HasQwenRouteRejection(
      decode_attention.rejected,
      strix::hip::QwenRouteRejection::kSsmEpilogueRequiresSsm));
  assert(strix::hip::HasQwenRouteRejection(
      decode_attention.rejected,
      strix::hip::QwenRouteRejection::kPrefillFfnSwiGluRequiresPrefill));
  assert(strix::hip::HasQwenRouteRejection(
      decode_attention.rejected,
      strix::hip::QwenRouteRejection::kPrefillSsmEpilogueRequiresPrefill));

  const auto prefill_ssm = strix::hip::ResolveQwenLayerRouteWithReasons(
      all_routes, strix::hip::QwenExecutionMode::kPrefill, false);
  assert(strix::hip::HasQwenRouteRejection(
      prefill_ssm.rejected,
      strix::hip::QwenRouteRejection::kDecodeFfnSwiGluRequiresDecode));
  assert(strix::hip::HasQwenRouteRejection(
      prefill_ssm.rejected,
      strix::hip::QwenRouteRejection::kDecodeSsmEpilogueRequiresDecode));
  assert(strix::hip::HasQwenRouteRejection(
      prefill_ssm.rejected,
      strix::hip::QwenRouteRejection::kRmsNormProjectionRequiresDecode));
  assert(strix::hip::HasQwenRouteRejection(
      prefill_ssm.rejected,
      strix::hip::QwenRouteRejection::kPrefetchRequiresDecode));

  const auto graph_rejections = strix::hip::ResolveQwenGraphRejections(
      false, true, true, false);
  assert(strix::hip::HasQwenGraphRejection(
      graph_rejections,
      strix::hip::QwenGraphRejection::kLogitsNotRequested));
  assert(strix::hip::HasQwenGraphRejection(
      graph_rejections,
      strix::hip::QwenGraphRejection::kSplitKAttentionRequired));
  assert(strix::hip::HasQwenGraphRejection(
      graph_rejections,
      strix::hip::QwenGraphRejection::kLayerPrefetchEnabled));
  assert(strix::hip::HasQwenGraphRejection(
      graph_rejections, strix::hip::QwenGraphRejection::kGraphDisabled));
  assert(strix::hip::ResolveQwenGraphRejections(true, false, false, true) ==
         strix::hip::QwenGraphRejection::kNone);

  const auto workload_seed = strix::hip::BeginQwenGraphWorkloadIdentity();
  const auto workload_a = strix::hip::ExtendQwenGraphWorkloadIdentity(
      workload_seed, decode_plan.Fingerprint());
  const auto workload_a_repeat = strix::hip::ExtendQwenGraphWorkloadIdentity(
      workload_seed, decode_plan.Fingerprint());
  const auto workload_b = strix::hip::ExtendQwenGraphWorkloadIdentity(
      workload_seed, prefill_plan.Fingerprint());
  assert(workload_a == workload_a_repeat);
  assert(workload_a != workload_b);
}

// --- 2. Unfused fallback: attn pre-norm RMSNorm (RMSNormProjection fusion off) ---
void TestUnfusedFallback_Norm() {
  const ModelConfig config = make_small_qwen_config();
  auto sw = build_synthetic_qwen_weights(config);
  const auto& layer = sw.weights.layers[0];
  const std::size_t hidden = config.hidden_size;

  std::mt19937 rng = strix::test::make_seeded_rng(0xF0U);
  std::vector<float> x = strix::test::make_random_tensor(hidden, rng, -1.0F, 1.0F);

  NormLayerView view = MakeAttnNormView(layer, config);
  const CpuModuleContext ctx;
  std::vector<float> out(hidden, 0.0F);
  NormForward(ctx, view, x, out);

  // Independent F32 RMSNorm reference (different path than the module's FP64
  // oracle): ref[i] = (x[i] / rms(x)) * w[i].
  std::vector<float> ref(hidden, 0.0F);
  double sum = 0.0;
  for (float v : x) {
    sum += static_cast<double>(v) * static_cast<double>(v);
  }
  const double rms = std::sqrt(sum / static_cast<double>(hidden) +
                               static_cast<double>(view.eps));
  const auto w = layer.attn_norm.AsFloatSpan();
  for (std::size_t i = 0; i < hidden; ++i) {
    ref[i] = static_cast<float>(static_cast<double>(x[i]) / rms *
                                static_cast<double>(w[i]));
  }

  auto res = strix::test::compare_module_logits(ref, out);
  assert(res.match);
  assert(res.finite);
}

// --- 3. Unfused fallback: residual add (SSM gate+residual and residual+norm off) ---
void TestUnfusedFallback_Residual() {
  std::vector<float> dst = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> src = {0.5F, -1.0F, 2.5F, -0.5F};
  std::vector<float> expect = {1.5F, 1.0F, 5.5F, 3.5F};

  const CpuModuleContext ctx;
  ResidualAdd(ctx, dst, src);  // dst += src, in-place
  for (std::size_t i = 0; i < dst.size(); ++i) {
    assert(std::abs(dst[i] - expect[i]) < 1e-6F);
  }
}

// --- 4. Unfused fallback: RoPE (QK-norm+RoPE+KV-write fusion ENABLED; standalone
//        module must still be correct as the unfused alternative) ---
void TestUnfusedFallback_Rope() {
  const std::uint32_t head_dim = 4;
  RopeLayerView view{/*num_heads*/ 1, /*num_kv_heads*/ 1, head_dim, head_dim,
                     /*rope_theta*/ 10000.0F};
  std::vector<float> q = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<float> k = {5.0F, 6.0F, 7.0F, 8.0F};

  const CpuModuleContext ctx;

  // pos=0 -> RoPE is the identity rotation (cos=0, sin=1) -> q/k unchanged.
  {
    std::vector<float> q0 = q;
    std::vector<float> k0 = k;
    RopeForward(ctx, view, q0, k0, 0u);
    for (std::size_t i = 0; i < q0.size(); ++i) {
      assert(std::abs(q0[i] - q[i]) < 1e-5F);
      assert(std::abs(k0[i] - k[i]) < 1e-5F);
    }
  }
  // pos>0 -> finite output; the first head pair actually rotates (changes value).
  {
    std::vector<float> q5 = q;
    std::vector<float> k5 = k;
    RopeForward(ctx, view, q5, k5, 5u);
    for (float v : q5) {
      assert(std::isfinite(v));
    }
    for (float v : k5) {
      assert(std::isfinite(v));
    }
    const bool rotated = std::abs(q5[0] - q[0]) > 1e-4F ||
                         std::abs(q5[1] - q[1]) > 1e-4F;
    assert(rotated);
  }
}

}  // namespace

int main() {
  TestFusionToggleValues();
  TestUnfusedFallback_Norm();
  TestUnfusedFallback_Residual();
  TestUnfusedFallback_Rope();
  std::cout << "All Qwen fusion-route toggle tests passed.\n";
  return 0;
}
