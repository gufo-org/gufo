// Graph identity must separate production execution from operator references.
#include "src/models/qwen/hip/execution_policy.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
  using namespace gufo::hip;
  const auto production = QwenExecutionPolicy::Production();
  assert(production.UsesFp16AttentionKv());
  assert(!production.UsesBf16RecurrentState());
  assert(QwenRecurrentStateElementBytes(QwenRecurrentStateStorage::kFp32) ==
         sizeof(float));
  assert(QwenRecurrentStateElementBytes(QwenRecurrentStateStorage::kBf16) ==
         sizeof(std::uint16_t));

  auto fp32_kv = production;
  fp32_kv.kv_cache_storage = QwenKvCacheStorage::kFp32;
  auto bf16_recurrent = production;
  bf16_recurrent.recurrent_state_storage = QwenRecurrentStateStorage::kBf16;
  auto separate_qk_norm = production;
  separate_qk_norm.fuse_qk_norm_rope_kv = false;
  assert(fp32_kv.Fingerprint() != production.Fingerprint());
  assert(bf16_recurrent.Fingerprint() != production.Fingerprint());
  assert(separate_qk_norm.Fingerprint() != production.Fingerprint());

  const auto decode = ResolveQwenLayerRouteWithReasons(
      production, QwenExecutionMode::kDecode, true);
  const auto prefill = ResolveQwenLayerRouteWithReasons(
      production, QwenExecutionMode::kPrefill, true);
  const auto ssm = ResolveQwenLayerRouteWithReasons(
      production, QwenExecutionMode::kDecode, false);
  assert(decode.plan.fuse_qk_norm_rope_kv);
  assert(prefill.plan.fuse_qk_norm_rope_kv);
  assert(!ssm.plan.fuse_qk_norm_rope_kv);
  assert(decode.rejected == QwenRouteRejection::kNone);
  assert(HasQwenRouteRejection(
      ssm.rejected, QwenRouteRejection::kQkNormRopeKvRequiresAttention));
  assert(decode.plan.Fingerprint() != prefill.plan.Fingerprint());
  assert(decode.plan.Fingerprint() != ssm.plan.Fingerprint());

  const auto rejections = ResolveQwenGraphRejections(false, true, false);
  assert(HasQwenGraphRejection(rejections,
                               QwenGraphRejection::kLogitsNotRequested));
  assert(HasQwenGraphRejection(rejections,
                               QwenGraphRejection::kSplitKAttentionRequired));
  assert(HasQwenGraphRejection(rejections, QwenGraphRejection::kGraphDisabled));
  assert(ResolveQwenGraphRejections(true, false, true) ==
         QwenGraphRejection::kNone);

  const auto seed = BeginQwenGraphWorkloadIdentity();
  const auto first =
      ExtendQwenGraphWorkloadIdentity(seed, decode.plan.Fingerprint());
  assert(first ==
         ExtendQwenGraphWorkloadIdentity(seed, decode.plan.Fingerprint()));
  assert(first !=
         ExtendQwenGraphWorkloadIdentity(seed, prefill.plan.Fingerprint()));
  std::cout << "Qwen graph identity and reference policy checks passed.\n";
}
