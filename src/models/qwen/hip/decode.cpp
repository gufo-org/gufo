#if defined(ENGINE_ENABLE_HIP)
#include <stdexcept>

#include "src/core/hip/detail/dispatch_telemetry.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/models/qwen/hip/detail/attention_policy.hpp"
#include "src/models/qwen/hip/detail/decode_step.hpp"
#include "src/models/qwen/hip/executor.hpp"

namespace strix::hip {

tokenization::TokenId QwenGpuExecutor::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  if (pos >= arena_.GetMaxContext()) {
    throw std::length_error("token position exceeds the GPU context length");
  }

  if (replaying_ssm_state_ && !compute_logits) {
    if (arena_.CanReplaySsmPosition(pos)) {
      ReplaySsmState(pos);
      return 0;
    }
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
  } else if (replaying_ssm_state_) {
    replaying_ssm_state_ = false;
    arena_.DisableSsmReplayCapture();
  }

  arena_.MarkSsmReplayPosition(pos);
  last_hidden_offset_ = 0;
  const auto& config = weights_.config;
  const std::size_t sequence_length = static_cast<std::size_t>(pos) + 1;
  const bool use_split_k_decode = detail::IsSplitKDecodeAttentionSupported(
      sequence_length, config.num_attention_heads, config.num_key_value_heads,
      config.head_dim);

  auto scratch = arena_.GetScratchView();
  auto* d_out_token = scratch.decode.sampled_token.data();

  // Copy token_id and pos to GPU device memory
  const std::uint32_t in_params[2] = {token_id, pos};
  HIP_CHECK(hipMemcpyAsync(scratch.decode.prompt_tokens.data(), in_params,
                           sizeof(in_params), hipMemcpyHostToDevice,
                           arena_.stream));

  EmitDecodeRouteTelemetry(weights_, policy_);
  const QwenGraphRejection graph_rejections = ResolveQwenGraphRejections(
      compute_logits, use_split_k_decode, policy_.prefetch_next_layer,
      graph_executor_.IsEnabled());
  detail::EmitQwenGraphEligibility(
      graph_key_.execution_identity, graph_key_.workload_identity,
      static_cast<std::uint32_t>(graph_rejections));

  const auto launch_captured_graph = [&]() {
    if (!graph_executor_.Launch(arena_.stream, graph_key_)) {
      graph_executor_.Reset();
      throw std::runtime_error(
          "Qwen HIP graph launch failed; captured graph was invalidated");
    }
  };

  if (graph_rejections == QwenGraphRejection::kNone) {
    if (graph_executor_.IsCapturedFor(graph_key_)) {
      launch_captured_graph();
    } else {
      const bool ok =
          graph_executor_.TryCapture(arena_.stream, graph_key_, [&]() {
            ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos,
                              compute_logits);
          });
      if (ok) {
        launch_captured_graph();
      } else {
        ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos,
                          compute_logits);
      }
    }
  } else {
    ExecuteDecodeStep(arena_, weights_, policy_, token_id, pos, compute_logits);
  }

  if (!compute_logits) {
    return 0;
  }

  std::uint32_t next_token_id = 0;
  HIP_CHECK(hipMemcpyAsync(&next_token_id, d_out_token, sizeof(std::uint32_t),
                           hipMemcpyDeviceToHost, arena_.stream));
  HIP_CHECK(hipStreamSynchronize(arena_.stream));

  return next_token_id;
}

}  // namespace strix::hip
#endif  // defined(ENGINE_ENABLE_HIP)
