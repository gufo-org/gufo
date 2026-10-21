#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_

#include <cstdint>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/ssm.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen/vision/device_input.hpp"

namespace gufo::models {
struct QwenModelWeights;
}

namespace gufo::hip {

class QwenGpuArena;
struct QwenExecutionPolicy;

/// Routed expert formats that decode sends to the vendored llama.cpp vector
/// kernels (qfn_mmq_moe_gated_vec and qfn_mmq_moe_vec).
[[nodiscard]] constexpr bool IsMmqRoutedFormat(core::GgmlType type) noexcept {
  return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kQ4_K;
}

/// Emits pure decode route decisions before entering graph capture. This must
/// never be called from the capture callback.
void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy);

/// Runs one graph-capture-safe decode layer stack using stable arena storage.
/// `recurrent_state` is the pass's QwenGpuArena::BeginRecurrentWrite result;
/// a captured graph bakes its pointers, so the graph key must carry
/// QwenGpuArena::RecurrentIoIdentity.
void ExecuteDecodeStep(
    QwenGpuArena& arena, const models::QwenModelWeights& weights,
    const QwenExecutionPolicy& policy, tokenization::TokenId token_id,
    std::uint32_t pos, bool compute_logits,
    QwenRecurrentStateIo recurrent_state,
    models::qwen::vision::DeviceInput* image_input = nullptr);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
