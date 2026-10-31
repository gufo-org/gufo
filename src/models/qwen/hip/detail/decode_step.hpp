#ifndef GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
#define GUFO_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_

#include <cstdint>

#include "src/families/qwen/kernels/ops/ssm.hpp"
#include "src/families/qwen/text/tokenizer.hpp"
#include "src/families/qwen/vision/device_input.hpp"

namespace gufo::models {
struct QwenModelWeights;
}

namespace gufo::hip {

class QwenGpuArena;
struct QwenExecutionPolicy;

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
