#ifndef STRIX_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
#define STRIX_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_

#include <cstdint>

#include "src/models/qwen/tokenizer.hpp"

namespace strix::models {
struct QwenModelWeights;
}

namespace strix::hip {

class QwenGpuArena;
struct QwenExecutionPolicy;

/// Emits pure decode route decisions before entering graph capture. This must
/// never be called from the capture callback.
void EmitDecodeRouteTelemetry(const models::QwenModelWeights& weights,
                              const QwenExecutionPolicy& policy);

/// Runs one graph-capture-safe decode layer stack using stable arena storage.
void ExecuteDecodeStep(QwenGpuArena& arena,
                       const models::QwenModelWeights& weights,
                       const QwenExecutionPolicy& policy,
                       tokenization::TokenId token_id, std::uint32_t pos,
                       bool compute_logits);

}  // namespace strix::hip

#endif  // STRIX_MODELS_QWEN_HIP_DETAIL_DECODE_STEP_HPP_
