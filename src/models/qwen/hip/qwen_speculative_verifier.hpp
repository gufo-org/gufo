#ifndef GUFO_MODELS_QWEN_HIP_QWEN_SPECULATIVE_VERIFIER_HPP_
#define GUFO_MODELS_QWEN_HIP_QWEN_SPECULATIVE_VERIFIER_HPP_

#include <memory>

#include "src/core/speculative/speculative_verifier.hpp"

namespace gufo::hip {

#if defined(ENGINE_ENABLE_HIP)

class QwenGpuExecutor;

/// Binds the model-owned Qwen GPU executor to the core-neutral speculative
/// verifier through the core ISpeculativeTargetExecutor contract. The verifier
/// owns the adapter for its lifetime.
[[nodiscard]] std::unique_ptr<speculative::SpeculativeVerifier>
CreateQwenGpuSpeculativeVerifier(
    QwenGpuExecutor& target_executor,
    std::unique_ptr<speculative::IDraftBackend> draft_backend,
    speculative::SpeculativeOptions options = {});

/// Component-cache slots own the executor and the draft backend and outlive
/// the bound runner.
[[nodiscard]] std::unique_ptr<speculative::SpeculativeVerifier>
CreateQwenGpuSpeculativeVerifier(QwenGpuExecutor& target_executor,
                                 speculative::IDraftBackend& draft_backend,
                                 speculative::SpeculativeOptions options = {});

#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_QWEN_SPECULATIVE_VERIFIER_HPP_
